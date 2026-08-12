"""Agent-layer contract tests.

The deterministic tools are covered by `test_pipeline.py`. What is covered here
is the *handoff*: that a model proposing physical constraints in tool-call form
reaches the adjudication functions intact, and that what comes back is a
verdict Python computed rather than anything the model asserted.

These run with `FunctionModel`, a scripted stand-in that emits a fixed sequence
of tool calls. No API key, no network, no non-determinism -- so the agent layer
is exercised on every CI run rather than only when someone has credentials.

That matters beyond convenience. The claim the architecture rests on is that
the model cannot state a statistic. A test that only ever runs against a live
model cannot establish this, because a live model's silence on any given run is
not evidence. Scripting the model lets us assert the stronger property: even
when the model proposes a constraint that is wrong, or narrates a violation
rate of its own invention, the adjudicated result is the one Python computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dra.agent import AgentDeps, build_agent

pydantic_ai = pytest.importorskip("pydantic_ai", reason="agent extra not installed")

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402


@pytest.fixture
def press_shop() -> pd.DataFrame:
    """A miniature of the demo dataset carrying three of its planted faults:
    negative absolute pressure, a broken torque/speed/power identity, and
    sub-meters that stop summing to the declared total half-way through."""
    rng = np.random.default_rng(0)
    n = 900
    ts = pd.date_range("2026-01-01", periods=n, freq="10s")
    speed = 1500 + rng.normal(0, 8, n)
    torque = 40 + rng.normal(0, 1.2, n)
    power = torque * speed * 2 * np.pi / 60 / 1000

    pressure = 4.2 + rng.normal(0, 0.15, n)
    pressure[:9] = -1.0                      # 1% physically impossible

    a, b, c = (30 + rng.normal(0, 1, n), 25 + rng.normal(0, 1, n), 20 + rng.normal(0, 1, n))
    total = a + b + c
    total[n // 2:] *= 1.18                   # meter drift after replacement

    return pd.DataFrame({
        "timestamp": ts, "speed_rpm": speed, "torque_nm": torque, "power_kw": power,
        "coolant_pressure_bar": pressure,
        "station_a_units": a, "station_b_units": b, "station_c_units": c,
        "line_total_units": total,
    })


def _scripted(calls: list[tuple[str, dict]]):
    """Build a FunctionModel that emits `calls` one per turn, then stops.

    Mirrors how a real run unfolds: one hypothesis per turn, each informed by
    the previous verdict, rather than a single batched guess.
    """
    def fn(messages, info: AgentInfo) -> ModelResponse:
        turn = sum(1 for m in messages if isinstance(m, ModelResponse))
        if turn < len(calls):
            name, args = calls[turn]
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[TextPart("Assessment complete.")])
    return FunctionModel(fn)


def _run(df: pd.DataFrame, calls: list[tuple[str, dict]]) -> AgentDeps:
    deps = AgentDeps(df=df, checks=[])
    agent = build_agent(model=_scripted(calls))
    agent.run_sync("Assess this dataset.", deps=deps)
    return deps


# --------------------------------------------------------------------------
# The handoff itself
# --------------------------------------------------------------------------

def test_proposed_constraints_reach_the_adjudicator(press_shop):
    """A range hypothesis on absolute pressure is adjudicated, not echoed."""
    deps = _run(press_shop, [
        ("test_range", {"column": "coolant_pressure_bar", "lo": 0.0, "hi": None,
                        "rationale": "Absolute pressure cannot be negative."}),
    ])
    assert len(deps.checks) == 1
    check = deps.checks[0]
    assert check.holds is False
    assert check.violation_rate == pytest.approx(9 / 900, abs=1e-6)


def test_verdict_is_computed_not_asserted(press_shop):
    """The load-bearing test.

    The scripted model proposes a constraint the data satisfies. If any part of
    the pipeline let the model's framing colour the outcome, a hypothesis
    phrased as an accusation would come back as a violation. It must not.
    """
    deps = _run(press_shop, [
        ("test_range", {"column": "speed_rpm", "lo": 0.0, "hi": 3000.0,
                        "rationale": "Catastrophic overspeed: the line trips well below this."}),
    ])
    check = deps.checks[0]
    assert check.holds is True
    assert check.violation_rate == 0.0


def test_algebraic_identity_between_torque_speed_and_power(press_shop):
    """Shaft power = torque x angular velocity. Invisible to any profiler:
    all three columns are individually well-behaved."""
    deps = _run(press_shop, [
        ("test_identity", {
            "expression": "power_kw - torque_nm * speed_rpm * 2 * 3.141592653589793 / 60 / 1000",
            "tolerance": 0.01,
            "rationale": "Mechanical power is torque times angular velocity.",
        }),
    ])
    assert deps.checks[0].holds is True


def test_conservation_catches_the_drifting_sub_meter(press_shop):
    """Stations must sum to the line total. They stop doing so at the midpoint."""
    deps = _run(press_shop, [
        ("test_conservation", {
            "parts": ["station_a_units", "station_b_units", "station_c_units"],
            "total": "line_total_units", "tolerance": 0.02,
            "rationale": "Sub-metered stations must sum to the declared line total.",
        }),
    ])
    check = deps.checks[0]
    assert check.holds is False
    assert check.violation_rate == pytest.approx(0.5, abs=0.02)


# --------------------------------------------------------------------------
# Failure modes a live model will actually produce
# --------------------------------------------------------------------------

def test_hypothesis_on_a_nonexistent_column_does_not_crash_the_run(press_shop):
    """Models hallucinate column names. The tool must return a failed check the
    agent can read and revise from, not raise and abort the assessment."""
    deps = _run(press_shop, [
        ("test_range", {"column": "vibration_mm_s", "lo": 0.0, "hi": 10.0,
                        "rationale": "Assumed an ISO 10816 vibration tag exists."}),
        ("test_range", {"column": "coolant_pressure_bar", "lo": 0.0, "hi": None,
                        "rationale": "Revised: use the tag that does exist."}),
    ])
    assert len(deps.checks) == 2
    assert deps.checks[0].holds is False
    assert deps.checks[1].violation_rate == pytest.approx(9 / 900, abs=1e-6)


def test_malformed_identity_expression_is_contained(press_shop):
    """A syntactically invalid expression is surfaced as a failed check rather
    than propagating an exception out of the tool call."""
    deps = _run(press_shop, [
        ("test_identity", {"expression": "power_kw - -- torque_nm *", "tolerance": 0.01,
                           "rationale": "Malformed on purpose."}),
    ])
    assert deps.checks[0].holds is False


def test_every_check_carries_its_rationale(press_shop):
    """Traceability: a client disputing a finding must be able to see the
    physical argument that motivated the test, not just the number."""
    deps = _run(press_shop, [
        ("test_range", {"column": "coolant_pressure_bar", "lo": 0.0, "hi": None,
                        "rationale": "Absolute pressure cannot be negative."}),
        ("test_conservation", {
            "parts": ["station_a_units", "station_b_units", "station_c_units"],
            "total": "line_total_units", "tolerance": 0.02,
            "rationale": "Sub-metered stations must sum to the declared line total.",
        }),
    ])
    assert all(c.rationale.strip() for c in deps.checks)
    assert all(c.columns for c in deps.checks)


def test_agent_exposes_the_expected_tool_surface():
    """The tool set is the contract between the two layers. Adding a tool that
    returns a model-authored number would break the architecture silently, so
    the surface is pinned."""
    captured: dict = {}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        captured["tools"] = {t.name for t in info.function_tools}
        return ModelResponse(parts=[TextPart("done")])

    agent = build_agent(model=FunctionModel(fn))
    agent.run_sync("Assess.", deps=AgentDeps(df=pd.DataFrame({"x": [1.0]}), checks=[]))

    assert captured["tools"] == {
        "profile", "test_range", "test_monotonic", "test_conservation",
        "test_rate_limit", "test_identity", "trace_lineage",
    }
