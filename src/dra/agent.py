"""Agent layer.

Division of labour, and the whole point of the architecture:

  deterministic Python  ->  every number, every verdict, every score
  the LLM               ->  reading column semantics to *propose* which
                            physical constraints are worth testing,
                            sequencing the tool calls, and writing prose

The LLM never sees a chance to assert a statistic. It proposes hypotheses;
the tools adjudicate them; the report renders only adjudicated results. This
also means the pipeline degrades gracefully: with no model configured, the
deterministic half still produces a complete readiness report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from dra.models import (
    ConstraintCheck,
    DatasetProfile,
    Dimension,
    Finding,
    ReadinessReport,
    RoadmapItem,
    Severity,
)
from dra.tools import lineage as lineage_tools
from dra.tools import physics, profiling
from dra.tools import quality as quality_tools

SYSTEM_PROMPT = """\
You are a data-readiness assessor working with small and medium manufacturers.
An SME has handed you an export from its historian or MES and wants to know
whether the data can support a modelling project.

You have deterministic tools. Use them. You must never state a statistic you
have not obtained from a tool call, and never estimate a violation rate.

Your distinctive contribution is domain reasoning: given the column names,
units and observed ranges, infer which *physical* constraints ought to hold
for this kind of equipment, and test them. Examples of the reasoning expected:
 - a rotating machine: shaft power should track torque times angular velocity
 - a PV array: power rises with irradiance and falls with cell temperature
 - a wind turbine: the implied power coefficient cannot exceed the Betz limit
 - sub-metered assets: the components must sum to the aggregate
 - anything with thermal mass: bounded rate of change

Propose constraints one at a time with an explicit physical rationale, test
them, and revise when a test fails for an uninteresting reason (wrong unit
assumption, wrong column pairing) rather than reporting a spurious violation.

Finish by judging feasibility: go, conditional-go, or no-go, with the single
most important blocker named. Address an operations manager, not a data
scientist.
"""


@dataclass
class AgentDeps:
    df: pd.DataFrame
    checks: list[ConstraintCheck]


def build_agent(model: str | None = None):
    """Construct the PydanticAI agent. Imported lazily so the deterministic
    pipeline has no hard dependency on any LLM SDK."""
    from pydantic_ai import Agent, RunContext

    model = model or os.getenv("DRA_MODEL", "anthropic:claude-sonnet-4-6")
    agent = Agent(model, deps_type=AgentDeps, system_prompt=SYSTEM_PROMPT)

    @agent.tool
    def profile(ctx: RunContext[AgentDeps]) -> DatasetProfile:
        """Full structural and statistical profile of the dataset."""
        return profiling.profile_dataset(ctx.deps.df)

    @agent.tool
    def test_range(ctx: RunContext[AgentDeps], column: str, lo: float | None,
                   hi: float | None, rationale: str) -> ConstraintCheck:
        """Test that a column stays inside physically admissible bounds."""
        c = physics.check_range(ctx.deps.df, column, lo, hi, rationale)
        ctx.deps.checks.append(c)
        return c

    @agent.tool
    def test_monotonic(ctx: RunContext[AgentDeps], x: str, y: str, direction: str,
                       rationale: str) -> ConstraintCheck:
        """Test that y varies monotonically with x. direction: increasing|decreasing."""
        c = physics.check_monotonic(ctx.deps.df, x, y, direction, rationale)
        ctx.deps.checks.append(c)
        return c

    @agent.tool
    def test_conservation(ctx: RunContext[AgentDeps], parts: list[str], total: str,
                          tolerance: float, rationale: str) -> ConstraintCheck:
        """Test that component columns sum to an aggregate column."""
        c = physics.check_conservation(ctx.deps.df, parts, total, tolerance, rationale)
        ctx.deps.checks.append(c)
        return c

    @agent.tool
    def test_rate_limit(ctx: RunContext[AgentDeps], column: str, max_delta_per_second: float,
                        timestamp_column: str, rationale: str) -> ConstraintCheck:
        """Test that a column cannot change faster than physical inertia allows."""
        c = physics.check_rate_limit(ctx.deps.df, column, max_delta_per_second,
                                     timestamp_column, rationale)
        ctx.deps.checks.append(c)
        return c

    @agent.tool
    def test_identity(ctx: RunContext[AgentDeps], expression: str, tolerance: float,
                      rationale: str) -> ConstraintCheck:
        """Test an algebraic identity, written as a residual expected to be zero,
        e.g. 'power_kw - torque_nm * speed_rpm * 2 * pi / 60 / 1000'."""
        c = physics.check_identity(ctx.deps.df, expression, tolerance, rationale)
        ctx.deps.checks.append(c)
        return c

    @agent.tool
    def trace_lineage(ctx: RunContext[AgentDeps]) -> list:
        """Infer which columns are derived from others (leakage risk)."""
        return lineage_tools.infer_lineage(ctx.deps.df)

    return agent


def deterministic_report(df: pd.DataFrame, name: str, sector: str | None = None) -> ReadinessReport:
    """The LLM-free baseline. Runs anywhere, produces a complete report, and
    serves as the ground truth the agent's additions are layered onto."""
    prof = profiling.profile_dataset(df, name)
    lin = lineage_tools.infer_lineage(df)
    findings = quality_tools.derive_findings(prof, [], lin)
    scores = quality_tools.score_dimensions(findings)
    return ReadinessReport(
        dataset_name=name, sector=sector, profile=prof, lineage=lin, findings=findings,
        scores=scores, overall_score=quality_tools.overall_score(scores),
        roadmap=build_roadmap(findings),
    )


def build_roadmap(findings: list[Finding]) -> list[RoadmapItem]:
    """Order remediation by what unblocks the most downstream capability.

    Blockers first, then whichever dimension accumulated the most damage --
    the sequencing an SME needs when it can only fund one fix this quarter.
    """
    order = {Severity.BLOCKER: 0, Severity.MAJOR: 1, Severity.MINOR: 2, Severity.INFO: 3}
    effort = {Dimension.COMPLETENESS: "medium", Dimension.VALIDITY: "medium",
              Dimension.CONSISTENCY: "high", Dimension.TIMELINESS: "low",
              Dimension.UNIQUENESS: "low", Dimension.INTEROPERABILITY: "low",
              Dimension.TRACEABILITY: "low"}
    unlocks = {
        Dimension.COMPLETENESS: "any supervised model on the affected tags",
        Dimension.VALIDITY: "trustworthy anomaly detection",
        Dimension.CONSISTENCY: "physics-informed and hybrid modelling",
        Dimension.TIMELINESS: "time-series forecasting and drift monitoring",
        Dimension.UNIQUENESS: "honest train/test separation",
        Dimension.INTEROPERABILITY: "merging data across sites or vendors",
        Dimension.TRACEABILITY: "leakage-free feature selection",
    }
    ranked = sorted(findings, key=lambda f: (order[f.severity], f.dimension.value))
    items: list[RoadmapItem] = []
    seen: set[tuple] = set()
    for f in ranked:
        if f.severity == Severity.INFO:
            continue
        key = (f.dimension, f.severity)
        if key in seen:
            continue
        seen.add(key)
        items.append(RoadmapItem(
            priority=len(items) + 1, action=f.remediation, effort=effort[f.dimension],
            unlocks=unlocks[f.dimension], addresses=f.columns,
        ))
    return items[:8]
