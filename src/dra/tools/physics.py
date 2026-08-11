"""Adjudication of physical constraints.

The LLM proposes a constraint in domain terms ("shaft power should equal
torque times angular velocity"); these functions decide whether the data
supports it. The verdict is arithmetic, not linguistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from dra.models import ConstraintCheck, ConstraintKind

_SAFE_FUNCS = {
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
    "sin": np.sin, "cos": np.cos, "pi": np.pi,
}


def check_range(
    df: pd.DataFrame, column: str, lo: float | None, hi: float | None, rationale: str
) -> ConstraintCheck:
    """Values must lie inside a physically admissible interval.

    Distinct from statistical outlier detection: a value outside these bounds
    is not unusual, it is impossible, and therefore an instrumentation fault.
    """
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    viol = pd.Series(False, index=s.index)
    if lo is not None:
        viol |= s < lo
    if hi is not None:
        viol |= s > hi
    rate = float(viol.mean()) if len(s) else 0.0
    return ConstraintCheck(
        name=f"range({column})",
        kind=ConstraintKind.RANGE,
        columns=[column],
        rationale=rationale,
        holds=rate < 0.001,
        violation_rate=rate,
        evidence={
            "bounds": [lo, hi],
            "observed_min": float(s.min()) if len(s) else None,
            "observed_max": float(s.max()) if len(s) else None,
            "n_violations": int(viol.sum()),
        },
    )


def check_monotonic(
    df: pd.DataFrame, x: str, y: str, direction: str, rationale: str
) -> ConstraintCheck:
    """y should increase (or decrease) with x, e.g. PV power against irradiance.

    Uses Spearman rather than Pearson because the physical relation is
    monotone but rarely linear.
    """
    sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 30:
        return ConstraintCheck(
            name=f"monotonic({x}->{y})", kind=ConstraintKind.MONOTONIC, columns=[x, y],
            rationale=rationale, holds=False, violation_rate=1.0,
            evidence={"error": "insufficient paired observations", "n": len(sub)},
        )
    rho, p = stats.spearmanr(sub[x], sub[y])
    expected = 1.0 if direction == "increasing" else -1.0
    holds = bool(np.sign(rho) == expected and abs(rho) > 0.3 and p < 0.01)
    return ConstraintCheck(
        name=f"monotonic({x}->{y})",
        kind=ConstraintKind.MONOTONIC,
        columns=[x, y],
        rationale=rationale,
        holds=holds,
        violation_rate=0.0 if holds else 1.0,
        evidence={"spearman_rho": float(rho), "p_value": float(p),
                  "expected_direction": direction, "n": len(sub)},
    )


def check_conservation(
    df: pd.DataFrame, parts: list[str], total: str, tolerance: float, rationale: str
) -> ConstraintCheck:
    """Sum of components must equal the aggregate within tolerance.

    The workhorse check on any plant with sub-metering: string inverters
    against site meter, line stations against total throughput. Failures
    localise the faulty meter rather than merely flagging the dataset.
    """
    cols = [*parts, total]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if not len(sub):
        return ConstraintCheck(
            name=f"conservation({total})", kind=ConstraintKind.CONSERVATION, columns=cols,
            rationale=rationale, holds=False, violation_rate=1.0,
            evidence={"error": "no complete rows"},
        )
    lhs = sub[parts].sum(axis=1)
    rhs = sub[total]
    denom = rhs.abs().clip(lower=1e-9)
    rel_err = (lhs - rhs).abs() / denom
    viol = rel_err > tolerance
    return ConstraintCheck(
        name=f"conservation({total})",
        kind=ConstraintKind.CONSERVATION,
        columns=cols,
        rationale=rationale,
        holds=bool(viol.mean() < 0.01),
        violation_rate=float(viol.mean()),
        evidence={"tolerance": tolerance, "median_relative_error": float(rel_err.median()),
                  "p95_relative_error": float(rel_err.quantile(0.95)),
                  "mean_signed_bias": float((lhs - rhs).mean())},
    )


def check_rate_limit(
    df: pd.DataFrame, column: str, max_delta_per_second: float,
    timestamp_column: str, rationale: str
) -> ConstraintCheck:
    """Physical inertia bounds how fast a quantity can change.

    A thermal mass cannot move 40 degrees in one second. Violations are
    almost always transmission glitches or historian interpolation artefacts,
    not process events -- an important distinction for an SME deciding
    whether to trust its historian.
    """
    sub = df[[timestamp_column, column]].copy()
    sub[timestamp_column] = pd.to_datetime(sub[timestamp_column], errors="coerce", format="mixed")
    sub[column] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna().sort_values(timestamp_column)
    if len(sub) < 3:
        return ConstraintCheck(
            name=f"rate_limit({column})", kind=ConstraintKind.RATE_LIMIT,
            columns=[column], rationale=rationale, holds=False, violation_rate=1.0,
            evidence={"error": "insufficient observations"},
        )
    dt = sub[timestamp_column].diff().dt.total_seconds()
    rate = (sub[column].diff().abs() / dt.replace(0, np.nan)).dropna()
    viol = rate > max_delta_per_second
    return ConstraintCheck(
        name=f"rate_limit({column})",
        kind=ConstraintKind.RATE_LIMIT,
        columns=[column],
        rationale=rationale,
        holds=bool(viol.mean() < 0.001),
        violation_rate=float(viol.mean()),
        evidence={"limit_per_second": max_delta_per_second,
                  "observed_p999_rate": float(rate.quantile(0.999)),
                  "observed_max_rate": float(rate.max()),
                  "n_violations": int(viol.sum())},
    )


def check_identity(
    df: pd.DataFrame, expression: str, tolerance: float, rationale: str
) -> ConstraintCheck:
    """An algebraic identity between columns, e.g. 'power - torque*speed*2*pi/60'.

    The expression is evaluated with pandas.eval and must be written as a
    residual that should be zero. Only column names and a small whitelist of
    numpy functions are in scope -- the LLM cannot execute arbitrary code here.
    """
    try:
        residual = pd.eval(expression, local_dict={c: pd.to_numeric(df[c], errors="coerce")
                                                   for c in df.columns},
                           global_dict=_SAFE_FUNCS, engine="python")
        residual = pd.Series(residual).dropna()
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent as a failed check
        return ConstraintCheck(
            name=f"identity({expression})", kind=ConstraintKind.IDENTITY, columns=[],
            rationale=rationale, holds=False, violation_rate=1.0,
            evidence={"error": f"{type(exc).__name__}: {exc}"},
        )
    if not len(residual):
        return ConstraintCheck(
            name=f"identity({expression})", kind=ConstraintKind.IDENTITY, columns=[],
            rationale=rationale, holds=False, violation_rate=1.0,
            evidence={"error": "expression yielded no finite values"},
        )
    scale = float(np.abs(residual).median()) or 1.0
    viol = residual.abs() > tolerance * max(scale, 1.0)
    return ConstraintCheck(
        name=f"identity({expression})",
        kind=ConstraintKind.IDENTITY,
        columns=[c for c in df.columns if str(c) in expression],
        rationale=rationale,
        holds=bool(viol.mean() < 0.01),
        violation_rate=float(viol.mean()),
        evidence={"median_abs_residual": float(residual.abs().median()),
                  "p95_abs_residual": float(residual.abs().quantile(0.95))},
    )
