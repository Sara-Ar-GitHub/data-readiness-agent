"""Column-level lineage inference.

SMEs almost never have documented lineage. What they have is a CSV exported
from a historian in which some columns are raw tags and others are derived
quantities computed upstream -- often by a formula nobody remembers. Training
a model on both an input and its own derivative is a classic leakage failure,
so this module recovers the dependency structure from the data itself.

Performance note: the naive form of this search fits least squares over every
combination of columns, which is cubic in the column count and unusable on a
real historian export (60 tags took roughly ten minutes). Two observations
remove almost all of that cost:

  * For a single parent, R-squared is exactly the squared Pearson correlation.
    No regression is needed at all -- one correlation matrix answers every
    single-parent question simultaneously.
  * Detecting a relation that holds at R-squared >= 0.995 does not require
    every row. A few thousand rows settle it, and the residual risk is a
    missed relation in a dataset with extreme heteroscedasticity, which is
    an acceptable trade against a tool nobody waits for.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from dra.models import LineageEdge

MAX_LINEAGE_ROWS = 5_000
MAX_LINEAGE_COLS = 120
TOP_K_CANDIDATES = 8


def _numeric_frame(df: pd.DataFrame, max_rows: int = MAX_LINEAGE_ROWS) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    num = num.loc[:, num.nunique(dropna=True) > 1]
    if num.shape[1] > MAX_LINEAGE_COLS:
        num = num.iloc[:, :MAX_LINEAGE_COLS]
    if len(num) > max_rows:
        # Systematic rather than random sampling: preserves temporal coverage,
        # so a relation that only holds in one operating regime is still seen.
        num = num.iloc[:: max(1, len(num) // max_rows)]
    return num


def _corr(num: pd.DataFrame) -> pd.DataFrame:
    """One vectorised pass replaces the pairwise dropna loop."""
    return num.corr(method="pearson", min_periods=30)


def find_duplicate_columns(
    df: pd.DataFrame, threshold: float = 0.999, _num: pd.DataFrame | None = None
) -> list[LineageEdge]:
    """Columns that are near-perfect copies, possibly rescaled (unit variants)."""
    num = _numeric_frame(df) if _num is None else _num
    if num.shape[1] < 2:
        return []
    corr = _corr(num).abs()
    edges: list[LineageEdge] = []
    cols = list(num.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.at[a, b]
            if not np.isfinite(r) or r < threshold:
                continue
            sub = num[[a, b]].dropna()
            if len(sub) < 30:
                continue
            ratio = float((sub[b] / sub[a].replace(0, np.nan)).median())
            relation = "duplicate" if abs(ratio - 1) < 1e-6 else f"scaled (x{ratio:.4g})"
            edges.append(LineageEdge(source=[str(a)], target=str(b),
                                     relation=relation, confidence=float(r)))
    return edges


def find_linear_derivations(
    df: pd.DataFrame,
    max_parents: int = 2,
    r2_threshold: float = 0.995,
    _num: pd.DataFrame | None = None,
) -> list[LineageEdge]:
    """Columns explainable as a linear combination of others.

    Single-parent relations are read straight off the correlation matrix.
    Two-parent relations are searched only among each target's most correlated
    neighbours: a documented heuristic that can in principle miss a pair of
    individually uncorrelated columns which jointly explain the target, in
    exchange for making the search tractable. Parents are capped at two
    because beyond that a high R-squared is more likely overfitting than
    genuine derivation, and a false lineage claim in a client report is worse
    than a missing one.
    """
    num = _numeric_frame(df) if _num is None else _num
    if num.shape[1] < 2 or len(num) < 50:
        return []

    corr = _corr(num)
    edges: list[LineageEdge] = []
    single_thresh = np.sqrt(r2_threshold)

    for target in num.columns:
        others = [c for c in num.columns if c != target]
        r_row = corr[target].drop(labels=[target], errors="ignore").abs()

        # One parent: R-squared is r-squared. No fit required.
        best_single = r_row.idxmax() if r_row.notna().any() else None
        if best_single is not None and r_row[best_single] >= single_thresh:
            edges.append(LineageEdge(
                source=[str(best_single)], target=str(target),
                relation="linear combination", confidence=float(r_row[best_single] ** 2)))
            continue

        if max_parents < 2:
            continue

        # Two parents: search only the most correlated neighbours.
        candidates = [c for c in r_row.sort_values(ascending=False).index[:TOP_K_CANDIDATES]
                      if c in others]
        if len(candidates) < 2:
            continue
        sub = num[[target, *candidates]].dropna()
        if len(sub) < 50:
            continue
        y = sub[target].to_numpy(dtype=float)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot == 0:
            continue

        best: tuple[float, list[str]] | None = None
        for combo in itertools.combinations(candidates, 2):
            X = np.column_stack([sub[c].to_numpy(dtype=float) for c in combo]
                                + [np.ones(len(sub))])
            try:
                coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = y - X @ coef
            r2 = 1.0 - float((resid**2).sum()) / ss_tot
            if r2 >= r2_threshold and (best is None or r2 > best[0]):
                best = (r2, [str(c) for c in combo])
        if best is not None:
            edges.append(LineageEdge(source=best[1], target=str(target),
                                     relation="linear combination", confidence=best[0]))
    return edges


def infer_lineage(df: pd.DataFrame) -> list[LineageEdge]:
    """Combine both detectors, then report each derived column exactly once.

    Without this pass the same derivation surfaces two or three times (a
    duplicate pair reported in both directions, a scaled copy also matching
    as a linear combination), which inflates the traceability penalty and
    makes the client report look mechanical.
    """
    num = _numeric_frame(df)          # computed once, shared by both detectors
    edges = find_duplicate_columns(df, _num=num) + find_linear_derivations(df, _num=num)

    # Prefer the simplest explanation: an exact copy or rescaling beats a
    # fitted linear combination, and fewer parents beats more.
    def rank(e: LineageEdge) -> tuple:
        simple = 0 if e.relation.startswith(("duplicate", "scaled")) else 1
        return (simple, len(e.source), -e.confidence)

    best: dict[str, LineageEdge] = {}
    for e in sorted(edges, key=rank):
        if e.target in best:
            continue
        if any(e.target in prior.source and prior.target in e.source
               for prior in best.values()):
            continue
        best[e.target] = e
    return list(best.values())
