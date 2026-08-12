"""Deterministic profiling. No LLM involvement whatsoever."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from dra.models import ColumnProfile, DatasetProfile

# Unit hints are surfaced, never trusted. Their purpose is to let the agent
# ask the SME "is this really kW or kWh?" -- the single most common cause of
# silent modelling failure on industrial data.
_UNIT_PATTERNS: list[tuple[str, str]] = [
    (r"(^|[_\W])(kwh|mwh|wh)($|[_\W])", "energy"),
    (r"(^|[_\W])(kw|mw|watt|power|puissance)($|[_\W])", "power"),
    (r"(^|[_\W])(temp|degc|degf|celsius|temperature)($|[_\W])", "temperature"),
    (r"(^|[_\W])(bar|psi|pa|pressure|pression)($|[_\W])", "pressure"),
    # Linear speed is matched before rotational: a bare "speed" on a vehicle tag
    # is kph, not rpm. Cheap to get right, and getting it wrong is exactly the
    # unit confusion this whole heuristic exists to surface.
    (r"(^|[_\W])(kph|kmh|mph|ms|knots)($|[_\W])", "linear_speed"),
    (r"(^|[_\W])(rpm|omega)($|[_\W])", "rotational_speed"),
    (r"(^|[_\W])(speed|vitesse)($|[_\W])", "speed_unspecified"),
    (r"(^|[_\W])(km|mi|miles|metres|meters|distance)($|[_\W])", "distance"),
    (r"(^|[_\W])(kg|tonnes|lbs|mass|weight|payload)($|[_\W])", "mass"),
    (r"(^|[_\W])(l|litres|liters|gal|volume)($|[_\W])", "volume"),
    (r"(^|[_\W])(nm|torque|couple)($|[_\W])", "torque"),
    (r"(^|[_\W])(amp|ampere|current|courant)($|[_\W])", "current"),
    (r"(^|[_\W])(volt|voltage|tension)($|[_\W])", "voltage"),
    (r"(^|[_\W])(irr|ghi|dni|irradiance)($|[_\W])", "irradiance"),
    (r"(^|[_\W])(flow|debit|m3h)($|[_\W])", "flow"),
]


def _infer_unit(name: str) -> str | None:
    low = name.lower()
    for pattern, unit in _UNIT_PATTERNS:
        if re.search(pattern, low):
            return unit
    return None


def _max_identical_run(s: pd.Series) -> int:
    """Longest run of identical consecutive values.

    A long run on an analogue sensor means a stuck transmitter or a frozen
    tag in the historian -- invisible to a missing-value check, and lethal
    for anomaly detection because the model learns the frozen value.
    """
    if len(s) == 0:
        return 0
    v = s.to_numpy()
    changed = np.ones(len(v), dtype=bool)
    changed[1:] = v[1:] != v[:-1]
    group = np.cumsum(changed)
    counts = np.bincount(group)
    return int(counts.max()) if len(counts) else 0


def _outlier_rate(s: pd.Series) -> float | None:
    x = s.dropna().to_numpy(dtype=float)
    if len(x) < 10:
        return None
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return 0.0
    return float(np.mean(np.abs(x - med) / (1.4826 * mad) > 3.0))


def _near_key_duplicate_rate(s: pd.Series, n_rows: int) -> float | None:
    """Duplicate rate for a column that looks like a business key.

    Exact-duplicate-row detection misses the case that actually costs an SME
    money: a consignment, work-order or batch identifier that repeats while the
    surrounding measurements differ. The row is not a duplicate, so no row-level
    check fires -- but the entity it claims to identify has been counted twice,
    and any join or group-by on that key silently double-counts.

    "Key-like" is decided by cardinality, not by name. A column unique across
    at least 80% of rows is being used as an identifier whether or not anyone
    declared it as one, and that threshold does not depend on the column being
    called `id` -- which matters, because the naming conventions differ in every
    export this tool will ever see.
    """
    # Floats are excluded outright. A continuous measurement is never a business
    # key, but a frozen transmitter makes one look like a near-unique column with
    # a few repeats -- which is a stuck-sensor finding, already reported as such,
    # and would be a false uniqueness finding here.
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return None
    v = s.dropna()
    if n_rows < 50 or len(v) < 0.5 * n_rows:
        return None
    ratio = v.nunique() / len(v)
    if ratio < 0.8 or ratio == 1.0:
        return None
    return float(v.duplicated(keep=False).mean())


def parse_timestamps(s: pd.Series) -> pd.Series:
    """Parse a timestamp column, trying cheap paths before the expensive one.

    ``format="mixed"`` infers a format per element and is roughly two orders
    of magnitude slower than a vectorised parse. Historian exports are almost
    always uniformly formatted, so it is worth attempting the vectorised path
    first and reserving element-wise inference for genuinely ragged columns.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    sample = s.dropna().head(200)
    if not len(sample):
        return pd.to_datetime(s, errors="coerce")
    for fmt in ("ISO8601", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d"):
        try:
            probe = pd.to_datetime(sample, format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        if probe.notna().mean() > 0.95:
            return pd.to_datetime(s, format=fmt, errors="coerce")
    try:
        return pd.to_datetime(s, errors="coerce")
    except (ValueError, TypeError):
        return pd.to_datetime(s, errors="coerce", format="mixed")


def _detect_timestamp(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if re.search(r"(time|date|ts|horodat)", str(col).lower()):
            # Probe on a sample first: no point parsing a million rows only to
            # discover the column is not a timestamp at all.
            probe = parse_timestamps(df[col].dropna().head(500))
            if len(probe) and probe.notna().mean() > 0.9:
                return col
    return None


def profile_dataset(df: pd.DataFrame, name: str = "dataset") -> DatasetProfile:
    """Full structural and statistical profile of a tabular industrial dataset."""
    ts_col = _detect_timestamp(df)
    median_dt = regularity = dup_ts = None
    gap_count = None

    if ts_col is not None:
        ts = parse_timestamps(df[ts_col]).dropna().sort_values()
        if len(ts) > 2:
            deltas = ts.diff().dt.total_seconds().dropna()
            deltas = deltas[deltas > 0]
            if len(deltas):
                median_dt = float(deltas.median())
                regularity = float(np.mean(np.abs(deltas - median_dt) <= 0.1 * median_dt))
                gap_count = int((deltas > 5 * median_dt).sum())
        dup_ts = float(ts.duplicated().mean()) if len(ts) else 0.0

    columns: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        columns.append(
            ColumnProfile(
                name=str(col),
                dtype=str(s.dtype),
                missing_rate=float(s.isna().mean()),
                n_unique=int(s.nunique(dropna=True)),
                is_constant=bool(s.nunique(dropna=True) <= 1),
                stuck_max_run=_max_identical_run(s.dropna()) if numeric else 0,
                min=float(s.min()) if numeric and s.notna().any() else None,
                max=float(s.max()) if numeric and s.notna().any() else None,
                mean=float(s.mean()) if numeric and s.notna().any() else None,
                std=float(s.std()) if numeric and s.notna().any() else None,
                outlier_rate=_outlier_rate(s) if numeric else None,
                inferred_unit=_infer_unit(str(col)),
                near_key_duplicate_rate=_near_key_duplicate_rate(s, len(df)),
            )
        )

    return DatasetProfile(
        n_rows=len(df),
        n_cols=int(df.shape[1]),
        duplicate_row_rate=float(df.duplicated().mean()) if len(df) else 0.0,
        timestamp_column=ts_col,
        median_sampling_seconds=median_dt,
        sampling_regularity=regularity,
        duplicate_timestamp_rate=dup_ts,
        gap_count=gap_count,
        columns=columns,
    )
