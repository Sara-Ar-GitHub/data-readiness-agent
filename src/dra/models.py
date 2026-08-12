"""Typed contract for every artefact the agent produces.

Nothing reaches the report unless it passes through one of these models,
which is what keeps LLM-generated prose from smuggling in invented numbers:
all numeric fields are populated by deterministic tools only.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    BLOCKER = "blocker"      # makes downstream modelling impossible
    MAJOR = "major"          # materially degrades any model built on this data
    MINOR = "minor"          # worth fixing, not on the critical path
    INFO = "info"


class Dimension(str, Enum):
    """Data-readiness dimensions. Deliberately aligned with the vocabulary
    used in EU DIH maturity assessments so findings map onto a roadmap."""

    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    INTEROPERABILITY = "interoperability"
    TRACEABILITY = "traceability"


class ConstraintKind(str, Enum):
    RANGE = "range"
    MONOTONIC = "monotonic"
    CONSERVATION = "conservation"
    RATE_LIMIT = "rate_limit"
    IDENTITY = "identity"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    missing_rate: float
    n_unique: int
    is_constant: bool
    stuck_max_run: int = Field(
        0, description="Longest run of identical consecutive values (sensor stuck-at)."
    )
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    outlier_rate: float | None = Field(None, description="Fraction beyond 3x MAD.")
    near_key_duplicate_rate: float | None = Field(
        None,
        description="For a column unique enough to be a business key, the fraction of "
                    "rows whose value is not unique. None if the column is not key-like.",
    )
    inferred_unit: str | None = Field(
        None, description="Guessed from the column name only. Never trusted, only flagged."
    )


class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    duplicate_row_rate: float
    timestamp_column: str | None = None
    median_sampling_seconds: float | None = None
    sampling_regularity: float | None = Field(
        None, description="Fraction of intervals within 10% of the median interval."
    )
    duplicate_timestamp_rate: float | None = None
    gap_count: int | None = Field(None, description="Intervals exceeding 5x the median.")
    columns: list[ColumnProfile]


class ConstraintCheck(BaseModel):
    """A physical hypothesis proposed by the LLM, adjudicated by Python."""

    name: str
    kind: ConstraintKind
    columns: list[str]
    rationale: str = Field(description="Why this constraint should hold, in domain terms.")
    holds: bool
    violation_rate: float
    evidence: dict[str, Any] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    source: list[str]
    target: str
    relation: str
    confidence: float


class Finding(BaseModel):
    dimension: Dimension
    severity: Severity
    columns: list[str]
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    remediation: str


class RoadmapItem(BaseModel):
    priority: int
    action: str
    effort: str = Field(description="low | medium | high")
    unlocks: str = Field(description="What modelling capability this enables once fixed.")
    addresses: list[str] = Field(default_factory=list)


class ReadinessReport(BaseModel):
    dataset_name: str
    sector: str | None = None
    profile: DatasetProfile
    constraints: list[ConstraintCheck] = Field(default_factory=list)
    lineage: list[LineageEdge] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    scores: dict[Dimension, float] = Field(
        default_factory=dict, description="0-5 maturity score per dimension."
    )
    overall_score: float = 0.0
    roadmap: list[RoadmapItem] = Field(default_factory=list)
    executive_summary: str = ""
    feasibility_verdict: str = Field(
        "", description="Go / conditional-go / no-go for the proposed modelling use case."
    )
