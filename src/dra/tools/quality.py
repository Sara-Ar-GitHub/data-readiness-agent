"""Rule-based findings and 0-5 maturity scoring.

Scoring is deterministic and auditable: the same dataset always yields the
same score, and every point deducted traces back to a named finding. An SME
that disputes its score can be shown exactly which rows caused it -- which is
the difference between a consulting deliverable and an opinion.
"""

from __future__ import annotations

from dra.models import (
    ConstraintCheck,
    DatasetProfile,
    Dimension,
    Finding,
    LineageEdge,
    Severity,
)

_PENALTY = {Severity.BLOCKER: 2.5, Severity.MAJOR: 1.2, Severity.MINOR: 0.4,
            Severity.INFO: 0.0}


def derive_findings(
    profile: DatasetProfile,
    constraints: list[ConstraintCheck] | None = None,
    lineage: list[LineageEdge] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    constraints = constraints or []
    lineage = lineage or []

    for col in profile.columns:
        if col.missing_rate > 0.5:
            findings.append(Finding(
                dimension=Dimension.COMPLETENESS, severity=Severity.BLOCKER,
                columns=[col.name],
                description=f"{col.name} is missing in {col.missing_rate:.0%} of rows.",
                evidence={"missing_rate": col.missing_rate},
                remediation="Trace the acquisition path for this tag. Above 50% missing, "
                            "imputation fabricates more signal than it recovers.",
            ))
        elif col.missing_rate > 0.05:
            findings.append(Finding(
                dimension=Dimension.COMPLETENESS, severity=Severity.MAJOR,
                columns=[col.name],
                description=f"{col.name} is missing in {col.missing_rate:.1%} of rows.",
                evidence={"missing_rate": col.missing_rate},
                remediation="Determine whether gaps are random or coincide with plant "
                            "states; the latter biases any model trained on this tag.",
            ))

        if col.is_constant:
            findings.append(Finding(
                dimension=Dimension.VALIDITY, severity=Severity.MINOR, columns=[col.name],
                description=f"{col.name} never varies and carries no information.",
                evidence={"n_unique": col.n_unique},
                remediation="Drop from the feature set, or confirm the sensor is live.",
            ))
        elif profile.n_rows and col.stuck_max_run > max(20, 0.05 * profile.n_rows):
            findings.append(Finding(
                dimension=Dimension.VALIDITY, severity=Severity.MAJOR, columns=[col.name],
                description=(f"{col.name} holds one identical value for up to "
                             f"{col.stuck_max_run} consecutive samples, indicating a "
                             "stuck transmitter or a frozen historian tag."),
                evidence={"max_identical_run": col.stuck_max_run},
                remediation="Cross-check against the instrument's maintenance log. Mask "
                            "frozen segments before training rather than dropping the tag.",
            ))

        if col.outlier_rate is not None and col.outlier_rate > 0.02:
            findings.append(Finding(
                dimension=Dimension.VALIDITY, severity=Severity.MINOR, columns=[col.name],
                description=f"{col.name} shows {col.outlier_rate:.1%} extreme values (>3 MAD).",
                evidence={"outlier_rate": col.outlier_rate, "min": col.min, "max": col.max},
                remediation="Separate genuine process excursions from acquisition faults "
                            "before deciding to clip or keep.",
            ))

        if col.inferred_unit:
            findings.append(Finding(
                dimension=Dimension.INTEROPERABILITY, severity=Severity.INFO,
                columns=[col.name],
                description=f"{col.name} appears to carry a {col.inferred_unit} quantity, "
                            "but no unit is declared anywhere in the schema.",
                evidence={"inferred_unit": col.inferred_unit},
                remediation="Attach explicit units and a semantic identifier. Undeclared "
                            "units are the most common cause of silent errors when merging "
                            "data across sites.",
            ))

    if profile.duplicate_row_rate > 0.001:
        findings.append(Finding(
            dimension=Dimension.UNIQUENESS, severity=Severity.MAJOR, columns=[],
            description=f"{profile.duplicate_row_rate:.2%} of rows are exact duplicates.",
            evidence={"duplicate_row_rate": profile.duplicate_row_rate},
            remediation="Deduplicate at ingestion; duplicates inflate apparent sample size "
                        "and leak between train and test splits.",
        ))

    if profile.timestamp_column is None:
        findings.append(Finding(
            dimension=Dimension.TIMELINESS, severity=Severity.BLOCKER, columns=[],
            description="No usable timestamp column was found.",
            evidence={},
            remediation="Without time ordering, no temporal model, no drift monitoring and "
                        "no causal analysis is possible. This is the first thing to fix.",
        ))
    else:
        if profile.sampling_regularity is not None and profile.sampling_regularity < 0.9:
            findings.append(Finding(
                dimension=Dimension.TIMELINESS, severity=Severity.MAJOR,
                columns=[profile.timestamp_column],
                description=(f"Only {profile.sampling_regularity:.0%} of sampling intervals "
                             "are close to the median rate."),
                evidence={"median_sampling_seconds": profile.median_sampling_seconds,
                          "regularity": profile.sampling_regularity},
                remediation="Resample onto a fixed grid with an explicit aggregation rule, "
                            "and record which samples were interpolated.",
            ))
        if profile.duplicate_timestamp_rate and profile.duplicate_timestamp_rate > 0.001:
            findings.append(Finding(
                dimension=Dimension.UNIQUENESS, severity=Severity.MAJOR,
                columns=[profile.timestamp_column],
                description=f"{profile.duplicate_timestamp_rate:.2%} of timestamps repeat.",
                evidence={"duplicate_timestamp_rate": profile.duplicate_timestamp_rate},
                remediation="Usually a clock reset or a merge of overlapping exports. "
                            "Resolve before treating the series as a single asset.",
            ))
        if profile.gap_count:
            findings.append(Finding(
                dimension=Dimension.TIMELINESS, severity=Severity.MINOR,
                columns=[profile.timestamp_column],
                description=f"{profile.gap_count} gaps exceed five sampling periods.",
                evidence={"gap_count": profile.gap_count},
                remediation="Check whether gaps align with shutdowns or with historian "
                            "outages; only the latter should be interpolated.",
            ))

    for check in constraints:
        if not check.holds:
            findings.append(Finding(
                dimension=Dimension.CONSISTENCY, severity=Severity.MAJOR,
                columns=check.columns,
                description=(f"Physical constraint '{check.name}' fails on "
                             f"{check.violation_rate:.2%} of observations. {check.rationale}"),
                evidence=check.evidence,
                remediation="A violated physical law points to instrumentation or unit "
                            "error, not to process behaviour. Resolve before modelling.",
            ))

    for edge in lineage:
        findings.append(Finding(
            dimension=Dimension.TRACEABILITY, severity=Severity.MAJOR, columns=[edge.target],
            description=(f"{edge.target} is reproducible from {', '.join(edge.source)} "
                         f"({edge.relation}, R={edge.confidence:.4f}) and is therefore a "
                         "derived quantity, not an independent measurement."),
            evidence={"confidence": edge.confidence, "relation": edge.relation},
            remediation="Keep either the parents or the derivative, never both, or the "
                        "model will appear accurate while learning an identity.",
        ))

    return findings


def score_dimensions(findings: list[Finding]) -> dict[Dimension, float]:
    scores = {d: 5.0 for d in Dimension}
    for f in findings:
        scores[f.dimension] = max(0.0, scores[f.dimension] - _PENALTY[f.severity])
    return {d: round(v, 2) for d, v in scores.items()}


def overall_score(scores: dict[Dimension, float]) -> float:
    """Weighted toward the dimensions that block modelling outright."""
    weights = {
        Dimension.COMPLETENESS: 0.20, Dimension.VALIDITY: 0.20,
        Dimension.CONSISTENCY: 0.20, Dimension.TIMELINESS: 0.15,
        Dimension.UNIQUENESS: 0.10, Dimension.INTEROPERABILITY: 0.10,
        Dimension.TRACEABILITY: 0.05,
    }
    return round(sum(scores[d] * w for d, w in weights.items()), 2)
