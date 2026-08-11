"""Markdown rendering of a ReadinessReport."""

from __future__ import annotations

from dra.models import ReadinessReport, Severity

_ICON = {Severity.BLOCKER: "[BLOCKER]", Severity.MAJOR: "[MAJOR]",
         Severity.MINOR: "[minor]", Severity.INFO: "[info]"}


def to_markdown(r: ReadinessReport) -> str:
    L: list[str] = []
    L.append(f"# Data readiness assessment — {r.dataset_name}")
    if r.sector:
        L.append(f"*Sector: {r.sector}*")
    L.append("")
    L.append(f"**Overall maturity: {r.overall_score} / 5**")
    if r.feasibility_verdict:
        L.append(f"**Feasibility: {r.feasibility_verdict}**")
    L.append("")
    if r.executive_summary:
        L.append(r.executive_summary)
        L.append("")

    L.append("## Maturity by dimension\n")
    L.append("| Dimension | Score |")
    L.append("|---|---|")
    for d, s in r.scores.items():
        L.append(f"| {d.value} | {s:.2f} / 5 |")
    L.append("")

    p = r.profile
    L.append("## Dataset structure\n")
    L.append(f"- {p.n_rows:,} rows x {p.n_cols} columns")
    L.append(f"- Duplicate rows: {p.duplicate_row_rate:.2%}")
    if p.timestamp_column:
        L.append(f"- Timestamp column: `{p.timestamp_column}`")
        if p.median_sampling_seconds:
            L.append(f"- Median sampling interval: {p.median_sampling_seconds:.1f} s")
        if p.sampling_regularity is not None:
            L.append(f"- Sampling regularity: {p.sampling_regularity:.1%}")
        if p.gap_count:
            L.append(f"- Gaps beyond 5 sampling periods: {p.gap_count}")
    else:
        L.append("- No timestamp column detected")
    L.append("")

    if r.constraints:
        L.append("## Physical constraints tested\n")
        L.append("| Constraint | Holds | Violations | Rationale |")
        L.append("|---|---|---|---|")
        for c in r.constraints:
            L.append(f"| `{c.name}` | {'yes' if c.holds else 'NO'} | "
                     f"{c.violation_rate:.2%} | {c.rationale} |")
        L.append("")

    if r.lineage:
        L.append("## Inferred lineage\n")
        for e in r.lineage:
            L.append(f"- `{e.target}` <- {', '.join('`'+s+'`' for s in e.source)} "
                     f"({e.relation}, confidence {e.confidence:.4f})")
        L.append("")

    if r.findings:
        L.append("## Findings\n")
        for sev in (Severity.BLOCKER, Severity.MAJOR, Severity.MINOR, Severity.INFO):
            group = [f for f in r.findings if f.severity == sev]
            if not group:
                continue
            L.append(f"### {sev.value.title()} ({len(group)})\n")
            for f in group:
                cols = f", columns: {', '.join('`'+c+'`' for c in f.columns)}" if f.columns else ""
                L.append(f"- {_ICON[sev]} **{f.dimension.value}**{cols} — {f.description}")
                L.append(f"  - *Remediation:* {f.remediation}")
            L.append("")

    if r.roadmap:
        L.append("## Prioritised roadmap\n")
        L.append("| # | Action | Effort | Unlocks |")
        L.append("|---|---|---|---|")
        for i in r.roadmap:
            L.append(f"| {i.priority} | {i.action} | {i.effort} | {i.unlocks} |")
        L.append("")

    return "\n".join(L)
