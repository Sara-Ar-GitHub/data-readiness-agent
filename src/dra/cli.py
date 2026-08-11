"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

from dra.agent import AgentDeps, build_agent, build_roadmap, deterministic_report
from dra.report import to_markdown
from dra.tools import lineage as lineage_tools
from dra.tools import profiling
from dra.tools import quality as quality_tools


def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise SystemExit(f"Unsupported file type: {path.suffix}")


async def run_agent(df: pd.DataFrame, name: str, sector: str | None, model: str | None):
    agent = build_agent(model)
    deps = AgentDeps(df=df, checks=[])
    prompt = (
        f"Assess dataset '{name}'"
        + (f" from the {sector} sector" if sector else "")
        + f". Columns: {list(df.columns)}. Begin by profiling, then propose and test "
        "the physical constraints that should hold for this equipment."
    )
    result = await agent.run(prompt, deps=deps)

    prof = profiling.profile_dataset(df, name)
    lin = lineage_tools.infer_lineage(df)
    findings = quality_tools.derive_findings(prof, deps.checks, lin)
    scores = quality_tools.score_dimensions(findings)

    from dra.models import ReadinessReport
    return ReadinessReport(
        dataset_name=name, sector=sector, profile=prof, constraints=deps.checks,
        lineage=lin, findings=findings, scores=scores,
        overall_score=quality_tools.overall_score(scores),
        roadmap=build_roadmap(findings),
        executive_summary=str(result.output),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="SME industrial data-readiness assessment")
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--sector", default=None)
    ap.add_argument("--model", default=None,
                    help="e.g. anthropic:claude-sonnet-4-6, ollama:qwen2.5")
    ap.add_argument("--no-llm", action="store_true", help="Deterministic pipeline only")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    df = load(args.dataset)
    name = args.dataset.stem

    if args.no_llm:
        report = deterministic_report(df, name, args.sector)
    else:
        try:
            report = asyncio.run(run_agent(df, name, args.sector, args.model))
        except ImportError:
            print("pydantic-ai not installed; falling back to deterministic mode.",
                  file=sys.stderr)
            report = deterministic_report(df, name, args.sector)

    md = to_markdown(report)
    if args.output:
        args.output.write_text(md, encoding="utf-8")
        json_path = args.output.with_suffix(".json")
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {args.output} and {json_path}")
    else:
        print(md)


if __name__ == "__main__":
    main()
