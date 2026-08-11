[![CI](https://github.com/Sara-Ar-GitHub/data-readiness-agent/actions/workflows/ci.yml/badge.svg)](...)

# Data Readiness Agent

An agentic assessment tool for industrial datasets. It takes a raw export from an SME's historian, MES or SCADA system and returns a data-readiness report: a profile, a set of *physical* constraints tested against the data, an inferred column lineage, a prioritised remediation roadmap, and a feasibility verdict on the proposed modelling use case.

The intended user is an engineer running a short, hypothesis-driven feasibility study for a company that has data but does not yet know whether it can support a model.

## The design decision that matters

**The language model never computes anything.**

Every statistic, violation rate and maturity score is produced by deterministic, unit-tested Python. The LLM contributes three things and nothing else:

1. reading column semantics to *propose* which physical constraints are worth testing,
2. sequencing tool calls and revising when a hypothesis fails for an uninteresting reason (wrong unit assumption, wrong column pairing),
3. writing the executive summary from adjudicated results.

This is not a stylistic preference. An assessment tool that lets a language model assert a violation rate on plant data is unusable in a client engagement, because nothing in the report can be defended when the client disputes it. Here every number traces to a function in `src/dra/tools/` and to a row range in the input file.

The corollary is that the pipeline degrades gracefully. With `--no-llm` the deterministic half runs alone and still produces a complete report — useful for regression testing, and useful when a client will not allow any external API call at all.

## What each layer catches

The demo dataset (`examples/make_demo.py`) is synthetic press-shop data with nine deliberately injected faults, which makes the split measurable rather than rhetorical.

| Layer | Recovers |
|---|---|
| Deterministic | missing tags, frozen/stuck transmitters, constant columns, duplicate rows and timestamps, irregular sampling, historian gaps, undeclared units, derived columns (leakage risk) |
| Agent | negative absolute pressure, sub-meter drift breaking conservation, impossible thermal rates of change, algebraic identities between torque, speed and shaft power |

The second row is the interesting one: those faults are invisible to any schema-level or statistical profiler, because detecting them requires knowing what the equipment *is*. A 45 °C jump in ambient temperature is not a statistical outlier — three points in six thousand — but it is physically impossible, and that judgement is what the domain-reasoning layer supplies.

## Performance

Lineage inference is the only part of the pipeline that is not linear in the input, and the naive form of it (least squares over every column combination, on every row) is cubic in the column count — roughly ten minutes on a 60-tag export. Two observations remove almost all of that cost:

- for a single parent, R² is exactly the squared Pearson correlation, so one correlation matrix answers every single-parent question at once and no regression is needed;
- a relation holding at R² ≥ 0.995 is settled by a few thousand rows, so the search runs on a systematic subsample rather than the full series.

A full deterministic report on 120,000 rows × 62 columns takes about 3 seconds. The trade-offs are documented under Limitations.

## Readiness dimensions

Findings are scored 0–5 across seven dimensions — completeness, validity, consistency, timeliness, uniqueness, interoperability, traceability — deliberately aligned with the vocabulary used in EU Digital Innovation Hub maturity assessments, so that a technical finding maps directly onto a roadmap item an SME can act on and fund.

Scoring is deterministic and auditable: identical input, identical score, and every point deducted traces to a named finding with its evidence.

## Data sovereignty

The model provider is configurable and includes a fully local path via Ollama, so an assessment can be run without any plant data leaving the client's premises. For SMEs unwilling to send process data to a hosted API — a common and reasonable position — this is the difference between a tool that can be used on site and one that cannot.

## Usage

```bash
pip install -r requirements.txt          # deterministic pipeline only
pip install -r requirements-agent.txt    # add the agent layer
pip install -e .                         # install the `dra` entry point
cp .env.example .env                     # then set a provider

python examples/make_demo.py                      # generate the demo dataset
dra examples/line3_press_shop.csv --sector manufacturing -o report.md
dra examples/line3_press_shop.csv --no-llm        # deterministic only
pytest                                            # fault-recovery tests
```

`sample_report.md` is the committed output of the `--no-llm` run above, so the
shape of the deliverable can be judged without installing anything.

Output is written as Markdown and as JSON conforming to the `ReadinessReport` schema in `src/dra/models.py`, so downstream tooling consumes the structured form rather than parsing prose.

## Layout

```
src/dra/
  models.py           typed contract for every artefact
  agent.py            tool registration, orchestration, roadmap construction
  report.py           Markdown rendering
  cli.py              entry point
  tools/
    profiling.py      structural and statistical profile
    physics.py        constraint adjudication (range, monotonic,
                      conservation, rate limit, algebraic identity)
    lineage.py        derived-column and leakage detection
    quality.py        rule-based findings and maturity scoring
tests/                fault-recovery tests against known injected faults
examples/
  make_demo.py        synthetic press-shop data with nine injected faults
sample_report.md      committed output of the deterministic run
.github/workflows/    lint, tests and an end-to-end no-LLM run on 3.11 / 3.12
```

## Limitations

- Lineage inference covers exact copies, rescalings and linear combinations of at most two parents. Non-linear derivations are not recovered, and the parent cap is deliberate: a false lineage claim in a client report is worse than a missing one.
- Unit inference is name-based and is surfaced as a question to ask the client, never as an assertion.
- The maturity weighting is a defensible default, not a validated instrument. It should be calibrated against real assessments before being used comparatively across companies.
- Two-parent lineage search is restricted to each target's most correlated neighbours, so a pair of individually uncorrelated columns that jointly explain a target can be missed. Single-parent detection is exact.
- Lineage runs on a systematic subsample of at most 5,000 rows, which could miss a relation that only holds under extreme heteroscedasticity.
- Constraint proposals depend on informative column names. On fully anonymised exports the agent layer contributes little, and the deterministic layer carries the assessment.
