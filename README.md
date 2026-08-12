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

### Why this is the explainability story, not a disclaimer about it

Post-hoc explanation of a model-authored number explains how the text was produced, not whether the number is right. Constraining the model so that it never authors a number moves the property from something asserted about the system to something structural in it — and structural claims can be tested.

That test is `tests/test_agent_contract.py`. It drives the agent with a **scripted model** (`pydantic-ai`'s `FunctionModel`), so the agent layer is exercised on every CI run with no API key and no network. The load-bearing case feeds the agent a constraint the data satisfies, phrased as an accusation, and asserts the verdict comes back clean: the adjudication is arithmetic, and the model's framing has no purchase on it. A test that only ran against a live model could not establish this, because a live model's silence on one run is not evidence.

Three properties follow, and each is a test rather than a claim:

| Property | How it is enforced |
|---|---|
| **Auditability** — every figure traces to code and rows | statistics only ever leave `src/dra/tools/`; the tool surface is pinned by a test, so adding a tool that returns a model-authored number breaks the build |
| **Traceability** — every finding carries its physical rationale | `ConstraintCheck.rationale` is required, and asserted non-empty for every adjudicated check |
| **Robustness** — a wrong hypothesis degrades, never aborts | a constraint proposed on a hallucinated column returns a failed check naming the missing tag, so the agent revises and the assessment continues |

The third matters more than it looks. A model reasoning from column names *will* assume tags that do not exist — an ISO 10816 vibration channel on a line that never had one. That assumption is worth stating and worth recording. What it must not do is raise, abort the run, and discard every constraint already adjudicated.

## What each layer catches

The demo dataset (`examples/make_demo.py`) is synthetic press-shop data with nine deliberately injected faults, which makes the split measurable rather than rhetorical.

| Layer | Recovers |
|---|---|
| Deterministic | missing tags, frozen/stuck transmitters, constant columns, duplicate rows and timestamps, irregular sampling, historian gaps, undeclared units, derived columns (leakage risk) |
| Agent | negative absolute pressure, sub-meter drift breaking conservation, impossible thermal rates of change, algebraic identities between torque, speed and shaft power |

The second row is the interesting one: those faults are invisible to any schema-level or statistical profiler, because detecting them requires knowing what the equipment *is*. A 45 °C jump in ambient temperature is not a statistical outlier — three points in six thousand — but it is physically impossible, and that judgement is what the domain-reasoning layer supplies.

## Two sectors, one assessment layer

A tool validated on one dataset has shown that it works on that dataset. The claim worth making is that the *abstraction* transfers — so there is a second demo in a domain that shares no vocabulary with the first.

`examples/make_demo_logistics.py` is synthetic cold-chain distribution: no torque, no shaft power, different units, different failure modes. It carries its own nine planted faults, and the constraints have the same *shape* as the press shop's while having none of the same names — a reefer body has thermal inertia, pallets per route must reconcile against the vehicle manifest, fuel burned is litres per km times distance, payload cannot be negative.

Nothing in the profiler or the quality rules is told which sector it is looking at. `tests/test_logistics.py` asserts the recovery, and passing it is the falsifiable version of "the tool generalises".

The second sector paid for itself immediately by exposing two defects in the first:

- **Duplicate business keys were invisible.** Row-level duplicate detection misses a consignment ID that repeats while the surrounding measurements differ — the row is not a duplicate, but the entity has been counted twice and every join on that key silently double-counts. Detection is by cardinality, not by column name, since naming conventions differ in every export.
- **Unit inference read `vehicle_speed_kph` as rpm**, a bare `speed` pattern written when the only speeds in view were rotational. Precisely the confusion the unit hint exists to surface, committed by the hint itself.

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

python examples/make_demo.py                      # press shop
python examples/make_demo_logistics.py            # cold-chain distribution

dra examples/line3_press_shop.csv --sector manufacturing -o report.md
dra examples/line3_press_shop.csv --no-llm        # deterministic only
dra examples/fleet_cold_chain.csv --sector "transport and logistics" --no-llm

pytest                                            # fault recovery, both sectors
```

`sample_report.md` and `sample_report_logistics.md` are the committed outputs of the two `--no-llm` runs above, so the shape of the deliverable can be judged without installing anything.

The full test suite — including the agent layer, which runs against a scripted model — needs no API key and no network. A key is required only to run the agent against a live provider.

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
tests/
  test_pipeline.py    fault recovery, press shop
  test_logistics.py   fault recovery, cold chain — the transfer claim
  test_agent_contract.py
                      the agent/tool handoff, driven by a scripted model
examples/
  make_demo.py        synthetic press-shop data, nine injected faults
  make_demo_logistics.py
                      synthetic cold-chain data, nine injected faults
sample_report.md      committed output of the deterministic run
sample_report_logistics.md
.github/workflows/    lint, tests and an end-to-end no-LLM run on 3.11 / 3.12
```

## Limitations

- Lineage inference covers exact copies, rescalings and linear combinations of at most two parents. Non-linear derivations are not recovered, and the parent cap is deliberate: a false lineage claim in a client report is worse than a missing one.
- Unit inference is name-based and is surfaced as a question to ask the client, never as an assertion.
- The maturity weighting is a defensible default, not a validated instrument. It should be calibrated against real assessments before being used comparatively across companies.
- Two-parent lineage search is restricted to each target's most correlated neighbours, so a pair of individually uncorrelated columns that jointly explain a target can be missed. Single-parent detection is exact.
- Lineage runs on a systematic subsample of at most 5,000 rows, which could miss a relation that only holds under extreme heteroscedasticity.
- Constraint proposals depend on informative column names. On fully anonymised exports the agent layer contributes little, and the deterministic layer carries the assessment.
- The rate-limit check adjudicates on the *fraction* of violating intervals. That is the right rule for transmission glitches and the wrong one for a single catastrophic discontinuity: one odometer reset in 5,000 samples does not move a rate, so it survives in the evidence but not in the verdict. Recovering it properly needs monotonicity-in-time for cumulative counters, which is not yet one of the five constraint kinds. `tests/test_logistics.py` asserts the current behaviour rather than hiding it.
- Business-key detection is by cardinality, so a genuinely low-cardinality key — a batch code with ten legitimate values — is not treated as a key at all. Composite keys are not detected.
