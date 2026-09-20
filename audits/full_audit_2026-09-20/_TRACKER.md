# full audit 2026-09-20 -- master tracker

Ten read-only reports over `src/mlframe/`, one per area, produced by one agent each. `src/mlframe/training/composite/`
is deliberately absent: the open [composite audit 2026-09-19](../composite_audit_2026-09-19/_TRACKER.md) owns it, and
this wave was scoped around it so the two do not fix the same code twice.

Reports: [training_core.md](training_core.md) (suite orchestration, splits, booster dataset reuse),
[feature_selection.md](feature_selection.md), [feature_engineering.md](feature_engineering.md) (plus preprocessing),
[metrics.md](metrics.md) (metrics and calibration), [predict_persistence.md](predict_persistence.md) (serving path and
artifact round-trip), [ensembling_models.md](ensembling_models.md) (model zoo, blends, thresholds, votenrank),
[evaluation_reporting.md](evaluation_reporting.md) (diagnostic verdicts, not chart aesthetics),
[performance.md](performance.md) (measured, not guessed), [concurrency_resources.md](concurrency_resources.md),
[config_contracts.md](config_contracts.md).

Statuses: **RESOLVED** (fixed in code; the note names the test that pins it), **PARTIAL** (part fixed; the note says
what remains), **TODO** (open), **REJECTED** (measured and declined; evidence in the note), **NOT A DEFECT** (the
claimed behaviour does not occur, or was already fixed before implementation started; the note names the commit).
Each report's own `- **Disposition**:` line is updated together with its row here.

## Summary

| File | Findings | RESOLVED | PARTIAL | TODO | REJECTED | NOT A DEFECT |
|---|---|---|---|---|---|---|
| `training_core.md` | 13 | 1 | 0 | 12 | 0 | 0 |
| `feature_selection.md` | 21 | 3 | 0 | 18 | 0 | 0 |
| `feature_engineering.md` | 14 | 2 | 0 | 12 | 0 | 0 |
| `metrics.md` | 17 | 3 | 1 | 13 | 0 | 0 |
| `predict_persistence.md` | 18 | 3 | 0 | 15 | 0 | 0 |
| `ensembling_models.md` | 14 | 3 | 0 | 11 | 0 | 0 |
| `evaluation_reporting.md` | 16 | 3 | 0 | 13 | 0 | 0 |
| `performance.md` | 6 | 0 | 0 | 6 | 0 | 0 |
| `concurrency_resources.md` | 12 | 1 | 0 | 11 | 0 | 0 |
| `config_contracts.md` | 32 | 0 | 0 | 32 | 0 | 0 |
| **Total** | **163** | **19** | **1** | **143** | **0** | **0** |

## Per-report status

One row per report, status first so a count can read it. The per-finding dispositions live in each report's own
table; this rolls them up to the coarsest status that is true of the whole report (a report is **TODO** until at
least one of its findings moves).

| Status | Report | Findings | Area |
|---|---|---|---|
| **PARTIAL** | [training_core.md](training_core.md) | 13 | suite orchestration, splits, booster dataset reuse (TRC-02 fixed) |
| **PARTIAL** | [feature_selection.md](feature_selection.md) | 21 | feature selection (FS-01, FS-02, FS-03 fixed) |
| **PARTIAL** | [feature_engineering.md](feature_engineering.md) | 14 | feature engineering and preprocessing (FE-01, FE-02 fixed) |
| **PARTIAL** | [metrics.md](metrics.md) | 17 | metrics and calibration (MET-01..MET-03 fixed, MET-04 partial) |
| **PARTIAL** | [predict_persistence.md](predict_persistence.md) | 18 | serving path and artifact round-trip (PRD-02, PRD-04, PRD-05 fixed) |
| **PARTIAL** | [ensembling_models.md](ensembling_models.md) | 14 | model zoo, blends, thresholds, votenrank (ENS-01, ENS-02, ENS-04 fixed) |
| **PARTIAL** | [evaluation_reporting.md](evaluation_reporting.md) | 16 | diagnostic verdicts (EVR-01, EVR-02, EVR-03 fixed) |
| **TODO** | [performance.md](performance.md) | 6 | measured performance |
| **PARTIAL** | [concurrency_resources.md](concurrency_resources.md) | 12 | concurrency and resources (CNC-01 fixed) |
| **TODO** | [config_contracts.md](config_contracts.md) | 32 | config contracts |

## What the wave is about

Three failure modes account for most of the list, and they cut across every area:

1. **Neutral substitution.** A value that could not be computed is replaced by one that wins: `0.0` into a signed
   importance array (`FS-02`, `FS-03`), a NaN loss term set to `0.0` in a lower-is-better metric (`MET-03`), a skip
   sentinel of `0` regardless of metric direction (`MET-04`), empty-input quantile losses returning `0.0` (`MET-11`),
   `0.0` pair-SU read as "maximally non-redundant" (`FS-12`), and five FE gates whose missing baseline makes the gate
   a no-op (`FS-19`, `FS-20`).
2. **Train/serve and train/report divergence.** State fitted on train never reaches predict (`FE-01`/`PRD-01`,
   `PRD-08`), blend weights and gate survivors are computed and then dropped (`ENS-01`, `ENS-02`, `ENS-03`), and the
   two predict entry points are each missing correctness steps the other has (`PRD-02`, `PRD-06`, `PRD-07`, `PRD-11`).
3. **Claims the code does not keep.** Three ICE paths advertise bit-exactness and return three different numbers
   (`MET-01`, `MET-02`), a memoisation that provably never hits is documented as the fix for a measured cost
   (`EVR-01`), a catalogue cites a test that does not exist (`EVR-05`), and roughly a dozen public config fields are
   validated and then ignored (`CFG-03`, `CFG-05`, `CFG-15`, `CFG-16`, `CFG-17`).

Highest-severity rows first: `MET-01` (P0: the metric driving early stopping is not the metric in the report, 38%
gap reproduced), then `ENS-01`/`ENS-04`, `PRD-02`/`PRD-04`/`PRD-05`, `CNC-01`, `TRC-01`/`TRC-02`, `FE-01`/`FE-02`,
`FS-01`/`FS-02`/`FS-03`, `EVR-02`/`EVR-03`, `CFG-01`, and the two import-cost items `PRF-01`/`PRF-02` (~25-30 s per
worker process and per notebook import, both import-graph edits).

## Per-report rows

Each report carries its own table with `| Sev | ID | Finding | Evidence (file:line + quote) | Failure scenario |
Suggested fix |`. Rows are dispositioned there and mirrored into the summary counts above as they are implemented.
