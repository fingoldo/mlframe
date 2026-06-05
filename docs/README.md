# mlframe documentation index

Conceptual guides and design notes for mlframe. Copy-pasteable per-feature
snippets live in [`docs/examples/`](examples/README.md); this index covers the
deeper material in the docs root.

## User-facing guides

| File | Topic |
|---|---|
| [`baseline_diagnostics_guide.md`](baseline_diagnostics_guide.md) | `BaselineDiagnostics` — ablation table + init_score baseline; the default-ON diagnostic that tells you whether composite targets will help |
| [`dummy_baselines_guide.md`](dummy_baselines_guide.md) | Parameter-free dummy baselines (median / mean / mode / lag / AR(1)) and the dummy-floor gate |
| [`calibration_policy.md`](calibration_policy.md) | `pick_best_calibrator` — OOF-ECE-with-bootstrap-CI selection between NoCal / Sigmoid / Isotonic |
| [`honest_diagnostics_guide.md`](honest_diagnostics_guide.md) | The honest-diagnostics aggregator: holdout-only verdicts, dummy-floor delta, calibration ECE, DeLong significance |
| [`feature_handling_examples.md`](feature_handling_examples.md) | `FeatureHandlingConfig` cookbook — text / categorical / per-model handling recipes wired via `feature_handling_config=` |
| [`composite_targets_tutorial.ipynb`](composite_targets_tutorial.ipynb) | End-to-end TVT walkthrough notebook for composite-target discovery |
| [`MULTI_OUTPUT.md`](MULTI_OUTPUT.md) | Multilabel / multiclass classification design notes |
| [`MULTI_TARGET_REGRESSION.md`](MULTI_TARGET_REGRESSION.md) | Multi-target (K-column) regression support matrix and integration roadmap |
| [`SELECTION_BIAS.md`](SELECTION_BIAS.md) | PU-learning and drift-correction wrappers for biased-sampling problems |
| [`DEBUGGING_UPSTREAM_ERRORS.md`](DEBUGGING_UPSTREAM_ERRORS.md) | Decoding cryptic errors raised by sklearn / LightGBM / CatBoost under the suite |

See also [`docs/examples/composite_targets.md`](examples/composite_targets.md) for the tiered composite-target recipes.

## Internal / research notes

Design audits, literature surveys, and forward-looking roadmaps. These are
working notes (some items have since shipped — each doc carries a status note
where relevant), not user API documentation.

| File | Topic |
|---|---|
| [`MRMR_RESEARCH_2026_05_28.md`](MRMR_RESEARCH_2026_05_28.md) | MRMR / feature-selection literature survey (scorers, MI estimators); several items shipped |
| [`pysr_fe_upgrade_research.md`](pysr_fe_upgrade_research.md) | PySR symbolic-regression FE tuning survey + operator-preset design (shipped) |
| [`date_features_kaggle_research.md`](date_features_kaggle_research.md) | Calendar / cyclical date-feature survey behind `create_date_features` |
| [`NUMERICAL_STABILITY_REPORT.md`](NUMERICAL_STABILITY_REPORT.md) | Catastrophic-cancellation audit + benchmark of the numba moment kernels |
| [`audit_2026_05_24_summary.md`](audit_2026_05_24_summary.md) | Snapshot summary of a calibration / diagnostics audit pass |
| [`WAVE5_GPU_ROADMAP.md`](WAVE5_GPU_ROADMAP.md) | GPU-acceleration roadmap notes |
| [`internal/`](internal/) | Harness-internal notes (e.g. `fuzz_audit.md` — the combinatorial fuzz harness) |
