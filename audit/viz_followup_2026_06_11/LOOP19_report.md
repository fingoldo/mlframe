# LOOP19 — data-aware binary panel emphasis threaded end-to-end

## Verdict
data_aware is now LIVE in real `train_mlframe_models_suite` runs. Fully threaded; no FUTURE hop. Back-compat byte-identical under the default config.

## Call-chain trace
- `train_mlframe_models_suite` -> `_trainer_train_and_evaluate.py:271` reads `binary_panels = reporting.binary_panels` (the ReportingConfig).
- `_trainer_train_and_evaluate.py:795` passes `binary_panels` + the whole `reporting_config` into `_compute_split_metrics` (`_eval_helpers.py`).
- `_eval_helpers.py:494` `report_model_perf(... binary_panels=binary_panels, reporting_config=reporting_config ...)`.
- `report_model_perf` (`training/reporting/_reporting.py`) builds the dispatch kwargs at ~L756 and calls `mlframe.reporting.auto_dispatch.render_multi_target_panels`.
- `render_multi_target_panels` already had the `panel_emphasis` / `binary_panels_is_default` / `emphasis_imbalance_lo/hi` params and consulted `select_binary_emphasis_panels` — but the report path passed NONE of them, so the dispatcher always saw the defaults (`panel_emphasis="all"`, `binary_panels_is_default=False`). That was the dormant gap.

## What was threaded (single file: training/reporting/_reporting.py)
At the `render_multi_target_panels` call site:
- `panel_emphasis = getattr(reporting_config, "panel_emphasis", "all")`
- `emphasis_imbalance_lo/hi = getattr(reporting_config, ...)` (0.2 / 0.8 fallback)
- `binary_panels_is_default = (binary_panels == _reporting_field_default("binary_panels"))`
- All four passed into `render_multi_target_panels(...)`.

New module helper `_reporting_field_default(field_name)`: reads `ReportingConfig.model_fields[field_name].default` (pydantic), cached on the function dict, returns None if the config import fails. This is the config-boundary computation of "left at default" required by scope item 2 — `binary_panels` flowing through the report path is exactly `reporting.binary_panels`, so equality vs the field default is the correct is-default test.

`y_true` for the base rate is the `targets` the report already holds (passed as `targets=` to the dispatcher); no extra full-n pass, no frame copy. RAM-safe.

Edge-safe: missing `reporting_config` -> getattr fallbacks give `panel_emphasis="all"` (no-op). Single-class / tiny-n / missing y_true -> `select_binary_emphasis_panels` falls back to the requested template inside the dispatcher (unchanged behavior).

## Back-compat proof
- Default `ReportingConfig()` -> `panel_emphasis == "all"` -> dispatcher short-circuits in `select_binary_emphasis_panels` (returns `requested_panels` unchanged) AND the data_aware branch in the dispatcher is gated on `panel_emphasis == "data_aware"`, so the binary template is byte-identical to today.
- New test `TestReportModelPerfThreading::test_all_mode_passes_default_through_unchanged` asserts the dispatcher receives the exact `_DEFAULT` string under a default config.

## Tests (tests/reporting/test_panel_emphasis.py — new class TestReportModelPerfThreading)
- `test_data_aware_default_template_reorders_pr_led` — imbalanced (0.03) + default template + data_aware -> dispatcher gets PR-led template, ROC dropped.
- `test_all_mode_passes_default_through_unchanged` — default config -> dispatcher gets `_DEFAULT` unchanged (back-compat sensor).
- `test_custom_template_not_reordered_even_in_data_aware` — custom binary_panels + data_aware -> `binary_panels_is_default=False` -> template untouched.
- `test_single_class_falls_back_to_all` — single-class y_true + data_aware -> falls back to `_DEFAULT`.
- Spies on `compose_binary_figure` to capture the effective template without rendering.

Results: `tests/reporting/test_panel_emphasis.py` 19 passed (15 pre-existing + 4 new). Broader: test_auto_dispatch + test_default_panels_e2e + test_multi_target_panels_e2e + test_reporting_integration_w5int2 = 39 passed.

## FUTURE-skipped hops
None. The full path was git-clean (`_reporting.py`, `_eval_helpers.py`, `_trainer_train_and_evaluate.py`, `_reporting_configs.py`, `auto_dispatch.py` all clean at edit time; only `_reporting.py` + the test needed edits). Parallel composite session held `training/composite/*` + CHANGELOG.md + a few unrelated files dirty — none touched.

## Commit
See commit hash in final reply.
