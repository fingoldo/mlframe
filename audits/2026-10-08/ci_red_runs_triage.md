# CI red runs on master, triage of 2026-10-08

Source: latest master run of every workflow (`gh run list --workflow <id> --branch master`), failed-job logs downloaded and grouped by failing test. All runs are on `354ce6241` or earlier. Status values: OPEN, RESOLVED, NOT A DEFECT, FUTURE.

## Runs that were not green

| Workflow | Run | Result | Failing jobs |
|---|---|---|---|
| CI | 37267018338 | failure | macOS shard 6 (1 test); "CI required checks" fails because of it; 6 other macOS shards cancelled |
| MyPy | 37267018132 (+2 earlier) | failure | whole-project mypy, `no-any-return` |
| deep-nightly | 37433617637 | failure | 12 of 20 shards |
| numba-coverage-nightly | 37445533095, 37603760151 | failure | about 12 of 16 shards each |
| fs-benchmark-nightly | 37440716614, 37596669838 | cancelled | not a failure |
| Dependency review, Dependabot auto-merge, GPU test matrix, release | | no master runs | not applicable |

## Findings

| ID | Finding | Evidence | Status |
|---|---|---|---|
| C1 | `test_group_demotion_prunes_public_rosters` fails its fixture precondition on macOS: with `max_runtime_mins=2` the control fit runs out of budget on the slower runner, so no FE survivor reaches a roster. Locally the two fits take 117 s. | CI log shard 6 | OPEN |
| C2 | MyPy: new `no-any-return` errors in `_iterative_stratification_njit.py` (lines 66, 99), `_lazy_host_codes.py:36`, `_numerical_numba.py:570` | MyPy run log | OPEN |
| D1 | Eleven `test_kalman_filter_njit_identity` cases fail by about 3e-13: the test's "verbatim baseline" still divides by `innovation_var + 1e-12` while the filter dropped the epsilon. Reproduced locally with and without JIT. | deep, numba nightlies, local | RESOLVED (`9c234ee9a`) |
| D2 | deep-nightly checks out with a shallow clone, so `test_no_new_stale_todos` reports a shallow-clone finding | log: "fetch-depth: 0" | OPEN |
| D3 | `test_meta` failures at `354ce6241`: function complexity (`apply_cmi_redundancy_gate` 26 > 25), 1k-LOC (`_fe_auto_escalation.py` 1009, `_usability_aware_selection.py` 1003), import cycle in `_feature_engineering_pairs`, `sentinel_guard_mismatch` in `_pairs_score_steps.py:758`, pydoclint baseline, fail-open handlers, stale todos, floorless assert loop, long functions, public annotations, meta-meta private reach, code_audit tests baseline, zero-tolerance baseline list, stdlib json in two test files | deep log | OPEN, recheck after merging origin/master |
| D4 | deep-nightly MRMR tests: `test_top_k_precision_5_signals_15_noise`, `test_clean_library_form_preferred_over_monotone_prewarp`, `test_feature_engineering_example_single_compound`, `test_linearly_usable_raw_operands_kept` (2), `test_mrmr_info_log_noise` | deep log | OPEN |
| D5 | `test_save_load_predict_parity_simple_mrmr` and `..._mrmr_with_fe`: `load_mlframe_suite` fails to load 2 of 2 models (`Could not load model from file`) | deep log | OPEN, possible production defect |
| D6 | `test_biz_val_per_group_baseline_polars_faster_than_pandas_at_1m_rows` | deep log | OPEN |
| D7 | `ValueError: y contains previously unseen labels` and 600 s / 900 s timeouts in deep shard 16 | deep log | OPEN |
| N1 | `test_yj_forward_speedup_gate`: with `NUMBA_DISABLE_JIT=1` the "numba" path is interpreted (0.04x). A ratio has no valid answer there. | numba nightlies | OPEN |
| N2 | `test_perf_screen_n1000_under_threshold` uses a fixed 5 s threshold (6.7 s interpreted) instead of `perf_time_budget` | numba nightlies | OPEN |
| N3 | `test_prewarm_numba_cache_completes_without_aborting` (`'function' object has no attribute 'signatures'`) and three `test_fold_cell_stats` cases (`no attribute '_cache'`) assume compiled dispatchers | numba nightlies | OPEN |
| N4 | `test_compact_codes_selection_equivalent[1]` and `test_fe_max_polynoms_fit_transform_no_keyerror` hit the 3600 s timeout interpreted | numba nightlies | OPEN |
| N5 | `test_selected_features_surface_for_inspection` selects no engineered features interpreted (budget exhausted) | numba nightlies | OPEN |
| N6 | `test_column_major_speedup_vs_row_major_reference` and `test_perf_regression` speedup/time gates | numba nightlies | OPEN |
| N7 | pytest-timeout kills cause `INTERNALERROR` (`tb_lineno=None`) that ends a shard before it writes coverage | numba nightlies | OPEN |
| E1 | `highspy` "undefined symbol Highs::releaseMemory" printed by CVXPY while importing the optional HIGHS solver | every nightly shard log | NOT A DEFECT: a logged import failure of an optional solver, no test depends on it |

## Open questions from the owner

- Whether `lint (advisory)` can become blocking, and what the pydoclint count is.
- Why `kaleido` is not installed in CI, and whether other pytest `importorskip` dependencies are missing from the CI install.

## Owner questions, answered

| ID | Question | Answer | Status |
|---|---|---|---|
| L1 | `lint (advisory)` pydoclint findings | 244 findings in the last run: DOC201 52, DOC101 43, DOC103 43, DOC107 36, DOC301 30, DOC106 12, DOC001 9, DOC111 6, DOC104 6, DOC105 6, DOC404 1. A baseline ratchet (`test_pydoclint_baseline`) already exists, but the advisory job reports the raw count and nothing drains it. | OPEN: fix all 244, then make the pydoclint step blocking |
| L2 | Make every advisory step blocking | Decide per step from the logs: a step that is clean becomes blocking now, a step with findings gets them fixed first. The census of the other advisory steps is pending. | OPEN |
| K1 | `kaleido` is not installed in CI, so `tests/reporting/test_kaleido_*` skip | `kaleido` appears only in the deptry ignore lists of `pyproject.toml`, never in an extra, so `.[all,dev]` cannot install it. The code under test (static plotly export, hang and recovery paths) is therefore never exercised in CI. | OPEN: add to the `viz` extra and regenerate `uv.lock` |
| K2 | Other forgotten modules | Test skips of the form "could not import X" in the downloaded CI logs: `mlxtend` (3 files), `ngboost` (2), `kaleido` (2), `gudhi`, `concepts`, `ndd`, `ml_insights`, `mlflow` (1 each), plus intentionally heavy `tensorflow`, `autogluon.tabular`, `lightautoml`, `cupy` (GPU only), and the external `mrmr` comparison package. None of `kaleido`, `mlxtend`, `ngboost`, `gudhi`, `ndd`, `ml_insights` is declared in an extra. | OPEN: declare the light ones in `dev`, leave the heavy ones skipped on purpose |
