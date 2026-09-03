# core infra B (data/, data_valuation/, inference/, signal/, system/, integrations/, inspection/, testing/, top-level mlframe/*.py) -- mlframe audit

## Scope

All 56 `.py` files in the assigned cluster were opened and read in full this session. None were skipped or partially covered.

- `src/mlframe/__init__.py` (429 LOC)
- `src/mlframe/config.py` (140 LOC)
- `src/mlframe/version.py` (8 LOC)
- `src/mlframe/data/__init__.py` (12)
- `src/mlframe/data/datasets.py` (79)
- `src/mlframe/data/synthetic.py` (387)
- `src/mlframe/data/_benchmarks/__init__.py` (0)
- `src/mlframe/data/_benchmarks/bench_assign_classes_njit.py` (75)
- `src/mlframe/data_valuation/__init__.py` (55)
- `src/mlframe/data_valuation/_weights.py` (70)
- `src/mlframe/data_valuation/_adversarial_validation.py` (105)
- `src/mlframe/data_valuation/_adversarial_reweighting.py` (180)
- `src/mlframe/data_valuation/_knn_shapley.py` (114)
- `src/mlframe/data_valuation/_mc_sampling.py` (143)
- `src/mlframe/data_valuation/_benchmarks/__init__.py` (0)
- `src/mlframe/data_valuation/_benchmarks/profile_adversarial_reweighting.py` (73)
- `src/mlframe/data_valuation/_benchmarks/profile_data_valuation.py` (98)
- `src/mlframe/inference/__init__.py` (20)
- `src/mlframe/inference/predict.py` (282)
- `src/mlframe/inference/explainability.py` (224)
- `src/mlframe/inference/native_gpu_shap.py` (74)
- `src/mlframe/inference/postanalysis.py` (38)
- `src/mlframe/inference/logical_constraints.py` (313)
- `src/mlframe/inference/group_zero_sum_constraint.py` (191)
- `src/mlframe/inference/time_budget_ensemble.py` (134)
- `src/mlframe/inference/recursive_forecast.py` (148)
- `src/mlframe/inference/entity_prediction_collapse.py` (133)
- `src/mlframe/inference/_ktc_dispatch.py` (145)
- `src/mlframe/inference/_benchmarks/bench_entity_prediction_collapse.py` (67)
- `src/mlframe/inference/_benchmarks/bench_time_budget_ensemble.py` (73)
- `src/mlframe/inference/_benchmarks/bench_recursive_forecast.py` (78)
- `src/mlframe/inference/_benchmarks/bench_logical_constraints.py` (90)
- `src/mlframe/inference/_benchmarks/bench_group_zero_sum_constraint.py` (91)
- `src/mlframe/inference/_benchmarks/profile_native_gpu_shap.py` (93)
- `src/mlframe/inference/_benchmarks/bench_shap_oof_per_fold.py` (136)
- `src/mlframe/inspection/__init__.py` (20)
- `src/mlframe/inspection/interaction.py` (153)
- `src/mlframe/integrations/__init__.py` (10)
- `src/mlframe/integrations/mlflow.py` (172)
- `src/mlframe/signal/__init__.py` (55)
- `src/mlframe/signal/dtw.py` (594)
- `src/mlframe/signal/gp_smoothing.py` (235)
- `src/mlframe/signal/hull_moving_average.py` (137)
- `src/mlframe/signal/changepoint_detection.py` (126)
- `src/mlframe/signal/_pelt_l2_njit.py` (82)
- `src/mlframe/signal/_benchmarks/__init__.py` (0)
- `src/mlframe/signal/_benchmarks/bench_changepoint_detection.py` (77)
- `src/mlframe/signal/_benchmarks/bench_hull_moving_average.py` (80)
- `src/mlframe/signal/_benchmarks/bench_dtw_gpu_banded.py` (83)
- `src/mlframe/signal/_benchmarks/bench_gp_smoothing.py` (83)
- `src/mlframe/system/__init__.py` (9)
- `src/mlframe/system/_gpu_guard.py` (65)
- `src/mlframe/system/kernel_tuning_cache/__init__.py` (180)
- `src/mlframe/system/kernel_tuning_cache/__main__.py` (7)
- `src/mlframe/testing/__init__.py` (3)
- `src/mlframe/testing/parametric.py` (409)

Total: **56 files, 6878 LOC reviewed** (matches `wc -l` across the cluster's file list exactly).

Overall impression: this cluster is unusually well-engineered relative to typical audit targets -- extensive docstrings documenting intentional tradeoffs, guarded division-by-zero, deliberate bit-identity contracts, and (mostly) solid test coverage under `tests/inference/`, `tests/data_valuation/`, `tests/signal/`, `tests/inspection/`, `tests/system/`. The findings below are real but narrower in blast radius than the P0-heavy pattern the excluded MRMR audit found.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| B1 | P1 | correctness/silent-failure | src/mlframe/signal/dtw.py:570-578 | `dtw_dispatch` has no runtime fallback if the chosen `cupy`/`cuda` backend raises at call time (OOM, driver hiccup, `CUDA_VISIBLE_DEVICES=""` set after cupy already imported) -- unlike the sibling dispatcher in `logical_constraints.py`, which explicitly catches and falls back to CPU. |
| B2 | P1 | GPU/CPU parity | src/mlframe/system/_gpu_guard.py:21-33 (consumed by src/mlframe/signal/dtw.py:44) | `try_import_cupy()` / `_HAS_CUPY` only proves cupy **imports**, not that a CUDA device is actually usable (import alone does not probe the device); combined with B1 this means a host with `CUDA_VISIBLE_DEVICES=""` or a broken CUDA runtime can still have `dtw_dispatch` pick `"cupy"` and crash instead of falling back, reproducing the exact GPU-native-crash class this repo's own test conventions work around elsewhere (`CUDA_VISIBLE_DEVICES=""` full-suite CPU-only runs). |
| B3 | P1 | silent-failure/target-type | src/mlframe/data_valuation/_knn_shapley.py:85-89 | `knn_shapley` validates only `y_train`'s dtype for "is this actually classification" and raises `NotImplementedError` for continuous `y_train`; it never checks `y_val`. If `y_train` is integer-coded but `y_val` is continuous (e.g. accidentally passed a probability/score column), the function does not raise -- it silently computes near-meaningless Shapley values via exact-float-equality label matching instead of failing loudly. |
| B4 | P2 | API contract / hidden mutation | src/mlframe/integrations/mlflow.py:41-56, 58-67 | `flatten_classification_report` and `log_classification_report_to_mlflow` both mutate the caller's `cr` dict in place via `cr.pop(metric)`, undocumented in either docstring. Calling both functions on the same `classification_report(output_dict=True)` object (a plausible pattern -- flatten for one destination, log for another) silently drops the popped scalar metrics (`accuracy`, etc.) on the second call. |
| B5 | P2 | test coverage | src/mlframe/integrations/mlflow.py (whole file, 172 LOC) | Zero test coverage: `tests/integrations/` does not exist at all. None of `flatten_classification_report`, `log_classification_report_to_mlflow`, `embed_website_to_mlflow`, `get_or_create_mlflow_run`, `create_mlflow_run_label` has any unit test. |
| B6 | P2 | robustness/security | src/mlframe/integrations/mlflow.py:69-85 | `embed_website_to_mlflow` writes an iframe with `sandbox='allow-same-origin allow-scripts'`; that specific combination is a well-known weak sandbox (a same-origin + scripting frame can largely defeat the sandbox's isolation), for a URL that is only `html.escape`d, not otherwise validated -- acceptable if `url` is always an internally-controlled MLflow artifact link, but the function has no docstring caveat about untrusted URLs and no origin allowlist. |
| B7 | P2 | edge case / div-by-zero | src/mlframe/inference/recursive_forecast.py:118 | `diagnose_error_accumulation`'s `growth_ratio = recursive_mse / recursive_mse[0]` divides by zero (producing `inf`/`nan` silently, no `np.errstate` guard or check) whenever the model's step-1 prediction is exact (`recursive_mse[0] == 0`), a realistic edge case on a deterministic/overfit model or a trivial fixture. |
| B8 | P2 | input validation | src/mlframe/inference/group_zero_sum_constraint.py:107, 132-135 | `apply_group_zero_sum_constraint` never validates that `weights` (or `extra_constraint_coefs`, which reuse the same metric) are non-negative, unlike the sibling `collapse_predictions_by_group` in the same package (`entity_prediction_collapse.py:111-112`), which explicitly raises on negative weights. A caller passing a signed weight (e.g. a signed "confidence" column by mistake) gets a silently wrong-signed correction instead of an error. |
| B9 | P2 | robustness / error UX | src/mlframe/data/synthetic.py:80-87 | `sample_random_variable`'s `kind` parameter is validated by an `if/elif/elif` chain with no `else`; an invalid `kind` value (anything other than `"cont"`/`"cat"`/`"mixed"`) leaves `source` unassigned and raises a cryptic `UnboundLocalError: local variable 'source' referenced before assignment` at line 87 instead of a clear `ValueError` naming the bad input. |
| B10 | P2 | dead code | src/mlframe/signal/dtw.py:91-113 | `_numba_cuda_diagonal_step` (the full-matrix numba.cuda cost kernel) is defined but never called anywhere in the codebase or tests -- `dtw_cuda` always uses the banded `_numba_cuda_banded_step`. Unlike its cupy analogue `dtw_cupy_full` (kept deliberately, with an explicit "retained for A/B reference, REJECTED!=DELETED" docstring and a benchmark harness exercising it), this numba.cuda full-matrix kernel has no such comment and no benchmark reference -- it reads as leftover dead code rather than an intentional kept-reference variant. |
| B11 | P2 | test coverage | src/mlframe/inspection/interaction.py | Only one test file (`tests/inspection/test_interaction.py`) covers both public functions (`friedman_h_statistic`, `pairwise_interaction_strength`); no dedicated `test_biz_val_*` file exists for this module even though it is exactly the kind of ML-behavior-affecting utility (interaction-strength ranking feeding feature-engineering decisions) the repo's own testing convention calls out for a quantitative business-value test. |

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| P1 | perf/architecture | src/mlframe/signal/dtw.py:531-578 | `dtw_dispatch` could reuse the same `try/except Exception: fall back to CPU` guard pattern already proven in `inference/logical_constraints.py:190-194` for its own cupy call -- a small, low-risk, high-value robustness fix that also closes findings B1/B2. |
| P2 | test coverage | src/mlframe/integrations/mlflow.py | Add `tests/integrations/test_mlflow.py` covering `flatten_classification_report`/`log_classification_report_to_mlflow` (including the in-place-mutation behavior from B4, once documented or fixed), `create_mlflow_run_label`, and `get_or_create_mlflow_run`'s "already active" retry path (mockable via `mlflow.start_run`). |
| P3 | ML best practice | src/mlframe/data_valuation/_knn_shapley.py | Extend the dtype guard to check both `y_train` and `y_val` (fixes B3), and add a regression test mirroring the existing `test_knn_shapley_regression_target_raises` but with a continuous `y_val` / integer-coded `y_train`. |
| P4 | robustness | src/mlframe/inference/recursive_forecast.py:118 | Guard `growth_ratio` with `np.errstate(invalid="ignore", divide="ignore")` plus an explicit `np.where(recursive_mse[0] > 0, ..., np.nan)`, matching the zero-guard pattern already used elsewhere in this cluster (e.g. `group_zero_sum_constraint.py`'s `np.where(group_weight_sum_by_code > 0, ...)`). |
| P5 | docs | src/mlframe/integrations/mlflow.py:69 | Document `embed_website_to_mlflow`'s intended trust boundary (internal MLflow-server URLs only) in its docstring, or drop `allow-scripts` from the sandbox if script execution inside the embed is not actually required. |
| P6 | perf | src/mlframe/data_valuation/_knn_shapley.py:41-54 | `_score_one_batch`'s per-row Python loop (`for r in range(Xb.shape[0])`) calling the njit recursion once per validation row could itself be pushed into a second njit/`prange` layer for very large `batch_val * n_train` products; currently only the inner recursion is JIT-compiled, so the outer per-row dispatch still pays Python loop overhead per validation point (likely negligible relative to the O(n log n) sort, but not benchmarked at very large `n_val`). |

## Coverage notes

- All 56 files in the assigned scope were read in full; nothing was skipped due to size or complexity (the largest file, `signal/dtw.py` at 594 LOC, was read completely in one pass).
- I did not execute any code (per the read-only mandate), so findings B1/B2/B3/B7 are static-analysis-derived from the guard logic and dtype checks visible in source, not confirmed via a live crash/repro. They are high-confidence based on direct code-path tracing (e.g. B3 is corroborated by the existing test `test_knn_shapley_regression_target_raises`, which only exercises the `y_train`-continuous path and never the `y_val`-continuous / `y_train`-integer combination).
- Excluded per the task's scope boundary: `feature_selection/filters/**`, `feature_selection/shap_proxied_fs/**`, and their test mirrors -- not opened, not analyzed, no findings reported about them.
- `pyutilz.performance.kernel_tuning.registry` / `pyutilz.performance.kernel_tuning.cache` (imported by `signal/dtw.py`, `inference/_ktc_dispatch.py`, `system/kernel_tuning_cache/`) are third-party (first-party sibling package) and out of this cluster's file scope; I read only the call sites and their docstrings, not the registry/cache internals, so I cannot rule out bugs inside `KernelTuningCache.get_or_tune`/`choose` itself.
