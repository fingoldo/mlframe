# Wave 8 audit 2026-11-02 -- tracker

Three read-only reports on `src/mlframe` by Opus 5.5 low-effort agents (NaN, inf and empty-input handling in metrics; resource leaks; reproducibility of seeded results). The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | NAN8-1 | nan_handling | High | `ks_statistic` looped forever on a NaN score (`_classification_extras.py`) |
| **RESOLVED** | NAN8-2 | nan_handling | High | `optimize_decision_threshold` picked a NaN-scored threshold (`threshold_optimizer.py`) |
| **RESOLVED** | NAN8-3 | nan_handling | Medium | `hosmer_lemeshow_test` dropped a group with E=0 and observed positives |
| **RESOLVED** | NAN8-4 | nan_handling | Medium | `entropy_calibration_index` returned 0.0 (perfect) for empty input |
| **RESOLVED** | NAN8-5 | nan_handling | Medium | `fast_log_loss_binary` returned 0.0 (perfect) for empty input |
| **RESOLVED** | RES8-1 | resource_leak | Medium | shared edge pool shut down while another thread still mapped on it |
| **RESOLVED** | RES8-2 | resource_leak | Low | held-back render task never settled after `close(wait=False)` |
| **RESOLVED** | RES8-3 | resource_leak | Low | `embed_website_to_mlflow` leaked the temp descriptor when `fdopen` failed |
| **RESOLVED** | RNG8-1 | reproducibility | Medium | BorutaShap lists built from string-set iteration (`PYTHONHASHSEED`) |
| **RESOLVED** | RNG8-2 | reproducibility | Medium | MRMR post-drop raw floor tie-break by set order |
| **REJECTED** | RNG8-3 | reproducibility | Low | vendored InfoNet `compute_smi_mean` uses the global numpy RNG |
| **RESOLVED** | RNG8-4 | reproducibility | Low | data-valuation functions defaulted to an entropy-seeded generator |

## Counts

12 findings: 2 High, 6 Medium, 4 Low. Dispositions: 11 RESOLVED, 1 REJECTED, 0 FUTURE.

RNG8-3 is REJECTED: `vendored/infonet/infer.py` begins with `from model.decoder import Decoder`, a module that does not exist in this tree, so the file cannot be imported and nothing reaches the global-RNG call.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| NAN8-1 | Return NaN from ks_statistic when a score is NaN instead of hanging | `tests/metrics/test_ks_statistic_nan_score_returns.py` |
| NAN8-2 | Skip NaN scores when picking the best decision threshold | `tests/calibration/test_threshold_optimizer_nan_scores.py` |
| NAN8-3 | Count a degenerate Hosmer-Lemeshow group that disagrees with its prediction | `tests/metrics/test_hosmer_lemeshow_degenerate_group.py` |
| NAN8-4, NAN8-5 | Return NaN, not a perfect score, for empty input in log loss and the entropy calibration index | `tests/metrics/test_log_loss_empty_input_is_nan.py`, `tests/calibration/test_edge_quality.py` (re-framed) |
| RES8-1 | Keep the replaced edge pool usable when the shared pool grows | `tests/feature_selection/test_edge_pool_growth_keeps_old_pool_usable.py` |
| RES8-2 | Settle held-back render tasks when the queue is closed before their predecessor ends | `tests/reporting/test_async_render_close_settles_held_back_tasks.py` |
| RES8-3 | Close the temp descriptor in embed_website_to_mlflow when fdopen fails | `tests/integrations/test_embed_website_fd_not_leaked.py` |
| RNG8-1 | Order BorutaShap's accepted, rejected and tentative lists by column order | `tests/feature_selection/boruta_shap/test_boruta_decision_lists_follow_column_order.py` |
| RNG8-2 | Break post-drop raw floor ties by column order | `tests/feature_selection/mrmr/core/test_post_drop_floor_pick_tie_break.py` |
| RNG8-4 | Seed the default generator of the data-valuation functions | `tests/data_valuation/test_adversarial_validation_default_is_deterministic.py` |

## Not covered by this wave

- The agent's lower-confidence notes were not acted on: NaN handling differs between the sequential and parallel log-loss kernels, `_anderson_darling_kernel` is compiled with `fastmath=True`, and the `fast_cv_rmse` docstring says NaN for mean <= 0 while the code checks `== 0`.
- Three tests in `tests/reporting/test_async_render_queue.py` that use the process backend fail with `BrokenProcessPool` on this machine with or without the RES8-2 change; not investigated here.
- Hand-traced failing inputs for NAN8-3 to NAN8-5 and RNG8-1/2 were confirmed by the regression tests, not by an independent reproduction of the original report.
