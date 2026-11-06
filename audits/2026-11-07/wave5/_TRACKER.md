# Wave 5 tracker: full attack-class audit of mlframe (2026-11-07)

Nine read-only Opus 5.5 (low effort) audits, one per attack class. Reports sit next to this file. Dispositions: RESOLVED, OPEN (verified, not yet fixed), FUTURE (real, deferred, with reason), DOC (documented, with reason), REJECTED (not a defect, with reason).

| ID | Sev | Status | Location | Note |
|---|---|---|---|---|
| NC-1 | High | RESOLVED | `metrics/_core_brier.py:81`, `metrics/_log_loss_and_separation.py:81`, `metrics/regression/_regression_metrics.py:72,93`, `metrics/_core_auc_brier.py:58,92` | Unguarded prange metrics reached from the shap-proxy revalidation thread pool. Fixed: `_loss_from_predictions` (`shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_loss.py`) enters `parallel_kernel_entry`; test `tests/feature_selection/shap_proxied/test_revalidate_loss_enters_parallel_guard.py` fails on the old code (no guard entry) and passes now; 15 tests pass. Other metric call sites in `metrics/` are not guarded by this change (see NC-3/NC-4/NC-5). |
| NC-2 | High | RESOLVED | `training/composite/transforms/unary.py:58,86`, `transforms/nonlinear.py:83`, `transforms/_nonlinear_ewma_fracdiff.py:68,108` | Unguarded prange transform kernels reached from composite-estimator pools. Fixed at the single gateway: `call_transform` (`training/composite/transforms/_call_gateway.py`) enters `parallel_kernel_entry` around every `fit`/`forward`/`inverse`. Test `tests/training/composite/transforms/test_call_transform_enters_parallel_guard.py` fails on the old code (no guard entry) and passes now. |
| NC-3 | Medium | RESOLVED | `metrics/_core_auc_brier.py:58,92` via `training/baselines/_baseline_diagnostics_ablation.py:96` | joblib threading backend, up to 8 workers, calls `fast_roc_auc`. Fix: guard the AUC sort on the large-n path. |
| NC-4 | Medium | OPEN | `feature_selection/filters/_fleuret_confidence_prange.py:22` and the other unguarded kernels reached from `MRMR.fit` through `stability.py:222` | Bootstrap stability threads enter MRMR's prange kernels. Fix: guard at the kernel call sites. |
| NC-5 | Medium | OPEN | `tests/test_meta/test_no_unguarded_nested_parallel.py:60`, `_nested_parallel_scan.py:151` | The gate treats only `ThreadPoolExecutor(...)` as a root; joblib `Parallel(backend="threading"/prefer="threads")` (about 20 call sites) is not scanned. Fix: extend the scanner so NC-1..NC-4 are caught. |
| NC-6 | Low | FUTURE | `training/_feature_importances.py:366` | joblib threading around sklearn `permutation_importance`; the scorer is sklearn r2/accuracy, so the only route into a prange kernel is a composite estimator's predict. Deferred until NC-2 is fixed. |
| NC-7 | Info | DOC | `_mi_greedy_cmi_fe_steps.py:62` | The scanner reached it through a name collision; the module already imports the guard. |
| NC-8 | Info | REJECTED | `_edge_pool.py:50`, `utils/daemon_task.py:34` | False path from the generic name `run`; the daemon task runs one thread. |
| NC-9 | Info | REJECTED | `_numerical_counts.py:211` | `enrich_ensemble_preds_with_numaggs` starts no pool; the match was on a nested name. |
| W5-SUP-01 | Low | OPEN | `.github/workflows/release.yml:43`, `ci.yml:241,495,747,960`, `deep-nightly.yml`, `dep-floors.yml`, `fs-benchmark-nightly.yml`, `macos-abort-probe.yml:184` | Some `astral-sh/setup-uv` steps are pinned by tag, not SHA. Fix: pin to the SHA of the same tag (one-line per step). |
| W5-SUP-02 | Info | DOC | `audits/2026-11-06/wave4/_TRACKER.md:16` | Tracker row W4-SUP-01 says the three revs are SHAs; they are still tags (owner decision, commit b987f5bee). Wording fix only. |
| DYN5-1 | Low | OPEN | `feature_engineering/windowed_edge_diff.py:73`, `cross_sectional_neighbors.py:59-61` | `getattr(np, agg)` / `getattr(np, stat)` with a config-supplied name. Fix: allow-list the reducer names. |
| DYN5-2 | Info | DOC | `training/composite/_estimator_dispatch.py:68-79` | EXE-10 / DYN4-1 is RESOLVED on HEAD; restated for the record. |
| DYN5-3 | Info | DOC | `training/feature_handling/hf_provider.py:230`, `providers.py:213` | `trust_remote_code` can be enabled by a URI query string. Operator-supplied configuration, documented. |
| NET5-01 | Info | DOC | `integrations/mlflow.py:95-97` vs `:106-110` | Same docstring issue as NET4-01 (still open in wave 4 as DOC). |
| W5-INP-01 | Low | OPEN | `utils/_param_oracle.py:525-531` | `_all_combos` materialises the full Cartesian product: 8 knobs x 10 values is 1e8 dicts. Fix: iterate lazily or cap the count. |
| W5-INP-02 | Info | DOC | `feature_selection/shap_proxied_fs/_shap_proxy_interactions.py:528,557-558` | Joint bincount length is `nb_a*nb_b` from data bins; bounded by the existing bin cap. |
| W5-FS-01 | Info | OPEN | `utils/private_temp.py:36-45`, callers | Bookkeeping: FS-02..FS-05 in `audits/2026-10-31/_TRACKER.md` are already done on HEAD (`private_temp_dir`, `mkdtemp`) but still listed FUTURE; mark RESOLVED. |

Checked and clean: subprocess (0), deserialization (0), secrets (0).
