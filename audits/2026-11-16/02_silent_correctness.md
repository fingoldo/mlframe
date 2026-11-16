# 02 Silent correctness audit (src/mlframe, MRMR/benchmarks/profiling excluded)

Method: pattern sweeps (except/pass: 201 hits in 126 files, mutable defaults, astype(int)/int8 truncation, nan_to_num) plus reading the hits in metrics, calibration, inference, data, training/core. Read-only; no tests were needed. Coverage was shallow: the remaining ~190 except/pass sites (disk_cache.py 12, composite/cache_store.py 18, feature_handling/*, shap_proxied_fs/*) were NOT reviewed and are listed as an open sweep (S-9).

| ID | severity | file:line | finding | evidence | disposition |
|----|----------|-----------|---------|----------|-------------|
| S-1 | low | metrics/_ice_metric.py:776 | `except (AttributeError, KeyError): pass` when re-attaching unsafe params to `_init_params` | the next statement retries via `set_params`, so the value is still attached | REJECTED: fallback path exists, no wrong result |
| S-2 | low | calibration/quality.py:358 | label/title setting swallowed | cosmetic only, no numeric effect | REJECTED: plotting cosmetics |
| S-3 | low | training/core/_phase_train_one_target.py:549 | `move_to_end` KeyError swallowed | comment states the value in hand is still correct, only LRU touch is lost | REJECTED: documented benign race |
| S-4 | low | metrics/_core_brier.py:174 | `(y_proba > 0.5).astype(int)` | NaN/out-of-range already rejected via non-finite Brier guard at :172; multiclass y raises at :182 | REJECTED: guarded |
| S-5 | low | inference/explainability.py:43 | int8 cast of binary labels | only for nclasses==2; multiclass branch uses argmax at full width (:44) | REJECTED: no truncation |
| S-6 | low | data/datasets/_sapp.py:76 | `target > target.mean()` binarize | NaN target (log of X3 guarded by 1e-12 floor at :67) cannot occur | REJECTED: guarded |
| S-7 | low | metrics/_core_precision_mape.py:129 | `np.nan_to_num(arr, nan=zero_division)` mutates per-class arrays in place before averaging | precision/recall/f1 already assigned zero_division for empty denominators (:103-107), so nan_to_num is a no-op safety net | REJECTED: consistent with sklearn zero_division semantics |
| S-8 | low | grep `def f(x=[]/{}/set())` over src/mlframe | mutable default arguments | zero matches | REJECTED: none present |
| S-9 | low | disk_cache.py, composite/cache_store.py, feature_handling/{locking,cache_backend,fingerprint}.py, shap_proxied_fs/_shap_proxy_explain.py | ~190 other `except ...: pass` sites not individually reviewed | counted by sweep only | FUTURE: not reviewed in this pass; cache/lock sites are most likely benign (miss => recompute), SHAP explain/loss sites deserve a read first |

No confirmed silent-correctness defect was found in the reviewed sample. The codebase already carries extensive guards and comments for the classes searched, so no TODO rows and no regression tests are proposed.
