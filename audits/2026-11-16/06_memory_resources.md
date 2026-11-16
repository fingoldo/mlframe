# 06 Memory and resources audit (src/mlframe, MRMR/benchmarks excluded)

| ID | severity | file:line | finding | evidence | disposition |
|----|----------|-----------|---------|----------|-------------|
| M1 | med | competition/leak_scan.py:119 | Full-frame `df.copy()` solely to insert a `__target__` column, against the no-whole-frame-copy rule. Only the mask `_nonnull_nonzero_mask(work)` is needed. | `work = df.copy(); work.insert(0, "__target__", np.asarray(target))` | TODO: regression test `test_leak_scan_with_target_does_not_copy_whole_frame` (tracemalloc/peak or assert input df identity and columns unchanged, no copy of float blocks) |
| M2 | low | metrics/_fairness_metrics.py:122 | `feature_vals.copy().replace(...)`: `.replace` already returns a new object, so the `.copy()` is a redundant extra column copy. | `feature_vals = feature_vals.copy().replace({cat: rare_group_name ...})` | TODO: `test_fairness_merge_rare_cats_does_not_mutate_input_series` (also guards removal of redundant copy) |
| M3 | low | feature_engineering/cat_cooccurrence_svd.py:299,318 | Early-exit returns `X.copy()` of the whole input frame when no columns qualify. Callers may rely on copy semantics, so it cannot simply be dropped. | `return X.copy(), [], []` | FUTURE: no-op path, only matters for huge frames; changing return-aliasing is an API decision |
| M4 | low | feature_engineering/bruteforce.py:343 | `tmp_df[target_col].copy()` copies one column only (not a frame) and is needed since the frame is later modified. | line 343 | REJECTED: single column, mutation protection justified |
| M5 | low | preprocessing/cleaning.py:140-163 | Defragmenting `df.copy()`. | Already guarded by `_DEFRAG_COPY_MAX_BYTES` and RAM-growth check | RESOLVED |
| M6 | low | estimators/pipelines.py:161 | `tempfile.mkdtemp` scratch dir. | Removed in `finally: shutil.rmtree(...)` | RESOLVED |
| M7 | low | reporting/_async_render.py:500 | Array snapshot `v.copy()`. | Only when `nbytes <= _snapshot_max`; larger arrays get `readonly_view` | RESOLVED |
| M8 | low | feature_selection/filters/_neural_mi.py:345,608; _oof_warp_service.py:134; info_theory/_cmi_cuda.py:84; _null_max_perm_gpu.py:214 | Module-level caches checked for unbounded growth. | Each evicts: `len > 8` pop, `>= 4` clear, weakref purge, keep-latest-only | RESOLVED |
| M9 | low | feature_selection/boruta_shap/_fit_explain.py:344 | `self.X_` stored as shallow copy (`deep=False`) for pandas. | line 344 | RESOLVED (view, not full copy) |
| M10 | low | feature_selection/filters/fleuret.py:230 | `data.copy()` of the data array inside a joblib worker (per worker). Filters tree is MRMR-adjacent. | `data_copy = data.copy()` | FUTURE: out of primary scope (MRMR-family filters); copy likely guards numba in-place mutation, needs separate review |

Scan method: grep of `.copy()`, deepcopy, mkdtemp/NamedTemporaryFile, unbounded `lru_cache`/module dict caches; candidate sites read. Bare `open()` without context manager found none in the grep hit set. No tests run (read-only audit).
