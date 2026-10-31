# Untrusted input and resource exhaustion audit (2026-11-01)

Read-only. No code or git state changed. Follow-up to `audits/2026-10-31/untrusted_input.md` (INP-01..09). Those items are not re-reported here.

## Scope and method

- Grep sweep of `src/mlframe` (excluding `_benchmarks/`) for: caller-sized `np.empty/zeros/full`, `while True` loops, recursion, `itertools.combinations`, `struct`/`frombuffer`/`fromfile`/`mmap`, raw `f.read()` of files, `2 ** max_depth`-style size math.
- Full reads of: `shap_proxied_fs/_shap_proxy_search.py` (generate_combinations, brute_force_top_n, total_subsets), `_shap_proxied_fit_search.py`, `_shap_proxied_fit.py:430-460`, `_shap_proxied_resolvers.py:14-68`, `_shap_proxy_interactions.py:245-285`, `engineered_recipes/_recipe_name_simplify.py`, `boruta_shap/_shadow_stats.py:350-380`, `filters/_mi_greedy_cmi_fe.py:555-580`, `training/composite/_synthetic_bases.py:30-70`, `training/core/_metadata_loader.py`, `inference/predict.py:95-125`, `training/cb/_cb_gpu_budget.py:45-70`, `filters/_gpu_resident_materialise.py:315-336`.
- Sampled only. Not a line-by-line read of the roughly 880 files with `np.empty/zeros` hits; most of those are sized by already-loaded data, as the prior audit also found.

## Summary

| ID | file:line | severity | proposed status |
|---|---|---|---|
| INP2-1 | src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_search.py:354 (reached from _shap_proxied_fit_search.py:25) | Medium | FUTURE |
| INP2-2 | src/mlframe/feature_engineering/transformer/multi_aux_ensemble.py:60,69,77,82 (also focal_lgb.py:42, rf_proximity.py:46, pseudo_smote.py:55, active_virtual.py:49) | Low | FUTURE |
| INP2-3 | src/mlframe/feature_selection/filters/engineered_recipes/_recipe_name_simplify.py:72-91 | Info | RESOLVED |
| INP2-4 | src/mlframe/inference/predict.py:111-112 | Info | DOC |
| INP2-5 | src/mlframe/training/cb/_cb_gpu_budget.py:58-59 | Info | REJECTED |
| INP2-6 | src/mlframe/training/core/_metadata_loader.py:27-29 | Info | RESOLVED |
| INP2-7 | src/mlframe/feature_selection/boruta_shap/_shadow_stats.py:366-378, training/composite/_synthetic_bases.py:40, filters/_mi_greedy_cmi_fe.py:567-574 | Info | RESOLVED |

Totals: 0 High, 1 Medium, 1 Low, 5 Info (INP2-3, INP2-6, INP2-7 RESOLVED; INP2-5 REJECTED; INP2-4 DOC).

## Details

### INP2-1 Medium: exhaustive combination enumeration materialises every C(n, r) row with no byte budget on the forced path

- Location: `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_search.py:354` calls `generate_combinations(seq, r)` (defined at :36-50). It builds the full `(C(n, r), r)` int32 array before the numba kernel sees any of it. `brute_force_top_n` (:314) is the public entry.
- Caller path: `_shap_proxied_fit_search.py:22-35` runs `brute_force_top_n` for `optimizer="bruteforce"` with no size check. The automatic dispatcher is gated (`_shap_proxied_resolvers.py:17-68`: `brute_force_max_features`=28, `brute_force_n_sub_gate`=80M, work budget 4e9 subset-rows), but the explicit forced path is not.
- The prescreen cap that bounds `n` is skipped when the user sets it. `_shapproxied_fit.py:443-444` only applies `effective_brute_force_cap` when `self.prescreen_top is None`. A user-set `prescreen_top` therefore lets `n_proxy` reach any width.
- Failure scenario (input values): `optimizer="bruteforce"`, `prescreen_top=40`, `max_features=None`, so n=40 and r ranges 1..40. At r=10, C(40,10) = 8.5e8 rows x 10 x 4 bytes = 34 GB, allocated in one piece. At r=20, C(40,20) = 1.4e11 rows, which cannot be allocated. The recursive builder also copies each intermediate level, so peak memory is several times the final array. The failure is a hang, then `MemoryError` or an OS paging-file error on Windows, rather than a clear message.
- Note: forcing exhaustive search at any size is documented intent (`_shapproxied_resolvers.py:67`), so the exhaustiveness itself is not the finding. The gap is the unchecked allocation: nothing states the required bytes or refuses before allocating.
- Fix: before enumerating, compute `total_subsets(n, min, max)` (already exists) and the implied bytes, `total * r * 4`, and raise `ValueError` naming the budget when it exceeds a cap; or stream combinations in fixed-size chunks through the kernel, which the chunked `_topn_fixed_r_parallel_colmajor` path could absorb.
- Disposition: FUTURE. Real and reachable from a user config value. Needs a chunked generator (bit-identical top-N output required) or a documented byte gate with a test; not a one-line change.

### INP2-2 Low: LightGBM `num_leaves = 2 ** max_depth` with no cap in five transformer constructors

- Location: `src/mlframe/feature_engineering/transformer/multi_aux_ensemble.py:60, 69, 77, 82`; `focal_lgb.py:42`; `rf_proximity.py:46`; `pseudo_smote.py:55`; `active_virtual.py:49`.
- Contrast: `boosting_leaf.py:102,106` already uses `min(2 ** max_depth, 63)`.
- Failure scenario: `max_depth=20` gives `num_leaves=1048576`, which LightGBM rejects at its 131072 ceiling (an exception, not a memory event). `max_depth=16` gives 65536 leaves, and LightGBM allocates per-leaf histogram state proportional to leaves x features x bins, which is large for wide frames. No explicit validation of `max_depth` exists in these constructors.
- Fix: clamp to `min(2 ** max_depth, 63)` as `boosting_leaf.py` does, or validate `max_depth` at construction.
- Disposition: FUTURE. Low because the hard ceiling is enforced by LightGBM itself, and the memory growth needs a deliberately large `max_depth`.

### INP2-3 Info (RESOLVED): recursive-descent op-name parser

- `_recipe_name_simplify.py:72-91` recurses once per nesting level of `(`. Depth is bounded by Python's recursion limit (about 1000), so a name with about 1000 nested calls raises `RecursionError`.
- `simplify_fe_name` (:165-180) wraps parse, simplify and render in `except Exception` and returns the original name. `RecursionError` is an `Exception` subclass, so the failure is caught and logged at debug. Input names come from the internal namer, with depth bounded by FE depth settings.
- Disposition: RESOLVED (caught at :178). No change needed.

### INP2-4 Info: features JSON read whole after integrity check

- `inference/predict.py:111-112` reads the whole sidecar JSON with `f.read()` before `orjson.loads`. The sha256 gate at :106 runs first, so this only reads a file the sidecar matches.
- The sidecar is an integrity check, not an authenticity check (noted at :122). Size is driven by the local model directory, which is trusted.
- Disposition: DOC. Intended local reader; state in the docstring that the model directory is trusted.

### INP2-5 Info (REJECTED): CatBoost snapshot read whole

- `training/cb/_cb_gpu_budget.py:58-59` reads the full snapshot blob into memory to find an embedded JSON header. The snapshot is written by CatBoost into mlframe's own temp dir, and its size tracks model size. No untrusted source, and the read is proportional to a file the process itself produced.
- Disposition: REJECTED as a risk. Not attacker-reachable.

### INP2-6 Info (RESOLVED): metadata raw read

- `training/core/_metadata_loader.py:27-29` reads the full raw metadata, then passes `pkl.zst` through `decompress_bounded`, which carries the streaming ceiling from `_bounded_zstd.py`. The sidecar gate at :22 runs first. Bounded.
- Disposition: RESOLVED.

### INP2-7 Info (RESOLVED): `while True` loops checked for termination

- `boruta_shap/_shadow_stats.py:366-378`: the loop exits on KS p>0.95 or after the last sample size (`element >= len(size)`), with `iteration` reset every 20 misses. Bounded.
- `training/composite/_synthetic_bases.py:40`: consumed by `parse_synthetic`-style scans; the inner `while` at :41-47 advances `start` by at least one each pass. Bounded by name length.
- `filters/_mi_greedy_cmi_fe.py:567-574`: halves `step` on GPU OOM and exits at `step <= 1`. Bounded by log2(k).
- Disposition: RESOLVED.

## Not checked

- Recursion beyond the two sites above (`generate_combinations` in INP2-1 is recursive too; its depth equals the proxy width, which is capped in the automatic path but not the forced one; noted under INP2-1).
- Per-file sizing from user window or lag parameters in `feature_engineering/` (for example `windowed_shape.py`, `stationarity.py`, `hurst.py`, `spatial.py`): grep showed many `np.empty/zeros` sites, not read individually.
- Polars and pandas loaders and their row or column limits: the prior audit covered the readers and found no size gate, which it judged intended.
- GPU allocation paths (`_gpu_resident_materialise.py`, `_fe_cmi_perm_null_gpu.py`) beyond the pinned-buffer helper read above; the VRAM headroom logic is outside this class.
- Dataset-driven memory in `data_valuation` and `competition` beyond the prior audit's INP-03.
