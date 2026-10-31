# Unsafe deserialization audit: mlframe (src/mlframe)

Read-only audit. No code was edited and no state-changing git commands were run.

## Summary

The grep for pickle/joblib/cloudpickle/torch.load/dill/marshal loads, plus `np.load`, `safe_load`, `safe_joblib_load` and `load_mlframe_model`, found about 40 sites. Every `torch.load` call in `src` passes `weights_only=True`. Most sites are either sidecar-gated (`safe_pickle.safe_load`, fail-closed) or use `np.load(allow_pickle=False)`.

The main weaknesses are in the sites that trust a pickle with no integrity gate, or only a sha256 gate that proves integrity and not authenticity:

- `FeatureCache` / `CompositeCache`: `cache_store.py:385` always loads unsigned pickles from `cache_dir` (`allow_unverified=True`). This is the only High finding, because it is the default path and the cache directory can be shared.
- `load_mlframe_model` (`io.py`) is the default loader for model bundles. It has an allowlist but no sha256 check. The allowlist is the only barrier.
- `safe_joblib_load` (`io.py:244`) is a denylist, not an allowlist. It is used for `infer/` models and for joblib metadata.

Trust model: a writer to a model directory or cache directory is treated as already able to execute code in most of these designs. The findings below say where mlframe's own code makes that boundary weaker than its comments claim.

## Findings table

| ID | file:line | Severity | Status proposal |
|---|---|---|---|
| DESER-01 | `src/mlframe/training/io.py:601` | Medium | DOC |
| DESER-02 | `src/mlframe/training/io.py:524` (`load_mlframe_model`, callers `_trainer_train_and_evaluate_parts.py:126`, `core/_phase_persist_post.py:49`) | Medium | FUTURE |
| DESER-03 | `src/mlframe/training/_unpickle_policy.py` (`resolve_restricted_class`, allowlist prefixes and `_DENIED_NAMES`) | Medium | FUTURE |
| DESER-04 | `src/mlframe/inference/predict.py:245` (`safe_joblib_load` in `read_trained_models`) | Medium | FUTURE |
| DESER-05 | `src/mlframe/inference/predict.py:123` (`features.dump` joblib fallback) | Low | FUTURE |
| DESER-06 | `src/mlframe/estimators/pipelines.py:111` (`replay_cv_results`, plain `joblib.load`) | Medium | FUTURE |
| DESER-07 | `src/mlframe/training/core/_metadata_loader.py:20-21` (legacy joblib metadata, no sidecar) | Medium | FUTURE |
| DESER-08 | `src/mlframe/training/feature_handling/cache.py:574-580` (pickle fallback, `allow_pickle=True`) | Medium | DOC |
| DESER-09 | `src/mlframe/training/feature_handling/cache.py:557` (`np.load` with `allow_pickle=False` default) | Info | RESOLVED |
| DESER-10 | `src/mlframe/training/composite/cache_store.py:385` (`safe_load(..., allow_unverified=True)`) | High | FUTURE |
| DESER-11 | `src/mlframe/training/_trainer_train_and_evaluate_parts.py:117-119` (`trusted_root` defaults to the file's own directory) | Low | FUTURE |
| DESER-12 | `src/mlframe/utils/disk_cache.py:368` | Info | RESOLVED |
| DESER-13 | `src/mlframe/feature_selection/wrappers/rfecv/__init__.py:598` | Info | RESOLVED |
| DESER-14 | `src/mlframe/feature_engineering/transformer/_key_bank.py:179-182, 182, 204` | Info | RESOLVED |
| DESER-15 | `src/mlframe/training/composite/discovery/_tiny_rerank_process.py:92-93` (`cloudpickle.loads`) | Low | REJECTED |
| DESER-16 | `src/mlframe/training/_unpickle_policy.py:145` (`torch.load(..., weights_only=True)`) | Info | RESOLVED |
| DESER-17 | `src/mlframe/training/neural/_recurrent_wrapper_base.py:581` | Info | RESOLVED |
| DESER-18 | `src/mlframe/training/neural/_flat_torch_module/__init__.py:591` | Info | RESOLVED |
| DESER-19 | `src/mlframe/feature_selection/filters/_vendored/infonet/infer.py:51` | Info | RESOLVED |
| DESER-20 | Dev benchmark caches: `feature_selection/_benchmarks/fs_hybrid/round4_*.py` (47, 64, 78, 132, 244), `reporting/_benchmarks/bench_gil_contention.py:50,70`, `bench_async_render_overlap.py:87` | Low | REJECTED |
| DESER-21 | Same-process round-trips: `round4_fe_median_gated_recipe_proof.py:47`, `test_hybrid_tree_member.py:136` | Info | REJECTED |
| DESER-22 | `src/mlframe/feature_selection/_benchmarks/fs_hybrid/_realdata_cache.py:197` (`np.load(allow_pickle=False)`) | Info | RESOLVED |

Count by severity: High 1, Medium 7, Low 4, Info 10. Total 22.

## Per-finding details

### DESER-01: `dill.load` on the unrestricted path (`io.py:601`)
- **Loads:** a dill bundle, using the unrestricted `dill.load(zf)`.
- **Source:** the model file path given to `load_mlframe_model(file, safe=False)`.
- **Untrusted reach:** only if a caller passes `safe=False` with an untrusted file. No `src` caller passes `safe=False`; the only one is in `training/composite/estimator/_estimator.py:464`, which is `clone(..., safe=False)` and unrelated. The site emits a `UserWarning` and is documented as opt-in.
- **Impact:** arbitrary code execution on load.
- **Severity:** Medium (opt-in, unsafe by design, not reachable by default).
- **Disposition: DOC.** The docstring at `io.py:531` already says "unsafe, RCE risk from untrusted sources". Add the same warning to the public docs and the `safe` parameter reference. Keep the `nosec` comment.

### DESER-02: `load_mlframe_model` default path has no sha256 gate (`io.py:524-612`)
- **Loads:** zstd-compressed model bundles, through `_SafeUnpickler` (dill allowlist).
- **Source:** `_trainer_train_and_evaluate_parts.py:126` (cached model at `model_file_name`, `trusted_root` defaulting to the file's own directory) and `core/_phase_persist_post.py:49` (`path` from the suite). Both come from the suite/model directory.
- **Untrusted reach:** a caller-supplied model directory or a planted file in one. `load_mlframe_model` runs only `validate_load_meta_sidecar` (non-strict, version warning). It never calls `verify_sidecar`, so there is no integrity check at all. The comment at `_trainer_train_and_evaluate_parts.py:117-121` says this openly ("authenticity half is still open").
- **Impact:** code execution on load for any class the allowlist admits. The allowlist is broad (see DESER-03).
- **Severity:** Medium.
- **Disposition: FUTURE.** Reason: the sidecar is written by the save site, and adding a fail-closed `verify_sidecar` gate changes the load contract for every existing bundle. Proposed fix: have the save site emit a `.sha256` sidecar, then call `safe_load` (fail-closed) in `load_mlframe_model` when `safe=True`, with an env-var opt-out for legacy bundles.

### DESER-03: `_SafeUnpickler` allowlist has broad prefixes (`_unpickle_policy.py`, `resolve_restricted_class`)
- **Loads:** every global referenced by a dill bundle or by `_SafeJoblibUnpickler` (`io.py:189`, `io.py:233`).
- **Source:** same as DESER-02 and DESER-04.
- **Untrusted reach:** a tampered bundle chooses the globals. The allowlist is the only control.
- **Impact:** the allowlist sets `numpy`, `pandas`, `polars`, `sklearn`, `scipy`, `catboost`, `lightgbm`, `xgboost`, `joblib` and `pyarrow` to `False`, so any attribute there resolves unless it is in `_DENIED_NAMES`, `_DENIED_NAME_PREFIXES` or `_DENIED_MODULE_PREFIXES`. The name check matches exact dotted components, so names such as `load_model` (xgboost `Booster.load_model`, catboost `load_model`) are not denied. `mlframe` and `collections` are `True` (class-only). Any reachable class whose `__setstate__` or `__reduce__` does IO would then be reachable.
- **Not verified by execution:** I did not build a gadget. This is a read of the policy tables only.
- **Severity:** Medium.
- **Disposition: FUTURE.** Reason: a gadget review of each allowlisted library is needed before tightening. Proposed fix: switch the `False` entries to class-only (`True`) for the ML libraries, and add `load_model`, `load_config`, `save_raw` and similar to `_DENIED_NAMES`. Add a regression test that loads a fixture bundle and asserts a `load_model` reference is refused.

### DESER-04: `safe_joblib_load` denylist for `infer/` models (`predict.py:245`)
- **Loads:** joblib model files under `infer/<featureset>/`, through `_SafeJoblibUnpickler` (`io.py:233`, denylist).
- **Source:** the inference folder. `read_trained_models` pins the folder with `validate_trusted_path` (`predict.py:172-175`), and checks a sha256 sidecar (`_verify_sidecar`, integrity only).
- **Untrusted reach:** a caller-chosen `featureset` is contained. A writer to the `infer/` directory can replace a model and its sidecar. The code comment says so.
- **Impact:** code execution for anything the denylist misses. The denylist is weaker than the allowlist in DESER-03.
- **Severity:** Medium.
- **Disposition: FUTURE.** Proposed fix: move `infer/` models to `load_mlframe_model` (allowlist) with the sidecar gate from DESER-02, or accept the denylist explicitly and document that writes to `infer/` are code execution.

### DESER-05: joblib `features.dump` fallback (`predict.py:123`)
- **Loads:** the features file through `safe_joblib_load` when no `.json` sidecar exists.
- **Source:** `infer/<featureset>/features.dump`.
- **Untrusted reach:** the same as DESER-04. The JSON sidecar is preferred, so this is only reached for legacy bundles.
- **Severity:** Low.
- **Disposition: FUTURE.** Proposed fix: write features as JSON only and drop the joblib fallback, after a migration period.

### DESER-06: `replay_cv_results` uses plain `joblib.load` (`estimators/pipelines.py:111`)
- **Loads:** a joblib dump of CV results, through unrestricted `joblib.load`, after `_verify_sidecar`.
- **Source:** `fname`, a caller-supplied path, checked with `trusted_root` (required).
- **Untrusted reach:** a caller who can write next to `fname` (and its sidecar) gets code execution. The sidecar is integrity only.
- **Impact:** code execution on load.
- **Severity:** Medium.
- **Disposition: FUTURE.** Proposed fix: replace with `safe_joblib_load` (the payload is dicts of metric arrays, so the denylist fits). The code change is one line plus a test.

### DESER-07: legacy joblib metadata with no sidecar (`training/core/_metadata_loader.py:20-21`)
- **Loads:** `metadata.joblib` through `safe_joblib_load`. The `kind == "joblib"` branch returns before the sha256 check that the `pkl`/`pkl.zst` branch enforces (line 24).
- **Source:** the suite directory.
- **Untrusted reach:** a writer to the suite directory.
- **Impact:** code execution, through the denylist.
- **Severity:** Medium.
- **Disposition: FUTURE.** Proposed fix: apply `verify_sidecar` to the joblib branch too, with the same `MLFRAME_ALLOW_UNVERIFIED_PICKLE` opt-in the other branch uses.

### DESER-08: pickle fallback in `FeatureCache` (`feature_handling/cache.py:574-580`, `_serialize` at 528)
- **Loads:** a pickle payload from the cache directory, only when `CacheConfig.allow_pickle=True`. The fallback calls `safe_load(path, allow_unverified=not strict)`, so unsigned legacy payloads load unless `MLFRAME_FEATURE_CACHE_STRICT=1`.
- **Source:** the cache directory.
- **Untrusted reach:** a writer to a cache directory, when the user has opted in.
- **Impact:** code execution.
- **Severity:** Medium.
- **Disposition: DOC.** The default (`allow_pickle=False`) is safe, and the opt-in is explained at `cache.py:528-532` and `config.py:128-134`. Document that `allow_pickle=True` needs a private cache directory. A future change could make strict mode the default.

### DESER-09: `np.load` in `FeatureCache` (`cache.py:557`)
- **Loads:** npz payloads with `allow_pickle=allow_pickle`, which is False by default.
- **Severity:** Info. **Disposition: RESOLVED.** The default is `allow_pickle=False`, so object arrays are refused. The `kind` sentinel is a uint8 array (`cache.py:503-507`), so the numeric path needs no pickle.

### DESER-10: unsigned pickle loads from the composite cache (`training/composite/cache_store.py:385`)
- **Loads:** `<cache_dir>/<key>.pkl` through `safe_load(..., allow_unverified=True)`, with no env-var or config opt-in.
- **Source:** `cache_dir`, which can be a shared path.
- **Untrusted reach:** any principal with write access to `cache_dir` can plant a pickle with no sidecar. The `allow_unverified=True` is hard-coded (comment at `cache_store.py:373-381` says it is for migration), so the fail-closed default in `pyutilz.core.safe_pickle` is bypassed on this path.
- **Impact:** code execution on the next read.
- **Severity:** High.
- **Disposition: FUTURE.** Reason: legacy entries written before sidecars exist would turn into cache misses and be recomputed, which is a cost, not a correctness problem. Proposed fix: drop the hard-coded `allow_unverified=True`, so missing sidecars fail closed and become cache misses. Keep a one-time migration by env var `MLFRAME_ALLOW_UNVERIFIED_PICKLE`. Add a test that a sidecar-less planted `.pkl` is refused and treated as a miss.

### DESER-11: `trusted_root` defaults to the file's own directory (`_trainer_train_and_evaluate_parts.py:117-119`)
- **Loads:** a cached model path, checked with `validate_trusted_path`.
- **Issue:** when `trusted_root` is None, the root is `dirname(model_file_name)`, so the containment check always passes. `core/helpers.py:17-30` says this exact no-op default must not exist.
- **Severity:** Low (the path comes from the caller's configuration, and DESER-02 is the real exposure).
- **Disposition: FUTURE.** Proposed fix: make `trusted_root` required here, like `replay_cv_results` does, or pass the suite directory explicitly.

### DESER-12: `DiskCache.get` (`utils/disk_cache.py:368`)
- **Loads:** via `safe_load`, which fails closed on a missing sidecar (`pyutilz` `verify_sidecar`, line 197). The key is also checked against the cache root (`_key_path`).
- **Severity:** Info. **Disposition: RESOLVED.**

### DESER-13: RFECV checkpoint (`feature_selection/wrappers/rfecv/__init__.py:598`)
- **Loads:** the resume checkpoint through `safe_load`, which fails closed. A mismatch starts fresh (line 605).
- **Severity:** Info. **Disposition: RESOLVED.**

### DESER-14: `KeyBank` cache (`feature_engineering/transformer/_key_bank.py:179-204`)
- **Loads:** `projections.npy`, `k_proj.npy`, `y_train.npy` with `np.load` at its default (`allow_pickle=False`). `metadata.pkl` and `ann_h*.pkl` go through `safe_load` (fail-closed). The save side writes sidecars (line 266).
- **Severity:** Info. **Disposition: RESOLVED.**

### DESER-15: `cloudpickle.loads` in the rerank pool (`_tiny_rerank_process.py:92-93`)
- **Loads:** `transform_blob` and `fitted_params_blob` from the task dict that the parent process builds and passes to a worker.
- **Source:** same-process memory via `ProcessPoolExecutor`, not disk or network.
- **Severity:** Low. It would become High if the task dict were ever read from disk or a socket.
- **Disposition: REJECTED.** Reason: the bytes never leave the parent/child pair on one host and no external input reaches them. Add a comment at the call site so a future change to read tasks from disk does not inherit this assumption.

### DESER-16: `_safe_torch_load_from_bytes` (`_unpickle_policy.py:145`)
- **Loads:** torch storage bytes with `weights_only=True`. This replaces the unrestricted nested `torch.load`.
- **Severity:** Info. **Disposition: RESOLVED.**

### DESER-17: `_recurrent_wrapper_base.py:581`
- **Loads:** a checkpoint with `torch.load(path, map_location="cpu", weights_only=True)`.
- **Severity:** Info. **Disposition: RESOLVED.**

### DESER-18: `_flat_torch_module/__init__.py:591`
- **Loads:** the best checkpoint with `weights_only=True`. A failure is logged, not retried with an unrestricted load.
- **Severity:** Info. **Disposition: RESOLVED.**

### DESER-19: InfoNet checkpoint (`feature_selection/filters/_vendored/infonet/infer.py:51`)
- **Loads:** a state dict with `weights_only=True`. The docstring says `torch < 1.13` raises `TypeError` rather than falling back.
- **Severity:** Info. **Disposition: RESOLVED.**

### DESER-20: dev benchmark caches (`_benchmarks/` trees)
- **Loads:** `pickle.load` of per-script caches, for example `round4_gated_recall_bench.py:78` (cache at `D:/Temp/gated_recall_cache`, line 44), `round4_knockoff_fdr_bench.py:47`, `round4_noise_floor_bench.py:64,244`, `round4_union_backward_bench.py:132`, `bench_gil_contention.py:50,70`, `bench_async_render_overlap.py:87`.
- **Source:** the script's own output directory, which can be a shared temp path.
- **Severity:** Low. These are not on any runtime path, and each one has an `nosec` comment saying the script wrote the file.
- **Disposition: REJECTED.** Reason: development-only scripts that load files they wrote. If the benchmarks are ever run from shared storage, switch them to `safe_dump`/`safe_load` (this is the one-line fix to apply then).

### DESER-21: same-process round-trips (`round4_fe_median_gated_recipe_proof.py:47`, `test_hybrid_tree_member.py:136`)
- **Loads:** `pickle.loads(pickle.dumps(x))` in memory.
- **Severity:** Info. **Disposition: REJECTED.** No external bytes.

### DESER-22: `_realdata_cache.py:197`
- **Loads:** `np.load(npz_path, allow_pickle=False)`.
- **Severity:** Info. **Disposition: RESOLVED.**

## Notes on scope

- I did not run any exploit or gadget. Severities are from reading the code paths.
- `pyutilz.core.safe_pickle` (`C:\Users\Admin\Machine learning\pyutilz\src\pyutilz\core\safe_pickle.py`) was read only for the fail-closed sidecar default. Its behaviour is the basis for DESER-10 through DESER-14.
- The `safe_load` default (missing sidecar is refused unless the env var is set) is the reason most sites are RESOLVED.
