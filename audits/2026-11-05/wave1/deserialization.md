# Wave 1 (2026-11-05), attack class A1: unsafe deserialization and model/cache I/O

Read-only audit by one Opus 5.5 low-effort agent. Nothing was edited and no tests were run. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| D1-01 | Medium | `src/mlframe/training/feature_handling/cache.py:557` (also 565-578, 579-583) | With `allow_pickle=True`, `np.load` unpickles a pure-pickle cache file before any sidecar check, so the sidecar written at line 376 is never verified. The fallback path that does call `safe_load` is also fail-open by default. |
| D1-02 | Low | `src/mlframe/training/_trainer_train_and_evaluate_parts.py:132-133` | When no `trusted_root` is passed it defaults to the model file's own directory, so the containment check can never fail. |
| D1-03 | Low | `src/mlframe/inference/predict.py:227` then `:245`; `predict.py:118` then `:123`; `src/mlframe/training/core/_metadata_loader.py:20` then `:26-28`; `src/mlframe/training/io.py:609` (`_verify_bundle_digest`) then `:624` | Each loader hashes the file through one open and then reopens it by path to unpickle (a check-then-load race): the bytes checked are not necessarily the bytes loaded. |

### D1-01 (Medium)

`FeatureCache` writes a `.sha256` sidecar for pickled payloads (line 376), but `_deserialize` calls `np.load(path, allow_pickle=allow_pickle, mmap_mode="r")` first. For a file with no npy/npz magic header, numpy with `allow_pickle=True` calls `pickle.load` itself; the code's own comment at lines 579-580 says it "returns the unpickled object directly". The `safe_load` route at line 578 is reached only if `np.load` raises, so the sidecar is never consulted for a valid pickle. Even on that route, `allow_unverified=not _strict` means a file with no sidecar loads unless `MLFRAME_FEATURE_CACHE_STRICT=1` is set, the opposite of the fail-closed default used by `DiskCache`, the composite cache and the RFECV checkpoints. An object-dtype `.npy`/`.npz` is also unpickled by `np.load` with no sidecar check.

Failure scenario: a user turns on `CacheConfig(allow_pickle=True)` to cache DataFrames, with the cache directory on a shared or NFS path. Another principal drops a crafted pickle at a cache-key filename, or replaces an existing entry and leaves the old sidecar in place. The next cache read runs arbitrary code. The sidecar the user believed protected them is never checked, and plain `pickle` is used, not the `_SafeUnpickler` allowlist.

### D1-02 (Low)

`core/helpers.py:24-30` documents this exact pattern (root = the file's own dirname) as the no-op variant that was removed; this call site reintroduces it. A path's directory always contains the path, so `_validate_trusted_path` never rejects anything when `trusted_root` is omitted. The impact is limited because `load_mlframe_model` now refuses bundles without a recorded sha256 (`io.py:537-555`) and uses the allowlist unpickler.

Failure scenario: a caller passes a caller-influenced `model_file_name` such as `../../other_project/model.dump`. The traversal is not caught; the only guard left is a valid `.meta.json` digest, which every legitimately saved bundle elsewhere on disk has. A cached model from an unrelated project, or a stale one, is silently loaded and used as the "cached" model.

### D1-03 (Low)

None of these sites holds the file handle open from verification to load. On the composite and suite caches the same hash-then-reopen may exist inside pyutilz's `safe_load`; that was not checked.

Failure scenario: a principal who can write the bundle file but not its sidecar (a different ACL on `*.sha256`, or a sidecar on read-only storage next to a writable artefact directory) swaps the bundle between the hash read and the load read. The sidecar check passes on the original bytes and the swapped bytes are deserialised. For `infer/` models that is the denylist-only `safe_joblib_load`, so it can run code through a gadget not on the denylist. This sits outside the stated writer-controls-both threat model, but it is the one route where the integrity check is bypassed with the sidecar left intact.

## Checked and found sound

- `DiskCache` key traversal guard and fail-closed sidecar.
- `private_temp_dir`: rejects symlinks via `lstat` and checks owner and mode.
- RFECV checkpoint: fail-closed `safe_load`.
- `replay_cv_results`: requires an explicit `trusted_root`.
- All torch loads use `weights_only=True`; infonet uses `yaml.safe_load`.
- `_tiny_rerank_process` `cloudpickle.loads`: the blobs come from the parent process, not from disk.
- `_metadata_loader`: verified, allowlist unpickler.
- `load_mlframe_model` with `safe=False`: explicit opt-in, with a warning.
- Benchmark-only `pickle.load` calls read files the scripts wrote themselves.

## Not covered

- The internals of pyutilz `safe_pickle.safe_load` / `verify_sidecar` (which unpickler it uses, and whether it has the same race).
- `training/composite/cache_store.py` beyond the decided item.
- `_key_bank.py` metadata and ANN index loads beyond the grep (they go through `safe_load`; their `np.load` calls use the default `allow_pickle=False`).
- Native model loaders (CatBoost/LightGBM/XGBoost `load_model` from a path): no direct calls showed up in the grep.
- `suite_artefact_cache.py` load path beyond confirming it routes through `safe_load`.
- `mlflow.py` artefact download handling.
