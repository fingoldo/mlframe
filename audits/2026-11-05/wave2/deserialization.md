# Wave 2: Deserialization audit (src/mlframe)

Scope: every `pickle`/`dill`/`joblib`/`cloudpickle`/`torch.load`/`np.load(allow_pickle=True)` site found by grep under `src/mlframe`.
Production loaders (`load_mlframe_model`, `inference.predict` model reads, `_metadata_loader`) route through `_SafeUnpickler` / `safe_joblib_load_bytes` with the digest checked on the same bytes that are unpickled.
All `torch.load` sites use `weights_only=True`. Benchmark `pickle.load` sites read files the same script wrote (dev-only).

| id | severity | file:line | statement | failure scenario | disposition |
|---|---|---|---|---|---|
| DESER-W2-1 | Medium | training/feature_handling/cache.py:557-565 | With `allow_pickle=True`, the sidecar is checked by path, then `np.load(path, allow_pickle=True)` reopens the file: a check-then-use race, and a missing sidecar passes unless `MLFRAME_FEATURE_CACHE_STRICT=1`. | An attacker with write access to the cache dir deletes the `.sha256` (or swaps the payload after verification) and gets arbitrary code execution through numpy's unpickle. Opt-in only (default `allow_pickle=False`). | RESOLVED-needed: read bytes once, verify, then `np.load(io.BytesIO(b))`; make a missing sidecar fatal when `allow_pickle=True` or document it explicitly. |
| DESER-W2-2 | Low | training/feature_handling/cache.py:578-586 | The legacy pickle fallback uses `safe_load(allow_unverified=True)` by default, so an unsigned pickle file runs unrestricted. | Same as W2-1: a planted pickle with no sidecar is executed when `allow_pickle=True`. | DOC (or flip strict to default ON with an opt-out, per the "corrective default on" rule). |
| DESER-W2-3 | Info | training/io.py:668-676, inference/predict.py:266-269 | The sha256 sidecar is an integrity check, not authenticity: an attacker who can write the directory rewrites payload and sidecar. | Wrong-model substitution with allowlisted classes (silent wrong predictions); the allowlist still blocks gadget RCE. `safe=False` is an explicit, warned opt-in. | DOC (already noted in code comments); HMAC signing would be FUTURE. |
| DESER-W2-4 | Info | training/composite/discovery/_tiny_rerank_process.py:92-93 | `cloudpickle.loads` on task blobs. | Blobs are produced in-process by the parent and passed to its own worker process; there is no disk or network source. | REJECTED: same trust domain. |
| DESER-W2-5 | Info | reporting/_benchmarks/*, feature_selection/_benchmarks/fs_hybrid/* | Unrestricted `pickle.load` of local bench cache files. | Only reachable by running dev bench scripts against a tampered local cache. | REJECTED: dev-only, not imported by production paths. |

No High findings. `core/helpers.validate_trusted_path` is fail-closed and resolves symlinks with realpath.
