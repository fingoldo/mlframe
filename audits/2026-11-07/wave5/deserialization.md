# Unsafe deserialization, wave 5 (2026-11-07)

Scope: `src/mlframe` on HEAD, `_benchmarks/` excluded per DESER-20/21. Prior dispositions read first: `audits/2026-10-31/_TRACKER.md` (DESER-01..22), `audits/2026-10-31/deserialization.md`, `audits/2026-11-06/wave4/deserialization.md` and `audits/2026-11-06/wave4/_TRACKER.md`.

Method: grep for `pickle.load(s)`, `Unpickler(`, `joblib.load`, `dill.load`, `cloudpickle.loads`, `torch.load`, `np.load`, `allow_pickle=True`, `yaml.load`, `read_pickle`, `from_pretrained`, `load_model`, `onnx`, and keras or xgboost model loaders. Each hit was traced to its input source and checked for a guard.

## Findings

| ID | Severity | file:line | Failure scenario | Disposition |
|----|----------|-----------|------------------|-------------|
| - | - | - | No new defect. Every loader call on HEAD maps to a DESER-01..22 row or a wave-4 checked-safe entry, and those rulings still hold. | - |

## Checked and found safe (HEAD)

- `training/io.py:193,237,273,672`: restricted `_SafeUnpickler` / `_SafeJoblibUnpickler`. `io.py:679` runs vanilla `dill.load` only when the caller passes `safe=False` explicitly and gets a warning (DESER-01 DOC).
- `training/_unpickle_policy.py:145`: nested storage loads use `torch.load(weights_only=True)`. Lines 177 and 191 raise on the denylist and the allowlist (DESER-03/16).
- `training/core/_metadata_loader.py:35`: restricted unpickler (DESER-07).
- `training/feature_handling/cache.py:577`: `allow_pickle=True` runs only on bytes already checked against the sha256 sidecar. Line 579 is the unverified path and uses `allow_pickle=False`. Line 593 is the pickle fallback through `_SafeUnpickler` (DESER-08/09).
- `training/neural/_recurrent_wrapper_base.py:581`, `training/neural/_flat_torch_module/__init__.py:600`, `feature_selection/filters/_vendored/infonet/infer.py:51`: all use `weights_only=True` (DESER-17/18/19). `_neural_mi.py:340` reaches the infonet loader after the digest sidecar check.
- `feature_engineering/transformer/_key_bank.py:187-189`: `np.load` with the default `allow_pickle=False` (DESER-14).
- `training/composite/discovery/_tiny_rerank_process.py:92-93`: `cloudpickle.loads` reads in-process task blobs, not files (DESER-15 REJECTED).
- `core/helpers.py:33`: callers that load from a caller-supplied path must pass `trusted_root`. `_trainer_train_and_evaluate_parts.py:126` and `estimators/pipelines.py:100` route through it (DESER-06/11).
- `training/feature_handling/hf_provider.py:259,267,275,454`: `trust_remote_code` is False by default (line 230) and enabling it warns. The revision is pinned by `resolve_revision` (line 240), and an unpinned model warns. The `.bin` weight path is handled by transformers' own `torch.load` and falls under NET-06 (FUTURE), so it is not re-reported.
- `feature_selection/filters/_neural_mi.py:493`: `from_pretrained` is pinned by revision.
- Outside `_benchmarks/` there are no `yaml.load`, `pd.read_pickle`, `shelve`, `marshal`, raw `pickle.load(s)` or keras/xgboost/ONNX file loaders in `src/mlframe`.
