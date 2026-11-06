# Unsafe deserialization, wave 4 (2026-11-06)

Scope: `src/mlframe`, HEAD. Prior dispositions: `audits/2026-10-31/_TRACKER.md` DESER-01..22.

## Findings

| ID | Severity | file:line | Failure scenario | Disposition |
|----|----------|-----------|------------------|-------------|
| - | - | - | No new defect found; every loader call on HEAD maps to a DESER-01..22 row whose disposition still holds. | - |

## Checked and found safe (HEAD)

- `training/io.py:193,237,273,672`: `_SafeUnpickler` / `_SafeJoblibUnpickler` restricted `find_class` (DESER-02/03/04).
- `training/_unpickle_policy.py:145,191`: `torch.load(weights_only=True)`, denylist raise (DESER-16).
- `training/core/_metadata_loader.py:35`: restricted unpickler (DESER-07).
- `training/feature_handling/cache.py:577,579,593`: `allow_pickle=True` only after sha256 sidecar check; unverified branch `allow_pickle=False`; pickle fallback via `_SafeUnpickler` (DESER-08/09; integrity-only residual already DOC).
- `training/neural/_recurrent_wrapper_base.py:581`, `neural/_flat_torch_module/__init__.py:600`, `feature_selection/filters/_vendored/infonet/infer.py:51`: `weights_only=True` (DESER-17/18/19).
- `feature_engineering/transformer/_key_bank.py:187-189`: `np.load` default `allow_pickle=False` (DESER-14).
- `training/composite/discovery/_tiny_rerank_process.py:92-93`: `cloudpickle.loads` of in-process task blobs, no file source (DESER-15 REJECTED holds).
- `core/helpers.py:33`: `trusted_root` required for caller-path loads.
- `feature_selection/filters/_neural_mi.py:493`: `from_pretrained` pinned by revision.
- No `yaml.load`, `pd.read_pickle`, `shelve`, `marshal`, or JSON `object_hook` loaders in `src/mlframe` outside `_benchmarks/` (DESER-20/21 REJECTED).
