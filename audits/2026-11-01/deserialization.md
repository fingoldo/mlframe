# Unsafe deserialization audit (re-check): mlframe (src/mlframe)

Read-only. No source, test or config file was edited; no state-changing git command was run.
Baseline: `audits/2026-10-31/deserialization.md` and `_TRACKER.md`. Prior RESOLVED rows are not re-reported.

## Scope and method

- Grepped every loader name across `src/mlframe` (excluding `_benchmarks/` trees, which the prior audit covered and REJECTED as dev-only): `pickle.load(s)`, `joblib.load`, `dill.load`, `cloudpickle.loads`, `torch.load`, `np.load`, `allow_pickle`, `yaml.load`/`unsafe_load`, `marshal`, `shelve`, `read_pickle`, `safe_load`, `safe_joblib_load`, `load_mlframe_model`, `Unpickler`.
- Traced the input path for each, and whether a sha256 sidecar or path guard gates it.
- Result: no `yaml.load` without SafeLoader, no `marshal`/`shelve`, no `pd.read_pickle`, no `torch.load` without `weights_only=True`, and no `np.load` with `allow_pickle=True` on a non-benchmark path. The new finding is a path-containment gap (DSR2-01), not a missing gate.

## Status of prior findings (re-verified against current tree)

- **DESER-10 (High)**: RESOLVED. `training/composite/cache_store.py:385` now calls `_safe_pickle_load(path, allow_unverified=None)`, which defers to `MLFRAME_ALLOW_UNVERIFIED_PICKLE` (unset means reject). Commit `303e54be4` per tracker. The prior FUTURE row is stale; the tracker status should be updated to RESOLVED.
- **DESER-08**: still open by default. `training/feature_handling/cache.py:578` calls `_sload(path, allow_unverified=not _strict)`, so unsigned legacy pickles load unless `MLFRAME_FEATURE_CACHE_STRICT=1`. Only reached when `allow_pickle=True`. Disposition unchanged (DOC).
- **DESER-07**: still open. `training/core/_metadata_loader.py:20-21` returns `safe_joblib_load(metadata_file)` for `kind == "joblib"` before any `verify_sidecar` call (line 22 is only on the pickle branch).
- **DESER-05 / DESER-04**: still open. `inference/predict.py:123` (features `.dump` fallback, after `_verify_sidecar` at 118) and `:245` (`infer/` models, after `_verify_sidecar` at 227) both use the denylist `safe_joblib_load`.
- **DESER-06**: still open. `estimators/pipelines.py:111` calls unrestricted `joblib.load(fname)` after an integrity-only sidecar check (line 108).
- **DESER-02 / DESER-03**: still open. `training/io.py:524-612`: `load_mlframe_model` calls `validate_load_meta_sidecar` (non-strict, no sha256 enforcement) and then `_SafeUnpickler(zf).load()` (line 594). Allowlist breadth in `_unpickle_policy.py` unchanged.
- **DESER-01**: `io.py:601` `dill.load(zf)` still reachable only with `safe=False`, which warns (line 596). No `src` caller passes `safe=False`. DOC.
- **DESER-11**: `_trainer_train_and_evaluate_parts.py:117-126` still calls `load_mlframe_model(model_file_name)` with the file's own directory as root. Unchanged.

## Findings

### DSR2-01: `validate_trusted_path` does not resolve symlinks (Low)
- **Location:** `src/mlframe/core/helpers.py:34` and `:42` (`os.path.abspath` on both root and path, then `os.path.commonpath`, lines 44-50).
- **Issue:** The containment check is lexical. A symlink inside `trusted_root` that points to a file outside it passes the check, and the loader then opens the link target. `os.path.realpath` is not used.
- **Failure scenario:** a principal with write access to the trusted root (e.g. a shared cache directory) creates `root/x.joblib -> /elsewhere/y.joblib`. `replay_cv_results(fname=root/x.joblib, trusted_root=root)` passes the guard and loads `/elsewhere/y.joblib`. The attacker still needs a pickle at the target that they can write, or a pickle that is already trusted there. Writing a payload into the root directly already gets a sidecar-gated path. So the symlink adds reach to files outside the root, but not a new write primitive. Note: `replay_cv_results` also requires a matching sha256 sidecar at `fname`, which the link path shares only if the attacker writes it too.
- **Severity:** Low.
- **Proposed disposition: FUTURE.** Reason: cheap hardening. Use `os.path.realpath` on both sides in `validate_trusted_path`, and add a test with a symlink (skip on Windows when symlink creation is denied). Not a bug with a demonstrated exploit path on its own.

### DSR2-02: `_resolve_denylisted_class` denies by module prefix, not by resolved object (Info)
- **Location:** `src/mlframe/training/io.py:212-230` (`safe_joblib_load` path, used by `inference/predict.py:245`, `:123`, and `training/core/_metadata_loader.py:21`).
- **Issue:** The denylist is applied to the `(module, name)` pair from the pickle stream, before `real_find_class`. Gadgets reached through an allowed module (e.g. a `numpy`/`pandas` callable that internally shells out, or a `torch` helper outside `torch.hub`/`torch.jit`) are not blocked by the denylist. This is the same weakness as prior DESER-04 and DESER-03 (denylist is weaker than the allowlist), not a new site.
- **Severity:** Info (duplicate class of DESER-04; listed for completeness of the denylist contract).
- **Proposed disposition: DOC.** Reason: the sidecar is documented as the trust boundary (`io.py:201`, `predict.py:122`). State that `infer/` and metadata joblib files are code-execution-equivalent for anyone who can write the directory.

## Carried-forward findings (IDs unchanged, still hold)

| ID | Severity | Location | Proposed disposition |
|---|---|---|---|
| DESER-01 | Medium | `src/mlframe/training/io.py:601` | DOC |
| DESER-02 | Medium | `src/mlframe/training/io.py:524-612` (`load_mlframe_model`, default `safe=True`, no sha256 gate, only `validate_load_meta_sidecar` at 575) | FUTURE |
| DESER-03 | Medium | `src/mlframe/training/_unpickle_policy.py` (allowlist breadth) | FUTURE |
| DESER-04 | Medium | `src/mlframe/inference/predict.py:245` | FUTURE |
| DESER-05 | Low | `src/mlframe/inference/predict.py:123` | FUTURE |
| DESER-06 | Medium | `src/mlframe/estimators/pipelines.py:111` | FUTURE |
| DESER-07 | Medium | `src/mlframe/training/core/_metadata_loader.py:20-21` | FUTURE |
| DESER-08 | Medium | `src/mlframe/training/feature_handling/cache.py:578` | DOC |
| DESER-11 | Low | `src/mlframe/training/_trainer_train_and_evaluate_parts.py:117-126` | FUTURE |

## Counts (this pass)

- New: 2 (DSR2-01 Low, DSR2-02 Info). Re-verified prior: 9 carried forward. DESER-10 changed to RESOLVED.
- By severity, new items only: High 0, Medium 0, Low 1, Info 1.

## Notes on scope

- No exploit or gadget was built. Severities come from reading the code paths.
- `pyutilz.core.safe_pickle` behaviour (fail-closed sidecar default) is taken from the prior audit and from the wrapper in `src/mlframe/utils/safe_pickle.py`, not re-read.
- The `_benchmarks/` trees were excluded from this grep, per the prior audit's REJECTED disposition.
