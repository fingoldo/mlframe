# Wave 1 (2026-11-05), attack class A2: dynamic execution, subprocess and filesystem

Read-only audit by one Opus 5.5 low-effort agent. Nothing was edited and no tests were run. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed.

Coverage: one grep over `src/mlframe` (without `_benchmarks`) for `eval`/`exec`, `subprocess`/`os.system`/`os.popen`, `mkdtemp`/`mkstemp`/`gettempdir`/`NamedTemporaryFile`, `rmtree`, `extractall`, `importlib`/`__import__` and the deserialisation loaders, then a read of the risky sites. The sites fixed by the previous audit are still in place.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| E2-01 | Low | `src/mlframe/training/feature_handling/hf_provider.py:118-122` | If the private lock directory cannot be created, the lock file falls back to a predictable name in the shared system temp directory. |

### E2-01

The function puts the HuggingFace download lock under `~/.cache/mlframe-hf-locks/.mlframe-fhc-locks` with mode `0o700`. If `os.makedirs` raises `OSError` (read-only or missing home, quota, `HOME` pointing at an unwritable place), it falls back to `tempfile.gettempdir()`. The lock path there is `hf_<blake2b(model@revision)>.lock`, which any local user can compute from the public model name. That brings back the earlier shared-temp-root issue the comment at line 113 says was fixed.

Failure scenario: on a shared Linux host the victim has a read-only home or a containerised `HOME=/nonexistent`. An attacker pre-creates `/tmp/hf_<hash>.lock`, either as a file they own and keep locked or as a symlink. Feature extraction for that model then blocks indefinitely (denial of service), or the lock open goes through the attacker's symlink.

Proposed fix: use `mlframe.utils.private_temp.private_temp_dir("mlframe-hf-locks")` as the fallback instead of the raw `gettempdir()`.

## Checked and clean

- Temp files: `estimators/pipelines.py` (`mkdtemp` plus `rmtree` in a `finally`); `_cb_border_cache.py` (`mkstemp` inside `private_temp_dir`); `_joblib_safe.py:353` (`mkstemp` in the default temp directory, exclusive create, mode 0600); `rfecv` checkpoints, `integrations/mlflow.py:119` and `composite/cache_store.py` (`mkstemp` in the destination directory followed by a replace); `_cb_gpu_monitor.py` (`mkdtemp` plus `rmtree`).
- Deletions: in `_key_bank.py` the temp directory has a uuid suffix inside the cache directory, and `rmtree` only targets that directory.
- Subprocess: `native_import_probe`, `kernel_tuning_cache/_ensure`, `_polars_native_support`, `_gpu_state_probe` and `pysr` pass a fixed argument list, no `shell=True`.
- Dynamic imports and eval: every `importlib`/`__import__` takes a hard-coded module table or a registry key; `_sapp.py:68` runs `df.eval` on a constant expression.
- Deserialisation: `torch.load` uses `weights_only=True` throughout; `dill.load` in `training/io.py:637` runs only on explicit `safe=False` with a warning; `joblib`/`dill` loads go through `validate_trusted_path`.

## Not covered

- Whether the `trusted_root` defaults (`_trainer_train_and_evaluate_parts.py:126` uses the model file's parent directory) limit anything; a parent-directory default makes the check close to a no-op.
- Path traversal through caller-supplied cache or fingerprint names in `utils/disk_cache.py` and `feature_handling/cache.py` (only grepped).
- The atomic-write naming in `training/io.py` beyond its comments.
- File handles left open across the whole tree.
- `cloudpickle.loads` in `_tiny_rerank_process.py`: its input is a blob built inside the process; the origin was not traced.
