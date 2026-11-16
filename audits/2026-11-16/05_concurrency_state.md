# 05 Concurrency and state audit (src/mlframe, MRMR excluded)

Method: grep sweep of module-level mutable globals, `global` writers, os.environ writes; spot-read of the hits. Read-only; no tests run. Not exhaustive.

| ID | severity | file:line | finding | evidence | disposition |
|---|---|---|---|---|---|
| C1 | low | metrics/_core_precision_mape.py:210,244 | `_MAPE_ZERO_WARN_SEEN` set check-then-add is unlocked | comment at :207 admits "NOT thread-safe"; worst case is a duplicate warning | DOC: benign, documented; no correctness impact |
| C2 | low | competition/__init__.py:28,33 | `_WARNED_COMPETITION_ONLY` same unlocked check-then-add | `if qualname not in ...: ...add(qualname)` | FUTURE: duplicate warning only; other registries here use locks, so align when touched |
| C3 | low | metrics/_gpu_metrics.py:116-125,222 | `_GPU_AVAILABLE` / `_CUPY_SSE_PER_COL` lazy init unlocked: two threads can both run the probe or build the kernel | `if _GPU_AVAILABLE is not None: return` then probe, then assign; no lock | TODO: double probe is a wasted driver call, and the probe runs a daemon thread with timeout, so two concurrent probes can both stall; wrap in a lock. Test: `test_gpu_available_probe_runs_once_under_concurrent_callers` |
| C4 | low | utils/misc.py:126 | `set_random_seed(set_hash_seed=True)` writes `os.environ["PYTHONHASHSEED"]` and never restores it | line 126 | DOC: it only affects child processes, not the current interpreter's hash seed; leaks to later subprocesses by design. Document it in the docstring |
| C5 | low | utils/native_import_probe.py:92 | probe verdicts exported into os.environ and never cleaned | `os.environ[_env_key(name)] = ...` under `_lock` | REJECTED: intentional, so child processes inherit verdicts; the write is lock-guarded and first-writer-wins |
| C6 | low | utils/log_sanitize.py:72-80 | `install_log_sanitizer` check-then-set of the global record factory is not atomic | `getLogRecordFactory()` check, then `_original_factory = current`, then `setLogRecordFactory` | TODO: two concurrent installs could both wrap; the factory reads `_original_factory` late-bound, so the damage is bounded but the chain could double-wrap. Test: `test_install_log_sanitizer_concurrent_installs_wrap_once` |
| C7 | low | reporting/_async_render.py:213 | worker `os.environ[key]=val` | runs in a spawned render worker process (`state.get("env")`) | REJECTED: process-local to the worker, nothing leaks to the parent |
| C8 | low | signal/dtw.py:521 | `_DEFAULT_GPU_MIN_CELLS` mutated by a setter, plain int assignment | `global _DEFAULT_GPU_MIN_CELLS` | REJECTED: atomic rebind of an int; readers see old or new value |
| C9 | info | reporting/_shared_arrays.py, _diagnostics_prep.py, _diagnostics_adversarial.py, renderers/_trend.py, _kaleido.py, ensembling/predict.py, post_import_hook.py, warning_filters.py, __init__.py:483 | module caches with a paired Lock/RLock | each cache has a named `_LOCK` next to it | RESOLVED: locked registries seen in the sweep; not deep-verified for every access path |

Counts: RESOLVED 1, FUTURE 1, DOC 2, REJECTED 3, TODO 2 (C3, C6).
