# Security audit 2026-11-01: filesystem, temp files and path handling

Scope: `src/mlframe` (read-only). Builds on `audits/2026-10-31/filesystem.md` and `audits/2026-10-31/_TRACKER.md`. RESOLVED items from that round are not re-reported.

Method: grep of runtime code (benchmarks excluded) for `mktemp`, `gettempdir`, `rmtree`, `os.remove`, `unlink`, `symlink`, `chmod`, `umask`, octal modes and `/tmp`, then a read of each deletion or key-to-path site that takes a name from a parameter or cache key.

## Summary

- 4 findings: 0 High, 0 Medium, 2 Low, 2 Info.
- Re-verified prior state: FS-02 (`estimators/pipelines.py:156`, `tempfile.mkdtemp`) is RESOLVED in code, so the stale row in the 2026-10-31 `filesystem.md` should be read as superseded by the tracker. FS-03, FS-04 and FS-05 remain present at the cited lines and stay FUTURE.
- No new deletion path takes network or untrusted input. The one new real gap is key-to-path mapping in `SuiteArtefactCache`, which has no traversal guard and no in-repo caller that passes a crafted key.

## Findings

| ID | file:line | Severity | Proposed disposition |
|---|---|---|---|
| FS2-1 | `src/mlframe/training/suite_artefact_cache.py:199-201` (and `:284, 317, 363, 415, 440`) | Low | FUTURE |
| FS2-2 | `src/mlframe/training/suite_artefact_cache.py:65-73, 203-205` | Low | FUTURE |
| FS2-3 | `src/mlframe/training/_training_loop_refit.py:70-80` | Info | REJECTED |
| FS2-4 | `src/mlframe/feature_engineering/transformer/_key_bank.py:168, 251-291` | Info | REJECTED |

### FS2-1 SuiteArtefactCache builds paths from raw keys (Low, FUTURE)
`_path` returns `os.path.join(self.cache_dir, f"{key}.pkl")` with no validation. `get`, `put`, `evict`, `clear` and `__contains__` all go through it, and `put`/`evict` remove or write files at that path. The sibling caches already guard this: `composite/cache_store.py:310-313` routes through `_safe_key` (hash-wraps any non-hex key), and `utils/disk_cache.py:343-360` resolves the candidate and refuses paths outside `cache_dir`. Scenario: a caller using the public API passes `key="../../somefile"`; `put` writes a pickle outside the cache directory, and `evict_lru` or `clear` later removes files at the same traversal target. In-repo risk is nil today: `cache_artefact` builds keys with `SuiteKeyBuilder.build` from a blake2b digest (`suite_artefact_cache.py:501-510`), so the key is hex. Proposed FUTURE: apply the `composite/cache_store.py` `_safe_key` approach here, plus a regression test that a `../` key stays under `cache_dir`. Deferred because it changes on-disk filenames for any non-hex key.

### FS2-2 SuiteArtefactCache directory and pickles are not created with a private mode (Low, FUTURE)
`_ensure_dir` (`suite_artefact_cache.py:203-205`) calls `Path.mkdir(parents=True, exist_ok=True)` with no `mode`, so the directory gets the process umask default. The default location is `~/.mlframe/cache/suite` (`:73`), or an env-var override (`:65-70`). Its files are pickles that `get` loads (`pickle` deserialization, see the deserialization class). The sibling `training/feature_handling/cache.py:334` and `cache_backend.py:104-106` create their roots with `mode=0o700`. Scenario: on a POSIX host with a permissive umask, or with a shared `MLFRAME`-style override pointing at a shared directory, another local user can read the pickles or plant one that is then unpickled. Impact is limited to hosts where the home directory is group- or world-readable, and to an env override that points at a shared path. Proposed FUTURE: `mkdir(mode=0o700)` and refuse a cache dir owned by another UID. Deferred to pair with the deserialization audit's FS and pickle work.

### FS2-3 CatBoost snapshot removal from config path (Info, REJECTED)
`_training_loop_refit.py:70-80` removes `params["snapshot_file"]` with `os.remove`, joining a relative value onto `train_dir`. The value comes from the trainer's own configuration (`save_snapshot` and `snapshot_file` are set by `_cb_gpu_budget`), not from request or dataset input. An absolute path is accepted, but only the operator who wrote the config can set it, and that operator can already delete any file. Not a risk in the trust model. REJECTED.

### FS2-4 Key-bank `rmtree` on a fingerprint-derived path (Info, REJECTED)
`_key_bank.py:251, 278, 284, 291` call `shutil.rmtree(tmp_dir, ignore_errors=True)` where `tmp_dir = cache_dir / (fingerprint + ".tmp." + uuid)` and `bank_dir = cache_dir / fingerprint` (`:168`). A `fingerprint` containing `..` would widen the target. The only producer is `_key_bank_fingerprint` (`:96-148`), which returns a `hashlib.sha256().hexdigest()`, so the value is always 64 hex characters. The function is internal and no caller passes an external string. REJECTED on the current call graph. If the function ever becomes public, add a `re.fullmatch(r"[0-9a-f]{64}", fingerprint)` check at entry.

## Checked and not a finding (for the record)

- Temp creation: runtime code uses `mkstemp`, `mkdtemp` and `O_EXCL` (see prior FS-07 to FS-14). The only new-looking hit, `training/cb/_cb_gpu_monitor.py:558`, is `rmtree` of its own `mkdtemp` directory and is fine.
- Deletion from input: `training/feature_handling/cache_backend.py:141-146` rejects keys with separators or `..` before any unlink. `utils/numba_cache_guard.py:27` deletes only a fixed stamp file under `__pycache__`.
- Shared memory: `reporting/_shared_arrays.py:91, 127` unlink the segments this process created. The naming scheme was not traced in this pass, so treat the shared-memory naming as unverified.
- Permissions: still no `chmod`, `umask`, `0o666` or `0o777` in `src/mlframe` (prior FS-16 holds).

## Coverage limits

Read-only grep and targeted reads. Benchmark directories were excluded. Shared-memory naming in `_shared_arrays.py` and the `composite/cache_store.py` deletion paths were sampled, not fully traced. No tests were run.
