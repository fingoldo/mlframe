# Concurrency and TOCTOU audit (CON2), 2026-11-01

Scope: `src/mlframe`, read-only. Grep sweep over locks, threading, `global` mutation, `os.replace`/`os.rename`/`mkstemp`, `os.remove`/`unlink`, `exists()` check-then-act, and `prange` writes. Read the sites that matter. Cross-checked against `audits/2026-10-31/_TRACKER.md` (no RESOLVED item re-reported).

Limits: the kernel-tuning-cache write path lives in external `pyutilz` and is not in this tree. `suite_artefact_cache._safe_pickle_dump` is imported, and its definition was not located, so its atomicity is taken from the module's own docstring and was not verified. No runtime interleaving was executed.

## Findings

### CON2-1 (medium), FUTURE
- **Location:** `src/mlframe/system/kernel_tuning_cache/_ensure.py:132-147` (`single_flight`).
- **Failure scenario:** The machine-wide tuning lock is created with `open(path, "x")` and the PID is written only after the handle is opened. A second process can open the file in that window and read an empty string. `_lock_holder` maps empty to 0, and 0 counts as stale. Process B then unlinks A's live lock and creates its own, so two sweeps run at once. The code's own comment says two concurrent sweeps measure each other and the fits around them, so this defeats the guard. A second race exists in the stale branch. P1 and P2 both read a dead holder PID. P2 unlinks, then acquires. P1 unlinks P2's live lock and acquires as well, leaving two holders.
- **Impact:** Duplicate concurrent tuning sweeps. Kernel-tuning measurements are skewed and the wrong crossover can be persisted.
- **Proposed disposition: FUTURE.** Fix by writing the PID to a temp file and publishing it with `os.link` (atomic create-with-content), and by re-reading the holder under the same check immediately before unlinking. The fix changes lock semantics, so it needs its own regression test that fails on the current code.

### CON2-2 (medium), FUTURE
- **Location:** `src/mlframe/feature_selection/filters/_analytic_mi_null.py:82, 90-96, 113-114` (`analytic_null_min_n_lowered`).
- **Failure scenario:** `_min_n_override` is a process-wide global restored by a context manager that saves and restores the previous value, with no lock. Two threads (for example joblib's threading backend) interleave. A enters (previous=None, set 100). B enters (previous=100, set 100). A exits (set None). B exits (restore 100). The override stays at 100 for the rest of the process. Later fits then use the analytic null at a row count the calibration did not cover.
- **Impact:** Silent change to MI-null selection that persists across fits, which is the "global state mutated across fits" case.
- **Proposed disposition: FUTURE.** Replace with `contextvars.ContextVar` (per thread, correct on restore), or hold a lock across enter and exit. Add a regression test that runs two threads with overlapping blocks and asserts the value is restored to None.

### CON2-3 (low), FUTURE
- **Location:** `src/mlframe/training/feature_handling/locking.py:97-148` (`PIDAwareFileLock` stale-reclaim branch, unlink at 131).
- **Failure scenario:** The holder PID is read and found dead. Before this process calls `os.unlink(self.path)`, another process may have reclaimed the lock and be holding it. The unlink then deletes a live lock, and a second process acquires it, so two holders exist. The comment at 120-129 covers the retry window but not unlinking a live lock. The code's own comment at 130-136 notes that on Windows the unlink can fail when the file is still open, and the next retry then proceeds. On POSIX, nothing stops the live-lock removal.
- **Impact:** Two holders of a lock meant to be exclusive, for the feature-handling cache writes that it protects.
- **Proposed disposition: FUTURE.** Re-check the holder PID against the file contents immediately before unlinking, or reclaim by renaming the stale file to a unique name and deleting only that rename target.

### CON2-4 (low), DOC
- **Location:** `src/mlframe/training/_training_loop_refit.py:62-80` (`_discard_stale_catboost_snapshot`).
- **Failure scenario:** The function deletes a CatBoost snapshot at `snapshot_file`, resolved relative to `train_dir`. Two concurrent fits that share the same `train_dir` and snapshot name can delete each other's live snapshot, and the refit then fails to resume or resumes from the wrong fit. The snapshot name is not checked for uniqueness in this file. The default name was not traced and I did not confirm whether two fits can produce the same path.
- **Proposed disposition: DOC.** Document that concurrent fits need distinct `train_dir`. If the default snapshot name is fixed, make it unique per fit.

### CON2-5 (info), RESOLVED
- **Location:** `src/mlframe/feature_engineering/transformer/_key_bank.py:240-293` (`save_key_bank`), with the reader at 165-223.
- **Why it is safe:** Each writer uses a UUID-stamped tmp dir. Publication is `rename` to the content-addressed final dir. A lost race cleans up its own tmp and returns. A reader treats any missing or corrupt file as a cache miss, and the existing published dir is never removed while a reader may be loading it. No shared mutable state is trusted.

### CON2-6 (info), RESOLVED
- **Location:** `src/mlframe/training/io.py:103-181` (`atomic_write_bytes`); `src/mlframe/training/composite/cache_store.py:175-199, 555-613`; `src/mlframe/utils/_param_oracle_store.py:77-84`; `src/mlframe/feature_selection/filters/_fe_family_budget.py:262-274`; `src/mlframe/utils/disk_cache.py:415-448`; `src/mlframe/training/slicing/_slice_pareto_plot.py:224-227`.
- **Why it is safe:** Each writes to a same-directory temp (pid, counter, or uuid-named, or `mkstemp`), fsyncs where durability matters, and publishes with `os.replace`. A reader therefore sees either the old or the new file. The non-atomic `write_text` that `_fe_family_budget` used to do has been replaced.

### CON2-7 (info), DOC
- **Location:** `src/mlframe/training/suite_artefact_cache.py:161, 288-298, 372-385` (`put` / `_evict_lru_locked`).
- **Observation:** The module states that cross-process writers race on rename and that last writer wins, which is acceptable because same key implies same content. Eviction under another process's open read fails on Windows with an `OSError` and breaks out of the sweep. The reader gets a miss, not corrupt data.
- **Proposed disposition: DOC.** The cross-process contract is already in the docstring. Not a defect.

## Summary

| ID | Severity | Disposition | Location |
|---|---|---|---|
| CON2-1 | medium | FUTURE | `system/kernel_tuning_cache/_ensure.py:132-147` |
| CON2-2 | medium | FUTURE | `feature_selection/filters/_analytic_mi_null.py:90-96` |
| CON2-3 | low | FUTURE | `training/feature_handling/locking.py:130-136` |
| CON2-4 | low | DOC | `training/_training_loop_refit.py:62-80` |
| CON2-5 | info | RESOLVED | `feature_engineering/transformer/_key_bank.py:240-293` |
| CON2-6 | info | RESOLVED | see finding body |
| CON2-7 | info | DOC | `training/suite_artefact_cache.py:161, 372-385` |

Counts: high 0, medium 2, low 2, info 3.
