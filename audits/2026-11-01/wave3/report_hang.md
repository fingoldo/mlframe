# Wave 3 audit 2026-11-01: non-termination and hang surfaces

Scope: `src/mlframe` (read-only, no code executed). Class: loops that data can keep from ending (NaN, inf, zero step), loops with no cap,
unbounded waits (locks, queues, joins, subprocess). Builds on the three earlier trackers (INP2-7 already says the `while True` loops it
checked terminate).

The auditor could not write this file itself; the content below is its returned text. `wavedec_batched` was not read for HNG4-1.

## Summary

7 findings: 0 High, 2 Medium, 3 Low, 2 Info.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| HNG4-1 | Medium | `feature_engineering/wavelet_dwt.py:99-103` (`_symmetric_reflect`, reached from `wavedec` `:350-358`) | FUTURE |
| HNG4-2 | Medium | `estimators/early_stopping.py:277-298` (`_fit_warm`) | FUTURE |
| HNG4-3 | Low | `training/_io_save.py:462` | DOC |
| HNG4-4 | Low | `feature_selection/filters/_joblib_safe.py:202` | DOC |
| HNG4-5 | Low | `reporting/_async_render.py:605`, `:424` | DOC |
| HNG4-6 | Info | `feature_selection/filters/engineered_recipes/_recipe_name_simplify.py:82` | REJECTED |
| HNG4-7 | Info | `timeseries.py:917`, `votenrank/leaderboard/_rules.py:122`, `state_history.py:38, 101` | REJECTED |

## Findings

### HNG4-1 `_symmetric_reflect` never ends for an empty signal (Medium, FUTURE)
The `while idx < 0 or idx >= n` loop. With `n == 0` every integer fails `0 <= idx < 0`; the two flips map -1 to 0 (`-idx-1`) and 0 to -1
(`2*0-1-0`), so it cycles forever. It runs inside njit, so Ctrl-C cannot stop it. `wavedec` stops only when `out_len = (n+flen-1)//2` is below
1; for `n == 0` and `flen >= 3` (db2 and wider) `out_len >= 1`, so the kernel runs. An empty group or window passed to a wavelet feature does
this. Shortest hang: `wavedec(np.array([]), "db2", 1)`. Fix: `wavedec` and the kernel must return early or raise when `n == 0`.

### HNG4-2 `warm_start_step <= 0` keeps the warm-start loop from ending (Medium, FUTURE)
`while n < self.max_iter` in `_fit_warm`. Only `None` takes the `max(1, ...)` path; with `warm_start_step=0` (or negative),
`n = min(max_iter, i*step)` stays at 0 or goes negative. The default `max_runtime_mins=None` gives a deadline of `+inf`, and the only other
exit is `_consider` returning true, which was not traced for a constant score. `max_iter=None` is rejected at `:334`, but `step` is not
validated. Shortest hang: `EarlyStoppingWrapper(warm_estimator, max_iter=10, warm_start_step=0).fit(X, y)` with `max_runtime_mins` unset.
Fix: require `warm_start_step >= 1`, or apply `max(1, step)`.

### HNG4-3 Save lock taken without a timeout (Low, DOC)
`_SAVE_MUTATION_LOCK.acquire()`. A save that hangs on a stuck filesystem or slow serialiser leaves the lock held and every other saving
thread blocks. The serialisation is intended.

### HNG4-4 Helper-thread join without a timeout (Low, DOC)
`t.join()`: the user function running in the helper thread may never return. The call is synchronous by contract.

### HNG4-5 Async render close and byte gate waits (Low, DOC)
`close(wait=True)` calls `join()` with no timeout, so a worker that never finishes blocks it. `_ByteGate.acquire` waits in 1 s slices; a lost
`release()` (for example from a dead worker) would block `submit` forever. `close(wait=False)` is the escape hatch.

### HNG4-6 Recipe-name simplifier (Info, REJECTED)
The inner `while True` either consumes a `,` or breaks, so every pass advances. Malformed names cannot loop; very deep nesting raises
`RecursionError`, not a hang.

### HNG4-7 Three loops that only look data-dependent (Info, REJECTED)
`find_next_cumsum_left_index` returns an index strictly below `windows_r` or 0 and the loop breaks at `windows_l <= 0`; a NaN `window_size`
does not break that progress. Baldwin's rule removes the minimum-Borda rows each pass; an all-NaN table raises at `.astype(int)` instead of
looping. The `while changed` relabel loops merge two runs per change, so the run count falls.

## Checked and already safe

- Subprocess calls all have timeouts: `_ensure.py:199`, `_gpu_state_probe.py:98, 109`, `_polars_native_support.py:90`, `native_import_probe.py:55`.
- File locks have timeouts: `locking.py:96`, `cache_store.py:149` (30 s), `cache_backend.py:225` (30 s), `hf_provider.py:255` (600 s).
- `power_rescale.py:95, 99` expansion loops are capped at 200, and `brentq` has `maxiter=500`.
- GPU tile loops step by at least 1; CUDA kernel strides use `blockDim.x` or `_WARP_SIZE`.
- `range` with a zero step raises `ValueError` rather than looping.
