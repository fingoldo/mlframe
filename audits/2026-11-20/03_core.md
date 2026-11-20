# 03 core: read-only audit

Scope: `src/mlframe/core/_benchmarks/` (manifest 03_core.txt: `__init__.py`, `bench_recency_step_weight.py`, `profile_robust_location.py`). Callers and definitions checked in `src/mlframe/core/recency_step_weight.py`, `src/mlframe/core/robust_location.py` and `tests/core/`.

| ID | Severity | file:line | Class | Finding | Failure scenario | Suggested fix |
|---|---|---|---|---|---|---|
| 03-F1 | P2 | src/mlframe/core/_benchmarks/bench_recency_step_weight.py:44-53 | perf | The cProfile section reruns the full 10M x 500 workload for each of the three variants, on top of the timed grid at lines 37-42 which already ran that same size. | none, perf/maintainability: the harness does roughly twice the work it needs to, so a full run is slow. | Profile at n=1_000_000 or fewer calls; the 10M size adds nothing to the call-graph breakdown. |
| 03-F2 | P2 | src/mlframe/core/_benchmarks/profile_robust_location.py:59-70 | ML correctness (test gap) | `bench_parallel` only times `_robust_mean_irls_parallel` against `_robust_mean_irls`. It never asserts that the two return the same value. | Input x = normal(0,1, n=100k), tukey: if the prange reduction ever diverged from the serial result, the harness would still print a speedup and report success. | Add `np.isclose(s_result, p_result, rtol=1e-9)` (capture the return values from `_best_of`'s warm call) and fail loudly on mismatch. |
| 03-F3 | P2 | src/mlframe/core/robust_location.py:116-139 and :181-182 | test coverage | No test in `tests/core/` references `_robust_mean_irls_parallel` or the dispatch branch `x.shape[0] >= _ROBUST_MEAN_PARALLEL_MIN_N`. The only hit for the knob is in `tests/feature_selection/test_env_knobs_read_per_call.py:59`, which checks the env read, not the numerical result. | Input x with n = 50_000 (the default threshold): the parallel kernel is selected, and nothing checks its output against the serial kernel, so a numeric regression in the prange path would pass CI. | Add a test comparing `robust_mean_mestimator(x)` at n just below and just above the threshold, for each weight in `WEIGHTS`, with `np.isclose`. |
| 03-F4 | P3 | src/mlframe/core/_benchmarks/profile_robust_location.py:72-73 | maintainability | `bench_parallel` is hooked in with a module-level `if __name__ == "__main__" and "PARALLEL" in os.environ` check placed before the main block at lines 76-78. The entry point is split across two places and the env var is undocumented in the docstring (lines 1-8). | none, maintainability: a user running `python -m ...profile_robust_location` will not discover the parallel bench without reading the source. | Move the call into `bench()` or the main block behind an explicit `--parallel` argument, and document the variable in the module docstring. |
| 03-F5 | P3 | src/mlframe/core/_benchmarks/profile_robust_location.py:52 | dead code | `weight="meshalkin"` is passed explicitly, but `robust_mean_mestimator` already defaults to `"meshalkin"` (robust_location.py:145). | none, maintainability: redundant argument. | Drop the keyword, or pass a non-default weight so the profile covers a different branch. |
| 03-F6 | P3 | src/mlframe/core/_benchmarks/bench_recency_step_weight.py:17-33 | maintainability | `_run`, `_run_tiers` and `_run_smooth` discard their return values. The harness measures time only, so a change that silently returns wrong weights would not show up. | none, maintainability: a wrong-result regression in `recency_step_weight` is invisible to this harness. | Have each runner return a checksum (e.g. `weights.sum()`) and assert it against the value from a reference call at the smallest n. |

## Checked, no finding

- `src/mlframe/core/_benchmarks/__init__.py` is empty; `python -m mlframe.core._benchmarks.<module>` resolves correctly.
- `bench_recency_step_weight.py:14` imports `recency_step_weight` from `mlframe.core.recency_step_weight`, which exists with the signature `(dates, cutoff_date, boost, base, tiers, smooth_window)` (recency_step_weight.py:28-35). The keyword names used by the harness match.
- The `tiers` tuples at bench_recency_step_weight.py:25 use ascending cutoffs (`n-3000 < n-500 < n-45`), which is the ordering the docstring requires (recency_step_weight.py:50-51).
- `profile_robust_location.py:18` imports `geometric_median` and `robust_mean_mestimator`; both exist (robust_location.py:142, :267) with the keyword names used.
- `bench_parallel` passes `WEIGHTS.index("tukey")` (= 2) and the positional arguments in the order `_robust_mean_irls` expects (robust_location.py:89), so the wcode mapping agrees with `_weight` (robust_location.py:77-78).
- `_best_of` (profile_robust_location.py:21-28) warms the JIT before timing, so the reported numbers do not include compile time.
- No P0 or P1 findings: the harness files do not feed production results, and the production functions they call were out of scope for this manifest.


## Dispositions

- 03-F3: FIXED. coverage gap closed: parallel and serial robust means agree at n above the dispatch threshold; tests/core/test_robust_mean_parallel_matches_serial.py
- 03-F1: DEFERRED. benchmark-script change not made in this pass; low impact
- 03-F2: DEFERRED. benchmark-script change not made in this pass; low impact
- 03-F4: DEFERRED. benchmark-script change not made in this pass; low impact
- 03-F5: DEFERRED. benchmark-script change not made in this pass; low impact
- 03-F6: DEFERRED. benchmark-script change not made in this pass; low impact
