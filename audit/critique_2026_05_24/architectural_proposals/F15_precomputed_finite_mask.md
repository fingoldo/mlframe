# F15: Precomputed finite_mask for residual-fit family (architectural deferral)

## Problem (from perf-hotspots-critique.md row 15)

`_additive_residual_fit`, `_diff_*`, `_ratio_*`, `_logratio_*` and similar single-column residual fit kernels in `composite_transforms.py` and `_composite_transforms_nonlinear.py` each compute a fresh `finite = np.isfinite(y) & np.isfinite(base)` mask per spec. Within the composite discovery loop a single (y, base) pair sees dozens of specs sharing the same finite-cell pattern, so we pay N_specs full-frame `np.isfinite` passes on identical inputs (n=1M: ~3-5ms per isfinite-pair pass, ~60-100ms cumulative per discovery target on 20+ specs).

## Why deferred to user-approve

Threading a precomputed `finite_mask` through the residual-fit family is an **API change spanning ~20 fit functions** (different files, different signatures, different downstream callers including disk-loaded specs that round-trip through JSON). Touching the public function shape risks breaking:

1. Downstream replay code that calls `_<x>_fit(y, base)` positionally
2. Disk-cached `CompositeSpec` payloads that may have been pickled with the old call shape
3. The `_disc_cfg` carrying a mutable `finite_mask` cross-instance could leak per-target masks across targets if not carefully reset

`feedback_perf_measure_first` calls for measure-then-ship; even at the upper bound (~100ms per target) this is well below the per-target wall (multi-second model fits), and the architectural blast radius is large.

## Options

| Option | LOC | Risk | Speed | Notes |
|---|---|---|---|---|
| A. New optional `finite_mask=` kwarg on every fit + skip recompute when supplied | ~80 LOC across 20 functions | Medium - callers passing positionally break unless every callsite migrates | ~60-100ms/target | Pure win on hot path; need to thread through `_disc_cfg`, `_dispatch_fit_*`, and the spec-replay path |
| B. Module-level `_FINITE_CACHE: WeakKeyDictionary[ndarray, ndarray]` keyed by `id(arr) + arr.shape + arr.dtype + content-hash-of-first-1k-cells` | ~40 LOC | Low - opt-in via wrapper helper `_isfinite_cached(arr)` | ~60-100ms/target | No API change; cache miss still pays full pass; cache hit ~us. Risk: cache must invalidate on array mutation (use content-hash) |
| C. Defer entirely; mark the cost as O(60-100ms) per target in the discovery docstring | 0 LOC | None | 0 | Cost is small per-target; defer until a profile shows it dominating |

## Recommendation

**Option B** when user approves. The cache helper is contained to one helper file, the API of the residual fits stays untouched, and the cache is keyed on content-hash so spec-replay paths stay safe. Cap cache size at 64 entries (LRU) so memory doesn't grow unbounded across long suites.

## Risks

- WeakKeyDictionary on a numpy array uses `id(arr)` and is sensitive to `id()` recycling (mirrors finding #22). Content-hash of the first/last 1k cells makes the cache stable against recycling.
- The `np.isfinite` pass on n=1M is itself ~3-5ms; the cache miss cost is invisible. Net win only at high spec counts (>=10).
