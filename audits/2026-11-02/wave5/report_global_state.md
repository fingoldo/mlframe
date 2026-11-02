# Wave 5 audit 2026-11-02: global state mutation, cross-call contamination and non-determinism

Scope: library code (benchmarks excluded), read-only, Opus 5.5 low effort. Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| STATE5-1 | Medium | `feature_selection/hybrid_selector.py:24`: `os.environ.setdefault("TQDM_DISABLE", "1")` at import | RESOLVED |
| STATE5-2 | Medium | `_mrmr_fit_impl/_eng_dedup_batch_corr.py` `_ORDER_CACHE` and `_grouped_coerce_shared.py` `_KEY_PREP`: strong references, no lock, never cleared | RESOLVED |
| STATE5-3 | Low | `utils/cupy_pinned.py:62`: process-wide warning filter | RESOLVED |
| STATE5-4 | Low | `training/__init__.py:95-96`: `LOKY_MAX_CPU_COUNT` set at import | DOC |
| STATE5-5 | Low | `filters/_prewarm.py:313-315`: prewarm draws from numba's RNG stream | DOC |
| STATE5-6 | Info | vendored `infonet/infer.py` unseeded `np.random.randn` | REJECTED |

STATE5-1: the variable is now set for the duration of `HybridSelector.fit` and restored (also on error); importing the module changes nothing. Test: `tests/feature_selection/test_hybrid_selector_leaves_tqdm_alone.py`.

STATE5-2: entries hold a weak reference (a dead one never matches a reused id), eviction runs under a lock with `pop(key, None)`, and the end of `MRMR.fit` clears both. Test: `tests/feature_selection/filters/test_id_keyed_caches_hold_no_arrays.py`. The in-place-refill case (same object, new values) is not addressed by identity checks and stays a property of these caches: both are populated from columns that are not mutated within a fit.

STATE5-3: the filter goes through `install_filter_once`; Python already de-duplicates identical `filterwarnings` entries, so the visible effect is unchanged.

STATE5-4: documented import-time default, like the macOS `OMP_NUM_THREADS` one.

STATE5-5: `numba_rng_scope` without a seed does not restore the stream, so wrapping the prewarm would not help; the draw only moves numba's per-thread stream, which is entropy-seeded unless the user seeded it.

STATE5-6: dead vendored example code that nothing imports.

Checked and safe: `set_random_seed` is a top-of-script helper; library seeding is inside scopes that restore state; environment writes are depth-counted and restored; warning filters live in `catch_warnings`; no mutable defaults; id-keyed caches elsewhere use weakrefs; no `as_completed`; no results stored on classes.
