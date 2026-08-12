# Audit report: cluster `fe_root_a`

**Scope**: the loose `*.py` files directly inside `src/mlframe/feature_engineering/` (not subdirectories),
sorted alphabetically, first half by file count (40 of 79 files: `__init__.py` through
`holiday_calendar_features.py`). `feature_selection/filters/**` and `feature_selection/shap_proxied_fs/**`
are out of scope (separately audited 2026-07-25, tracker closed). The `transformer/` subpackage is a
subdirectory and therefore out of scope even though `__init__.py` re-exports from it.

**Files reviewed**: 40 (full read, every line)
**LOC reviewed**: 10,882 (measured via line count of the 40 in-scope files; matches the ~22.2k/2 total the
task brief anticipated for this half of the split)

**Note on scope brief vs. actual split**: the task brief's example list ("include grouped.py, timeseries.py,
numerical.py, bayesian.py, spatial.py") does not match a literal alphabetical-first-half split — `timeseries.py`
and `spatial.py` alphabetically fall in the *second* half (`fe_root_b`). The explicit, unambiguous instruction
("sort alphabetically, take the first half by count") was followed; the resulting LOC total (10,882) matches
the brief's "~22.2k split with fe_root_b" almost exactly (10,882 + 11,296 = 22,178), confirming the split rule
used is the intended one and the example list is very likely a copy/paste artifact from a different cluster's
prompt.

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| FE_ROOT_A-1 | P1 | `_numerical_numba.py:86-87`, `:379-436` | `compute_numerical_aggregates_numba` returns a list whose length silently does not match `get_basic_feature_names()`'s length for empty/degenerate input, breaking the documented fixed-width contract | Route the `size==0` fast path (and the drawdown/duration recursive sub-calls when a sub-array collapses to length 0) through a NaN-filled vector of the length implied by the caller's own flags, not a hardcoded `[0.0]` | Property test: for every flag combination and `n` in `{0,1,2,3}`, assert `len(compute_numerical_aggregates_numba(arr, **kw)) == len(get_basic_feature_names(**kw))` |
| FE_ROOT_A-2 | P1 | `grouped.py:540-552` | `per_group_rolling_reduce(op in {std,var,median,min,max})` crashes with `ValueError` (`sliding_window_view` "window shape cannot be larger than input array shape") whenever `min_periods < window_K` and a group's length falls in `[min_periods, window_K)` — only the sum/mean branch implements the documented partial-window `min_periods` behavior | Give the std/var/median/min/max branch the same partial-window treatment as sum/mean (e.g. slice `seg[:seg_len]` through a variable-length window loop, or clamp the `sliding_window_view` window to `min(window_K, seg_len)` and mask), or explicitly document/raise a clear error for the unsupported combination instead of leaking a numpy internal exception | Parametrized test: for every `op`, sweep `seg_len` from `min_periods` to `window_K - 1` and assert no exception / correct output length |
| FE_ROOT_A-3 | P1 | `binned_unique_count.py:99-102` | `binned_unique_count` crashes with `ValueError: 'list' argument must have no negative elements` whenever `entity_col` contains any NaN/missing label together with at least one valid `value_col` observation — `pd.factorize`'s `-1` "missing" sentinel flows unmasked into `np.bincount` | Drop rows whose `entity_codes == -1` before building `combined_key` (mirroring how `valid` already excludes non-finite `value_col`), or raise an explicit, actionable `ValueError` up front if entity_col contains nulls | Fuzz test: for a small panel fixture, inject a NaN/None into `entity_col` on a row with a finite `value_col`; assert `binned_unique_count` neither crashes nor silently drops that entity from the accounting without a documented reason |
| FE_ROOT_A-4 | P1 | `bayesian.py:675-709`, `_bayesian_oblr.py:103-130` | Neither `bocpd_features` (NIG prior hyperparameters `kappa0`/`alpha0`/`beta0`) nor `online_bayesian_linear_regression` (`prior_precision`, `noise_sigma`) validate that their positivity-constrained hyperparameters are actually positive. `bocpd_features(alpha0=0.0, beta0=0.0)` crashes with an unhandled `ZeroDivisionError` raised from inside the njit kernel (ugly traceback, not a clean `ValueError`); `online_bayesian_linear_regression(prior_precision=0.0)` does not crash but silently returns an all-NaN `predictive_var` with no warning | Add an explicit upfront `if kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0: raise ValueError(...)` / `if prior_precision <= 0 or noise_sigma <= 0: raise ValueError(...)`, mirroring the positivity guards these functions rely on internally | Unit test: call each of `bocpd_features`, `online_bayesian_linear_regression`, `kalman_filter_posterior_1d` with each positivity-constrained kwarg set to `0.0` and `-1.0`; assert a `ValueError` is raised (not an unrelated internal exception, not silent NaN) |
| FE_ROOT_A-5 | P1 | `anchor.py:227-300`, `:303-376` | `add_anchor_extrapolation_features` — the module's primary/flagship function, listed first in `__all__` and the module docstring — has NO numba-accelerated code path. `_anchor_features_for_segment` always runs a pure-Python per-row loop using `list.append`/`list.pop(0)` (itself O(K) per pop) plus a fresh `np.asarray`+OLS-fit on every new anchor. Every OTHER function in the same file (`anchor_residual_rmse_features`, `anchor_quadratic_extrapolation_features`, `anchor_ewm_features`, `anchor_density_features`) explicitly received the "convert Python list append/pop to preallocated njit buffers" treatment (see each one's own docstring), leaving this one function as the sole holdout | Add an `_anchor_core_njit` mirroring the pattern already used 4x in this same file (preallocated `pos`/`val` buffers, O(1) incremental OLS update instead of a full re-fit per anchor) and dispatch to it when numba is available, keeping `_anchor_features_for_segment` as the documented no-numba fallback | Perf regression test: benchmark `add_anchor_extrapolation_features` at n=200k against a sibling (e.g. `anchor_ewm_features`) on the same data; assert wall-time is within the same order of magnitude once both are numba-accelerated |
| FE_ROOT_A-6 | P2 | `gmm_bic_membership_features.py:96-109` | The `gmm_shift_diagnostics` distribution-shift z-score is one-directional: `shift_zscore = (train_avg_loglik - new_avg_loglik) / standard_error` only goes positive (and can trip `distribution_shift_detected`) when `new_avg_loglik` is LOWER than train's. A shift where `new_df` fits noticeably BETTER (e.g. new rows collapse near a single mixture component, which degrades the membership-probability features' discriminative usefulness even though nominal fit improves) is silently never flagged — contradicting the docstring's stated purpose ("surfaced rather than passed through unflagged"). Additionally `standard_error` uses `train_std` only, ignoring `new_scores`'s own dispersion | Use `abs(shift_zscore)` (or a proper two-sample Welch z/t combining both `train_std` and `new_scores.std()`) so shifts in either direction trip the diagnostic | Unit test: construct a synthetic `new_df` whose average log-likelihood is HIGHER than train's (e.g. rows tightly clustered on one component mean) and assert `distribution_shift_detected` is still capable of firing when the shift is large |
| FE_ROOT_A-7 | P2 | `_recursion_autotune.py:148-166` | The module-level `for _kn, _ref in (...): kernel_tuner(...)` registration block (and its `from pyutilz.performance.kernel_tuning.registry import kernel_tuner` import) has no `try`/`except`, despite the adjacent comment claiming "Wrapped so a missing pyutilz / circular import never breaks the dispatcher." The actual protection exists only indirectly, in `_recursion_dispatch.py`'s call site, which wraps ITS OWN `from ._recursion_autotune import ...` in try/except. Any other import path (the file's own `python -m ...` CLI entry point, direct import, test collection) hits an uncaught exception if `pyutilz.performance.kernel_tuning.registry` is absent or its API changed | Wrap the module-level registration loop in its own `try/except Exception: logger.debug(...)`, or fix the comment to accurately describe that the safety net lives in the caller, not this file | Import-robustness test: monkeypatch/uninstall `pyutilz.performance.kernel_tuning.registry.kernel_tuner` (or simulate `ImportError`) and assert `import mlframe.feature_engineering._recursion_autotune` does not raise |
| FE_ROOT_A-8 | P3 | `_numerical_numba.py:3,247,294,679,700`; `_timeseries_emit.py:3` | Six comments embed "Wave NN (YYYY-MM-DD)" process/phase markers (`Wave 107 (2026-05-21)`, `Wave 47 (2026-05-20)` x4, `Wave 96 (2026-05-21)`), which is exactly the class of comment this repo's own CLAUDE.md convention prohibits ("No process/audit metadata in code comments: no phase/wave markers ... date stamps ... that belongs in git history / the PR description") | Strip the `Wave NN (YYYY-MM-DD)` prefix from each comment, keeping only the WHY-content (e.g. "zero-sum weights vector ... used to crash the njit kernel here" already stands alone without the wave/date tag) | Grep-based CI lint: `grep -rn 'Wave [0-9]\+\|([0-9]{4}-[0-9]{2}-[0-9]{2})' --include=*.py src/` outside of already-approved exemptions (e.g. the CLAUDE.md-mandated `bench-attempt-rejected (YYYY-MM-DD)` note format), failing the build on any match in a non-benchmark file |
| FE_ROOT_A-9 | P3 | `ensemble_features.py:119-120` | Stale comment on `predictor_disagreement_var`: "`_coerce_preds already raises for arr.shape[1] < 2, so the arr.shape[1] > 1 branch below was always taken`" — but the function body directly below is a single unconditional `return np.asarray(arr.var(axis=1, ddof=1))`; there is no branch left to describe. The comment is a leftover from a prior refactor that removed the `if/else` it originally explained | Delete the stale comment (or replace with a one-line note on why `ddof=1` — the actually non-obvious choice — is correct here) | Not independently automatable in general, but a lightweight heuristic checker flagging comments containing "branch below"/"the following if"/"case above" whose immediately-following statement is not an `if`/`match` would catch this class |
| FE_ROOT_A-10 | P3 | `numerical.py:413` | `compute_entropy_features`'s local variable `nonzero = (~np.isnan(arr)).sum()` is misleadingly named — it counts NON-NAN elements (including legitimate zero values), not literally "nonzero" values. The variable feeds `if nonzero < 10: return zeros`, i.e. it is really a "how many finite/valid observations" gate | Rename to `n_valid` or `n_finite` to match what it actually measures | N/A (naming-only); could be caught by a generic identifier-vs-usage sanity linter (e.g. flag `nonzero`/`is_zero`-named variables built from `isnan`/`notna` rather than `!= 0`/`> 0`) |

## Counts

- P0: 0
- P1: 5
- P2: 2
- P3: 3

## Coverage by dimension

1. **Correctness bugs**: FE_ROOT_A-1 (silently-wrong fixed-width output), FE_ROOT_A-2 (crash), FE_ROOT_A-3
   (crash), FE_ROOT_A-4 (crash / silent NaN from missing validation). No mutable-default-argument bugs found
   (grep confirmed zero occurrences across all 40 files). No bare `except:` found (grep confirmed zero). Every
   `except Exception` in scope (4 total, all in `_recursion_autotune.py`/`_recursion_dispatch.py`/`grouped.py`)
   logs before falling back — no silent exception-swallowing found.
2. **ML correctness (leakage / reproducibility / calibration / sample_weight / honest OOS)**: no leakage bugs
   found. `as_of_aggregate.py`'s cutoff semantics (`searchsorted(..., side="left")`) were verified to correctly
   implement the documented strict `time_col < as_of` contract. `anchor.py`/`grouped.py`'s group-boundary
   handling was spot-verified (causal rank, ordinal-tiebreak rank) against independent Python reference
   implementations across 900 randomized trials with zero mismatches — no leak-across-group bugs found.
   `particle_filter_posterior`'s per-group RNG already uses a distinct sub-stream per group (`np.random.default_rng([seed, g])`)
   rather than reseeding identically per group, so no correlated-noise-across-groups bug. FE_ROOT_A-4's missing
   validation is the one reproducibility/robustness gap found in this dimension (undefined/degenerate
   hyperparameters silently produce NaN features rather than a raised error a caller could catch).
3. **Computational efficiency**: FE_ROOT_A-5 is the one clear finding (asymmetric optimization within one
   file). No unnecessary `.copy()` on large frames found — every `.copy(deep=False)` observed is a deliberate,
   correctly-reasoned shallow copy (e.g. `entity_diff_features.py`, `categorical_group_concat.py`,
   `control_difference_augment.py` all comment on why deep copy is unnecessary). No O(n^2)-where-O(n log n)
   patterns found; `grouped.py` and its callers consistently reuse the shared `iter_group_segments` O(n log n)
   (or O(n) counting-sort) primitive rather than re-sorting per feature.
4. **Edge cases and robustness**: FE_ROOT_A-1/2/3/4 all fall in this dimension (empty/degenerate/malformed
   input handling). Beyond those, single-class-target / single-row / all-constant-column edge cases were
   spot-checked (`fuzzy_partition_fit`'s all-identical-value fallback, `cat_cooccurrence_svd_fit`'s
   zero-total contingency guard, `grouped.per_group_apply`'s "all groups failed -> escalate, don't
   silently return all-fill" design) and found correctly handled.
5. **Test coverage gaps**: confirmed for FE_ROOT_A-1 (`test_compute_numerical_aggregates_numba_empty`/
   `_drawdown` exist but only assert `len(res) > 0`, not the documented length-matches-`get_basic_feature_names`
   contract — the exact gap that let FE_ROOT_A-1 ship), FE_ROOT_A-2 (no test sweeps `min_periods < window_K`
   for the non-sum/mean ops), FE_ROOT_A-3 (no NaN-entity test), FE_ROOT_A-4 (no invalid-hyperparameter test
   for any of the three Bayesian entry points). No tests-against-own-output anti-pattern found in this
   cluster's own test files (not fully in scope, but every test referenced during investigation compared
   against an independent oracle — scipy, a hand-written naive reference, or a documented closed form).
6. **Code quality / architecture**: FE_ROOT_A-7 (misleading exception-safety comment), FE_ROOT_A-9 (stale
   comment), FE_ROOT_A-10 (misleading name). No overly-broad `except` clauses found (see dimension 1). API
   consistency across `grouped.py`/`anchor.py`/`entity_inter_event.py`'s per-group primitives is high (shared
   `iter_group_segments` convention, consistent NaN/leak-safety documentation). Docstrings throughout this
   cluster are unusually thorough and mostly accurate (the exceptions are called out above).
7. **OSS/hygiene**: FE_ROOT_A-8 (stale wave/date markers, the one systemic hygiene issue found). No mojibake
   found. No dashes-in-prose issues found (comments in this cluster consistently use " - " correctly). No
   missing type hints found beyond what mypy would itself flag (out of scope to run mypy here, but a manual
   read found no `param: T = None` implicit-Optional patterns — grep confirmed zero).

## Narrative

### FE_ROOT_A-1 — `compute_numerical_aggregates_numba` breaks its own fixed-width output contract on empty input
`get_basic_feature_names()` is documented ("Length and order are guaranteed to match the kernel's return
tuple") to describe exactly what `compute_numerical_aggregates_numba` returns. The kernel's very first lines
are `if size == 0: return [0.0]` — a single-element list regardless of which `return_*` flags were requested.
Verified directly: `compute_numerical_aggregates_numba(np.empty(0), return_exotic_means=True, ...)` (i.e. the
function's own default flags) returns a list of length 1 while `get_basic_feature_names()` expects 20. A
second, related manifestation: the `return_drawdown_stats=True` branch recurses into itself on `pos_dds[1:]`/
`neg_dds[1:]` etc — for a *parent* array of length 1 (reachable directly, since `compute_numaggs`'s own
`len(arr) <= 1` guard only protects the primary entry point, not this lower-level function, which is exported
in `__all__` and called directly by `mlframe/models/ensembling/base.py`), those sub-arrays have length 0 and
silently hit the same 1-element short-circuit — confirmed empirically: `compute_numerical_aggregates_numba(np.array([1.0]), return_drawdown_stats=True)` returns 24 elements vs. the 100 `get_basic_feature_names(return_drawdown_stats=True)` expects. Any caller (present or future) that assumes a fixed-width row in a stacked
feature matrix and blindly writes `row_features[i, :] = compute_numerical_aggregates_numba(...)` would get a
silent shape mismatch — for numpy this raises on assignment (a crash, not silent corruption) UNLESS the row
being written happens to have a length that "fits" by accident, which is precisely the kind of intermittent,
hard-to-repro failure this bug class produces. Existing tests only assert `len(res) > 0`, never the actual
length contract, so this shipped unnoticed.

### FE_ROOT_A-2 — `per_group_rolling_reduce` crashes for a documented, valid `min_periods < window_K` combination
The docstring explicitly documents `min_periods` as a general parameter ("The first `min_periods - 1` rows of
each group emit `fill_value`") applicable across all 6 supported `op` values. The `sum`/`mean` branch honors
this via an explicit prefix-sum partial-window loop. The `std`/`var`/`median`/`min`/`max` branch, however, goes
straight to `sliding_window_view(seg, window_K)` with no partial-window handling at all — and `numpy.lib.stride_tricks.sliding_window_view` raises `ValueError: window shape cannot be larger than input array shape`
whenever `seg.size < window_K`. Reproduced directly: `per_group_rolling_reduce(np.arange(5.0), zeros(5), window_K=10, op='std', min_periods=3)` raises exactly this `ValueError`, even though `min_periods=3 <= seg_len=5`
should (per the docstring) produce a valid partial-window output rather than crash. This is a plain, deterministic
crash on a documented and plausible call shape (any caller mixing a generous `window_K` with a permissive
`min_periods` on a not-yet-long-enough group — extremely common at the start of a panel/time-series).

### FE_ROOT_A-3 — `binned_unique_count` crashes on a NaN/missing entity id
`pd.factorize(df[entity_col], sort=False)` returns `-1` for any row whose grouping key is missing (`NaN`/`None`).
That `-1` flows unmasked into `combined_key = entity_codes[valid] * n_bins_total + bin_codes[valid]` (only the
VALUE column's NaN is filtered via `valid`, not the entity column), and then into `np.bincount(unique_entity_codes, minlength=len(entities))`, which numpy explicitly refuses for negative input:
`ValueError: 'list' argument must have no negative elements`. Reproduced directly with a 5-row frame containing
two `None` entity ids and finite values. A long-format panel with even a handful of dirty/unresolved entity
keys — a routine real-world data-quality issue this exact function's own docstring anticipates ("long-format
panel: one row per (entity, value) observation") — crashes the whole feature instead of either excluding the
unresolved rows or raising a clear, actionable error naming the real cause.

### FE_ROOT_A-4 — Bayesian family functions accept invalid hyperparameters without validation
`bocpd_features`'s `_bocpd_inner` njit kernel computes `scale_sq = beta[r] * (kappa[r] + 1.0) / (alpha[r] * kappa[r])`
for the NIG-conjugate Student-t predictive — division by `alpha[r] * kappa[r]`, which is guaranteed non-zero
only if `alpha0`/`kappa0` are validated positive on entry. They are not: `bocpd_features(y, alpha0=0.0, beta0=0.0)`
raises an unhandled `ZeroDivisionError` from deep inside the compiled numba frame (confirmed via direct repro;
traceback bottoms out in `_bocpd_inner`, not a clean Python-level `ValueError` a caller could reasonably catch
and diagnose). The sibling `online_bayesian_linear_regression` has the analogous issue on `prior_precision`
(`Sigma = np.eye(k) / prior_precision`): `prior_precision=0.0` does not crash but silently fills
`predictive_var` with all-NaN (confirmed via direct repro) with no warning logged anywhere — a caller who
passes a config-driven `0.0` by mistake gets a feature column that is silently useless rather than an error
surfaced at the point of misuse. Given both classes of Bayesian filters are meant to be config-driven,
production-facing feature generators (the module docstring lists finance/geosteering/medical use cases), a
malformed or zero-defaulted hyperparameter is a realistic failure mode this code does nothing to catch early.

### FE_ROOT_A-5 — `add_anchor_extrapolation_features` is the one un-optimized function in an otherwise fully-njit-accelerated file
`anchor.py` opens with `_ANCHOR_FASTMATH`-tuned njit cores for four sibling functions
(`_anchor_rmse_core`, `_anchor_quadratic_core`, `_anchor_ewm_core`, `_anchor_density_core`), each explicitly
documented as replacing an earlier "growing/shrinking Python list (append/pop)" implementation with
"preallocated buffers + window indices so a `@njit` can compile them." `_anchor_features_for_segment` — which
backs `add_anchor_extrapolation_features`, the function listed FIRST in `__all__` and described first in the
module's own docstring as the primary feature — is exactly that un-converted "growing/shrinking Python list"
form the other four explicitly moved away from: it still does `anchor_positions.append(i)` /
`anchor_positions.pop(0)` per anchor (with `pop(0)` itself being O(K)) and rebuilds a fresh `np.asarray(...)`
+ OLS fit from scratch on every new anchor, entirely in the Python interpreter, with no numba dispatch of any
kind (`_NUMBA_AVAILABLE` is checked in all four sibling functions but never referenced in
`add_anchor_extrapolation_features`/`_anchor_features_for_segment`). On the large panels this module's own
docstring targets (wellbore geosteering streams, high-frequency sparse-label streams), this is a genuine,
measurable performance gap relative to every other feature in the same file, and directly contradicts this
repo's own stated engineering convention ("Always try njit when numpy").

### FE_ROOT_A-6 — GMM shift diagnostic only detects shift in one direction
See findings table; verified by inspection of the formula (`shift_zscore = (train_avg_loglik - new_avg_loglik) / standard_error`, flagged when `shift_zscore > shift_zscore_threshold`) — a `new_df` whose average
log-likelihood is HIGHER than train's produces a negative `shift_zscore`, which can never exceed a positive
threshold, so `distribution_shift_detected` stays `False` regardless of magnitude. The docstring's stated goal
("silently-unreliable membership probabilities under covariate shift are surfaced rather than passed through
unflagged") is only half-delivered.

### FE_ROOT_A-7 — misleading exception-safety comment in `_recursion_autotune.py`
The comment directly above the module-level `for _kn, _ref in (...): kernel_tuner(...)` loop claims "Wrapped so
a missing pyutilz / circular import never breaks the dispatcher" — but inspection of the actual code shows no
`try`/`except` anywhere in this file around that loop or its `from pyutilz.performance.kernel_tuning.registry
import kernel_tuner` import. The real safety net is `_recursion_dispatch.py`'s `try: from ._recursion_autotune
import ...` wrapper at its OWN call site — a different file protecting a different failure mode (a lazy,
call-time import failure), not the module-level, import-time registration failure the comment describes. Any
code path that imports `_recursion_autotune` directly (its own `if __name__ == "__main__": _cli()` entry
point still triggers the same top-level module code; a test importing it directly; a future caller) would hit
an uncaught exception if `pyutilz.performance.kernel_tuning.registry.kernel_tuner` were ever missing or its
signature changed, exactly contradicting the comment's claim.

### FE_ROOT_A-8 — stale "Wave NN (date)" process markers in comments
Six comments across two files embed a wave number + date stamp (`Wave 107 (2026-05-21)`, four instances of
`Wave 47 (2026-05-20)`, `Wave 96 (2026-05-21)`). This repo's own CLAUDE.md is explicit that this class of
comment does not belong in source: "No process/audit metadata in code comments: no phase/wave markers ...
date stamps ... that belongs in git history / the PR description." These are harmless (they don't affect
behavior) but are exactly the debt category the project's own convention calls out for cleanup, and are easy
to grep-detect for the fix wave.

### FE_ROOT_A-9 — stale comment referencing a removed code branch
`predictor_disagreement_var`'s comment ("`_coerce_preds already raises for arr.shape[1] < 2, so the arr.shape[1]
> 1 branch below was always taken`") describes an `if/else` that is no longer present — the function body is a
single unconditional return statement. This is a comment left behind by a refactor that simplified away the
branch it was explaining, now pointing at nothing and mildly confusing to a future reader looking for the
"branch below."

### FE_ROOT_A-10 — misleading variable name `nonzero` for a non-NaN count
`compute_entropy_features`'s `nonzero = (~np.isnan(arr)).sum()` counts finite (non-NaN) elements, not elements
that are literally non-zero-valued — a column of all exact zeros with no NaNs would report `nonzero == len(arr)`
under this name, which reads backwards to anyone skimming the code. The gate it drives (`if nonzero < 10: return zeros`) is really "too few valid observations to compute entropy reliably," and should be named
accordingly (`n_valid`/`n_finite`).
