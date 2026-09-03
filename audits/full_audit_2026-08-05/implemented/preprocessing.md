# Audit: preprocessing

**Scope**: `src/mlframe/preprocessing/**` (all `.py` files, including `_benchmarks/`). `feature_selection/filters/**` and
`feature_selection/shap_proxied_fs/**` are out of scope (separate, already-closed audit).

**Files reviewed**: 51 (`__init__.py` + 22 module files + 28 `_benchmarks/*.py` files), all read in full.

**LOC reviewed**: ~6,387 (per `wc -l` over every `.py` file in scope).

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| PREPROCESSING-1 | P0 | `temporal_drift_augment.py:96-101` | `augment_temporal_drift`'s expanding mean/std uses the catastrophic-cancellation-prone `sum(x^2) - (sum(x))^2/n` formula; on realistic large-offset/small-spread columns (prices, balances, epoch timestamps) it goes negative, gets clipped to 0, and the resulting z-scored synthetic feature silently collapses to a constant `0.0` for every augmented row. | Replace with a numerically stable two-pass or Welford-style expanding variance (mirrors the `_centered_moments_njit` pattern already used elsewhere in the codebase for the identical bug class). | AST/grep scanner flagging `cumsum(x**2) - cumsum(x)**2/n`-shaped expressions repo-wide; a property test asserting expanding/rolling variance matches `pd.Series.expanding().var()` (or a two-pass reference) within tolerance across offset/scale sweeps (offset up to 1e9, scale down to 1e-1). |
| PREPROCESSING-2 | P0 | `auto_transform_select.py:149-201` | `select_column_transforms` computes every non-identity candidate transform (all sklearn scalers, RankGauss, and the NaN-median fill) on the WHOLE column BEFORE the K-fold CV split, then fits/scores the probe per fold on that pre-computed array — so cross-row-statistic transforms leak test-fold rows into the very statistics (mean/std/quantiles/ranks) used to transform themselves. The "honest" per-transform CV score this function exists to produce is optimistically biased for every candidate except `identity`/`log1p_signed`, systematically favoring scaler/RankGauss candidates over `identity`. | Move `_apply_transform` for cross-row-statistic candidates inside the fold loop, fitting only on `train_idx` and transforming `test_idx` with those train-only stats (mirror `gaussian_power_transform_search`'s fit/apply split, which already does this correctly). | A property test with a synthetic column where `identity` is genuinely optimal: assert the leaky pre-split CV score for `StandardScaler`/`RankGauss` does NOT exceed an honest per-fold-refit score by more than noise; a repo-wide scanner flagging "a `.fit`/`.fit_transform` call outside a CV fold loop whose output is later indexed by both `train_idx` and `test_idx` of that loop." |
| PREPROCESSING-3 | P1 | `sibling_group_cold_start_fill.py:69-75` | `interpolate=True` fills an entirely-missing sibling group by interpolating over the POSITIONAL rank of groups (`reset_index(drop=True).interpolate()`), not the actual `order_col` VALUE distance the docstring promises ("weighted by each sibling's distance in the ordering"). Confirmed empirically: groups at `order=(0, 1, 100)` with values `(10, NaN, 50)` fill the missing middle group with `30.0` (the positional midpoint) instead of a value close to `10` (its true near neighbor in `order_col`). | Interpolate against the actual `order_col` values per group (e.g. set the interpolation index to `group_order` values, not `reset_index(drop=True)`'s default RangeIndex) before falling back to ffill/global mean. | A regression test with 3 unevenly-spaced groups (as in the repro above) asserting the interpolated fill is closer to the near neighbor than the midpoint. |
| PREPROCESSING-4 | P1 | `category_support.py:26-51` | `smoothed_target_encode_column` computes `train_encoded = train_series.map(shrunk).fillna(global_mean)` where `shrunk` is derived from `y_train.groupby(train_series)` over the SAME rows — i.e. every row's own target directly informs its own category's shrunk mean (in-sample, non-OOF target encoding). If `train_encoded` is used as a training feature (the obvious use of a function named `*_encode_column` returning a `train_encoded` series), this is classic target leakage. No leakage-warning docstring accompanies it, unlike every other stateful-statistic module in this package (`cleaning.py`, `outlier_capping_or_missing.py`, `rare_count_pruning.py`, etc. all carry an explicit LEAKAGE WARNING / fit-on-train-only note). | Either (a) add an explicit K-fold/leave-one-out OOF variant for `train_encoded` (mirroring the codebase's own kfold target-encoding elsewhere), or (b) document the in-sample leakage risk as prominently as the sibling modules do and steer callers toward an OOF wrapper. | A test asserting `train_encoded`'s correlation with `y_train` on a purely-random (label-independent) categorical column is near the smoothing-implied floor, not inflated by self-leakage; a grep-based scanner for `<series>.groupby(<same-frame-column>)` results mapped straight back onto that same frame without a fold split. |
| PREPROCESSING-5 | P0 | `cleaning.py:773-783` | `analyse_and_clean_features`'s nunique==2 replacement branch computes `repl_value` only for `isinstance(real_val, str)`, `col_is_numeric`, or `col_is_boolean` — a 2-valued categorical column whose real (non-NaN) value is none of those (e.g. `decimal.Decimal`, `pd.Timestamp`, a tuple) falls through all three branches and raises `UnboundLocalError: cannot access local variable 'repl_value'`. Reproduced live: a `pd.Categorical` column with one `Decimal` value + NaN crashes `analyse_and_clean_features(df, update_data=True)`. | Add an `else` branch (e.g. `repl_value = ("NOT_" , real_val)` or fall back to leaving the NaN unmapped with a warning) instead of relying on exactly three type checks being exhaustive. | Property/fuzz test feeding `analyse_and_clean_features` 2-valued object/category columns across a matrix of Python value types (str, int, float, bool, Decimal, Timestamp, tuple, custom object) and asserting no `UnboundLocalError`. |
| PREPROCESSING-6 | P1 | `gaussian_power_transform_search.py:55-68, 200-220` | `apply_gaussian_power_transform`'s Box-Cox replay path calls `scipy.special.boxcox(x, fitted_params)` without re-validating positivity (the fit path DOES guard `if np.any(x <= 0): return None, None`). Reproduced: fitting Box-Cox on an all-positive train column then applying to inference data containing `0.0`/`-1.0` (schema drift / a genuinely observed edge value) returns those non-positive inputs UNCHANGED (raw, non-transformed) mixed into an otherwise Box-Cox-transformed column — a silent, undetectable scale/distribution mismatch within the same output column, not a crash or NaN that would be noticed. | Guard the apply-phase branch the same way the fit phase does; either raise, return `None`, or clip/impute non-positive apply-time values before calling `boxcox_apply`, and document the choice. | A regression test asserting `apply_gaussian_power_transform` either raises or clearly flags (not silently passes through raw values) when apply-time data violates the Box-Cox positivity precondition the fit-time data satisfied. |
| PREPROCESSING-7 | P2 | `outlier_policy.py` (whole file, docstring 86-111) | `apply_outlier_policy` recomputes per-column quantile capping bounds from whatever `X` it is called on every time (no fit/apply split, no persisted state) and, unlike its sibling `outlier_capping_or_missing.py` (which explicitly documents "fit-on-train discipline is the caller's responsibility"), carries no such caveat at all. Calling it separately on train and test computes different bounds from each split's own distribution, silently diverging the transform applied to train vs. test. | Add the same explicit fit-on-train-only caveat `outlier_capping_or_missing.py` carries, or (better) add a `fit_outlier_policy_bounds`/`apply_outlier_policy_bounds` split so callers can persist train-only bounds, matching the `fit_*`/`apply_*` pattern used elsewhere in this package. | A regression test asserting `apply_outlier_policy(train, ...)` and `apply_outlier_policy(test, ...)` called independently produce DIFFERENT cap bounds when train/test distributions differ, documenting (and eventually closing) the gap. |
| PREPROCESSING-8 | P2 | `__init__.py:19-33` | The package's `fit_*`/`apply_*` leakage-safe function pairs for `rare_count_pruning` (`fit_rare_category_collapse`, `apply_rare_category_collapse`), `missing_indicator_pairing` (`fit_missing_indicator_imputation`, `apply_missing_indicator_imputation`), and `regime_conditioned_imputation` (`fit_regime_conditioned_median`, `apply_regime_conditioned_median_fill`) are NOT re-exported at the package level — only the combined single-frame convenience wrapper is — even though each module's own docstring explicitly recommends the fit/apply split "for train/test consistency". Confirmed: `from mlframe.preprocessing import fit_rare_category_collapse` raises `ImportError`, while the analogous `apply_feature_direction` (from `align_feature_direction.py`) IS exported. Inconsistent with `align_feature_direction.py`/`gaussian_power_transform_search.py`, whose fit/apply pairs ARE both exported. | Add the missing `fit_*`/`apply_*` names to `__init__.py`'s import list for parity with the other fit/apply-split modules. | A test asserting every public name defined via a `fit_X`/`apply_X` pair in a `preprocessing/*.py` module's own `__all__` is also reachable from `mlframe.preprocessing` (package-level `__all__` superset check). |
| PREPROCESSING-9 | P2 | `cleaning.py:76-93` | `_get_nunique`'s float fast path only extracts `skip_vals[0]`/`skip_vals[1]` into `skip0`/`skip1`; any 3rd+ element of a longer `skip_vals` tuple is silently dropped with no validation, even though the docstring only promises support for "up to two". No current caller passes >2, so it's latent, but nothing guards against a future caller (or the `int_part`/`fract_part` call sites growing a 3rd skip value) silently under-counting distinct values. | Add an explicit `if skip_vals and len(skip_vals) > 2: raise ValueError(...)` (or extend the njit kernel to accept a variable-length skip array) instead of silently truncating. | A unit test calling `_get_nunique` with a 3-element `skip_vals` on a float array and asserting either correct behavior or a clear `ValueError`, not a silently wrong count. |
| PREPROCESSING-10 | P2 | `cluster.py:61-91` | `clusterize`'s `show_plot=True` path (the default) builds the full matplotlib figure, computes the scatter data, sets the title/annotations, and then immediately calls `plt.close(fig)` without ever calling `plt.show()`, saving the figure, or returning it — so `show_plot=True` produces NO visible or persisted output at all, contradicting the docstring's "render a scatter plot of the clusters." All the work in that branch is effectively dead code from the caller's point of view. | Either return the `Figure` object (so callers can `fig.savefig(...)`/display it themselves in the caller's environment) or accept a `save_path`/`ax` parameter; at minimum, document that `show_plot=True` currently only computes-and-discards the figure. | A test that calls `clusterize(show_plot=True)` and asserts a `Figure` object (or saved file) is actually produced/returned, not silently discarded. |
| PREPROCESSING-11 | P3 | `regime_conditioned_imputation.py:26-29` | `fit_regime_conditioned_median`'s docstring contains a stray, undefined cross-reference — "a train/serve statistic mismatch), same architectural gap as F1/F7\n    ." (odd trailing period on its own line) — leftover from an external planning document; "F1/F7" is meaningless to a reader of this module in isolation. | Remove the "F1/F7" reference (or replace with a concrete pointer, e.g. "same gap as `fit_missing_indicator_imputation`/`fit_rare_category_collapse`"), fix the stray line break. | A grep-based hygiene scanner for docstring references to bare alphanumeric codes (`F\d+`, `#\d+`, etc.) not defined anywhere in the same file/module, repo-wide. |
| PREPROCESSING-12 | P3 | `_benchmarks/check74.py:27` | The `"falsy_skip"` test case passes `skip_vals=(0.0)` — missing the trailing comma needed for a 1-tuple — so it is actually the scalar float `0.0`. Since `if skip_vals:` treats `0.0` as falsy, both the `OLD` reference and (via truthiness) the code path under test skip the skip-values branch entirely; the case's stated purpose (verify a tuple containing a falsy `0.0` is still honored) is never actually exercised. | Change to `skip_vals=(0.0,)`. | A grep/AST scanner flagging single-element parenthesized literals passed to a parameter whose name/usage implies a tuple/sequence (`skip_vals=(X)` without a trailing comma) across the repo's test/bench scripts. |

## Counts

- P0: 3
- P1: 3
- P2: 4
- P3: 2

## Narrative

### PREPROCESSING-1 (P0) — `augment_temporal_drift` catastrophic-cancellation variance collapses augmented features to 0

`augment_temporal_drift` computes each entity's expanding mean/std via `cumsum_x`/`cumsum_x2` and the textbook
`Var = (sum(x^2) - (sum(x))^2/n) / (n-1)` shortcut (`temporal_drift_augment.py:96-101`), then z-scores the truncated
history to build the "earlier vintage" synthetic training rows. This is the exact same raw-moment-cancellation bug
class documented repeatedly elsewhere in this codebase (`_binned_numeric_agg_fe.py`'s `_global_stats_all`, the
target-encoding skew/kurt fix) — here applied to the 2nd moment (variance) instead of the 3rd/4th. I reproduced it
directly: with a realistic large-offset/small-spread synthetic entity history (`offset=1e9, scale=0.5, n=8`, the
kind of scale real epoch timestamps, account IDs, or balances carry), the unstable formula returns **negative**
variance for 6 of 7 expanding windows (`-256`, `-170.7`, `-256`, ... vs. the numerically-stable two-pass reference's
correct `0.008`-`0.089`). `var.clip(lower=0)` in the real code turns every one of those negatives into exactly `0`,
so `expanding_std` becomes `0` for every row after the first, `safe_std.notna()` is `False` everywhere, and
`standardized.where(safe_std.notna(), 0.0)` sets the synthetic feature to a flat `0.0` regardless of the row's true
value — the augmentation silently destroys all signal in every affected feature for every entity whose values carry
a large offset relative to their spread, which is common in production panel data (prices, balances, IDs-as-features,
epoch timestamps). The sibling function in the SAME file, `select_true_last_standardized`, already avoids this by
using pandas' own `.transform("mean")`/`.transform("std")` (a numerically stable two-pass implementation) instead of
the cumsum shortcut — the unsafe formula was chosen purely for the perf win of using `cumsum` instead of
`expanding().mean()`/`.std()`, without carrying over the numerical-stability fix the codebase has already applied to
this exact bug class multiple times elsewhere.

### PREPROCESSING-2 (P0) — `select_column_transforms`'s CV leaks test-fold statistics into every scaler/RankGauss candidate

`select_column_transforms` (`auto_transform_select.py`) exists specifically to give an "honest cross-validated"
score per candidate transform. But `transformed = _apply_transform(finite_fill, transform_name)` (line 201) is
called ONCE, on the WHOLE column (`finite_fill`, itself median-filled from the whole column too), BEFORE the
`for train_idx, test_idx in fold_indices` loop begins. Every candidate transform other than `identity`/
`log1p_signed` — every sklearn scaler from `make_all_scalers()` (`StandardScaler`, `MinMaxScaler`, `RobustScaler`,
`PowerTransformer`, `QuantileTransformer`, ...) plus `rankgauss` — computes cross-row statistics (mean/std,
quantiles, rank distribution) that therefore see the test-fold rows before the probe model is even fit on
`train_idx`. I confirmed this concretely: fitting `StandardScaler` on the whole 1000-row column vs. fitting it only
on an 800-row "train" subset and applying to the 200-row "test" subset produces different means/stds on that same
test subset (`-0.109`/`0.871` honest vs. `-0.089`/`0.892` leaky) — exactly the train/serve statistic mismatch this
function is trying to measure honestly, silently present in its own scoring mechanism. Because the leak affects
every non-identity candidate uniformly, it systematically biases `best_transform` selection toward scalers/RankGauss
relative to `identity`, which is the entire point of comparison this function exists to make honest.
`gaussian_power_transform_search.py` (the module's own sibling, one file over) gets this right with an explicit
fit/apply split and a docstring explaining exactly why replay-not-refit matters — this function should follow the
same pattern.

### PREPROCESSING-3 (P1) — `sibling_group_cold_start_fill`'s `interpolate=True` weights by rank, not by the documented `order_col` distance

The docstring for `interpolate` promises "a linear interpolation between the two, weighted by each sibling's
distance in the ordering", and `order_col`'s own docstring gives "the group's first-seen timestamp" as an example —
i.e., real, plausibly-unevenly-spaced values. The implementation, however, calls
`last_known_per_group.reset_index(drop=True).interpolate(method="linear", limit_area="inside")` — a plain positional
(RangeIndex) interpolation — with an inline comment asserting "siblings are equally spaced by construction once
indexed by their rank," which is only true in RANK, never in the actual `order_col` VALUE the docstring's own
timestamp example invites. I reproduced this directly: three groups at `order=(0, 1, 100)` with values
`(10, NaN, 50)` — group B (`order=1`) is nearly coincident with group A (`order=0`) and far from group C
(`order=100`) — fill group B with `30.0`, the exact positional midpoint, instead of a value close to `10` a
true distance-weighted interpolation would produce. The bundled benchmark (`bench_sibling_group_cold_start_fill.py`)
never surfaces this because its synthetic `order_vals` happen to equal the group's own rank (perfectly evenly
spaced by construction), so the bug is invisible to the existing perf-only test infrastructure.

### PREPROCESSING-4 (P1) — `smoothed_target_encode_column` produces in-sample (non-OOF) target encoding

`smoothed_target_encode_column` (`category_support.py:26-51`) computes `stats = y_train.groupby(train_series)`
over the full train frame, then maps that SAME per-category statistic straight back onto `train_series` to produce
`train_encoded`. Every row's own target value therefore contributes to the shrunk mean subsequently used to encode
that very row — classic target-encoding leakage with no out-of-fold split. Every other stateful-statistic function
in this package (`cleaning.analyse_and_clean_features`, `outlier_capping_or_missing.outlier_cap_or_missing`,
`rare_count_pruning.fit_rare_category_collapse`, `missing_indicator_pairing.fit_missing_indicator_imputation`, ...)
carries an explicit "LEAKAGE WARNING" or "fit-on-train discipline" note in its docstring; this function, whose
entire purpose is to hand back a `train_encoded` series presumably meant to be used as a training feature, carries
none. A caller who follows the obvious naming (`train_encoded` — use it as a train-time feature) reproduces
in-sample target leakage without any warning telling them not to.

### PREPROCESSING-5 (P0) — `analyse_and_clean_features` crashes with `UnboundLocalError` on a 2-valued non-str/numeric/boolean column

In the "6. Replaces nan with some other value" branch (`cleaning.py:773-783`), `repl_value` is assigned only inside
`if isinstance(real_val, str): ... else: if col_is_numeric: ... elif col_is_boolean: ...` — there is no final
`else`, so a 2-valued categorical/object column whose real (non-NaN) value is neither a `str`, numeric, nor boolean
Python object falls through all three branches, leaving `repl_value` unbound, and the subsequent
`logger.info("feature %s: %s->%s in %s.", col, na_val, repl_value, ...)` / `repl_instructions = {na_val: repl_value}`
raises `UnboundLocalError`. I reproduced this live with a `pd.Categorical` column holding one `decimal.Decimal`
value plus `None` (nunique==2): `analyse_and_clean_features(df, update_data=True)` crashes with
`UnboundLocalError: cannot access local variable 'repl_value' where it is not associated with a value`. Any
category dtype whose levels are non-numeric non-string non-bool Python objects (Decimal, Timestamp, tuple, a custom
`__eq__`-comparable object) hits this on the main documented cleaning entry point, not an edge-case opt-in path.

### PREPROCESSING-6 (P1) — `apply_gaussian_power_transform`'s Box-Cox replay silently passes non-positive apply-time values through untransformed

`_apply_transform`'s fit phase (`gaussian_power_transform_search.py:56-57`) guards Box-Cox with
`if np.any(x <= 0): return None, None`, but the apply/replay phase (line 66-68,
`scipy.special.boxcox(x, fitted_params)`) has no equivalent guard. I reproduced this: fitting Box-Cox on an
all-positive synthetic train column, then calling `apply_gaussian_power_transform` on an inference frame that
contains `0.0` and `-1.0` (plausible schema drift or a genuinely-occurring boundary value not present in the
smaller train sample) returns those two values completely UNCHANGED — `-1.0` and `0.0` verbatim, not NaN, not an
error — silently mixed into an otherwise properly Box-Cox-transformed column. This is worse than a crash or NaN
because it produces plausible-looking numeric output with no signal that anything went wrong; a downstream model
consuming this column sees a handful of rows on a completely different (raw, untransformed) scale from the rest
with no diagnostic trail.

### PREPROCESSING-7 (P2) — `apply_outlier_policy` has no fit/apply split or leakage caveat, unlike its sibling module

`apply_outlier_policy` (`outlier_policy.py`) recomputes `np.nanquantile`-based cap bounds directly from whatever
`X` it receives on every call — there is no persisted "fitted" state to replay onto val/test consistently. Its
sibling module doing conceptually the same job, `outlier_capping_or_missing.outlier_cap_or_missing`, explicitly
documents this exact caveat ("Bounds are always computed from the values passed in `df`... fit-on-train discipline
is the caller's responsibility"). `apply_outlier_policy`'s docstring makes no such statement at all, despite having
the identical architectural gap — calling it independently on train and test (the natural-looking call pattern
given its name) computes different quantile bounds from each split's own distribution, silently diverging the cap
applied to train vs. test.

### PREPROCESSING-8 (P2) — Package-level `__init__.py` omits several modules' documented leakage-safe `fit_*`/`apply_*` pairs

Three modules (`rare_count_pruning.py`, `missing_indicator_pairing.py`, `regime_conditioned_imputation.py`) each
ship a `fit_*`/`apply_*` pair specifically so a caller can "fit once on train, apply the SAME learned [state] to
every other split" — each module's own docstring recommends this explicitly over the single-frame convenience
wrapper. `preprocessing/__init__.py`, however, only imports the convenience wrapper for each
(`collapse_rare_categories`, `impute_with_missing_indicator`, `regime_conditioned_median_fill`), never the `fit_*`/
`apply_*` names. I confirmed `from mlframe.preprocessing import fit_rare_category_collapse` (and the missing-
indicator/regime-imputation equivalents) raise `ImportError`, while `align_feature_direction.py`'s
`apply_feature_direction` (the analogous safe-replay function for THAT module) IS exported at package level. This
inconsistency actively works against the leakage-safe usage pattern these modules' own docstrings are trying to
steer callers toward, for anyone importing from the top-level `mlframe.preprocessing` package (a very natural thing
to do, since every other sibling exposes both halves of its pair there).

### PREPROCESSING-9 (P2) — `_get_nunique`'s float fast path silently drops any `skip_vals` beyond the first two

`_get_nunique` (`cleaning.py:76-93`) documents "excluding NaN and up to two `skip_vals`" and its slow (`np.unique`)
path genuinely supports an arbitrary-length `skip_vals` iterable (`for val in skip_vals: unique_vals = ...`), but
the float fast path only ever reads `skip_vals[0]`/`skip_vals[1]` into `skip0`/`skip1` — any 3rd+ element is
silently ignored, with no assertion or `ValueError` guarding the documented 2-element limit. Every current call
site in `cleaning.py` passes at most 2 elements, so this is latent rather than actively wrong today, but it is a
silent-wrong-count trap for the next caller (or refactor) that grows a 3rd skip value, since the float path and the
`np.unique` fallback path would then silently diverge in their answers for the exact same input.

### PREPROCESSING-10 (P2) — `cluster.clusterize`'s default plotting path computes and immediately discards its own figure

`clusterize`'s `show_plot=True` branch (`cluster.py:61-91`, `show_plot` defaults to `True`) constructs a full
matplotlib `Figure`/`Axes`, computes every scatter series, sets the title, and (if `true_labels` is given) adds
per-point annotations — then, on the very next line, calls `plt.close(fig)` without ever calling `.show()`, saving
the figure to disk, or returning it to the caller. Every bit of work in that branch is therefore computed and
thrown away; a caller relying on the docstring's promise to "render a scatter plot of the clusters" gets no visible
or persisted output whatsoever from the default call. The `plt.close(fig)` line's own comment ("library code must
not leak figures nor block on `plt.show()`") is a legitimate concern in isolation but was applied here without
providing any alternative output channel (return the figure, accept a `save_path`), leaving the feature
non-functional end-to-end.

### PREPROCESSING-11 (P3) — Stray "F1/F7" cross-reference in `regime_conditioned_imputation.py`'s docstring

`fit_regime_conditioned_median`'s docstring (`regime_conditioned_imputation.py:26-29`) reads "...a train/serve
statistic mismatch), same architectural gap as F1/F7\n    ." — an undefined external code ("F1/F7", presumably
referring to feature IDs from some planning document) with an odd trailing period on its own line. This is
meaningless to anyone reading the module in isolation and looks like a leftover artifact from an external planning
doc that leaked into the shipped docstring.

### PREPROCESSING-12 (P3) — `_benchmarks/check74.py`'s "falsy_skip" case is vacuous due to a missing tuple comma

`check74.py`'s `cases` list includes `("falsy_skip", np.modf(rng.uniform(-10, 10, 1000))[1], (0.0))` — `(0.0)` is
just the float `0.0` (no trailing comma), not a 1-tuple. Since both `OLD` and (by extension, via the same
truthiness pattern in `_get_nunique`) the real code use `if skip_vals:` to decide whether to loop over skip values,
and `0.0` is falsy in Python, this case silently never exercises the skip-value-subtraction logic the case's own
name says it is meant to check (a tuple containing a single falsy `0.0` value should still be honored as "skip
0.0", not treated as "no skip_vals given"). Both sides trivially agree (`o == n`) because neither ever runs the
loop, so the script reports "ALL_OK True" without having tested what it claims to.

## Additional coverage notes (explicit, per review dimensions)

- **Mutable-default-argument bugs**: none found — every module normalizes `Optional[...] = None` to a fresh
  list/dict inside the function body (confirmed via `grep -rn "def .*=\[\]\|def .*={}"`, zero matches).
- **Bare `except:` / overly broad exception swallowing**: no bare `except:` anywhere in scope. The two
  `except Exception:` sites (`transforms.py:154, 210`) both log a throttled warning and append to
  `skipped_columns` — reviewed and found to already follow the codebase's own documented fix pattern for this bug
  class (see CLAUDE.md's "broad except silently downgraded..." entries); no further action needed there.
- **GPU/numba dispatch correctness**: `_cleaning_kernels.py`'s three lazily-compiled njit kernels
  (`_outlier_mask`, `_span_fence`, `_count_distinct_*`) and `outliers.py`'s parallel/serial min-max + out-of-range
  kernels all have documented bit-identity verification against their numpy/pandas references in their own
  `_benchmarks/` scripts (`bench_naive_outlier_minmax.py`, `bench_count_outofranges99.py`,
  `bench_cleaning_cpx18_cpx19.py`, `bench_nunique74.py`, `bench_paired74.py`) — reviewed all of them, all assert
  identity before timing, no correctness gaps found in this dimension.
- **Empty/all-NaN/single-class/single-row edge cases**: `is_variable_truly_continuous`, `suggest_non_outlying_data_indices`,
  `compute_naive_outlier_score`, `batch_univariate_auc`, and `select_outlier_threshold` all have explicit, tested
  guards for empty/all-NaN/single-class input (raising `ValueError` or returning a documented degenerate result) —
  no gaps found there.
- **`scalers.py`**: clean; the module-level mutable-scaler-pollution bug this file's own comment describes as a
  "prior revision" issue is already fixed (factory pattern, fresh instance per call) — verified no residual
  aliasing.
