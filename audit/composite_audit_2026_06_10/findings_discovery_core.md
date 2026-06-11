# Composite Targets audit 2026-06-10 - Discovery core + leakage discipline

Scope: src/mlframe/training/composite/discovery/{__init__,_fit,_eval,_filter,_auto_base,screening}.py
(cross-checked against forward_stepwise.py, transforms/linear.py, core/_phase_composite_discovery.py,
_composite_target_discovery_config.py, and existing tests). Known-fixed items from
tests/composite_discovery_audit_notes.md (Welford, transform round-trip, LCB gate application,
rerank fold splitter) were re-checked and are NOT re-reported. All line numbers verified against current code.

---

## D1 - P0 / leak - _filter.py:203-212 (with screening.py:122-156) - mean-imputation dilution defeats the leak-corr gate for any column with a handful of NaNs

The vectorised corr filter mean-imputes non-finite cells (`X_train[non_finite_mask] = broadcast(col_means)`,
_filter.py:209-211) then computes `_safe_abs_corr_all(y, X)`. Imputed cells have `X_dev == 0`, contributing 0
to covariance and column variance, while `var_y` is over ALL rows (screening.py:142-149). Computed corr for a
column with NaN fraction f is `corr_true * sqrt(1-f)`. The default gate is
`forbidden_base_corr_threshold = 0.99999` (config line 622): `sqrt(1-f) >= 0.99999` requires `f <= 2e-5` -
**more than ~2 NaN rows per 100k lets an exact copy of y pass the leak filter.** Realistic leak columns
(target encodings with unseen-group NaNs, y-derived ratios with div-by-zero NaNs, lags with leading NaNs)
virtually always carry NaN. Once past the gate, the column has near-maximal MI(y,x), wins `_auto_base`,
and discovery builds a composite around a y-derived base - textbook leakage with fake gains. The
`finite_mask.sum() < 50` gate (_filter.py:162) bounds nothing (a 99.9%-NaN column passes with 50 finite rows,
corr diluted to ~0.03). The inline comment (180-184) calls this a "small approximation", but vs a 1e-5-wide
threshold the dilution is orders of magnitude larger. The pre-vectorisation per-column `_safe_corr` masked NaN
pairs per column and returned |corr|=1.0 - this is a regression introduced by the 2.2x vectorisation.

Compounding: the matrix is float32 and `_safe_abs_corr_all` accumulates dots in float32
(screening.py:144-153); on multi-million-row columns the float32 accumulation error (~1e-6..1e-5) is
comparable to the 1e-5 threshold band, and squares of values > ~1.8e19 overflow to inf, mapping the column to
corr 0 (silent pass).

**Fix.** Per-column renormalisation (vectorised: per-column var_y over each column's finite rows via one
masked matmul `(~non_finite_mask).T @ y_dev**2`, divide corr_j by `sqrt(var_y_finite_j/var_y_all)`), or
simplest/exact: route only columns having any NaN through scalar `_safe_corr` (per-pair masking). Do the corr
reductions in float64 (per-column promotion, no full-frame float64 copy). Regression test: exact y copy with
1% NaN MUST be dropped by the filter.

---

## D2 - P1 / bug - discovery/__init__.py:160-170 - iter_transform crashes on auto-promoted linear_residual_multi specs (ignores extra_base_columns)

`iter_transform` extracts only `spec.base_column` as 1-D (line 162) and calls
`transform.forward(y, base_full[valid], spec.fitted_params)`. For a multi-base spec (produced by the
**default-ON** auto-promotion, `multi_base_enabled=True`, config line 115), `fitted_params["alphas"]` has K>=2
entries and `_linear_residual_multi_forward` raises ValueError ("...base has 1 columns but fitted alphas has K
entries") (transforms/linear.py:364-368). Every other consumer was already fixed to stack
`(base_column,) + extra_base_columns` (ensemble/__init__.py:439,674,879; the export_specs comment at
discovery/__init__.py:250-259 documents this exact bug class) - iter_transform, the public
"apply all specs" API, was missed. No test exercises iter_transform with a multi-base spec.

**Fix.** Build `cols = (spec.base_column,) + tuple(getattr(spec, "extra_base_columns", ()) or ())`; when
len(cols) > 1, np.column_stack the columns and pass the (n, K) matrix to domain_check/forward (both already
accept 2-D, linear.py:360-391). Add a regression test with a promoted multi-base spec.

---

## D3 - P1 / bug - _fit.py:737-738 + forward_stepwise.py:114-128 - multi-base promotion fits joint OLS on unfiltered rows; NaN in any pool base silently kills (or poisons) the default-ON feature

The single-base path fits on domain_check-valid rows only (_eval.py:44-56). The multi-base path does not:
`_pool_arrays` are raw base_train slices (NaN included, _fit.py:350-351, 709-711), `_y_train_local` is raw
(line 696), and both per-trial CV fits (forward_stepwise.py:124) and the final promotion fit
(`_linear_residual_multi_fit(_y_train_local, _base_matrix)`, _fit.py:738) get NaN rows.
`_linear_residual_multi_fit` has NO finite-row masking (linear.py:285-356):

- K>=2 with NaN: col_norms are NaN, `np.any(NaN < 1e-12)` is False, SVD raises LinAlgError, caught ->
  cond=inf -> collinear fallback (alphas=0, beta=mean(y)). Every trial degenerates to a constant predictor ->
  never clears the 2% gate -> **promotion silently dead**.
- K==1 (seed baseline) with NaN in the seed: cond hardcoded 1.0 (linear.py:307-308), lstsq on NaN raises ->
  caught at _fit.py:725 -> single-base kept with a vague warning.

Lag features - the canonical composite base - have leading NaNs in the earliest train rows, and temporal splits
put those rows in train. The benchmark-validated 83% geo-mean win (config line 114) is silently never realised
on realistic data. Conversely, if CV folds dodge the NaNs but the full-train final fit at line 738 (NOT inside
the try at line 713) hits them, fit() raises mid-flight or stamps fallback/NaN-beta params onto an accepted
spec.

**Fix.** Mask rows to all-finite before fitting, mirroring `_linear_residual_multi_domain` (linear.py:383-391):
in `_cv_rmse_with_folds` compute `finite = isfinite(y) & all(isfinite(base_matrix), axis=1)` per trial and
fit/score on that subset; same mask before _fit.py:738. Biz-value regression test: promotion must fire on
synthetic y = b1 + b2 + eps where b1 is a lag with NaN in its first row.

---

## D4 - P2 / bug - _auto_base.py:119-124 - hint-strength gate reads strengths positionally from the RAW hint list, misaligned after drops

`_hint_strengths_pct` is built in core/_phase_composite_discovery.py:346-350 aligned with the raw
ablation-sorted hint list. `_auto_base` filters that list into `hint_kept` (lines 86-91) but then evaluates
`max(hint_strengths[:len(hint_kept)])` (line 123) - count-of-kept indexing into raw-ordered strengths.
Example: hint = [A(501%), B(3%)], A dropped by the corr filter (plausible - a dominant lag trips exactly that
filter) -> hint_kept=[B], hint_strengths[:1]=[501.0] -> is_strong_hint=True -> weak B granted FULL hint trust,
no cap, and immunity from demoters/dedup (lines 243, 327, 514).

**Fix.** Plumb strengths as {feature: delta_pct} (or zip and filter pairs together); gate on
max(strength[c] for c in hint_kept).

---

## D5 - P2 / bug - _fit.py:560-561 - alpha-drift z-score omits the sqrt(2) factor for a DIFFERENCE of two independent estimates

a1, a2 are fitted on independent halves of `half` rows each, so Var(a1-a2) = 2*sigma^2/(half*var_base) and
SE = sigma*sqrt(2)/(sqrt(half)*base_std). The code uses `se_alpha = sigma_resid/(sqrt(half)*base_std)` - the
SE of a SINGLE half's estimate - inflating every z by 1.41x. Default threshold 3.0 effectively becomes ~2.12
sigma: materially more false drift WARNINGs (lines 786-800) and over-rejection when
reject_on_alpha_drift=True. (ENS-Low-2 fixed the sigma numerator, not the difference-variance factor.)

**Fix.** `se_alpha = sigma_resid * np.sqrt(2.0) / (np.sqrt(half) * base_std)`; update tests pinning z.

---

## D6 - P2 / bug - screening.py:534-548 - stratified sampler's tail truncation systematically drops the highest (latest) row indices

`per_stratum = max(1, sample_n // n_strata)` over n_strata + 1 bins (extra non-finite-y bin, line 532) can
overshoot sample_n; `out.sort(); return out[:sample_n]` (547-548) then deletes the overshoot exclusively from
the LARGEST indices - the most recent rows - contradicting the docstring's own temporal-bias rationale for
sorting. With the default-on heavy-tail boost (mi_n_strata 10->30, _fit.py:228-263) and NaN-y rows present,
up to ~sample_n/30 (~3.3k of 100k) latest rows are excised - thinning exactly the regime closest to
deployment. Side effect of the floor division: with no overshoot the sample slightly UNDERSHOOTS sample_n
(30x3333 = 99,990).

**Fix.** When out.size > sample_n, drop the excess via seeded rng.choice(out, sample_n, replace=False) BEFORE
the final sort; distribute the sample_n % n_strata remainder across strata.

---

## D7 - P2 / bug - discovery/__init__.py:209-227 - fit_with_stability_check mutates the SHARED config's random_state; restoration is lost once the heavy-tail boost swaps self.config

The suite passes the shared composite_target_discovery_config object into discovery without a copy when no
hint is applied (_phase_composite_discovery.py:330,460). fit_with_stability_check does
`self.config.random_state = base_seed + i*7919` (line 210) - in-place mutation of the shared object, visible
to every other consumer mid-loop. Worse: if the heavy-tail boost fires during run i, fit replaces self.config
with a model_copy (_fit.py:252-255); the restore at line 223 then writes to the COPY, leaving the user's
original config permanently at base_seed + i*7919 - subsequent targets' discovery (and the cache-key config
signature) silently run with a different seed.

**Fix.** Take `self.config = self.config.model_copy()` once at entry; restore by re-assigning the saved
original object, not by writing the field.

---

## D8 - P2 / bug - _fit.py:477, 568-590, 630-646, 655, 736-760, 804 - report_ kept-flags frozen at the eps gate; later drops/upgrades invisible

entry["kept"]=True is stamped at the eps gate (477). Specs removed later by alpha-drift reject (569),
linres->diff collapse (631), tiny-model rerank (655), or stability filtering still show kept=True, reason=""
in report(); multi-base upgrades replace a spec with a NEW name that appears nowhere in report_ (740-760,
report built at 804 from the original candidates). report()'s contract is "all evaluated candidates including
rejected ones with reasons" - an operator auditing "why is X not in specs_" gets actively misleading output.

**Fix.** After kept_specs is final, reconcile: kept=False + reason="dropped_by_<gate>" for entries whose spec
name is absent from the final list; append synthetic entries for upgrade names (with upgraded_from).

---

## D9 - P2 / bug - _fit.py:736-760 - two seeds converging to the same multi-base set produce duplicate specs (different name order), doubling downstream training

Each kept linear_residual spec is independently seeded into forward_stepwise_multi_base. Seed X adding Y
yields [X, Y]; seed Y adding X yields [Y, X]. Both clear the gate; `"+".join(_kept_bases)` produces two
distinct names for the SAME joint-OLS transform (identical up to column permutation). No set-level dedup after
the upgrade loop -> downstream Phase B / suite training pays 2x for one composite, and the cross-target
ensemble gets two perfectly correlated members.

**Fix.** Dedup _upgraded_specs on (transform_name, frozenset((base_column,) + extra_base_columns)), keeping
the first (higher-ranked) spec.

---

## D10 - P2 / perf - _fit.py:348-389 - per-base np.delete copies of the full screen matrix (float32 + int64 prebinned) held alive for ALL bases simultaneously; mi_y baselines recomputed per base

For each base, np.delete(_full_x_matrix, j, axis=1) (355) and np.delete(_full_x_prebinned, j, axis=1) (357)
allocate (n_screen, F-1) copies (~12 B/cell combined), and _base_contexts keeps them ALL alive through
candidate evaluation (382). At 100k screen rows x 500 features x 10 explicit bases that is ~6 GB of avoidable
resident copies. mi_y_for_base also runs a full (F-1)-column MI pass per base (370-381), though with the
default aggregation="mean" it is derivable from ONE full-matrix per-feature pass:
mi_y_for_base_j = (S - mi_j)/(F-1) with S = sum(per_feature_mi) - _mi_per_feature_y_fixed already returns
exactly that vector.

**Fix.** Compute per-feature MI(y, x_j) once; derive each base's mean baseline arithmetically. For mi_t, pass
the FULL prebinned matrix plus an exclude_col: int param into _mi_to_target_prebinned (skip j in its
per-column loop) - zero copies, bit-identical means. Keep np.delete only for the non-"bin" estimator path or
build it lazily inside eval_one_transform.

---

## D11 - P2 / bug - screening.py:431-433 - _mi_to_target (knn path) global AND finite mask: one NaN-heavy column zeroes MI for the whole sweep

`finite = np.isfinite(target) & np.all(np.isfinite(feature_matrix), axis=1)`. A column with >=50 finite values
passes _filter_features but can be 99% NaN; the AND-mask then leaves <50 rows -> return 0.0 for EVERY
mi_y/mi_t evaluation -> all gains 0 -> discovery silently degenerates whenever mi_estimator="knn" is selected
(the documented heavy-tail-accurate option; default "bin" is protected by per-column -1 sentinels). The same
bug class was fixed for the prebinned path in Wave 24 (screening.py:306-313) but not here.

**Fix.** Mirror the prebinned design: per-column finite handling for the knn estimator, or drop >X%-NaN
columns before the AND-mask and log the row shrinkage.

---

## D12 - P2 / bug - _auto_base.py:184-203 - all-column AND finite mask silently shrinks the ranking sample with MNAR bias for mid-range-NaN columns

Columns with 10-90% NaN survive the _MIN_FRAC_FINITE=0.10 drop (167-169); the AND-mask (184) then intersects
their NaN patterns (five independent 50%-NaN columns leave ~3% of rows). As long as >=50 rows survive, MI
ranking for ALL features runs on this tiny non-random subsample (rows where every feature is observed - MNAR
selection bias), and the time-index/spatial demoters and dedup inherit it. The mean-impute fallback engages
only below 50 rows (185-196). _mi_pair_bin supports per-pair masking internally; the hoisted
_mi_per_feature_y_fixed traded that away (docstring assumes pre-masked input).

**Fix.** Log shrinkage when finite.sum() < 0.5*n_screen; for the bin estimator fall back to per-column
_mi_pair_bin (per-pair masks) when the AND-mask retains < ~50% of rows, not only below 50 rows.

---

## D13 - P2 / bug - _fit.py:397-413 + _eval.py:34-44, 277 - unary transforms scored against an arbitrary base's context and stamped with that base's name

Unary (requires_base=False) transforms are deduped to the FIRST base they appear under (409-413);
eval_one_transform computes mi_t/mi_y against THAT base's x_remaining_matrix (X minus a column the transform
never uses) and the spec gets base_column=<first base> plus a name composed with it (_eval.py:277-281).
Consequences: (a) a unary spec's mi_gain changes when auto-base ranking reorders an irrelevant degree of
freedom; (b) the spec/report claim a base dependence that does not exist and downstream consumers extract a
base column for no reason. The natural feature set for a unary y-transform is the FULL usable X.

**Fix.** Dedicated unary context: x_remaining = _full_x_matrix (+ _full_x_prebinned), mi_y computed once on
the full matrix, base_column=""/sentinel rendered base-free by the spec/name layer.

---

## D14 - P2 / bug - screening.py:66 - _extract_column_array crashes on pandas nullable dtypes containing NA

df[col].to_numpy(dtype=np.float32) on a nullable extension dtype (Int64, Float64, boolean) with missing values
raises ValueError: cannot convert to 'float32'-dtype NumPy array with missing values. _is_numeric_column
approves those dtypes (is_numeric_dtype True, line 92), so a frame with Arrow-backed / nullable pandas columns
crashes discovery at the first column pull instead of treating NA as NaN.

**Fix.** df[col].to_numpy(dtype=np.float32, na_value=np.nan) (branch on extension dtype if needed).

---

## D15 - P2 / usability - _fit.py:185 - boolean-mask train_idx accepted silently, detonates later with a cryptic IndexError

np.asarray(train_idx) accepts a boolean mask (common numpy idiom). y_full[mask] works, but
_sample_indices(train_idx.size, ...) treats mask LENGTH as row count, and train_idx[sample_idx] /
base_screen = base_train[sample_idx] (_fit.py:295, 352) index positionally into mask/short arrays ->
IndexError far from the misuse (or bool values reinterpreted as indices 0/1 when mi_sample_n=None). The
message names neither train_idx nor the cause.

**Fix.** At fit entry: if train_idx.dtype == bool: train_idx = np.flatnonzero(train_idx); reject float dtypes
with a clear TypeError. Same for val_idx/test_idx.

---

## D16 - P2 / extension - _fit.py:149-206 - no train/val/test disjointness or bounds validation despite the class's "Leakage discipline (CRITICAL)" contract

fit stores val_idx_/test_idx_ "for later integrity checks" (docstring 174-176) but performs none: overlapping
train/test indices (an easy caller bug after re-splitting/reindexing) silently fit transform params, MI
screens, and tiny-rerank on test rows - exactly the leakage the class documents as its core discipline.
Duplicates within train_idx (bias MI via repeated rows) and out-of-bounds indices are also unchecked. The
check is O(n log n) on int arrays - negligible vs MI screening.

**Fix.** At fit entry, when val/test provided: raise on non-empty np.intersect1d(train_idx, test_idx) (and
val); warn on duplicated train_idx; bounds-check max index vs len(df). Unit test asserting the raise.

---

## D17 - LOW / bug - _eval.py:261-271 - bootstrap-failure warning fires only when replicate index 0 fails, not on the first failure

Comment says "Log first per-spec failure" but the guard is `if b == 0:` - failures at replicates 1..N-1 are
silent; the LCB is computed over a reduced/biased bootstrap sample with no operator signal (the
>= bootstrap_n // 2 guard at 273 only catches extreme under-sampling).

**Fix.** _warned flag set on first exception; include running failure count in the message.

---

## D18 - LOW / perf - _fit.py:219, 288, 696 - y_full[train_idx] materialised three times per fit

y_train (219), y_train_for_strat (288), _y_train_local (696, with a dead `if y_full is not None` else-branch -
y_full is always bound by then) are the same fancy-index copy. Three O(n_train) allocations + gathers per fit;
trivially one.

**Fix.** Reuse y_train; delete the dead else-branch at 696.

---

## D19 - LOW / bug - _fit.py:608 - float(np.std(empty)) or 1.0 is NaN-truthy

With all-NaN y_train, np.std of the empty slice is NaN (+ RuntimeWarning); NaN or 1.0 evaluates to NaN
(bool(NaN) is True), so std_y=NaN and the linres->diff collapse silently disables itself (NaN compares False).
Practically unreachable today (no specs survive all-NaN y) but a guard-pattern bug that invites copy-paste.

**Fix.** Explicit `if not (isfinite(std_y) and std_y > 0): std_y = 1.0`.

---

## D20 - LOW / usability - _fit.py:191 - self._df_ref = df keeps a 100+GB frame alive on the instance with no __getstate__ exclusion

The fitted discovery instance holds a hard reference to the full training frame (for iter_transform).
Pickling/deep-copying the instance (diagnostics dump, provenance snapshot, joblib) serialises the whole frame;
the reference also pins the frame against GC for the instance lifetime. Per the project's
runtime-caches-break-pickle rule, frame-holding attributes should be excluded from state.

**Fix.** __getstate__/__setstate__ dropping _df_ref (and _auto_base_pool, which holds n_train-sized arrays per
base); document that iter_transform needs the explicit df argument after unpickling.

---

## D21 - LOW / extension - _eval.py:160-196 - near-duplicates of the base remaining in X_remaining attenuate mi_gain (conservative bias)

np.delete removes only the base's own column; a 0.99-correlated sibling (lag2, smoothed copy) stays in BOTH
halves of the gain: it inflates mi_y_compare (high MI with y) while contributing little to mi_t (T is
residualised against the base), biasing mi_gain DOWN for exactly the bases most likely to be genuine (lag
families). No leak (conservative direction), but real composites get under-ranked vs bases without duplicated
siblings. Auto-base dedup (0.95) filters base CANDIDATES, not the screening feature set.

**Fix.** Optionally exclude columns with |corr(base, x_j)| >= auto_base_dedup_corr_threshold from that base's
x_remaining (per-base column mask; integrates with the D10 exclude-col refactor at zero copies).

---

## D22 - LOW / docs - _fit.py:360-363 - "explicit base outside the FE pool" branch is unreachable

_resolve_base_candidates filters explicit bases to usable_features (__init__.py:344) and _auto_base returns
subsets of usable_features, so every base is in _col_index; the else-arm assigning the FULL matrix as
x_remaining (which would also be methodologically wrong - base present in its own "remaining" set) cannot
execute.

**Fix.** Delete or convert to a loud logger.error guard; note the load-bearing invariant in
_resolve_base_candidates.

---

## D23 - LOW / bug - _auto_base.py:197-203 - degenerate-sample fallback returns raw feature-list order, ignoring hint priority

When <50 finite rows survive even after impute, `return list(usable_features)[:auto_base_top_k]` discards
hint_kept entirely - the one signal (BD ablation) that does NOT depend on the broken screening sample and that
the function elsewhere treats as authoritative. Hint features may not even be in the first top_k of the
arbitrary column order.

**Fix.** return (hint_kept + [c for c in usable_features if c not in hint_kept])[:top_k].

---

## D24 - LOW / docs - screening.py:223-224 + _fit.py:194-202 - for 50 <= n < 5*mi_nbins all bin-MI values are silently 0.0; ranking degenerates to name order

The fit-level gate warns below 50 rows, but _mi_pair_bin / _mi_to_target_prebinned return 0.0 below 5*nbins
rows (80 at default mi_nbins=16). In the 50-79 band every mi_y/mi_t/mi_gain is exactly 0; with default
eps_mi_gain=-10 all candidates pass tied at 0 and top-K selection falls to the alphabetical tiebreaker
(_fit.py:481). Hybrid rerank arbitrates so results aren't wrong, but the MI "ranking" is fictitious and
undocumented in this band.

**Fix.** Log when min(train_idx.size, mi_sample_n) < 5*mi_nbins ("bin-MI inactive; ranking deferred to
rerank") or auto-shrink nbins to n // 5.

---

## D25 - LOW / usability - _fit.py:469-482 - gate uses the bootstrap LCB but ranking/top-K truncation uses the point estimate

With mi_gain_bootstrap_n > 0, a spec is admitted on mi_gain_lcb but competes for top_k_after_mi slots on
spec.mi_gain (481). A high-variance candidate with a big point estimate and barely-passing LCB outranks a
stable candidate with a better LCB - inconsistent with the gate's own noise-floor rationale. (Default
bootstrap_n=0 makes this opt-in-only.)

**Fix.** When bootstrap is enabled, rank by (-mi_gain_lcb, name); document the choice either way.

---

## Summary table

| id | sev | category | file:line |
|----|-----|----------|-----------|
| D1 | P0 | leak | discovery/_filter.py:203 |
| D2 | P1 | bug | discovery/__init__.py:160 |
| D3 | P1 | bug | discovery/_fit.py:738 |
| D4 | P2 | bug | discovery/_auto_base.py:123 |
| D5 | P2 | bug | discovery/_fit.py:560 |
| D6 | P2 | bug | discovery/screening.py:548 |
| D7 | P2 | bug | discovery/__init__.py:210 |
| D8 | P2 | bug | discovery/_fit.py:804 |
| D9 | P2 | bug | discovery/_fit.py:739 |
| D10 | P2 | perf | discovery/_fit.py:355 |
| D11 | P2 | bug | discovery/screening.py:431 |
| D12 | P2 | bug | discovery/_auto_base.py:184 |
| D13 | P2 | bug | discovery/_fit.py:409 |
| D14 | P2 | bug | discovery/screening.py:66 |
| D15 | P2 | usability | discovery/_fit.py:185 |
| D16 | P2 | extension | discovery/_fit.py:157 |
| D17 | LOW | bug | discovery/_eval.py:261 |
| D18 | LOW | perf | discovery/_fit.py:288 |
| D19 | LOW | bug | discovery/_fit.py:608 |
| D20 | LOW | usability | discovery/_fit.py:191 |
| D21 | LOW | extension | discovery/_eval.py:162 |
| D22 | LOW | docs | discovery/_fit.py:360 |
| D23 | LOW | bug | discovery/_auto_base.py:203 |
| D24 | LOW | docs | discovery/screening.py:223 |
| D25 | LOW | usability | discovery/_fit.py:481 |