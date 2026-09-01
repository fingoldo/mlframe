# xcut_numerical_stability

Moment computations examined: 34 | files: 24

## Summary

All **three** confirmed instances named in the audit brief are **already fixed** in this tree -- including
`_derive_cell_stats`, which `CLAUDE.md:443` and `docs/NUMERICAL_STABILITY_REPORT.md:233` still describe as OPEN.
`_binned_numeric_agg_fe.py:137` now takes `(cnt, mean, cm2, cm3, cm4)` centred sums from
`_per_cell_moments_stable`, carries no `+1e-12` denominator pad, and its GPU twin
`_binned_numeric_agg_resident.py:75/98` was converted in lockstep. (Verified independently after the sweep: the
docstring at `_binned_numeric_agg_fe.py:138-147` records the replacement and the reason.)

Of the 34 moment computations examined, 27 are stable -- two-pass centred, Welford, or delegated to scipy.

The sweep instead found the same defect class alive **one order down**: variance from raw power sums,
`var = E[x^2] - E[x]^2`, which is the identical `sum(x^k)` cancellation at k=2. **Nine live sites across four
files**, none of which the three prior fix rounds touched, because those rounds grepped for skew and kurtosis.
That is the lesson worth carrying forward: the recurring shape is `sum(x^k)` minus a power of the mean for ANY
k >= 2, not just k in {3, 4}.

## Findings

### XCUT_NUMERICAL_STABILITY-1 [P1] raw-power-sum-variance-defeats-reject-gate

**File:** `src/mlframe/feature_selection/filters/_usability_njit_pool.py` :300, :334, :380, :418, :625

**Summary:** Five copies (serial njit, parallel njit, two unary-table twins, and the cupy GPU twin) compute the
combo-rejection variance as `var = ss / n - mean * mean` from raw power sums, then gate on `var <= 1e-18` -- a
threshold far below the formula's own cancellation error.

**Failure scenario:** The cancellation noise floor of `ss/n - mean*mean` is `~eps * mean^2 = 2.2e-16 * mean^2`.
The gate constant `1e-18` sits below that floor for any `|mean| > 0.067`, i.e. essentially always. Concretely,
`_apply_binary(add)` on two epoch-second columns gives `v ~ 3.4e9`, so `mean^2 ~ 1.16e19` and the noise floor is
`~2.6e3`. Any combo whose TRUE variance is under `2.6e3` -- a standard deviation under 51 seconds on an epoch
column, which is routine -- computes a value that is pure noise with random sign. Roughly half the time it lands
negative, trips `var <= 1e-18`, and the combo is written out as the `-1.0` sentinel and silently dropped from
the retention pool. With an `exp` unary on a column of ~30 (`v ~ 1e13`, `mean^2 ~ 1e26`) the noise floor is
`2.2e10`, so a genuinely informative combo with true variance `1e9` is a coin flip. The failure is a SILENT
false-negative feature rejection, invisible downstream because `-1.0` is indistinguishable from a legitimately
constant combo. The converse also occurs: a truly constant combo whose noise lands positive passes the gate and
is quantile-binned as a degenerate single-bin column.

**Suggested fix:** Replace all five with a two-pass centred accumulation. `val[i]` is already materialised into
the `val` array inside the same loop, so the second pass costs nothing extra: accumulate `s` in the existing
loop, then `mean = s / n`, then one more pass over `val` accumulating `d = val[i] - mean; ss += d * d`. Add the
constant-column short-circuit `_global_stats_all` already has (`vmin == vmax` -> variance exactly 0), since
sequential summation of a huge-offset constant column is not otherwise bit-exact to `n * value`. Re-derive the
`1e-18` threshold only after the numerator is stable. The GPU twin at :625 must change identically to preserve
the module's stated bit-faithfulness contract (docstring :25-33).

**Evidence:** `_usability_njit_pool.py` :1-40 (module contract: three bit-faithful kernel versions dispatched by
`kernel_tuning_cache`), :280-341 (serial + parallel njit), :353-425 (both unary-table twins), :605-631 (cupy
twin, whose comment `# std<=1e-9 sentinel: var = E[v^2] - E[v]^2 <= 1e-18` makes the raw-sum intent explicit).
`_apply_unary` :110-137 confirms `sqr`/`qubed`/`exp` are in the op table, so operand magnitudes are amplified
before the variance is taken.

**Disposition:** RESOLVED at all five sites (centred two-pass accumulation plus an exact `vmin == vmax` short-circuit), but the finding is HALF CORRECT and the correction matters.

The converse failure it describes reproduces exactly: a genuinely constant combo whose cancellation noise lands positive passes the gate. Measured on 4000 identical values, `ss/n - mean*mean` returns 1.34e8 at scale 1e12 and 1.41e14 at 1e15, so a column with one distinct value was quantile-binned into a single bin. The `vmin == vmax` test now catches it exactly.

The headline scenario -- an informative combo silently rejected as the -1.0 sentinel -- does NOT reproduce, and the reason is that `_apply_binary` scrubs every value through `np.float32` before the variance is computed. The float32 ulp at scale m is ~1.2e-7 * m while the float64 cancellation floor is ~1.5e-8 * m, so any combo surviving the scrub as non-constant differs by at least one ulp and has a variance roughly 16x above the floor. Measured across scales 1.7e9 / 1e12 / 1e15 at 1, 2, 8 and 64 ulps of spread: the old formula agreed with the true variance to within 8% every time and never landed negative. The epoch-column example in the finding (v ~ 3.4e9, true variance 450) is not reachable -- a 20-unit signal on 3.4e9 is below float32 resolution and the combo is genuinely constant after the scrub.

`tests/feature_selection/test_combo_variance_gate_is_not_reading_noise.py`; the four constant-column cases at scale 1e12 fail on the pre-fix code.

### XCUT_NUMERICAL_STABILITY-2 [P2] raw-power-sum-variance-reintroduces-the-false-drop-it-guards

**File:** `src/mlframe/feature_selection/pre_screen.py` :219

**Summary:** The sparse-column closed-form variance is `var_val = sumsq_valid / n_valid - mean_valid**2`, feeding
a `var_val <= _var_cutoff` DROP decision.

**Failure scenario:** The comment at :189-196 says this branch exists specifically to stop informative sparse
columns being false-dropped; the raw-sum form reintroduces that false drop in a different regime. For a sparse
price column with `fill_value = 1e6` and stored values clustered near it, `mean^2 ~ 1e12` and the cancellation
floor is `2.2e-4`, while a genuine within-column variance of `1e-6` is three orders of magnitude below the
noise. The computed `var_val` is noise of random sign; when negative it satisfies the cutoff at :227 and the
column is added to `drops` and permanently removed by `apply_drops`. There is no `max(var, 0)` clamp, so a
cancellation-negative variance is treated as "less variance than the cutoff" rather than as a numerical failure.

**Suggested fix:** Compute the mean first, then form the variance from centred contributions -- still one pass
over the stored values, so no cost change: `mean = sum_valid / n_valid`, then
`var = (sum((finite_sp - mean)**2) + n_fill_valid * (fill_value - mean)**2) / n_valid`. The implicit fill cells
contribute a single closed-form centred term, so the sparse-awareness the branch exists for is preserved. Keep
the `n_fill_valid` truthiness guard at :215 that avoids `0 * nan`.

**Evidence:** `pre_screen.py` :185-231. `sum_valid`/`sumsq_valid` accumulate at :213-217 including the fill-mass
terms, combine at :218-219, and flow unclamped into the NaN check at :224 and the cutoff at :227.

### XCUT_NUMERICAL_STABILITY-3 [P2] raw-power-sum-variance-under-float32-accumulation

**File:** `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_prefilter_univariate.py` :211-214

**Summary:** The ANOVA F prefilter forms `sst = total_sumsq - correction` and `ssbn = (...) - correction` from
raw sums of squares, then takes `sswn = sst - ssbn` -- a difference of two already-contaminated quantities --
with the accumulator dtype inherited from the input (`acc_dtype = X.dtype`, so float32 for float32 input).

**Failure scenario:** At float32, `eps = 1.19e-7`. For an epoch-timestamp feature (~1.7e9) with a genuine
within-class std of 10 at `N = 1e6`: `total_sumsq ~ 2.89e24`, `correction` the same, so cancellation noise is
`3.4e17` against a true `sst` of `1e8` -- nine orders of noise over signal. `sst`, `ssbn` and `sswn` are all
garbage, `f` at :236 is meaningless, and the `f64 < 0.0` clamp at :241 sends the column to `-inf`, ranking a
genuinely discriminative feature WORST in the pool. Even at float64 the same column gives a noise floor of
`6.4e8` against `sst = 1e8`. The existing `cancel_floor` (:223) and `min == max` (:234) guards catch only
CONSTANT columns; a large-offset column with real but small variance passes both and gets a fabricated F.

**Suggested fix:** This site is constrained by an explicit sklearn bit-parity contract
(`test_f_classif_float32_input_matches_sklearn_float32`, referenced at :170), so the raw form cannot simply be
replaced. Add an opt-in stable path: compute the grand mean first, then accumulate centred sums of squares per
class, which yields the identical `sst`/`ssbn`/`sswn` algebraically with no cancellation. Gate it behind
`stable=True`, defaulting ON for the mlframe prefilter caller, retaining the bit-parity path under
`stable=False` for the parity test. At minimum, promote `acc_dtype` to float64 regardless of input dtype and
document the parity divergence.

**Evidence:** :185-244; the dtype contract at :164, :171, :187. The comment block at :215-222 shows the author
already diagnosed one large-mean/low-variance false drop and fixed the THRESHOLD; the underlying raw-sum formula
was left in place.

### XCUT_NUMERICAL_STABILITY-4 [P2] raw-power-sum-variance-inverts-a-redundancy-diagnostic

**File:** `src/mlframe/calibration/_independence_check.py` :52-54

**Summary:** `cov_ab`, `var_a` and `var_b` are all formed as `E[xy] - E[x]E[y]` / `E[x^2] - E[x]^2` from raw
sufficient statistics, and a cancellation-negative variance is silently converted into an "independent" verdict.

**Failure scenario:** Logits are bounded to +-16.1 by the `clip=1e-7` default, which caps the offset/scale ratio
and keeps ordinary cases safe. The break is a NEAR-SATURATED member: one that outputs essentially the same
confident logit on every row (`+16.0 +- 1e-7`). Then `mean_a^2 ~ 259`, the cancellation floor is `5.7e-14`, and
the true `var_a ~ 1e-14` is below it. `var_a` comes out noise-signed; when negative,
`np.clip(var_a * var_b, 0.0, None)` at :56 forces the product to `0.0`, the `where=denom > 1e-300` guard at :57
fails, and the function returns a correlation of exactly 0.0. The diagnostic's whole purpose (docstring :20-21:
separating redundant members from conditionally independent ones) is INVERTED -- a degenerate, maximally
redundant member is reported as perfectly independent and would be retained by any downstream pruning rule.
`cov_ab` at :52 has the same form with no guard at all.

**Suggested fix:** Centre once, then form the second moments from deviations. `row_sum`, `col_sum` and `logits`
are all already in memory, and the leave-one-out affine identity at :46-48 is translation-covariant, so the
closed form and its single `logits.T @ row_sum` BLAS pass survive intact. Additionally, replace the silent
`clip(..., 0.0, None)` with an explicit near-degenerate-variance branch returning NaN rather than 0.0, so a
saturated member is surfaced rather than laundered into a clean independence result.

**Evidence:** `_independence_check.py` :28-57. The docstring :31-35 documents the sufficient-statistic closed
form as a deliberate one-BLAS-pass optimisation; the fix preserves that property.

### XCUT_NUMERICAL_STABILITY-5 [P3] raw-power-sum-variance-plus-additive-mean-pad

**File:** `src/mlframe/votenrank/adversarial_stochastic_blend.py` :189-192

**Summary:** The convergence curve computes `cum_var = np.maximum(cum_sq_mean - cum_mean**2, 0.0)` from
cumulative raw power sums, then divides by an additively padded mean `np.abs(cum_mean) + 1e-12`.

**Failure scenario:** Blend weights are simplex-bounded in [0, 1], so the offset/scale ratio is mild and this is
the least severe instance. The regime that breaks it is a CONVERGED blend: all iterations produce nearly
identical weights (`w = 0.25 +- 1e-9`). Then `cum_mean^2 = 0.0625`, the cancellation floor is `1.4e-17`, the
true variance `1e-18` is below it, and `np.maximum(..., 0.0)` clamps the noise to exactly 0 -- giving
`cum_std = 0`, `per_iter_cov = 0` and `stability_score = 1.0`. A report of PERFECT convergence produced by a
numerical artifact rather than a measurement, and optimistic, which is the wrong direction for a
trustworthiness diagnostic. Separately the `+1e-12` at :192 is the additive-pad shape: for a member whose weight
legitimately converged toward zero, `|cum_mean| ~ 1e-13` and the pad dominates the true denominator,
understating that member's coefficient of variation by ~10x with no cancellation involved at all.

**Suggested fix:** `collected_weights` is fully materialised, so a stable cumulative variance is available
directly -- Welford across the iteration axis, or per-prefix means centred against. Replace the `+1e-12` pad
with `np.where(np.abs(cum_mean) > tol, cum_std / np.abs(cum_mean), np.nan)` so a near-zero-weight member reports
an undefined coefficient of variation rather than a silently deflated one.

**Evidence:** :180-199. `cum_var` -> `cum_std` (:191) -> `per_iter_cov` (:192) -> `convergence_curve` (:193) ->
the exported `stability_score` (:195), so the artifact reaches the caller-visible result.

### XCUT_NUMERICAL_STABILITY-6 [P3] stale-documentation-of-a-closed-finding

**File:** `docs/NUMERICAL_STABILITY_REPORT.md` :233, and `CLAUDE.md` :313, :443-446

**Summary:** Both documents state that `_binned_numeric_agg_fe.py::_derive_cell_stats` still uses the unstable
formula and is an open follow-up. It was converted to centred moments and is stable.

**Failure scenario:** No runtime failure. The cost is misdirected effort -- this sweep was commissioned on the
premise that the finding was open -- and, more importantly, the stale entry lists only three instances of the
bug class, which anchored three prior search rounds to skew and kurtosis. That anchoring is why the nine
`var = E[x^2] - E[x]^2` sites in findings 1-5 went unnoticed.

**Suggested fix:** Mark the `_derive_cell_stats` bullet fixed in both documents, add the GPU twin
`_binned_numeric_agg_resident.py::_per_cell_moments_stable_gpu` to the fixed list, and broaden the guidance at
:238-240 from "any new skew/kurt kernel" to cover VARIANCE as well: the shape is `sum(x^k)` minus a power of the
mean for any k >= 2.

**Evidence:** `_binned_numeric_agg_fe.py` :137-176 (signature takes `(cnt, mean, cm2, cm3, cm4)`; docstring
:138-147 records the replacement and the pad removal), :179-185, :399/:403 (both OOF call sites pass centred
moments). Repo-wide grep for `_raw_moments`: the only surviving raw-power-sum call site is
`profiling/bench_binned_numeric_agg_fold_gate.py` :69-84, a benchmark that intentionally preserves the old form
for A/B. `_per_cell_raw_moments_njit` (:36) survives in production but is called only for `cnt, s1` (:89) -- a
plain additive sum for the mean pass, not a defect.

## Verification of the three named instances

| Site | Status | Evidence |
|---|---|---|
| `_binned_numeric_agg_fe.py::_global_stats_all` | Fixed | `_centered_moments_njit` :209-236 (two-pass), constant-column fast path :259-275, no additive pad :290-295 |
| `_target_encoding_fe.py::_raw_moment_sums` | Fixed | Replaced by `_per_cat_centered_moments_njit` :86-121 and `_smooth_moments_from_centered` :124-160; pad removed, docstring :138-140 records why |
| `_binned_numeric_agg_fe.py::_derive_cell_stats` | **Fixed -- NOT still open** | :137 takes `cm2/cm3/cm4`; guards are bare `np.where(std > 1e-9, ...)` / `np.where(m2 > 1e-12, ...)` at :168/:172 with no `+1e-12` pad |

The OOF fold-structure additivity bug is also closed on both host and device: `_binned_numeric_agg_fe.py`
:393-399 recomputes train moments directly rather than `full - test`, and `_binned_numeric_agg_resident.py`
:110-113 documents the same tradeoff for the masked GPU pass.

## Coverage

Swept by grep then targeted read, across all of `src/mlframe/`: the shapes `s2`/`s3`/`s4`, `sum(x**k)`,
`bincount(..., weights=)`, `E[x^2] - E[x]^2` in its `/ n - mean`, `sumsq`, `ss / n` and `- mean*mean` spellings,
additive denominator pads (`+ 1e-12`, `+ 1e-9`), and `def *(skew|kurt|moment|variance|std)`.

Read and judged stable (27), no finding written: `_binned_numeric_agg_fe.py` (4 kernels),
`_target_encoding_fe.py` (2), `_binned_numeric_agg_resident.py` (3, including the fused `_skew_k`/`_kurt_k` at
:163/:173 -- no pad), `feature_engineering/_numerical_stable.py` (`welford_moments_seq`,
`naive_moments_two_pass_seq` :192-222), `_numerical_numba.py` (`_make_compute_moments_slope_mi` --
Kahan-compensated, centred on a precomputed `mean_value` at :540), `transformer/_aggregation.py` :246-257,
`utils/_param_oracle.py` :279-291, `_target_distribution_analyzer_stats.py` :18-53,
`loss_recommendation.py::_safe_moments` :85-110, `regression_residual_audit.py` :76-94,
`entity_inter_event.py` :99-124 (Welford add/remove, with a docstring explicitly rejecting the raw-sum
recurrence), `reporting/charts/drift.py` :362-368, `_regression_metrics.py::_fast_r2_variance_seq` :171-183,
`composite/transforms/unary.py` :176, `split_comparison.py::_var` (closed-form Mann-Whitney null variance, not
sample moments), `transformer/distributional_moments.py` (quantile-spread proxies, out of class).

Not reached: `src/mlframe/feature_selection/filters/_vendored/` (third-party, excluded as not first-party).
`_benchmarks/` and `profiling/` were grepped but not written up -- benchmark files intentionally retain the old
unstable forms for A/B comparison, and `profiling/bench_binned_numeric_agg_fold_gate.py` is the one such case.
