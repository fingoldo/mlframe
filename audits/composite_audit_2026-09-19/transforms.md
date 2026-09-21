# Composite targets audit 2026-09-19: direction 1 of 6, TRANSFORMS

**Scope**: `src/mlframe/training/composite/transforms/` (all 23 modules: `__init__.py`, `registry.py`, `_registry_extended.py`, `_registry_shared_adapters.py`, `_domain_shared.py`, `simple.py`, `linear.py`, `nonlinear.py`, `_nonlinear_ewma_fracdiff.py`, `unary.py`, `extended.py`, `_multi_extra.py`, `_grouped_extra.py`, `_seasonal.py`, `_second_diff.py`, `_volatility.py`, `_nadaraya_watson.py`, `_gaussian_copula.py`, `_rank_ecdf.py`, `_causal_anchor.py`, `categorical.py`, `interaction_bases.py`, `naming.py`) plus the transform helper at the composite root, `_quantile_edges.py`. I also read the fit/predict call sites in `estimator/_estimator.py` and `estimator/_predict.py` to confirm how `domain_check`, `domain_check_fitted`, `recurrent`, `row_index` and `groups` reach the transforms. I did not audit the estimator itself; that is another direction.

**Checkout**: `.claude/worktrees/agent-ab1823c081b446d9e` (origin/master a950f7e47). All file:line references are to that tree.

**Method**: I read every module in full. I cross-checked prior audits (`full_audit_2026-07-21/implemented/training_composite_blocks.md`, `full_audit_2026-08-05/implemented/training_composite_ensemble_estimator_transforms.md`, and the 2026-09-0x trackers) so that decided items are not raised again. Already resolved and not repeated here: seasonal `row_index` / `recurrent`, the grouped-recurrent `groups_full` fix, soft-shrink allow-list gaps, and dead duplicate kernels/constants. I reproduced every "concrete input -> wrong output" claim below with a tiny single-process snippet (`OMP_NUM_THREADS=2`) calling `get_transform(name).fit/forward/inverse` directly, and I quote the numbers from those runs. No pytest batch was run.

**Why the existing contract test misses most of these**: `tests/training/composite/transforms/test_composite_transforms_registry_contract.py` round-trips every transform on a single fixture: `_BASE = linspace(1, 10, 200)`, `_Y = 0.5*base + 1 + noise`. That fixture has small scale, positive values, a continuous target, n=200, and no out-of-range predict base. It also compares the **median** absolute error, not the maximum. Every scale-, tie-, tail- and OOD-dependent defect below passes it.

---

### TRF-01 [P1] `reciprocal_residual` inverse clamps `z = T_hat + 1/base` with an epsilon in base units, so every prediction collapses to a constant once |y| is larger than about 1e6/median|base|
- **Where**: `src/mlframe/training/composite/transforms/extended.py:445-457` (`_reciprocal_residual_inverse`, `eps_z = max(eps_b, _RECIPROCAL_EPS_FLOOR)`), with `eps_b` fitted at `extended.py:418-429`.
- **What**: `eps_b = 1e-6 * median|base|` is in *base* units. `z` is in *1/y* units (`z ~ 1/y`). The inverse replaces every `|z| < eps_z` with `sign(z)*eps_z`, so the clamp fires whenever `1/|y| < 1e-6*median|base|`. Repro: `base ~ U(900, 1100)`, `y = base * U(1.5, 2.5)` (y about 1600-2700). Fitted `eps_b = 0.00101`. For true z about 5e-4 the clamp gives `safe_z = 0.00101`, and **every** `y_hat` becomes `987.96` (= 1/eps_b). Actual y values are `[1869.5, 1609.8, 1973.1]` and the max round-trip error on train is `1714`, with T = the exact forward. The forward is fine because it floors `y` with `eps_y`. Only the inverse is wrong.
- **Why it matters**: `reciprocal_residual` is in the **default discovery pool** (`_composite_target_discovery_config_base.py:233`). With any target and base in the thousands (prices, revenue, durations in seconds), the transform's y-scale predictions are a constant. At best discovery throws away a transform that could have helped. At worst an explicit `transform_name="reciprocal_residual"` serves nonsense silently: no warning, finite output.
- **Suggested fix**: derive the z-floor from the **y** side and store it in params at fit. For example `eps_z = 1 / (_RECIP_Y_CAP_MULT * max|y_train|)`, which bounds `|y_hat|` to a multiple of the train y range. Alternatively reuse `eps_y` semantics: `eps_z = 1/(y_scale * 1e6)`. Keep the old key readable for pickled params.
- **Test to add**: round-trip `reciprocal_residual` on `base ~ U(900,1100)`, `y = 2*base`, and assert **max** abs error < 1e-6 * max|y|. Add a parametrised scale sweep (`base` scale in {1e-3, 1, 1e3, 1e6}) that fails on the current code at 1e3 and above.
- **Disposition**: COMPLETED - inverse floor derived from the train y range, bounding |y_hat| at 1000x max|y| (428a76f75; test_transform_defects_fits.py)

### TRF-02 [P1] `centered_ratio` shifts even a strictly positive base down to about 0 at the train minimum, so a predict base slightly below the train min flips the sign of `y_hat`
- **Where**: `src/mlframe/training/composite/transforms/extended.py:125-139` (`c = -b_min + 0.01 * max(b_scale, 1e-6)`); `_registry_shared_adapters.py:70-94` (`_centered_ratio_domain_fitted` gates only `|base + c| >= eps`).
- **What**: `c` is always set so that `min(base_train + c) = 0.01*median|base|`, whatever the sign of the base. The docstring says it is an "Extension of ratio to signed bases", yet an all-positive base is also moved next to a pole. Repro: `base ~ U(100, 200)`, `y = 3*base + noise`. Fitted `c = -98.51`, `eps = 1.5e-4`. For predict bases `[99.0, 98.5, 97.0, 150.0]`, the fitted domain returns `[True, True, True, True]`. The inverse at the train-median T gives `[4.23, -0.086, -13.04, 444.6]`, while the truth is about `[297, 295.5, 291, 450]`. A base 3% below the train min produces a **negative** prediction. On train, rows near the minimum have denominators of about 1.5, so T is extremely heavy-tailed.
- **Why it matters**: the transform is in the default discovery pool (`_composite_target_discovery_config_base.py:230`). A base drifting a few percent below its train range is routine (prices, lags on a new regime). The soft-shrink OOD guard does not cover `centered_ratio` because it is not additive-in-base, and the fitted-domain gate passes these rows.
- **Suggested fix**: (a) use `c = 0` when `b_min > 0` (then the transform matches `ratio`, which is correct for a positive base). Use a margin proportional to scale (for example `c = -b_min + max(b_scale, iqr)`), not `0.01*scale`, only when the base actually reaches zero or goes negative. (b) Make `_centered_ratio_domain_fitted` require `base + c > eps` (same sign as every train row), not `|base + c| >= eps`, so a sign-flipping row goes to the fallback.
- **Test to add**: fit on `base ~ U(100,200)`. Assert `inverse(median T, base=97)` is positive and within 10% of `3*97`, and assert the fitted domain is False for any predict row with `base + c <= 0`.
- **Disposition**: COMPLETED - a strictly positive base keeps c=0 and the fitted domain requires base+c >= eps, so sign-crossing rows take the fallback (428a76f75; test_transform_defects_fits.py)

### TRF-03 [P1] `monotonic_residual` sends under-populated knots to the global median before the cumulative max/min, which flattens half the spline on small and mid-size train sets
- **Where**: `src/mlframe/training/composite/transforms/nonlinear.py:442-478` (slab loop `if n_in_slab < min_knot_n: knots_y[k] = y_global_med`, then `np.maximum.accumulate` / `np.minimum.accumulate`).
- **What**: knot 0 sits at the base minimum and the last knot at the maximum, so the two edge slabs are only half a quantile wide: about `n / (2*(n_knots-1))` rows. With the defaults (`n_knots=12`, `min_knot_n=30`) the edge slabs fall below 30 rows whenever n is below about 660. The code then sets them to the **global median**. For an increasing relation, the cumulative max then drags every lower knot up to the global median. Repro with `y = base + N(0, 0.3)`, `base ~ U(0, 10)`:
  - n=300: `knots_y = [4.94 x7, 6.44 x3, 9.15 x2]`, `var_explained = 0.573`
  - n=500: `[4.76 x6, 5.26, 6.29, 7.24, 8.30, 9.07, 9.07]`, `var_explained = 0.697`
  - n=800: `[0.28, 1.01, ..., 9.81]`, `var_explained = 0.988`

  `is_degenerate` stays `False` in all three cases, so discovery keeps the crippled spline. The same path is reached through `chain_monres_cbrt`, `chain_monres_yj` and `monotonic_residual_grouped`. In the grouped variant each group is fitted on its own rows (often fewer than 660), so nearly every per-group spline is affected.
- **Why it matters**: the transform is in the default pool. The lower half of g(base) becomes a constant, so T keeps most of the signal and the composite loses its value. The cumulative max also hides the problem from the monotonicity assertions in the existing tests.
- **Suggested fix**: never inject the global median into the ordered knot sequence. Drop under-populated knots (fall back to the remaining knots), or merge an under-populated slab into its neighbour, or cap the threshold at `min(min_knot_n, max(3, n // (2*n_knots)))`. Set the edge knots from the nearest populated slab rather than the global median.
- **Test to add**: `y = base + N(0,0.3)` at n in {300, 500}. Assert `knots_y` is strictly increasing over at least 80% of knots and `var_explained > 0.95`. Repeat for `monotonic_residual_grouped` with 4 groups of 150 rows.
- **Disposition**: COMPLETED - the knot-population threshold scales with n and a sparse knot is interpolated from its populated neighbours (428a76f75; test_transform_defects_fits.py)

### TRF-04 [P1] `frac_diff` inverse amplifies any T-bias about 10x and makes each row's prediction depend on which other rows are in the predict batch
- **Where**: `src/mlframe/training/composite/transforms/_nonlinear_ewma_fracdiff.py:464-484` (`_frac_diff_forward` uses **observed** past y; `_frac_diff_inverse` rebuilds past y recursively from **predicted** T); `_grouped_extra.py:250-270` (same for `frac_diff_grouped`).
- **What**: the forward at train is `T_i = sum_k w_k * y_{i-k}` with the true past y. The inverse at predict is `y_i = (T_hat_i - sum_{k>=1} w_k * y_hat_{i-k}) / w_0`, which feeds its own reconstructions back in. With the defaults `d=0.5, lags=30`, `sum(w) = 0.1026`, so the recursion's steady-state gain is about `1/0.1026 = 9.75`. Repro: fit on 300 rows of a random walk, then shift T_hat by +0.1 on the next 100 rows. The y bias at the end of the batch is `0.919 ... 0.925`. Predicting the same 100 rows one at a time instead of as a batch changes y_hat by up to `14.56`, because every single-row call pads all 30 lags with the train mean anchor. The docstring says "Past y values are unknown at predict". The past y **are** known in the usual setting where lags are features, but the transform has no way to receive them.
- **Why it matters**: a small systematic inner-model bias becomes a 10x y-scale bias, and online (single-row or micro-batch) serving gives different predictions from batch scoring of the same rows. There is no warning, and `recurrence_continuation` only fixes the seed of the first row after train.
- **Suggested fix**: take the observed y history as the inverse's lag source. Route an observed-lag base (the y lag columns) and use it in place of the `y_hat_{i-k}` recursion. Document the gain `1/sum(w)` and log it at fit. At minimum, emit a one-time warning when `inverse` is called on a batch shorter than `lags`, and state the batch dependence in the registry description.
- **Test to add**: (1) a bias-amplification pin: shift T_hat by delta and assert the y shift stays within `1.1*delta` once the observed-lag inverse exists (currently about 9.75*delta). (2) Batch-invariance: assert that row-by-row inverse equals batch inverse on a held-out continuation.
- **Disposition**: COMPLETED - observed history_y seeds the lag terms, so per-row serving has gain 1; the fit logs the amplification (168e79ee8; test_transform_defects_recurrent.py)

### TRF-05 [P2] Predictions from the recurrent family (`ewma_residual`, `volatility_normalized_residual`, `rolling_quantile_ratio*`, and their `_grouped` variants) depend on the composition of the predict batch
- **Where**: `_nonlinear_ewma_fracdiff.py:358-371` (EWMA re-seeded at `anchor` on every call); `_volatility.py:284-302`; `simple.py:420-450` (rolling median over the predict batch only); `_grouped_extra.py:90-183`.
- **What**: each forward/inverse call restarts the recurrence at the start of the batch it receives. Repro (`ewma_residual`, k=7, random-walk base): inverting 100 consecutive rows as one batch versus one row at a time differs by up to `11.94` in y. With a single row, `EWMA = (1-a)*anchor + a*base` where `anchor = mean(base_train)`, while at train each row's EWMA had its full history. `rolling_quantile_ratio` in trailing mode on a 1-row batch returns `base` itself as the "rolling median". The registry descriptions only say "Caller is responsible for chronological row order". They do not say that the **batch boundary** is a state reset, or that val/test/prod batches must be contiguous continuations.
- **Why it matters**: train/serve skew in online serving and in any caller that splits a `predict` into chunks (micro-batch streaming, CV folds scored per fold). The inner model learned T relative to a warmed-up EWMA, and inverting it against a cold-started EWMA adds `(1-a)^i * (anchor - true_state)` to the first rows of every batch.
- **Suggested fix**: accept an optional warm-up prefix (the last `k`/`lags` observed base rows) through fit params or a predict kwarg and seed from it. Always store `tail_anchor` (already done) plus a short raw-base tail buffer. Warn once per estimator when a predict batch is shorter than the recurrence horizon and continuation is off.
- **Test to add**: for each recurrent transform, assert that batch inverse equals the concatenation of chunked inverses when a warm-up prefix is supplied. Add a pin that documents the current single-row skew until the fix lands.
- **Disposition**: COMPLETED - every recurrent forward/inverse warms up on history_base/history_y, so a chunked predict equals the one-batch predict (168e79ee8; test_transform_defects_recurrent.py)

### TRF-06 [P2] `quantile_normal_y` and `gaussian_copula_residual` clip the ECDF with an epsilon from the knot count, not the sample size, so their round-trip is lossy on tails (continuous y) and on the extreme values (tied/discrete y)
- **Where**: `unary.py:488-499` (`eps = 1.0 / (2.0 * len(knots_q))` with `n_knots = min(1000, n)`); `_gaussian_copula.py:241-246` (`eps = 1.0 / (2.0 * max(len(cdf), 2))` with `len(cdf)` = number of **unique** values).
- **What**:
  - `quantile_normal_y`: the knots carry `q in [0.5/n, (n-0.5)/n]`, but q is clipped to `[1/2000, 1-1/2000]` once n exceeds 1000. The bottom and top 0.05% of train rows collapse onto T = ±3.2905. Repro with n=100k standard normal: 0.014% of rows round-trip with error > 1e-3, and the max round-trip error is **0.80** (in sigma units).
  - `gaussian_copula_residual`: for a discrete target, `len(cdf)` is the number of distinct values. Repro with binary y in {0,1}: eps = 0.25, the top knot `u = (n-0.5)/n` is clipped to 0.75, and every y=1 row round-trips to **0.5229** (max error 0.477 at the exact T).
- **Why it matters**: `quantile_normal_y` is in the default pool and in `chain_linres_cbrt_qn`. It exists for heavy-tailed targets, yet it loses exactly the tail rows. `gaussian_copula_residual` silently mispredicts every top-category row of a count or rating target, even when the inner model is perfect. The contract test tolerates both (`rtol 0.5`, median error).
- **Suggested fix**: set eps from the smallest/largest stored plotting position: `eps = min(knots_q[0], 1 - knots_q[-1])` for quantile_normal, and `eps = min(cdf[0], 1 - cdf[-1])` for the copula, so the extreme knots stay invertible. For discrete y in the copula, place the knot at the tie's mid-rank rather than the last occurrence, so `u < 1`.
- **Test to add**: max-error round-trip on n=100k normal (assert < 1e-6 on all rows) and on binary / 5-level integer y for `gaussian_copula_residual` (assert exact recovery of every level).
- **Disposition**: COMPLETED - the ECDF clip is the tail mass beyond the extreme knots instead of an eps from the knot count (428a76f75; test_transform_defects_fits.py)

### TRF-07 [P2] `linear_residual` (and every transform built on `_linear_residual_fit`) returns a minimum-norm, non-zero alpha on a zero-variance base, which extrapolates wrongly on any different base value
- **Where**: `linear.py:92-132` (`_linear_residual_fit`, a plain `lstsq` on `[base, 1]` with no zero-variance guard). The fit is reused by `linear_residual_robust`, `chain_linres_*` and the per-group fits in `linear.py:656`. The docstring claim at `linear.py:142-145` ("Degenerate folds are guarded EXACTLY as `_linear_residual_fit`'s scalar path ... a zero-variance base (`den == 0`) returns `(0.0, mean(y))`") is false. `_causal_anchor.py:194-195` repeats the claim.
- **What**: with a constant base, the design matrix has rank 1 and `lstsq` returns the minimum-norm split of the level between alpha and beta. Repro: `y ~ N(100,1)`, `base = 5.0` (constant), fitted `alpha = 19.23, beta = 3.85`. At predict with `base = 10` and `T_hat = 0`, `y_hat = 196.2`, while mean(y) is 100.0. The closed-form sibling `_linear_residual_fit_closed` correctly returns `(0, mean y)`.
- **Why it matters**: in `linear_residual_grouped`, a base that is constant **within a group** (a store-level attribute, a group-level lag) is common. That group's alpha is then an artifact. It also enters the James-Stein spread statistics with `Var(base_g)=0`, which is floored to 1e-12 in `_james_stein_shrinkage_factor` and blows up that group's noise proxy. The same happens for a globally constant base in a short train fold.
- **Suggested fix**: add the same `den == 0` / `np.ptp(base) == 0` guard as `_linear_residual_fit_closed` (return `alpha=0, beta=mean(y)` or the weighted mean). Exclude zero-variance groups from `alphas_for_shrink`.
- **Test to add**: constant-base fit returns `alpha == 0` and `beta == mean(y)`, and inverse at a different base equals mean(y). Add a grouped variant with one constant-base group, asserting that group's alpha falls back to the global one.
- **Disposition**: COMPLETED - a zero-variance base returns (0, mean y); a constant-base group keeps the global slope and its own level (428a76f75; test_transform_defects_fits.py)

### TRF-08 [P2] The `logratio` soft-cap floor is `1e-3 * std(y)` in raw y units while the cap applies to log-scale T, so the cap switches off for large-scale targets and is scale-dependent
- **Where**: `linear.py:44-70` (`mad_floor = _MAD_FLOOR_FRAC * std_y`; `mad_eff = max(mad_train, mad_floor)`); constant rationale at `transforms/__init__.py:18-27`.
- **What**: `T = log(y) - log(base)` has no units, but the floor has units of y. Repro: y and base lognormal around e^12 (about 1.6e5), true `mad_train = 0.068`. The fitted `mad_eff = 97.3`, so the cap is `k*mad_eff = 973` **log units** and never binds: `exp(973)` overflows. The same data divided by 1e6 gives a cap of `0.68`. Rescaling y changes the inverse, which contradicts the "y-scale invariant" contract that other tests (`test_composite_y_scale_invariant.py`) pin for the wrapper.
- **Why it matters**: `logratio` is in the default pool. For money- or count-scale targets, the soft cap described as trading "exact round-trip on extreme in-domain train rows for bounded inverse blow-up" does nothing. Only the wrapper T-clip and y-clip remain.
- **Suggested fix**: floor on a unitless quantity, for example `mad_floor = max(_MAD_FLOOR_FRAC * std(t_train), 1e-6)` or an absolute log-scale floor such as 1e-3. Keep `mad_train` as is.
- **Test to add**: fit `logratio` on the same (y, base) at scales 1e-3, 1, 1e6 and assert that `mad_eff` and the capped inverse are identical up to the multiplicative scale.
- **Disposition**: COMPLETED - the soft-cap MAD floor is unitless instead of 1e-3*std(y) in raw y units (428a76f75; test_transform_defects_fits.py)

### TRF-09 [P2] `polynomial_residual_deg2` solves raw (uncentred) normal equations, so the quadratic fit breaks down once the base is offset from zero; the comment claims centring that the code does not do
- **Where**: `extended.py:175-200` (comment at `:186-187` says "with column-mean removal for numerical stability", but the code builds `[1, b, b^2]` raw and solves `X.T @ X`).
- **What**: normal equations square the condition number of `[1, b, b^2]`. With `b ~ N(c, s)` the columns are nearly collinear once `c/s` is large. Repro: `y = 0.5*x^2 + 2*x + N(0, 0.01)` with `x = (b - c)/s`. Residual std should be about 0.01. Measured `std(T)`: `c=1e4,s=1 -> 0.596`, `c=1e5,s=1 -> 0.718`, `c=1e6,s=10 -> 0.733`, against `std(y) ~ 2.11`. About a third of the signal stays in T. (At `c=1e3` it still works: 0.0099.) Also, the `< 10 finite rows` fallback returns `alpha1 = 1.0`, i.e. `T = y - base` (`diff`), while every sibling fallback uses zero slope.
- **Why it matters**: the transform is in the default pool. Offset bases (years, prices, sensor levels, timestamps) are the normal case, and the failure is silent: finite T, just badly fitted.
- **Suggested fix**: centre and scale the base (`z = (b - mean)/std`) before building `[1, z, z^2]`, solve with `lstsq` (or QR), then map the coefficients back to raw-base `(alpha1, alpha2, beta)`. Alternatively store the centring constants in params and evaluate in z space. Make the small-n fallback `alpha1=0, beta=mean(y)`.
- **Test to add**: the repro above at `c in {1e4, 1e6}`, asserting `std(T) < 0.05`.
- **Disposition**: COMPLETED - least squares on the centred/scaled (1, z, z^2) design, evaluated in z space; raw alphas kept for provenance (428a76f75; test_transform_defects_fits.py)

### TRF-10 [P2] Grouped and categorical transforms crash with `TypeError` on a group column holding missing values (`None`/`NaN` mixed with strings)
- **Where**: `np.unique(groups, ...)` at `linear.py:600`, `nonlinear.py:171` (`_row_alpha_beta`), `_grouped_extra.py:30` (`_group_segments`), `categorical.py:389` and `categorical.py:427`.
- **What**: `np.unique` sorts, so an object array mixing `str` with `None` or `float('nan')` raises. Repro: `target_encoding_residual.fit(y, None, groups=np.array(["a", None, "b", "a"], dtype=object))` gives `TypeError: '<' not supported between instances of 'NoneType' and 'str'`, and a NaN-for-missing pandas object column gives `'<' not supported between instances of 'float' and 'str'`. `_extract_groups` (`estimator/__init__.py:98-115`) passes the column through unchanged, and `_canonical_group_key` already maps NaN to `'nan'`, but it is only applied **after** the failing `np.unique`.
- **Why it matters**: nullable categorical columns are routine. The whole grouped family (`linear_residual_grouped`, `*_grouped`, `target_encoding_residual`) fails at fit or predict instead of treating missing as a level or routing it to the global fallback. A predict-time NaN group in an otherwise clean column fails the whole predict call.
- **Suggested fix**: factorise through `_canonical_group_key` first. For example use `pd.factorize(pd.Series(groups).map(_canonical_group_key))` or canonicalise to a `str` array before `np.unique`, so that missing becomes the `'nan'`/`'None'` level. Put this in one shared helper used by all five sites.
- **Test to add**: fit + forward + inverse of `linear_residual_grouped`, `quantile_residual_grouped` and `target_encoding_residual` with groups `["a", None, "b", np.nan, "a"]`, asserting finite output and that unseen/missing groups fall back to the global parameters.
- **Disposition**: COMPLETED - grouped and categorical transforms accept a group column mixing strings with None/NaN (428a76f75; test_transform_defects_extended.py)

### TRF-11 [P2] `seasonal_residual` period selection maximises in-sample fit with no complexity penalty: the largest candidate period always wins on noise and nested multiples beat the true period
- **Where**: `_seasonal.py:70-82` (argmin of in-sample residual variance over `(4, 5, 7, 12, 24, 52)`).
- **What**: a larger period has more phase means, and 24 and 52 **nest** 4 and 12, so their in-sample variance is never higher. Repro: pure noise with n=400 selects period **52 in 20/20 seeds**. A clean period-12 series with n=480 selects **24**. The fitted 52 phase means on noise have only about 7.7 rows each, so the inverse adds noise with variance about `sigma^2/7.7` to every prediction.
- **Why it matters**: the selected period is overfit. The transform subtracts a noise pattern at train and adds it back at predict, which inflates variance. A spurious "seasonality" is also recorded in the spec and the provenance formula.
- **Suggested fix**: select by held-out variance (for example fit phase means on even cycles and score on odd ones), or by a penalised criterion (AIC/BIC with `period` parameters). Prefer the smallest period within 1 SE, and let period 1 (no seasonality) compete as a candidate.
- **Test to add**: noise input must select period 1 (or report no seasonality). Period-12 data must select 12, not 24.
- **Disposition**: COMPLETED - period selection is held-out error over alternating cycles with period 1 a candidate, taking the smallest within one SE (168e79ee8; test_transform_defects_recurrent.py)

### TRF-12 [P2] At predict, `seasonal_residual` assumes the batch starts at phase 0 and there is no continuation offset, so a chronological test split usually gets wrong phase means
- **Where**: `_seasonal.py:85-104` (`phase = np.arange(t_f.size) % period` in forward and inverse); module docstring `_seasonal.py:7-9`.
- **What**: training phase 0 is the first train row. A test batch that continues the series begins at true phase `n_train % period`, but the inverse uses phase 0. Unlike EWMA/frac-diff (`tail_anchor` under `recurrence_continuation`), no `phase_offset` is stored. Example: daily data with period 7 and 1000 train rows (1000 % 7 = 6). Every test row gets the phase mean from 6 days away, a constant error of `means[true] - means[shifted]` on every prediction.
- **Why it matters**: a chronological train/val/test split is the main use case for a seasonal transform. The assumption is documented, but nothing enforces it or offers an alternative. The error is systematic and silent.
- **Suggested fix**: store `n_train` (or `phase_offset = n_rows_seen % period`) in params and apply it when `recurrence_continuation` is on. Better, accept an explicit `phase`/`row_index` column at predict (the fit side already takes `row_index`).
- **Test to add**: fit on 1000 rows of a clean period-7 series, predict the next 70 with `recurrence_continuation=True` and T_hat = 0, and assert exact recovery.
- **Disposition**: COMPLETED - forward/inverse take absolute row_index positions and continuation starts at the phase following the train series (168e79ee8; test_transform_defects_recurrent.py)

### TRF-13 [P2] `second_diff` with a 1-D base becomes `T = y - 2*b1`, which keeps the full (negated) level; the docstring calls this "still a valid, if weaker, detrend"
- **Where**: `_second_diff.py:157-168` (`b2 = zeros` when only one column is given); claim at `_second_diff.py:137-139` and in the registry description `_registry_extended.py:362`.
- **What**: with `b1 ~ y`, `T = y - 2*b1 ~ -y`. Repro: `y = linspace(100,200,50)`, `b1 = y - 2`, gives `T[:3] = [-96.0, -98.04, -100.08]`, while `diff` would give `[2, 2, 2]`. Nothing is detrended: T carries the full level with the sign flipped.
- **Why it matters**: a user who wires only `base_column` (forgetting `extra_base_columns=[lag2]`) silently gets a target that is harder than raw y, and the docstring tells them it is fine.
- **Suggested fix**: raise `ValueError` when fewer than two base columns reach fit, or degrade to `diff` (`T = y - b1`) with a warning. Correct the docstring either way.
- **Test to add**: calling `second_diff` fit with a 1-D base raises, or its T equals `y - b1`.
- **Disposition**: COMPLETED - a single-column fit records single_lag_as_diff so T = y - b1 and warns about the missing lag-2 wiring; old params keep their algebra (84c125ddf; test_biz_val_second_diff.py)

### TRF-14 [P3] `nadaraya_watson_residual` sets the bandwidth from the full n but only averages over at most 2000 single-observation knots, so accuracy stops improving with more data
- **Where**: `_nadaraya_watson.py:165-201` (`h = 0.9*spread*n**(-0.2)` with the full n; knots = 2000 raw `(base_j, y_j)` pairs).
- **What**: each knot is one noisy observation, not a local average, so g's variance is set by m=2000 while h keeps shrinking with n. Repro: `y = sin(6b) + N(0,1)`. At n=2000, h=0.056 and RMSE(g - truth) = 0.085. At n=200000, h=0.023 and RMSE = 0.076, so 100x more data barely helps. The docstring says the 2000 knots "reproduce g to well below the kernel smoothing error".
- **Why it matters**: the transform wastes most of the training data. It is not in the default pool.
- **Suggested fix**: set each knot's y to the mean of y in its rank bucket (and optionally weight by bucket count). Compute Silverman's h from the effective knot count, or keep h from n but average y within buckets.
- **Test to add**: RMSE(g - truth) at n=200k is at least 2x lower than at n=2k.
- **Disposition**: COMPLETED - count-weighted rank-bucket means replace the 2000 single-observation knots, so accuracy improves with n (428a76f75; test_transform_defects_extended.py)

### TRF-15 [P3] `monotonic_residual_grouped` shrinks per-group medians toward the global **mean** (`y_train_mean`), so skewed targets get a one-sided level shift instead of a shrink toward the global fit
- **Where**: `_grouped_extra.py:470-475` (`global_median_key="y_train_mean"`) combined with `_grouped_extra.py:288-321` (per-group level = `np.median(seg)`, offset = `c*(global_center - med_g)`); `nonlinear.py:499` (`y_train_mean` = `np.mean(y_clean)`).
- **What**: the deviations `global_mean - median_g` are all the same sign for a right-skewed y. The JS spread therefore includes the mean-median gap, and every eligible group's knots move **up** by `c*(mean - median_g)`, even a group identical to the global population. `quantile_residual_grouped` uses `global_median` and does not have this problem. (Found by reading the code; I did not run a numeric repro.)
- **Why it matters**: T gets a systematic per-group offset that the inner model has to absorb, and the JS factor is mis-estimated. The round-trip stays exact, so the output is not wrong, only miscentred.
- **Suggested fix**: centre on the median of the global y (store `y_train_median` in the monotonic params and use it as `global_median_key`).
- **Test to add**: lognormal y, 5 groups drawn from the same distribution. Assert per-group level offsets average to about 0 (currently all positive).
- **Disposition**: COMPLETED - the grouped monotonic fit shrinks toward the global MEDIAN (`_grouped_extra.py:573` global_median_key="y_train_median", stored by `nonlinear.py:428`)

### TRF-16 [P3] The `rank_ecdf_residual` / `gaussian_copula_residual` constant-column ECDF adds a synthetic knot at `value + 1.0` in raw units, so a tiny T error inverts to `y + 1`
- **Where**: `_rank_ecdf.py:68-72` (`knots = [v, v + 1.0]`, `u = [0.5, 0.5 + 1e-9]`).
- **What**: for a constant y, any recovered `u` above `0.5 + 1e-9` maps to `v + 1`. Repro: y = 7.0 everywhere, `inverse(T + 1e-6)` returns `8.0`. The `+1.0` is an arbitrary absolute unit (for y around 1e-6 it is a 1e6x error; for y around 1e9 it is invisible).
- **Why it matters**: this is a degenerate fold/segment edge case, but the output is a value that never occurred in train. It reaches production through a constant-target segment.
- **Suggested fix**: for a constant column, make the inverse return `v` for every u (store a `constant` flag), or use a scale-relative span `max(|v|, 1) * 1e-9`.
- **Test to add**: constant-y fit, and inverse of `T + small noise` equals the constant.
- **Disposition**: COMPLETED - the constant-column ramp knot is scale-relative instead of a raw +1.0 (428a76f75; test_transform_defects_extended.py)

### TRF-17 [P3] The `smoothing_spline_residual` registry description gives the wrong smoothing formula, and spline build/eval failures are swallowed at `debug` level
- **Where**: `_registry_extended.py:274` ("smoothing factor s = n_unique * std(y) * 1.0") compared with `extended.py:361-366` (Rice noise-variance estimator: `s = n_unique * 0.5*mean(diff(yc_avg)^2)`); `extended.py:383-392` (`except Exception` -> `logger.debug` -> constant `y_mean`).
- **What**: the description is out of date. Separately, any `UnivariateSpline` failure silently turns the transform into `T = y - mean(y)` with only a debug-level log. Fit and predict both rebuild the spline, so a failure that depends on the data seen at predict (it rebuilds from the stored knots, so it should be deterministic) would not be visible to the user.
- **Why it matters**: the documentation is inaccurate, and the transform can degrade silently (see the "silent error swallow" bug class).
- **Suggested fix**: update the description. Raise the swallow to `warning` with `log_throttle`, and store a `degenerate=True` flag in params so that discovery rejects the spec.
- **Test to add**: monkeypatch `UnivariateSpline` to raise, and assert a warning is logged and `degenerate` is set.
- **Disposition**: COMPLETED - spline build failures log at WARNING and set is_degenerate; the stale description is corrected (428a76f75; test_transform_defects_extended.py)

### TRF-18 [P3] `quantile_residual` per-bin IQR floor is an absolute `1e-6`, not scale-relative, so heteroscedastic scaling is silently disabled for small-scale targets
- **Where**: `nonlinear.py:198` (`bin_iqrs[b] = bin_iqr if bin_iqr > 1e-6 else global_iqr`) and `nonlinear.py:225` (`np.where(raw_iqr > 1e-6, raw_iqr, global_iqr)`), in contrast with the relative threshold in `_robust_y_scale` (`nonlinear.py:246-265`, `1e-9 * max(max|y|, 1)`).
- **What**: for y on a 1e-7 scale (probabilities of rare events, rates), every bin IQR is below 1e-6 and is replaced with the global IQR. The transform then reduces to a per-bin median shift. For y on a 1e9 scale, an effectively constant bin (IQR about 1e-3) passes the floor and inflates T by 1e3.
- **Why it matters**: the output depends on the units of y; the transform is in the default pool.
- **Suggested fix**: use the same relative threshold as `_robust_y_scale` (`raw_iqr > 1e-9 * ref`, or a fraction of `global_iqr`).
- **Test to add**: fit on `y*1e-8` and on `y`, and assert `bin_iqrs` scale by exactly 1e-8.
- **Disposition**: COMPLETED - the per-bin IQR floor is relative to the global IQR instead of an absolute 1e-6 (428a76f75; test_transform_defects_fits.py)

### TRF-19 [P3] `signed_power_y` never tries `p = 1` (identity), so an already-symmetric target is always compressed
- **Where**: `unary.py:147` (`_SIGNED_POWER_GRID = np.linspace(0.1, 0.9, 17)`), `unary.py:168-180` (`best_p = 1.0` is only an initial value; `best_skew = inf`, so any finite grid point replaces it).
- **What**: the docstring says "p=1 is identity" and presents p<1 as the compression choice, but a target that is already symmetric (skew about 0) still gets the grid point with the smallest |skew|, typically 0.9 or lower, and never identity. The inverse `|T|^(1/p)` then amplifies T_hat error with no benefit.
- **Why it matters**: discovery applies an unneeded nonlinear transform to already-symmetric targets.
- **Suggested fix**: add `1.0` to the grid (or evaluate identity first and keep it unless a grid point reduces |skew| by a margin).
- **Test to add**: y ~ N(0,1) must fit `p == 1.0`.
- **Disposition**: COMPLETED - p = 1 (identity) is in the search, so a symmetric target is no longer compressed (428a76f75; test_transform_defects_fits.py)

### TRF-20 [P3] `geometric_mean_residual` domain requires `y > 0` even though `T = y / geomean(bases)` is defined for any finite y
- **Where**: `extended.py:516-527`.
- **What**: the docstring's justification ("for a subsequent log/ratio-style downstream use") does not apply to this transform, whose forward and inverse are a plain division and multiplication. Rows with `y <= 0` are dropped at fit (or raise `DomainViolationError` when `drop_invalid_rows=False`). `ratio`, the single-base sibling, has no such restriction.
- **Why it matters**: training rows are discarded for no reason; a zero-inflated target loses all of its zero rows.
- **Suggested fix**: gate only on the bases (finite and > 0) plus finite y.
- **Test to add**: y containing zeros and negatives round-trips exactly with positive bases.
- **Disposition**: COMPLETED - y <= 0 rows are kept: T = y / geomean is defined for any finite y (428a76f75; test_transform_defects_extended.py)

### TRF-21 [P3] Chain transforms silently drop `sample_weight`, while their standalone bivariate half honours it
- **Where**: `nonlinear.py:563-570` and `nonlinear.py:609-616` (`_fit(y, base)` has no `sample_weight` parameter). The wrapper gates `sample_weight` on the signature (`estimator/_estimator.py:561`, `_callable_accepts_param(transform.fit, "sample_weight")`).
- **What**: `linear_residual` fits weighted OLS when given weights. `chain_linres_cbrt` / `chain_linres_yj` / `chain_linres_cbrt_qn` fit the same OLS **unweighted**, with no log, because the gate sees no parameter.
- **Why it matters**: when a user passes weights, the chain specs are fitted differently from their parent `linear_residual`, so discovery compares a weighted fit against an unweighted one.
- **Suggested fix**: add `sample_weight=None` to the chain `_fit` closures and forward it to `bivariate_fit` when that function accepts it.
- **Test to add**: fit `chain_linres_cbrt` with weights that zero half the rows, and assert its `bivariate_params` equal `linear_residual` fitted on the kept half.
- **Disposition**: COMPLETED - call_transform forwards sample_weight exactly when the target accepts it, so chains honour it (428a76f75; test_transform_defects_extended.py)

### TRF-22 [P3] `_grouped_extra` stores the global `anchor` as `tail_anchor`, so unseen groups under `recurrence_continuation` get the mean seed, unlike the ungrouped transforms
- **Where**: `_grouped_extra.py:63` (`tail_anchor = anchor`, never updated) and `_grouped_extra.py:223` (`"tail_anchor": anchor` in `frac_diff_grouped`).
- **What**: `_ewma_residual_fit` / `_frac_diff_fit` compute a real train-tail seed. The grouped variants label the global mean as `tail_anchor`, so the fallback for an unseen group under continuation is inconsistent with the ungrouped contract and the parameter name is misleading.
- **Why it matters**: minor inconsistency; a new entity in a streaming panel is seeded from the whole-history mean.
- **Suggested fix**: compute the global trace tail as the ungrouped fit does, or rename the key to make clear it is the mean.
- **Test to add**: a continuation predict for an unseen group equals the ungrouped continuation seed.
- **Disposition**: COMPLETED - the grouped seed is the ungrouped train-tail state: `_grouped_extra.py:90-94` computes the global EWMA trace tail instead of reusing the mean anchor

### TRF-23 [P3] `rank_residual` stores the full sorted train `y` and `base` arrays in params
- **Where**: `extended.py:251-274` (`"y_sorted": y_sorted, "b_sorted": b_sorted`, length n each).
- **What**: params grow as O(n): 10M rows means 160 MB in `fitted_params_`, pickled into every saved model and spec. The inverse is a nearest-bucket lookup, so about 1-10k quantile knots would give the same precision up to the bucket width. (`quantile_normal_y` caps its knots at 1000; `nadaraya_watson` caps at 2000.)
- **Why it matters**: model size and serialisation time grow with the training set; `rank_residual` is in the default pool.
- **Suggested fix**: store at most `_RANK_MAX_KNOTS` quantiles and interpolate the rank lookup, as `_ecdf_knots` does.
- **Test to add**: fit on n=1M and assert the size of the serialised params is bounded (under 1 MB).
- **Disposition**: COMPLETED - bounded knot tables replace the two full sorted train arrays (428a76f75; test_transform_defects_extended.py)

### TRF-24 [P3] `generate_interaction_bases` defaults `train_mask=None` (the divisor epsilon leaks test-set scale) and silently ignores a mask of the wrong length
- **Where**: `interaction_bases.py:32`, `interaction_bases.py:81-86` (`if train_mask is not None and train_mask.shape == b.shape:` otherwise falls back to the whole-array median).
- **What**: the default path computes `median|b|` over all rows, including test rows. The docstring admits this ("leaks test-set scale into the eps"). A mis-shaped mask is dropped without warning, so a caller who meant to prevent the leak still gets it.
- **Why it matters**: a small train/test leak that is on by default, and a guard that fails silently when misused.
- **Suggested fix**: raise `ValueError` on a shape mismatch. Emit a one-time warning when `train_mask is None` and any `div` op is requested.
- **Test to add**: a mis-shaped mask raises; the eps with a train mask equals the median over train rows only.
- **Disposition**: COMPLETED - a mis-shaped mask raises ValueError and a div without train_mask warns that the eps floor sees every row (1f8f92837; test_composite_interaction_bases.py::TestTrainMaskScale)

### TRF-25 [P3] The Yeo-Johnson fit failure log says "Box-Cox", and `_registry_extended.py` rebuilds all six unary adapters but uses only one
- **Where**: `unary.py:384` (`logger.debug("Box-Cox lambda MLE optimization failed, ...")` inside `yeo_johnson_y_fit`); `_registry_extended.py:172-201` (builds the `_cbrt_*`, `_log_*`, `_yj_*`, `_qn_*`, `_sp_*`, `_bc_*` adapter tuples; only `_bc_*` is referenced in `_TRANSFORMS_REGISTRY_EXTENDED`; `registry.py:123-152` already builds the other five).
- **What**: a mislabelled diagnostic (a YJ failure reads as a Box-Cox failure), plus duplicate adapter construction that can drift from the copies in `registry.py`. The fit failure also falls back to `lambda=1.0` with only a debug log.
- **Why it matters**: debugging friction and dead code.
- **Suggested fix**: fix the message (and raise it to `warning` via `log_throttle`); build only the Box-Cox adapter in `_registry_extended.py`, or import the others from a single shared place.
- **Test to add**: force `minimize_scalar` to raise and assert the logged message names Yeo-Johnson.
- **Disposition**: COMPLETED - the Yeo-Johnson failure log names Yeo-Johnson and _registry_extended no longer builds the five unregistered adapters (428a76f75; test_transform_defects_extended.py)

### TRF-26 [P3] `target_encoding_residual` fits category means in-sample: each train row's own y is included in its encoding
- **Where**: `categorical.py:336-404` (`smoothed = (sums + a*global_mean)/(counts + a)` over all train rows); the leak is acknowledged in the module docstring at `categorical.py:305-314`.
- **What**: for a category with `count` rows, a fraction `1/(count + a)` of each row's own y is folded into its encoding, so train T is shrunk toward 0 compared with predict-time T for the same category. With the default `a = 20`, a singleton's T is `20/21*(y - global_mean)` at train, while an unseen row of that category gets the full deviation. The docstring leaves OOF encoding to the caller, but no OOF option exists in the transform or its fit kwargs.
- **Why it matters**: a small, documented leak on high-cardinality columns; the inner model is trained on slightly too-optimistic residuals.
- **Suggested fix**: add an opt-in (or, per the "corrective mechanism default ON" convention, default) `oof_folds` fit kwarg that computes each train row's T from out-of-fold encodings, while keeping full-train encodings for predict.
- **Test to add**: with `oof_folds=5`, no train row's T uses its own y (verify on singleton categories: T equals `y - global-mean-based encoding`).
- **Disposition**: COMPLETED - train T is out-of-fold by default (predict keeps the full encoding), declared as oof_train_forward on the registry (428a76f75, 61c3bae14; test_composite_target_encoding_residual.py)
