# Composite Targets audit - Transform math correctness (2026-06-10)

Scope: src/mlframe/training/composite/transforms/{simple,linear,nonlinear,unary,extended,interaction_bases,naming,registry,__init__}.py
Method: full read of all 9 modules + call-site verification in estimator/_estimator.py, estimator/_predict.py, discovery/_eval.py, discovery/_fit.py, _composite_target_discovery_config.py. Every numerical claim below was VERIFIED BY RUNNING THE ACTUAL CODE (Python 3.14 store build, numpy 2.3.5, numba present, bottleneck absent - bottleneck path emulated faithfully). Known-clean items from tests/composite_discovery_audit_notes.md (Welford, MI LCB, benign-data round-trips) were not re-reported.

---

## T1 - P1 / bug - Yeo-Johnson inverse silently produces NaN outside the lambda-dependent asymptote

File: src/mlframe/training/composite/transforms/unary.py:200-213 (_yj_inverse_numpy), also :72-93 (_yj_inverse_numba_kernel), :174-181 (_yj_inverse_scalar).

The closed-form YJ inverse has a bounded domain: for lambda < 0 the positive branch needs t*lam + 1 > 0 (t < 1/|lam|); for lambda > 2 the negative branch needs 1 + t*(lam-2) > 0 (t > -1/(lam-2)). Fitted lambda ranges over [-2, 4]. No clamp exists, so np.power(negative, fractional) -> NaN.

VERIFIED: left-skewed lognormal target fits lambda = 2.1994; T_train in [-3.98, -0.035]; asymptote at t = -5.014; yeo_johnson_y_inverse([-5.11, -6.01]) -> [nan, nan]. Also _yj_inverse_numpy([-0.6], 4.0) -> [nan].

Why the wrapper does not save it: the predict-time T-clip (_estimator.py:614-619) is median(T) +/- 10*MAD EXPANDED to the observed T min/max - that envelope can cross the asymptote (here t_med - 10*MAD < -5.014 is easily reachable); the t_mad == 0 branch disables the clip entirely; and the y-clip cannot fix it because np.clip(nan) = nan (verified). Screening (_screening_tiny.py:662) calls transform.inverse on raw tiny-model t_hat with no T-clip at all. yeo_johnson_y, chain_linres_yj, chain_monres_yj are all in the DEFAULT discovery transforms list.

Fix: inside _yj_inverse_*, clamp the power base to a tiny positive floor: pos_base = np.maximum(t*lam + 1.0, 1e-12) and neg_base = np.maximum(-t*(2.0-lam) + 1.0, 1e-12) (equivalently clip t to the lambda-valid interval minus a margin, mirroring logratio soft-cap philosophy). Add a regression test for lambda < 0 and lambda > 2 inverse on out-of-envelope t_hat.

---

## T2 - P1 / bug - linear_residual_multi condition-number gate is scale-variant: independent mixed-unit bases are falsely declared collinear and the fit silently degenerates to y-mean

File: src/mlframe/training/composite/transforms/linear.py:306-327.

The Belsley/Kuh/Welsch kappa < 30 rule applies to a COLUMN-EQUILIBRATED (unit-norm) design matrix. The code computes the SVD condition number of the centered-but-UNSCALED base columns, so two perfectly independent bases of different units trip the gate purely from scale disparity.

VERIFIED: b1 ~ N(0,1), b2 ~ N(0,1e6) independent, y = 2*b1 + 3e-6*b2 + noise -> condition_number = 1.004e6, collinear_fallback = True, alphas = [0.0, 0.0] - the transform degenerates to T = y - mean(y) although the joint OLS is perfectly well-posed. No caller anywhere checks collinear_fallback (grepped repo), and the default-ON multi-base auto-promotion (discovery/_fit.py:738) calls this fit directly, so mixed-unit kept-base sets silently lose the upgrade they were promoted for.

Fix: normalise the centered columns to unit L2 norm before the SVD (base_centered / col_norms); kappa of that matrix is the BKW scaled condition index and is invariant to units. Add a biz_value test: independent mixed-scale bases must NOT fall back, near-duplicate columns must.

---

## T3 - P1 / bug - smoothing_spline_residual smoothing factor has wrong units (std instead of variance, signal instead of noise) -> massive oversmoothing

File: src/mlframe/training/composite/transforms/extended.py:359 (s = max(unique_b.size, 1) * float(np.std(yc_avg)) * _SPLINE_DEFAULT_S_MULT).

scipy UnivariateSpline(s=...) targets the residual SUM OF SQUARES; with unit weights the statistically correct scale is s ~= m * sigma2_noise. The code uses m * std(signal) - wrong units (std vs variance) AND the wrong quantity (signal spread instead of noise level).

VERIFIED: y = 5*sin(b) + N(0, 0.1), n = 3000: fitted s = 9990 (correct scale ~30), residual std after the transform = 1.825 vs the 0.1 noise floor; the spline absorbs only 70% of variance when ~99.96% is absorbable. At amp 0.2 / noise 0.05: residual std 0.123 vs 0.05, only 25% absorbed. The transform materially fails its registry description on exactly its target use case, and discovery comparisons against it are systematically pessimistic.

Fix: estimate noise variance from the data, e.g. Rice first-difference estimator on (b-sorted) y: sigma2 = sum((y[i+1]-y[i])**2) / (2*(m-1)), then s = m * sigma2. Also pass w = sqrt(count_per_unique_b) to UnivariateSpline since yc_avg are duplicate-averaged points with heteroscedastic effective noise. Pin a biz_value test (sine target, residual std <= 2x noise floor).

---

## T4 - P2 / bug - ratio forward/inverse eps asymmetry: in-domain rows with 0 < |base| < eps break round-trip with unbounded relative error

File: src/mlframe/training/composite/transforms/simple.py:311-318 (_ratio_forward floors with safe_base, _ratio_inverse multiplies by RAW base), domain at :321-325 admits any |base| > 0.

_ratio_domain accepts |base| > 0, so a row with base = 1e-9 is "valid"; forward computes T = y / eps (eps ~ 1e-6 * median|base|), inverse computes T * base = y * base/eps.

VERIFIED: base[0] = 1e-9, y[0] = 1.0, fitted eps = 9.27e-7 -> round-trip y_hat = 0.00108, 99.9% relative error on an in-domain row. Also distorts T_train: the floored division manufactures a huge T outlier (y/eps) that then poisons the T-MAD clip envelope.

Fix: apply the identical safe_base floor in _ratio_inverse (round-trip then exact by construction). Optionally also report rows with |base| < eps via fit-time stats so discovery can see how often the floor engages.

---

## T5 - P2 / bug - centered_ratio inverse: same eps asymmetry + sign-flipped predictions when predict-time base drops below the train envelope

File: src/mlframe/training/composite/transforms/extended.py:144-158.

(a) _centered_ratio_forward floors shifted = base + c at +/-eps; _centered_ratio_inverse multiplies by RAW (base + c) - same round-trip break as T4 for |base + c| < eps.
(b) c is fitted so min(base_train) + c = +0.01*scale; any predict row with base < min(base_train) - 0.01*scale makes (base + c) NEGATIVE, and the inverse flips the prediction sign. _centered_ratio_domain (:161-167) only checks finiteness, so the row passes as valid.

VERIFIED: train base in [5, 10], c = -4.928; t_hat = 1.5 inverts to +1.61 at base = 6.0 but -1.39 at base = 4.0.

Fix: floor (base + c) with the same signed-eps rule in the inverse; additionally treat base + c <= 0 rows as domain violations at predict (requires params-aware domain - see T15) or at minimum count them into runtime_stats_.

---

## T6 - P2 / bug - rolling_quantile_ratio inverse multiplies by the raw rolling median while forward divides by the eps-floored one

File: src/mlframe/training/composite/transforms/simple.py:355-376 (_rolling_quantile_ratio_forward uses safe; _rolling_quantile_ratio_inverse uses roll_med directly).

Third instance of the T4 asymmetry class: rows whose centred rolling median is within +/-eps of zero round-trip with error factor roll_med/eps (sign flip possible at exactly 0 -> +eps vs raw negative median). Fix: reuse the same safe = np.where(|roll_med| < eps, sign*eps, roll_med) line in the inverse.

---

## T7 - P2 / bug - James-Stein sigma2 proxy ignores per-group base variance (and per-group sizes), mis-scaling the shrinkage factor

File: src/mlframe/training/composite/transforms/nonlinear.py:174-183 (_james_stein_shrinkage_factor), consumed at linear.py:491-498.

The sampling variance of an OLS SLOPE is sigma2/(n_g * Var(base_g)), but the code uses sigma2_total / mean(n_g) - dropping the Var(base) term entirely and collapsing heterogeneous group sizes to a single mean. The shrinkage factor c = (K-3)*sigma2_proxy/sum(dev**2) is therefore wrong by a factor of Var(base): over-shrinks whenever Var(base) > 1, under-shrinks when < 1 - the formula is unit-dependent, which a correct JS estimator never is.

VERIFIED: synthetic 8-group fixture with base ~ N(10, 2) (Var = 4): noise proxy 4x overestimated -> shrinkage_factor = 1.0 (full collapse to global alpha) where the variance-correct proxy gives materially less shrinkage.

Fix: accumulate per-group Var(base_g) during the per-group OLS loop and use mean over groups of sigma2/(n_g*Var(base_g)) (or the proper per-group weighted JS with heteroscedastic noise) as the proxy.

---

## T8 - P2 / bug - grouped shrinkage rescales alpha_g but never re-centres beta_g -> systematic per-group offset in T after shrinkage

File: src/mlframe/training/composite/transforms/linear.py:501-507.

beta_g was fitted jointly with the UNSHRUNK alpha_g (OLS => beta_g = mean(y_g) - alpha_g*mean(base_g)). After replacing alpha_g with (1-c)*alpha_g + c*alpha_global while keeping beta_g, the per-group residual mean becomes (alpha_g - alpha_shrunk)*mean(base_g) != 0 - the transform manufactures a per-group bias that the per-group OLS had explicitly removed.

VERIFIED: 8 groups x 60 rows, noise sd 2 (group-mean SE ~ 0.26), shrinkage c = 1.0 -> per-group T means range -1.88 ... +1.90 - far beyond noise.

Fix: in the shrink loop also set per_group_betas[g] += (a_g - a_shrunk)*mean(base_g) (store per-group base means during the OLS pass). Round-trip is unaffected (forward/inverse share params) but the residualisation quality and the "residual mean ~= 0" reporting contract are restored.

---

## T9 - P2 / bug - monotonic_residual degenerate early-returns omit is_degenerate, so the MOST degenerate fits bypass discovery degeneracy rejection

File: src/mlframe/training/composite/transforms/nonlinear.py:426-434 and :442-450 (early returns), flag consumed at discovery/_eval.py:66.

The full-fit path stamps is_degenerate / var_explained (Pack D), and discovery rejects flagged specs to avoid wasting full training on T ~= y - const. But both degenerate early-return paths (too few finite rows; knots collapsed to < 3, e.g. constant base) return constant-g params WITHOUT the flag.

VERIFIED: _monotonic_residual_fit(y, ones(100)) returns keys [knots_x, knots_y, monotone_direction, n_knots_effective, y_train_mean] - no is_degenerate. fitted_params.get("is_degenerate") -> None -> spec proceeds to full evaluation despite being exactly the pathology the flag exists for.

Fix: add "is_degenerate": True, "var_explained": 0.0 to both early returns.

---

## T10 - P2 / bug - auto-knot rule n_unique // 200 conflates cardinality with discreteness: continuous bases on small/mid n get 3-5 knots instead of 12

File: src/mlframe/training/composite/transforms/nonlinear.py:417-421.

The Pack-3 auto-cap was aimed at DISCRETE/categorical bases, but for a CONTINUOUS base n_unique ~= n, so: n = 600 rows -> 3 knots; n = 1000 -> 5 knots; full 12 knots require n >= 2400. A 12-knot fit on n = 1000 continuous rows is perfectly healthy (~83 rows/slab >= min_knot_n = 30, which already guards under-populated slabs independently). Result: monotonic_residual systematically underfits sigmoid/saturating shapes on the very sample sizes the tiny-rerank screening uses, degrading both the fitted transform and discovery opinion of it.

Fix: only cap when the base is actually low-cardinality relative to n (e.g. if n_unique < 4 * n_knots: cap to max(3, n_unique // 2)), or soften the divisor so min_knot_n remains the binding constraint (e.g. n_unique // 50).

---

## T11 - P2 / bug - _rolling_median bottleneck path diverges from the pandas fallback: every interior row for even k, last k//2 rows for odd k

File: src/mlframe/training/composite/transforms/nonlinear.py:818-834.

(a) EVEN k: the left-shift by k//2 aligns the bn forward window to [i-k/2+1, i+k/2], while pandas rolling(center=True) uses [i-k/2, i+k/2-1] - off by one at EVERY position (verified k = 4: 19/20 positions differ).
(b) ODD k tail: the last k//2 positions are filled with a constant (_fwd[-1], the last FULL-window median) instead of re-centred truncated-window medians (verified k = 7: positions 17-19 = [16,16,16] vs pandas [16.5,17,17.5]). The docstring claim "boundary fallback consistent with min_periods=1 semantics" is false.

Consequence: rolling_quantile_ratio T values (train and predict) depend on whether bottleneck is installed -> non-reproducible composite targets across environments (prod has mlframe[all] incl. bottleneck; CI/user envs may not).

Fix: for even k align the shift to the pandas convention (use _fwd[i + (k-1)//2]); for the tail compute true truncated centred medians (np.median(arr[i-k//2:]) for the last k//2 rows - negligible cost). Add a backend-parity test that runs both paths.

---

## T12 - P2 / leak - rolling_quantile_ratio centred window reads future base rows: look-ahead in time-ordered deployment

File: src/mlframe/training/composite/transforms/simple.py:328-376 (design), window math in nonlinear.py:804-839.

T_i = y_i / RollingMedian_k(base)[i] with a CENTRED window uses base[i+1 ... i+k//2] - features of FUTURE rows - both to construct the train target and to invert predictions. In batch backtests this is silent look-ahead (future base values usually embed information unavailable at decision time of row i); in streaming inference the inverse for the newest k//2 rows is not even computable as fitted. EWMA residual, by contrast, is correctly causal (left recurrence). The transform is gated out of default discovery for ORDERING reasons (config :80), but nothing warns the explicit user about the future-window dependence.

Fix: add a mode="trailing" (default for time-series use) computing the median over [i-k+1, i]; keep "centred" as opt-in with a loud docstring/registry-description warning about look-ahead.

---

## T13 - P2 / bug - domain-filtering compacts the row sequence before time-recurrent forwards: EWMA / rolling / frac-diff T near gaps is computed on a different sequence than predict-time

Files: estimator/_estimator.py:458 (y_train = y_arr[valid]; base_train = base_arr[valid] before transform.fit/forward); affected transforms: nonlinear.py:558-578 (ewma), simple.py:355-366 (rolling_quantile_ratio), nonlinear.py:864-891 (frac_diff).

The wrapper drops domain-invalid rows and then runs the recurrent forward on the COMPACTED sequence: a dropped row neighbours become adjacent, so the EWMA state / rolling window / frac-diff lag stack at every subsequent row differs from what the same transform computes on the full frame at predict time (where non-finite rows carry state forward instead of vanishing). The fitted mapping near gaps is therefore trained on values that can never occur in deployment. The EWMA kernel already supports the correct semantics (non-finite => carry state), making the row-drop doubly unnecessary for these transforms.

Fix: for sequence transforms (tag them, e.g. TAG_SEQUENTIAL), compute forward on the FULL-LENGTH arrays (NaN-aware kernels) and apply the valid mask only afterwards to select training rows; never re-index the sequence before the recurrence.

---

## T14 - P2 / bug - frac_diff is y-only but registered requires_base=True: duplicate identical specs per base + needless base-driven row drops

File: src/mlframe/training/composite/transforms/registry.py:376-386 (no requires_base=False), forward/inverse ignore base (nonlinear.py:864-891), domain demands finite base (nonlinear.py:892-900).

(a) Discovery per-transform dedup (discovery/_fit.py:409-412) applies only to requires_base=False; with frac_diff configured, one IDENTICAL candidate is fitted/evaluated per base candidate, each with a misleading distinct name (y-fdiff-lag1, y-fdiff-lag2, ... all the same T).
(b) The domain check rejects rows where the (unused) base is non-finite, shrinking the fit set and compacting the y sequence (compounds T13).
Not in the default transforms list, so impact is config-opt-in - but the registry contract is simply wrong about the transform data dependence.

Fix: set requires_base=False on the registry entry and make _frac_diff_domain ignore base when y is not None (mirror y_quantile_clip).

---

## T15 - P2 / bug - fitted-params-dependent domains are unenforceable: log_y forward NaNs on screen/val rows below -offset; the params-aware log_y_domain(y, params) variant is dead code

Files: registry.py:179-183 (adapter wraps lambda y: _log_y_domain_raw(y) - drops the params arg), unary.py:145-150 (2-arg domain, never called with params), discovery/_eval.py:44-56 (domain gate runs BEFORE fit, so it cannot know offset).

log_y true domain is y > -offset where offset exists only after fit. The pre-fit gate checks only isfinite(y), so any screen/val row with y < min(y_train) - 1 sends log(negative) -> NaN into T (verified: offset = 1.0, log_y_forward([-2.0]) = [nan]), which then flows into MI screening / tiny-rerank silently. The same structural gap is why T5 base + c <= 0 rows pass as valid: domain_check(y, base) has no params channel at all.

Fix: add an optional post-fit validation hook to the Transform contract (e.g. domain_check_fitted(y, base, params) defaulting to the params-less check) and call it in discovery screening / eval after fit; wire log_y existing 2-arg domain and a shifted > 0 check for centered_ratio into it. Cheap and closes the whole class.

---

## T16 - LOW / docs - YJ numba forward claims "bit-identical" to the numpy path; it is not (fastmath=True)

File: src/mlframe/training/composite/transforms/unary.py:218-221 (docstring), kernel at :45-66.

VERIFIED: max |numba - numpy| = 7.1e-15 (lam=0.37), 1.14e-13 (lam=-1.3), 5.7e-14 (lam=3.1) on n = 50k - within the project ~1e-9 acceptance band, so the dispatch is fine, but the docstring bit-identity claim is false (and the inverse kernel comment at :69-71 even explains why fastmath breaks bit-exactness). Fix: correct the docstring to "equal within ~1e-13" and pin a tolerance regression test.

---

## T17 - LOW / bug (hygiene) - dead duplicated numba kernels and MAD constants across three modules

Files: transforms/__init__.py:31-58, linear.py:46-78 (duplicate _ewma_kernel + _frac_diff_inverse_kernel; both copies unused - the live ones are in nonlinear.py:53-119); linear.py:87-90 + nonlinear.py:127-136 (duplicate _MAD_FLOOR_FRAC/_MAD_SOFT_CAP_K - dead, since _logratio_fit lazy-imports the parent copies at linear.py:101); duplicated logger = lines linear.py:28/94.

Risk is drift: editing the sibling copy of a constant/kernel silently changes nothing. Fix: delete the dead copies; keep single definitions in nonlinear.py (kernels) and __init__.py (constants).

---

## T18 - LOW / perf - _finite_mask plumbing documented as wired into "the outer Transform.fit dispatcher" is never called by anyone

Files: simple.py:87,170,249,290,341, nonlinear.py:287,402,560,851 (8 fit functions take _finite_mask); grep over the repo shows ZERO call sites pass it - discovery/_eval.py:56 calls transform.fit(y_train[valid], base_train[valid]) plain.

The advertised optimisation (one isfinite pass per (y, base) shared across 10+ specs) was never realised; each fit recomputes O(n) isfinite masks. Impact small (~ms per spec) but the comments describe infrastructure that does not exist. Fix: either thread the mask from eval_one_transform (it already has valid) or delete the dead kwarg + comments.

---

## T19 - LOW / docs - Transform contract claims fitted params are "JSON-serialisable"; six transforms return ndarrays

File: transforms/__init__.py:106-110 (contract), violated by median_residual (simple.py:196-200), quantile_residual (nonlinear.py:357-365), monotonic_residual (nonlinear.py:512-520), rank_residual (extended.py:273-278), smoothing_spline_residual (extended.py:360-365), quantile_normal_y (unary.py:297).

frac_diff deliberately does .tolist() on its weights (nonlinear.py:863) for exactly this reason - the convention is inconsistent. A naive json.dumps(spec.fitted_params) crashes. Fix: either amend the contract docstring ("ndarray values allowed; use the spec serialiser") or .tolist() the small arrays (all are O(n_bins/knots), cheap).

---

## T20 - LOW / usability - is_composite_target_name substring heuristic false-positives on plausible user columns

File: naming.py:91-117.

Fragments include common tokens: -diff-, -ratio-, -spline-, -interact-. A user column price-diff-7d or debt-ratio-q is classified as a composite target and gets the MTRESID metric labelling. Fix: parse strictly - require exactly {known_target}-{alias}-{base} when the target list is available, or at least anchor the alias as the SECOND dash-separated token rather than any substring.

---

## T21 - LOW / usability - unary specs are named with a spurious base segment

Files: discovery/_eval.py:277 (compose_target_name(target_col, transform_name, base) unconditionally), work items built at _fit.py:397-413 (unary transforms keep the current loop base).

A cbrt_y spec evaluated during the lag1 iteration is named y-cbrtY-lag1 although the transform ignores lag1 entirely (verified naming). Misleading in reports, logs and dedup keys; two runs with different base-candidate orderings name the same unary composite differently. Fix: for requires_base=False compose with a sentinel base ("none"/"") and teach is_composite_target_name the 2-segment unary form.

---

## T22 - LOW / bug (hygiene) - silent except Exception: pass around the pandas-groupby fast paths can mask real bugs

Files: simple.py:161-166 (_median_residual_per_bin_medians), nonlinear.py:270-280 (_quantile_residual_per_bin_stats).

A genuine defect in the v2 pandas path (dtype regression, pandas API change) would silently reroute to v1 forever - the exact "silent error swallowing" class the project tracks. Fix: logger.warning (once) with the exception before falling back.

---

## T23 - LOW / docs - logratio soft-cap makes the inverse inexact for in-domain train rows beyond 10 MAD; behaviour is deliberate but unpinned

File: linear.py:120-130.

_logratio_inverse clips t_hat to median(T) +/- 10*MAD - rows whose true T legitimately exceeds the cap (heavy-tail logratio, |T-med| > 10*MAD ~ 6.7 sigma) do not round-trip. This is a sound predict-time safety tradeoff, but the registry description ("Inverse y_hat = base * exp(softcap(T_hat))") does not state the round-trip consequence and no test pins the capped branch on either side. Fix: one sentence in the description + a two-sided test (capped row inverts to the cap; in-cap row round-trips to 1e-12).

---

## T24 - LOW / extension - EWMA (and frac-diff) inverse cold-starts every predict batch from the train-mean anchor

Files: nonlinear.py:558-578 (fit stores only k + anchor), :785-796 (forward/inverse recompute EWMA from scratch per call).

Statelessness is a deliberate design (JSON params, predict-on-new-data), but for the canonical deployment - test window immediately following train - the first ~k rows of every predict batch use EWMA ~= anchor (train global mean) instead of the actual recent level, biasing y_hat = t_hat + EWMA exactly where recency matters most. Frac-diff has the same property but is self-consistent (forward and inverse share the anchor-padding convention). Extension: store the last train EWMA state in params and expose warm_start=True (use stored state when the predict frame chronologically continues train), or accept an optional base_history kwarg.

---

## T25 - LOW / test-gap - adversarial round-trip / parity coverage missing for every bug class found above

File: tests/training/composite/test_biz_val_composite_transforms.py (existing round-trips use benign data only).

Missing and would have caught T1/T4/T5/T6/T8/T11: (a) round-trip at eps-edge rows (|base| < fitted eps) for ratio/centered_ratio/rolling_quantile_ratio; (b) YJ inverse on t_hat beyond the lambda-asymptote for lambda < 0 and lambda > 2; (c) bottleneck-vs-pandas _rolling_median parity for even and odd k incl. boundaries; (d) grouped-transform per-group residual means ~= 0 after shrinkage fires; (e) linear_residual_multi non-fallback on independent mixed-scale bases; (f) frac_diff/ewma fit-vs-predict T equality on data containing scattered NaN rows. Each is a <=20-line synthetic test.

---

### Summary statistics

| severity | count | ids |
|---|---|---|
| P1 | 3 | T1, T2, T3 |
| P2 | 12 | T4-T15 |
| LOW | 10 | T16-T25 |

All numerical claims verified by execution on current code (see method note at top). No known-fixed items from composite_discovery_audit_notes.md re-reported.
