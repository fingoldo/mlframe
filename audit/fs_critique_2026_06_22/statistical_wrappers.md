# Statistical-wrapper feature-selection audit (2026-06-22)

Scope: `wrappers/_knockoffs.py`, `_univariate_ht.py`, `_noise_floor.py`, `_auto_tune.py`, `_helpers*.py`, `_enums.py`
(excludes `wrappers/rfecv/`, MRMR, worktrees). Python 3.14.3 (store build; anaconda absent on this host).

## Findings table

| # | File:line | Severity | Issue | Disposition |
|---|-----------|----------|-------|-------------|
| F1 | `_auto_tune.py:86` (pre-fix) | HIGH | `np.bincount(y_arr.astype(int))` raises `ValueError` on negative integer labels and over-allocates on sparse high integer codes. | RESOLVED — use `np.unique(..., return_counts=True)`. |
| F2 | `_univariate_ht.py:466` (pre-fix) | MED | `ml_task="classification"` counted labels with `np.unique(y_arr)` without dropping NaN; one stray NaN in a float-coded target inflates the count to 3 and mis-routes a binary target to Kruskal-Wallis. | RESOLVED — drop NaN via `_isnan_mask` before counting. |
| F3 | `_auto_tune.py:74` (pre-fix) | MED | Target-type cap `max(10, sqrt(n))` grows with n; a 200-distinct integer regression target at n=2e5 was mislabelled `multiclass` (the exact flaw `_univariate_ht` already fixed with `_MULTICLASS_MAX_LABELS`). | RESOLVED — absolute cap 50 + `<=0.05*n` ratio, mirroring the univariate module. |
| F4 | `_univariate_ht.py` MWU/KW kernels | PERF | `_mann_whitney_u_z` / `_kruskal_wallis_h` each ran TWO `argsort` passes over the same column (once for ranks, once for the tie sum). | RESOLVED — `_rank_and_tiesum` does both in one sort; `_mann_whitney_u_z_v2` / `_kruskal_wallis_h_v2` route through it. Old kernels kept. **2.43x kernel / 2.48x end-to-end, bit-identical**. |
| F5 | `_knockoffs.py:33-147` | DOC | The construction is **model-X** Gaussian knockoffs (Candès 2018), not fixed-design Barber-Candès 2015 as the docstring claims. The math (`X(I-Σ⁻¹diag s)+ZC`, equicorrelated s) is correct model-X; only the citation/name is wrong, which matters because model-X assumes X~N(μ,Σ) and gives FDR control *in expectation over X*, a different guarantee. | DOC (caveat below; not code-changed to avoid touching a passing hot path). |
| F6 | `_knockoffs.py:120` | LOW | `s_val = min(2*lam_min,1)*0.99` is the equicorrelated s; correct, but on near-singular Σ (collinear features) s→0 and knockoffs become near-copies (W≈0, zero power). Warned/raised already via `_KNOCKOFFS_STRICT_LAM_MIN`. | DOC — known degeneracy, surfaced. |
| F7 | `_knockoffs.py:146` | LOW | Knockoffs returned on original scale (`*stds+means`) while the joint fit standardises nothing; for a scale-sensitive linear `coef_` W-statistic the real/knockoff columns share scale so W is fair, OK. Constant columns get std=1, mean=const → knockoff=const+noise; harmless. | DOC. |
| F8 | `select_features_fdr:176-192` | LOW (correct) | On few positives (e.g. one strong + one negative W) returns `[]` even for an obvious driver — this is **correct** Barber-Candès low-power behaviour (the `1+#neg` offset cannot be beaten at small support), NOT a bug. The "knockoff+" data-splitting offset is already implicit. | DOC — power weakness, not a defect. |
| F9 | `_univariate_ht.py:401` | LOW | `_kendall_p_numeric_continuous` subsamples to 2000 rows with a hard-coded `default_rng(0)` ignoring any caller seed. Deterministic/reproducible but not controllable; also drops power on large n. | FUTURE — thread a seed; low priority (deterministic today). |
| F10 | `_noise_floor.py:102,136` | MED (assumption) | `scoring="roc_auc"` + `StratifiedKFold` hard-code a **binary** target; a multiclass/continuous y crashes inside `cross_val_score`, and a class with `< cv` members crashes the splitter. Docstring says "binary target" but there is no guard/clear error. | FUTURE — add an explicit binary-target check with a clear message. |
| F11 | `_univariate_ht.py:491` | LOW | Continuous-target × categorical-feature path `pd.qcut(...)` can emit NaN bins (dropped duplicates); `_chi2_independence_p` then includes a NaN level in `pd.crosstab` (NaN silently dropped by crosstab — acceptable but undocumented). | DOC. |
| F12 | `_chi2_independence_p:431` | LOW | Pearson chi-squared with no expected-cell-count guard (cells `<5` make the asymptotic χ² unreliable); BY-FDR partially compensates but small contingency tables give inflated significance. | FUTURE — consider Fisher/`lecturer` small-cell fallback. |

## Optimizations (measured)

- **Single-sort MWU/KW kernels** (F4): microbench `_mann_whitney_u_z` n=5000 → **1440→415 µs/call (2.43x)**; end-to-end
  `calculate_relevance_table` (binary, n=5000, p=200, ×3) **1.598s→0.645s (2.48x)**. Bit-identical on heavily-tied
  integer data (regression-tested). Old kernels retained under their original names per the keep-all-versions rule;
  the new `_rank_and_tiesum` + `*_v2` kernels are the dispatched default.
- No further actionable speedups: knockoff matrix ops are `np.linalg` (eig/inv/cholesky) on p×p and dominated by the
  user-supplied model fit; noise-floor cost is entirely `cross_val_score`. Both are O(model-fit), not our code.

## Strengths / weaknesses verdict

- **Univariate-HT** — best general default. Distribution-free (rank tests), numba-fast, BY-FDR is valid under
  arbitrary dependence (right call for correlated features), scales to wide p>>n. Weakness: per-feature marginal test
  only (misses pure interactions / XOR signal); chi-squared small-cell unreliability; Kendall subsample caps power on
  huge n. Highest power+lowest runtime of the three for screening.
- **Knockoffs** — strongest FDR *control* (provable model-X bound) and catches conditional (not just marginal)
  importance via the joint [X,X̃] fit. Weakness: degenerates on collinear/correlated features (s→0 ⇒ zero power — the
  madelon failure noted in `_noise_floor`'s own docstring), needs a model refit on 2p columns, low power at small
  support (F8), and assumes X~Gaussian. Use when FDR rigor matters and features are not near-collinear.
- **Noise-floor plateau** — the pragmatic cut when the CV curve is flat (noise-robust GBM) where both knockoffs
  (correlated probes) and band-rules over-select; empirically cuts madelon 251→~12 with downstream AUC gains. Weakness:
  binary-only (F10), expensive (real + n_perm permuted CV curves), post-hoc on a supplied ranking, n_perm<3 noisy.
  Best as a final parsimony cut, not a primary selector.

Recommended routing: univariate-HT for fast wide screening → knockoffs for FDR-controlled confirmation on a
de-correlated subset → noise-floor plateau as the final count cut on flat curves.

## Tests

`tests/feature_selection/wrappers/test_wrappers_statistical_audit_fixes.py` (5 tests, all green; each fails on pre-fix
code): MWU/KW v2 bit-identity on tied data, classification NaN-label routes to Mann-Whitney (not KW), auto-tune
negative-label no-crash, auto-tune high-card integer target → regression.
