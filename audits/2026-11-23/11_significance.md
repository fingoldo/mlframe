# Audit 11: MRMR significance gates and multiple testing (wave 3)

Scope: maxT / FDR gates, honest MDLP null, permutation nulls (non-GPU parts), analytic chi-square null, prescreen FDR floor, fallback maxT. Read-only, Haiku model, nothing executed. GPU-only files out of scope.

Tracker rows: AUD23-14, 15, 16, 17.

1. Count/n p-value with no +1 (serial confirm path). High in principle; deliberate. Confirmed.
   - `_confirm_predictor_helpers.py:119`: `confidence = 1 - nfailed / nchecked`. At the serial screen budget (`full_npermutations=3`), `nfailed=0` gives p=0 and confidence 1.0. Add-one `(1+nfailed)/(B+1)` gives a floor of 1/4 at B=3, so this path can never pass a p-value gate at alpha below 0.25.
   - Comment at 114-118 says intentional, to avoid starving the screen. Treat as a known deviation: at this budget the serial confirm path has no valid significance test.
   - `permutation.py:109-110` `_perm_pvalue` uses add-one only when `MLFRAME_MRMR_ADDONE_PVALUE` is set; default of the flag not verified.
   - Honest path (`_confirm_predictor.py:288-291`) uses `_perm_pvalue`; same flag dependence.
   - The maxT gate itself is correct: `_family_maxt_gate.py:598-609` (`exact_pvalue_pass`) implements `(1+R)/(B+1) <= alpha` with `searchsorted(side="left")`; `:844` and `:859-873` consistent.

2. Quantile convention for the maxT / FDR floor. Low. Convention mismatch, not a demonstrated error.
   - `_permutation_null.py:455`, `:735`, `:822` use `np.quantile(maxes, q)` with linear interpolation. `_honest_mdlp_null.py:365`, `:430` use `method="higher"` (more conservative).
   - Example K=B=200, q=0.95: linear sits at index 189.05 (0-based), between the 190th and 191st order statistics; the conservative (1-alpha) order statistic is ceil((B+1)(1-alpha)) = 191st. Linear floor slightly lower, so the family error rate can slightly exceed alpha.
   - `_mrmr_fit_impl/_fallback_maxt.py:3-6` and `:35-37` say the floor is the (1-alpha) quantile and alpha is applied by the caller; the code returns raw null maxima and ignores alpha. Docstring is stale; the decision in `exact_pvalue_pass` is correct.

3. Shared seed across candidates and families. Medium for FDR accounting, low for per-test validity. Confirmed in code; documented as intentional.
   - `_family_maxt_gate.py:690`, `:705`, `:911` default `seed=0`; `_permutation_pass_flags_with` (`:829`) calls `default_rng(seed)`. Every family draws the same permutation stream (cache comment `:268-271`).
   - `_honest_mdlp_null.py:404-405`, `:497-498` call `default_rng(0 if seed is None else seed)`; every candidate of one fit gets the same row permutations (docstring `:480`).
   - `_permutation_null.py:189` falls back to a fixed base `0x9E3779B9` when `random_seed` is None.
   - Consequence: each candidate's marginal size is still alpha, but exceedances are perfectly correlated across candidates sharing the null draws. Analyses that treat candidates' noise decisions as independent (expected count of false passes) understate variance. Magnitude not measured.

4. Analytic chi-square null outside its validity regime. Low. Pass-through, guarded at main call sites.
   - `_analytic_mi_null.py:351` and `:358`: `_reject = (fe_mi > 0) & _applicable & (_p >= alpha_reject)`. Where the analytic null is not applicable the column is never rejected and keeps its raw MI with no permutation fallback; comment `:347-350` says so.
   - Production guards: `_feature_engineering_pairs/_pairs_dispatch.py:317-319` checks `analytic_null_applicable` with declared `quantization_nbins` (conservative) before the batch gate; `_gpu_resident_pair_mi.py:91` applies the same check.
   - `_family_maxt_gate.py:757-818` (`_analytic_triage_core`) has its own density gate at `:783` (`n / (Kx*Ky) >= 4*5 = 20`); when it fails it returns None and the permutation test runs.
   - Not verified: the guard before the direct call at `_feature_engineering_pairs/_pairs_analytic_gpu.py:359`.
   - The batch gate's df uses declared bin counts (`:340`) while the applicability check uses occupied bins (`:351`); comments `:342-346` document this as conservative.

Not confirmed: an off-by-one in the order statistic of the scoped maxT/FDR floors. The `(1+R)/(B+1)` and `searchsorted` logic is correct; the only quantile issue is the interpolation convention in item 2.
