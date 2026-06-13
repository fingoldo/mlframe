# MRMR FE/FS audit findings (agentic optimization /loop)

Findings from the optimization loop's audit agents. FIXED items are listed for provenance; OPEN
items are precisely scoped for a fresh-context iteration to implement + test.

## FIXED (shipped this session)

- **div-by-zero in `resolve_adaptive_vote_k(min_rows_per_fold=0)`** (P2) -> clamp floor >=1. (cf03d387)
- **fail-OPEN `_operand_marginal_mi` returning 0.0** loosened the marginal-uplift gate -> return +inf
  (fail-closed) + log. `_pairs_core.py`. (7cd23bce)
- **usability greedy empty-fold crash/NaN** (random multinomial folds) -> balanced `arange%k` partition;
  **+ in-sample shortlist residual** -> held-out. `_usability_aware_selection.py`. (cb182c44)
- **non-row-wise-pure FE ops** (grad1/grad2/logn) admitted to FE pair candidates -> slice-replay
  corruption; excluded from the candidate dicts. `_fit_impl_core.py:5725`. (a375b580)
- **orth TRIPLET/QUADRUPLET cross-basis recipes refit the per-leg preprocess at replay** (P0/P1
  slice-replay corruption) -> froze the fit-time basis-preprocess params per leg (mirroring the pair
  BUG2 fix) across the 2 builders + 2 apply fns + 4 fit-emit sites, exported `_freeze_preprocess_params`.
  Direct slice-replay unit test (frozen -> byte-exact; unfrozen -> drifts on a slice). Was the
  PREREQUISITE for enabling triplet/quad FE by default. 117 existing triplet/quad tests green.

## VERIFIED CLEAN (no triggerable bug)

- `_fe_raw_redundancy_drop.py` -- keep/drop sign, nested clean-subexpression anchoring, fail-closed
  replay, determinism all correct; the one P2 (`cols.index` first-match) cannot trigger (raw names unique).
- pair/cluster recipe replay (`_recipe_unary_binary.py`, `_recipe_poly_cluster.py`) -- frozen anchors
  (log_shift, gate_med, prewarp preprocess, cluster member stats) all read from `recipe.extra`; continuous
  output; registry-mismatch raises. Clean.

## OPEN

(none currently -- the orth triplet/quadruplet replay P0 below was RESOLVED, see FIXED list above.)

### [RESOLVED] Orth TRIPLET / QUADRUPLET cross-basis recipes refit per-leg preprocess at replay

**Severity P0/P1** (silent wrong feature values on test/slice replay), gated behind the default-OFF
`fe_hybrid_orth_triplet_enable` / `fe_hybrid_orth_quadruplet_enable`. FIXED this session (the user asked
to enable these by default, which made the replay fix the mandatory prerequisite). Detail kept below for
provenance.

**The bug:** `_apply_orth_triplet_cross` / `_apply_orth_quadruplet_cross` call
`_eval_orth_basis_column(vals, basis, deg)` with `preprocess_params=None`
(`_orthogonal_triplet_fe_recipes.py:56-58`, `_orthogonal_quadruplet_fe_recipes.py:63-66`), so the sink
`engineered_recipes/_orth_basis_recipes.py:134-135` takes the REFIT branch `z,_=fit_fn(x)` -- the per-leg
z-score mean/std (or legendre/chebyshev min-max lo/hi, laguerre shift) is recomputed from the APPLY-time
rows, not frozen from fit. On a row-slice / drifted test frame the basis axis shifts and every value
differs (the same class as the pair "BUG2 FIX (2026-06-12)"). The builders
`build_orth_triplet_cross_recipe` / `build_orth_quadruplet_cross_recipe` don't even accept frozen params,
and the fit-emit sites (`_orthogonal_triplet_fe.py:175-183`, `_orthogonal_quadruplet_fe.py` equivalent,
and the triplet wrapper's univariate emit `_orthogonal_triplet_fe.py:600-603`) call
`_evaluate_basis_column` WITHOUT `return_params=True`, discarding the fit-time stats.

**The fix (mirror the pair BUG2 fix end-to-end):** the pair path already does this correctly --
`build_orth_pair_cross_recipe` freezes `preprocess_params_i/j` (`_orth_basis_recipes.py:199-206,261-289`)
and `_apply_orth_pair_cross` passes them through; `_evaluate_basis_column` supports
`return_params=True`/`preprocess_params=`; `_freeze_preprocess_params` exists. Replicate across triplet
(legs i,j,k) + quadruplet (legs i,j,k,l): (1) capture `return_params=True` at each fit-emit
`_evaluate_basis_column`; (2) plumb `preprocess_params_{i,j,k[,l]}` through the two builders into `extra`;
(3) read + pass them at each `_eval_orth_basis_column(..., preprocess_params=pp_x)` in the two apply fns
(cache by `(col,basis)`); (4) fix the triplet univariate emit at `:600`. Also freeze the NaN-fill
constant (`_orth_basis_recipes.py:116` `np.nanmean` is likewise data-dependent -- same fix bucket).

**Missing test (add with the fix):** no test does TRUE slice/test-frame replay of these recipes -- the
OOS tests (`test_triplet_cross_basis.py:339-361`, `test_layer77.py:346-367`) build a joint frame on the
full data and slice AFTER generation, sidestepping the drift. Add: fit on frame A, `apply_recipe` on a
drifted/sliced frame B, assert byte-equality vs the fit-time column slice (will fail RED on current code,
GREEN after the fix) -- mirror `test_fe_rowwise_pure_replay.py` / the pair adversarial slice-replay battery.
