# MRMR FE/FS audit findings (agentic optimization /loop)

Findings from the optimization loop's audit agents. FIXED items are listed for provenance; OPEN
items are precisely scoped for a fresh-context iteration to implement + test.

## FE DEFAULT-ON CANDIDATE BENCHES (which disabled generators to enable)

A scout categorized all `fe_*_enable=False` generators: most are correctly opt-in -- DATA-TYPE-SPECIFIC
(cat/missingness/grouped/temporal), RESEARCH-GRADE ALT SCORERS (ksg/copula/dcor/hsic/jmim/tc/cmim/
lasso/elasticnet/auto/ensemble/meta -- duplicate the tuned plug-in MI hot path over byte-identical
engineered values), EXPENSIVE/UNBOUNDED (gbm_seeder, gradient_interaction, conditional_gate's O(p^3)
n-driven select sweep), or SUBSUMED (mi_greedy, pairwise_ratio/log_ratio, rankgauss, adaptive_arity,
modular). Only two general-numeric (A) candidates surfaced; both BENCHED:

- **`fe_hybrid_orth_triplet_enable` / `fe_hybrid_orth_quadruplet_enable` -> ENABLED (shipped, a28e19f4).**
  3-way `a*b*c` linear test MAE 0.094 -> 0.049, no-harm additive, seed_k-bounded, replay-fixed.
- **`fe_gate_med_enable` -> REJECTED (keep OFF, harmful as default-on).** The scout flagged it (docstring
  claims +0.0355 single-column linear-usability). But a multi-seed end-to-end bench (gate target on
  shifted/skewed marginals, linear downstream) showed it HURTS: gate target mean linMAE 0.865 -> 1.256,
  smooth control 0.107 -> 0.157, and ~2x slower. DIAGNOSIS: gate_med is a pseudo-unary fed to the
  MI-GREEDY pair search, which wraps it into high-MI but linearly-USELESS composites
  (`div(gate_med(a),exp(b))`, `div(gate_med(c),neg(d))`) instead of the clean `mul(gate_med(a),b)` =
  `(a>med)*b` the linear model needs -- the SAME MI-vs-linear-usability blindness this session keeps
  hitting. The isolated docstring metric does not survive the pipeline. It would only help if its
  pseudo-unaries fed the USABILITY-AWARE list (linear-usability binary selection), not the MI-greedy
  search -- a deeper, speculative integration, not pursued. Do not enable by default.
- **`fe_numeric_decompose_enable` -> REJECTED (keep OFF, no-win + cost).** Scout's #2. Benched OFF vs ON
  multi-seed on a step/rounding target (`round(a/10)*10 + floor(c*5)/5`, where its rounding features
  should win) + a smooth control: IDENTICAL linMAE on both (step 0.1561, smooth 0.1073) -- its candidates
  do not surface/get-selected (the default hinge/binning FE already captures the step structure) -- and
  ~1.6x slower. No benefit, only cost. Keep OFF.

NET: triplet/quad were the genuine general-numeric default-on win; the FE-default-on lever is now worked
(every other disabled generator is correctly opt-in or benched-harmful).

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
- **orth DIFF_BASIS recipe refit-at-replay** (same P0 class, last orth straggler) -> froze the diff's
  basis-preprocess params. `_orthogonal_diff_basis_fe.py` + builder. Slice-replay test. (c714fa1e)
- **transform vs get_feature_names_out width mismatch on legacy pickles** (P1, breaks sklearn Pipelines)
  -> get_feature_names_out mirrors the `requires_refit_for_replay` filter transform applies. (8f517a77)
- **usability lists embedded full-n TRAINING DATA in the pickle** (privacy leak + ~n*8 bytes/candidate
  bloat) -> clear each selected candidate's `values` post-fit (transform_usability replays from recipe/
  name, never reads values). `_usability_lists.py`. (413d410f)

## VERIFIED CLEAN (no triggerable bug)

- `_fe_raw_redundancy_drop.py` -- keep/drop sign, nested anchoring, fail-closed replay, determinism.
- pair/cluster recipe replay -- frozen anchors (log_shift, gate_med, prewarp, cluster stats), continuous.
- spline/fourier/encoding/grouped recipe families -- all freeze fit-time state (no refit-at-replay).
- `row_argmax` / `conditional_gate` FE -- tau frozen + replayed, row-pure argmax, genuine slice-replay test.
- MRMR transform/validate path -- name-based selection (robust to reordered frames), NotFittedError on
  missing cols, no y leakage, ndarray dtype-promotion, empty-support fallback (the one P1 fixed above).
- MRMR screening CORE (greedy relevance/redundancy, both stop floors, Fleuret min-over-Z, unbiased
  permutation shuffle, degenerate handling, determinism) -- correct at default config.
- DISCRETIZATION / binning (quantile+MDLP edges monotone, separate-bin NaN per-column, constant single-bin,
  `factors_nbins == max(code)+1` by construction so no MI-kernel OOB) -- correct on all paths.

## OPEN

### [P2, OFF-DEFAULT, untested] screening-core `last_checked_k` stale resume under interactions_order >= 2

The greedy's per-candidate `partial_gains` resume (evaluation.py `evaluate_gain`, resumed at the
`cand_idx in partial_gains` branch) stores a global `last_checked_k` index into
`generate_combinations_recursive_njit(selected_vars, order)`. At the DEFAULT
`max_veteranes_interactions_order=1` a newly-selected var always appends at the END, so resuming from
`last_checked_k` correctly picks it up -- the audit confirmed NO bug at default. But with
`max_veteranes_interactions_order >= 2`, growing `selected_vars` inserts the new singleton in the MIDDLE
of the global k-sequence, so a resumed candidate with stored `last_checked_k` SKIPS the new singleton Z
-> its redundancy against the most-recently-selected feature is never measured -> a feature fully
redundant with that Z can survive (P1-class wrong-selection, but only on the off-default research path,
and currently UNTESTED). Fix: when `selected_vars` has grown since a candidate's `partial_gains` entry
was written, invalidate the entry (re-evaluate from `last_checked_k=-1`) rather than resume; key the
resume on the size of `selected_vars` at write time, not the raw `k`. Deferred (off-default + needs a
new test exercising order>=2 + the resume path -- craft with fresh context, do not rush untested code).

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
