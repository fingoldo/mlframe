# FE_FAMILIES_B — categorical-interaction / grouped-agg / missingness / target-encoding / density / synergy / FE-gates — audit (2026-07-25)

## Scope

The categorical-interaction cat-FE pipeline (`cat_interactions.py` facade + `_cat_*` siblings: bandit-UCB1 /
fixed / Westfall-Young permutation confirmation, Miller-Madow re-rank, k-way materialize, target-encoding &
weighted kernels, pair/triple hybrid crosses, `CatFEConfig`/`CatFEState`); the grouped / composite / binned
aggregation families; the missingness, LOF, Mahalanobis-density, target-encoding (moment + order-stat), count-freq,
extra-FE (rare-cat / conditional-residual / RankGauss / dispersion) families and their GPU-resident twins; the
synergy / edge-MI / meta-recommender detectors; and the shared FE gates (accuracy, linear-explainability,
pure-form retention CPU+GPU-resident, retention-subsumption, usability-signal, unary tuning, unified gate,
baselines). Verified against current source at git `d8091a138`. mypy note: not re-run cluster-wide here; findings
below are source-verified line-level claims.

Prior findings cross-checked and confirmed **FIXED** (not re-reported): `cat_interaction_a` A-1 (unweighted
marginal in II — now weighted, `_cat_interactions_step.py:367`), A-2 (GPU dispatch now calls
`gpu_globally_disabled()` at both `_cat_confirm_permutation.py:426` and `_cat_interactions_step.py:333`), A-3
(bandit auto-falls-back to the correctly-weighted fixed path + unconditional `logger.warning` when weighted,
`_cat_interactions_step.py:526-531`), A-7 (count-freq docstring corrected), A-8 (`_kfold_target_encode_codes`
now validates `n_folds`, `_cat_pair_fe.py:390`), A-9 (`CatFEConfig.__post_init__` now range-checks the 5 fields,
`cat_fe_state.py:249-258`); `cat_interaction_b` B-2 (count/nunique per-group-scale fallback), B-5 (lagged_diff
`entity_cols`), B-6 (two-dot import); `fe_redundancy_synergy` FE_REDUNDANCY_SYNERGY-1 (KTC-backed CPU pairs/sec,
`_fe_synergy_exhaustive.py:162`) and -8 (stale "NOT YET WIRED" docstring corrected).

## Findings

| ID | Severity | Category | File:Line | Summary | Repro / failure scenario |
|----|----------|----------|-----------|---------|--------------------------|
| FE_FAMILIES_B-1 | P1 | bug (silent wrong selection) | `_cat_mm_correction.py:256,278-279,289-291,314,322` (+ subtraction `331-332`) | `_maybe_rerank_with_mm` **double-corrects** the Miller-Madow II bias: it computes all six entropies with `_entropy_for_mode(..., use_mm=not use_kt, ...)` = **Miller-Madow** entropies (each already = plug-in + (k-1)/(2n)), then ALSO subtracts the closed-form telescoped bias `(k_a-1)(k_b-1)(k_y-1)/(2n)` at 331-332. The in-code comment at 325-328 claims "the six terms above are now all plug-in under MM mode" — factually false; they are MM. The sibling `_compute_pair_ii_mm` (:120-184) does it correctly (entropies with `use_mm=False`, single telescoped subtraction), so the two production MM paths **disagree** on the same pair's II. | Default cat-FE fit (`use_miller_madow=None`, `use_kt_smoothing=False`), any pair whose analytic gate fires (`(a-1)(b-1)(c-1) >= 6*sqrt(n)`, i.e. exactly the high-cardinality joints MM exists for). `_maybe_rerank_with_mm` is called at `_cat_interactions_step.py:446` and re-sorts `selected_idx` by the over-corrected `ii_mm_arr` (lines 337-344), so pair ranking and the downstream II-floor gate see an II biased too low by ~the per-term signed correction on top of the telescoped term — pairs that should survive can be dropped / reordered. Fix: pass `use_mm=False` in the MM branch (keep KT branch as-is), matching `_compute_pair_ii_mm`. Regression test: construct a high-card pair, assert `_maybe_rerank_with_mm` II == `_compute_pair_ii_mm(..., use_mm=True)` II (currently they differ). |
| FE_FAMILIES_B-2 | P2 | house-convention (leftover audit metadata) | 19 sites: `cat_fe_state.py:244`, `cat_interactions.py:255`, `_cat_confirm_fwer.py:3`, `_cat_confirm_permutation.py:419,648`, `_cat_interactions_step.py:325,367,520`, `_composite_group_agg_fe.py:334,527`, `_grouped_agg_fe.py:219`, `_count_freq_interaction_fe.py:264`, `_cat_pair_fe.py:390`, `_fe_linear_explainability.py:84`, `_fe_accuracy_gate.py:45`, `_unified_fe_gate.py:69`, `_fe_synergy_exhaustive.py:132,162,389`, `_fe_synergy_screen.py:26`, `_synergy_detector.py:174` | Code comments/docstrings embed audit finding-IDs (`CAT_INTERACTION_A-9 fix:`, `FE_REDUNDANCY_SYNERGY-1 fix:`, `FE_ORCH_BUDGET-2 fix:`, `X_EFFICIENCY_ARCHITECTURE-1 fix`, `INFO_THEORY_B-4 fix`, etc.). CLAUDE.md "Comment style": no finding-IDs / audit metadata in comments — that belongs in git history. The prior ~178-file cleanup missed these because the fix commits ADDED them after the sweep. | Grep `[A-Z_]+-[0-9]+ fix` over the cluster returns these 19; each is a WHAT/WHY comment prefixed with a finding-ID that should be stripped (keep the explanatory sentence, drop the ID). |
| FE_FAMILIES_B-3 | P2 | test_gap | `_cat_mm_correction.py:201` (`_maybe_rerank_with_mm`) | No test pins `_maybe_rerank_with_mm`'s MM II against `_compute_pair_ii_mm` (the two are supposed to agree), which is exactly why FE_FAMILIES_B-1's double-correction went unnoticed. `grep _maybe_rerank_with_mm tests/` = 0 direct hits. | Add a unit test on a high-cardinality synthetic pair (n small enough the analytic gate fires) asserting the rerank II equals the standalone MM II; it fails pre-fix, passes post-fix. |
| FE_FAMILIES_B-4 | P2 | test_gap | `_cat_target_encoding_and_weighted.py:72-80` (`_compute_target_encoding` multi-class "expected class index" semantics) | Still open (was `cat_interaction_a` A-11 / prior `c7c`): the documented nominal-multi-class-mean-is-expected-class-index behaviour has no test with a 3-class nominal target in `tests/feature_selection/fe/categorical/`. The naive-leak default-safe fallback (`n_oof_folds<=0` → 2-fold OOF unless `allow_naive_leak=True`) IS present and leak-safe; the untested part is the multi-class semantics only. | Add a 3-class nominal-y fixture asserting the emitted `te_values` are the per-cell mean class index (documented) so a future refactor to per-class one-vs-rest can't silently change it. |
| FE_FAMILIES_B-5 | P3 | architecture (LOC) | `_extra_fe_families.py` (906 LOC), `_grouped_quantile_fe.py` (789), `_cat_confirm_permutation.py` (952) | Approach/exceed the ~800-900 LOC carve guideline. `_extra_fe_families.py` already carved Family D to `_extra_fe_families_dispersion.py`; Family C (RankGauss, self-contained) is the next natural cut (repeats prior `fe_redundancy_synergy` FE_REDUNDANCY_SYNERGY-5, still open). `_cat_confirm_permutation.py` already carved `_cat_confirm_fwer.py` (188 LOC) out yet is still 952. | Advisory; no behavioural impact. |

## Non-findings / confirmed-clean angles

- **Target-encoding leak-safety** (the flagged prod-gap): `kfold_target_encode_fit` (`_target_encoding_fe.py:304`)
  and `_compute_target_encoding` (`_cat_target_encoding_and_weighted.py:50`) are both leak-safe — shuffled
  round-robin fold ids (not positional `arange % K`), per-fold train sums via `full - test` (a category present
  only in the test fold gets `cnt==0` → global fallback, no self-leak), transform-time replay reads only the
  frozen per-category lookup (no `y`). `_compute_target_encoding` defaults to a real 2-fold OOF and warns unless
  `allow_naive_leak=True`. Order-stat OOF path (`per_category_order_stats` on train rows only) is also leak-safe.
- **Density families RAM discipline**: LOF (`_lof_fe.py:141` `generate_lof_features`) subsamples a bounded
  reference (`max_ref=2000`, `rng.choice` no-replace) and freezes only `(X_ref, lrd_ref, k_distance_ref, k_eff)` —
  never the whole fit frame; transform scores against the frozen reference. Mahalanobis (`_mahalanobis_density_fe.py`)
  freezes only `mu`/`Sigma_inv` (p×p). Both return all-NaN on degenerate/non-finite input rather than raising.
- **Bandit-UCB1 budget** (`_cat_confirm_bandit.py`): budget math (`total_budget`, phase-1 min_perms, phase-2 burst
  capped to remaining) is sound; seeds are explicit (`_phase1_base_seed`, per-pair/per-burst LCG offsets) so runs
  are reproducible without touching numpy global RNG; the unweighted-only limitation is now guarded upstream (A-3
  fixed — never reached with weights).
- **MM cardinality-bias correction, order-1 vs joints**: `_compute_pair_ii_mm` correctly uses OCCUPIED marginal
  cardinalities (`len(freqs[freqs>0])`) and the single telescoped `(k_a-1)(k_b-1)(k_y-1)/(2n)` — matches
  `info_theory.entropy_miller_madow`'s occupied-k convention. (The bug is only the double-application in the
  rerank twin — FE_FAMILIES_B-1.)
- **Pure-form retention CPU/GPU parity**: `_fe_pure_form_retention_gpu_resident.py` documents and implements
  selection-equivalence (same 6-fn additive basis, mean-centered OLS, same gate scalars), consults
  `gpu_globally_disabled()`, returns `None` (→ exact CPU path) on any device/import fault, and classification is
  intentionally not ported (returns `None`). The normal-equations+ridge vs sklearn-lstsq substitution is a
  documented ~1e-10 shift; no evidence it flips a gate.
- **GPU-policy gating**: both cat-FE GPU dispatch points now consult `gpu_globally_disabled()` (A-2 fixed).

## Proposals (perf / refactor / test — not bugs)

1. After fixing FE_FAMILIES_B-1, factor the shared MM-II math so `_maybe_rerank_with_mm` and `_compute_pair_ii_mm`
   call ONE primitive (plug-in entropies + single telescoped bias), eliminating the class of "two MM paths drift
   apart" bug permanently; pin them equal in a test (FE_FAMILIES_B-3).
2. Sweep the 19 FE_FAMILIES_B-2 comment sites in one grep-driven pass (strip the finding-ID prefix, keep the WHY
   sentence), matching the house "grep ALL instances, fix one pass" convention.
