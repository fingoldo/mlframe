# gt_01: Faith-Shap / Shapley-Taylor interaction indices for ShapProxiedFS

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Problem & motivation

ShapProxiedFS's `proxy_mode="interaction"` re-scores candidate subsets under
`base + Σφ_j + 2·Σ_{i<j∈S} Φ_ij` where Φ is the raw TreeSHAP interaction tensor
(`compute_interaction_tensor`, `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_interactions.py:129`).
Two known weaknesses:
1. The TreeSHAP interaction tensor is O(P²) memory/compute and is hard-gated to P≤16 proxy columns
   (`max_interaction_features`), making it a no-op on wide frames.
2. The tensor's off-diagonal terms are the Shapley interaction index of Lundberg's construction,
   which is NOT "faithful": summed reconstructions of v(S) from {φ_j, Φ_ij} do not minimize any
   weighted reconstruction error, and higher-order effects leak into pairwise terms unpredictably.

Faith-Shap (Tsai, Yeh, Ravikumar, JMLR 2023, "Faith-Shap: The Faithful Shapley Interaction Index")
derives THE unique interaction index that (a) satisfies the natural extensions of the Shapley
axioms (linearity, symmetry, dummy, efficiency) and (b) is the best l-order polynomial
approximation of the game v under the Shapley kernel weighting — i.e., the interaction analogue of
"SHAP = weighted-least-squares projection". For order l=2 it gives coefficients
{a_∅, a_{i}, a_{ij}} minimizing Σ_S μ(S)·(v(S) − a_∅ − Σ_{i∈S} a_i − Σ_{ij⊆S} a_ij)² with the
Shapley kernel μ(S) ∝ (P−1) / (C(P,|S|)·|S|·(P−|S|)); the minimizer can be estimated by weighted
linear regression over SAMPLED coalitions — no tensor, no O(P²) blowup, cost is
O(n_coalitions · (1 + k + k²_candidate_pairs)) where the pairwise design is restricted to
candidate pairs only.

Key fit with ShapProxiedFS: **v(S) is the proxy loss** (`subset_loss`,
`src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_objective.py`) — evaluating a coalition
costs one O(n) vector reduction, so thousands of sampled coalitions are cheap. This makes Faith-Shap
estimable directly over the proxy game, whereas TreeSHAP interactions require the tree tensor.

## 2. Integration verdict

**Config extension of ShapProxiedFS** — new `proxy_mode="faith_interaction"` alongside
`"additive"`/`"interaction"`/(gt_08's) `"auto"`. NOT a new selector: the pipeline (prefilter, OOF-SHAP,
prescreen, search, trust guard, revalidation, refine) is unchanged; only the candidate-augmentation
stage gains a new scorer. If gt_08 has landed, `"auto"` MAY later route to faith_interaction instead
of the su_seeded sparse path — out of scope here; keep them parallel options.

## 3. Existing machinery to reuse (verified paths)

- `su_synergy_screen` (`_shap_proxy_interactions.py:342`) — O(P)+O(min(P,120)²) synergy screen with
  permutation-null SNR gate; returns candidate pairs. Reuse VERBATIM to pick the pairwise design
  columns (candidate pairs), keeping the regression design k + |pairs| wide, never k².
- `subset_loss(phi, base, y, idx_list, metric)` (`_shap_proxy_objective.py`) — the game oracle.
  For coalition sampling loops prefer the memoised `_Evaluator` from `_shap_proxy_heuristics.py`
  (`loss(idx)`) which caches by sorted tuple and uses the contiguous-transpose fast path.
- Candidate augmentation seam: `augment_candidates_with_interactions`
  (`_shap_proxied_fit_interactions.py:121`) — add a third block, mirroring the
  `proxy_mode == "interaction"` block at `:173-200` (its structure: compute scorer → generate
  candidates → merge into `merged` dict keyed by sorted tuple, keep min loss → re-sort; report dict).
- Constructor/validator: facade `__init__.py`, `proxy_mode` validation currently
  `("additive", "interaction")` at `:305` — extend.

## 4. Implementation steps

1. **New module** `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_faith_interactions.py`
   (keep <900 LOC; it should land well under 400):

   ```python
   def faith_shap_order2(
       evaluator,                      # _Evaluator over (phi, base, y, metric)
       n_features: int,
       candidate_pairs: list[tuple[int, int]],   # from su_synergy_screen, proxy-index space
       *, n_coalitions: int = 2048, rng: np.random.Generator,
       ridge: float = 1e-6,
   ) -> tuple[np.ndarray, dict[tuple[int, int], float], dict]:
       """Weighted ridge regression estimate of order-2 Faith-Shap coefficients.

       Returns (a_lin (n_features,), a_pair {(i,j)->coef}, info). Design matrix per sampled
       coalition S: [1 | 1{j in S} for all j | 1{i in S and j in S} for candidate pairs].
       Weights: Shapley kernel mu(|S|); sample |S| from the kernel-normalized size distribution,
       then uniform subsets of that size (standard KernelSHAP sampling); always include S=∅ and
       S=full with large finite weight (the standard 1e6 surrogate for the infinite-weight
       constraints). Solve weighted ridge via np.linalg.lstsq on the sqrt-weighted system.
       v(S) = -evaluator.loss(S)  (negate: game value = goodness; loss is lower-better).
       """
   ```

   ```python
   def faith_interaction_top_n(
       phi, base, y, *, classification, metric, candidate_pairs,
       min_card, max_card, top_n, n_coalitions, rng,
   ) -> list[tuple[float, tuple[int, ...]]]:
       """Rank subsets by the order-2 Faith-Shap surrogate game
       v̂(S) = a_∅ + Σ_{j∈S} a_j + Σ_{(i,j)⊆S} a_ij  — i.e. run the SAME search heuristics
       (beam over the surrogate is fine: build a surrogate-phi matrix trick is NOT possible since
       pairwise terms are not per-row; instead enumerate candidates greedily over the surrogate
       closed form, which is cheap: v̂ evaluation is O(k + |pairs∩S²|)). Return top_n candidates
       as (surrogate_loss, idx_tuple) with surrogate_loss = -v̂(S) to keep lower-is-better."""
   ```

2. **Wire into `augment_candidates_with_interactions`**: new block
   `if str(getattr(self, "proxy_mode", "additive")).lower() == "faith_interaction" and phi.shape[1] >= 2:`
   — run `su_synergy_screen` on X_proxy (reuse the resolve_su_seeded_pairs plumbing so the screen
   runs at most once per fit), call `faith_shap_order2` + `faith_interaction_top_n`, merge
   candidates (keep-min-loss rule; note surrogate losses and raw proxy losses are on the same
   scale — both are the proxy loss metric — because the surrogate regresses v = −loss).
   Report: `report["faith_interaction"] = dict(applied, n_pairs, n_coalitions, r2_of_fit,
   top_pairs=[(i_name, j_name, a_ij) first 10])`. `r2_of_fit` (weighted R² of the surrogate
   regression) is the key diagnostic: low R² means order-2 is insufficient — surface it.
3. **Constructor**: extend validator to include `"faith_interaction"`; new params
   `faith_n_coalitions: int = 2048`, reuse `interaction_proxy_top_k` naming conventions.
   Default `proxy_mode` stays whatever gt_08 set (do not fight over the default here — this mode
   is opt-in until its own bench).
4. **cProfile harness** `_benchmarks/profile_faith_interactions.py`: wall vs n_coalitions
   {512, 2048, 8192} at proxy widths {28, 112}; the regression solve is tiny — the loop cost is
   n_coalitions × O(n) subset_loss reductions; confirm ≤2s at n=3000, width 112, 2048 coalitions.

## 5. biz_val tests

File: `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_faith_interaction.py`.
Beds: reuse the generators in
`src/mlframe/feature_selection/_benchmarks/bench_shap_interaction_proxy.py` (competing-XOR,
additive-redundant) + one NEW saddle bed `y = x_a*x_b - x_c*x_d + additive terms` (tests two
simultaneous interactions).

1. `test_biz_val_faith_interaction_beats_additive_on_xor` — competing-XOR bed: XOR operand recall
   2/2 (additive: 0/2), downstream-AUC ≥ additive + 0.05.
2. `test_biz_val_faith_interaction_not_worse_than_treeshap_mode` — same bed:
   downstream-AUC(faith) ≥ AUC(proxy_mode="interaction") − 0.01, AND faith works at proxy width
   112 where the tensor mode no-ops (assert `report["proxy_mode_interaction"]` absent/not-applied
   but `report["faith_interaction"]["applied"]` true).
3. `test_biz_val_faith_interaction_additive_bed_no_regression` — pure additive bed:
   selected_features_ Jaccard ≥ 0.9 vs additive mode, no noise columns added.
4. `test_biz_val_faith_interaction_saddle_two_pairs` — saddle bed: both pairs' operands recalled
   (4/4) with faith vs ≤2/4 with additive.

Unit tests: `faith_shap_order2` on a HAND-CONSTRUCTED game (3 features, v defined by an explicit
table with one known interaction) recovers a_ij within 1e-2 of the analytic Faith-Shap value
(compute analytic value by exact weighted least squares over all 2³ coalitions — feasible closed
form for the test); efficiency check Σa_j + Σa_ij ≈ v(full) − v(∅); dummy feature gets |a| < 1e-6.

## 6. Acceptance criteria
- New module + wiring + params, mypy-clean, all unit/biz_val tests green locally
  (`CUDA_VISIBLE_DEVICES="" ... --no-cov -p no:anyio`).
- Analytic-recovery unit test proves estimator correctness (this is the core scientific claim —
  do not skip).
- Bench table (4 beds × 3 seeds × {additive, interaction, faith_interaction}) committed under
  `_benchmarks/`; keep mode opt-in unless it majority-wins.
- Existing shap_proxied suite green (3-batch recipe if needed).

## 7. Known risks / rejected alternatives
- Shapley-Taylor index: rejected as the primary target — it is not faithful (no least-squares
  optimality) and needs discrete-derivative estimation per pair; mention in the doc's theory
  section but implement Faith-Shap.
- Full k² pairwise design (all pairs, not screened candidates): rejected — design width explodes
  at proxy width 112 (6216 pair columns vs 2048 samples → underdetermined). The su-screen
  restriction is what makes this tractable; document that missed pairs (screen false negatives)
  degrade gracefully to the additive treatment.
- Sampling variance: n_coalitions=2048 default chosen for width≤112; add
  `info["coef_stderr_max"]` (from the ridge covariance diagonal) so callers can detect
  under-sampling; bench it before hardening the default.
