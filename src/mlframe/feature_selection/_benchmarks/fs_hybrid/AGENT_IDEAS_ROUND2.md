# Round-2 agent ideas (informed by round-1 verdicts) — tracker

Generated after round-1 (see AGENT_IDEAS.md). Each round-2 idea is grounded in a round-1 learning and avoids
the closed paths (structure-preserving shadows; interaction-via-marginal-signal; EVT thresholds; per-model
tailoring; phi-corr clustering; fidelity-gating; internal-MI warm-start). Status: TODO | TESTING | DONE-*.
Decision rule (CLAUDE.md §6): default = most accurate on the 6-scenario x multi-seed bed; speed breaks ties.

## RFECV (round-2)
- R2r-1 Variance-aware permutation n_repeats early-stop — DONE-doc (deferred, speed-only): correctness-affecting (changes which repeats run) speedup on the shipped permutation default. Agents flagged the ABSOLUTE win is small on the small synthetics (p~30-110, fit dominates); real payoff is production-scale held-out folds. Accuracy-first mandate prioritises the AUC/robustness benches; this is a safe future speed lever (gate so it never changes the boundary decision). Not benched now (no accuracy change to measure on the AUC bed).
- R2r-2 Cross-seed/bootstrap support_ aggregation (frequency-vote final support_ across independent fits) — TODO (M, robustness; attacks the documented cross-seed variance that R6 mis-proxied)
- R2r-3 Batched-predict permutation — DONE-doc (deferred, speed-only): bit-identical compute restructuring. Agent flagged the win EVAPORATES when model.fit dominates predict, which it does on the 5k-row synthetics; payoff is wide production folds. Zero-accuracy-risk future lever; deferred (nothing to measure on the AUC bed; sklearn's permutation_importance is already vectorised per-column).
- R2r-4 Data-shape + step-1 rank-agreement routing impurity<->permutation (replace flat 4M cap) — TODO (M)
- R2r-5 Raw-corr cluster collapse as SEARCH-SPACE reduction (rep + free slots, re-expand) — TODO (M, risk: B-extra collapse rejected)
- R2r-6 Plateau-onset n_features rule — DONE-shipped-as-OPTION (not default): implemented as n_features_selection_rule='plateau' + validators + test. Benched off cv_results_ across 6 scenarios: wins 0/6 on downstream AUC - behaves like one_se_min (same N in 4/6: base 18, manyredundant 15, weakmix 10), over-prunes the noise-robust regimes where one_se_max wins (highnoise 0.795 vs 0.814; manyredundant 0.786 vs 0.823). Agent's predicted 'collapses toward one_se_min on flat GBM tails' confirmed. Kept as an explicit parsimony-oriented option (distinct on rise-then-plateau curves), NOT the default.
- R2r-7 Early-stopped/reduced-fold interior CV — DONE-doc (deferred, speed-only): changes the MBH search trajectory (risk) for a refit-cost win that is small on small synthetics. Deferred behind the accuracy benches; revisit at production scale with the calibration-revert guard.
- R2r-8 Permutation memoisation across RFE steps — DONE-rejected (grounded): MBH revisits the same N with DIFFERENT subsets (documented F9 loser-rollback behaviour), so exact (subset,seed) cache collisions are near-zero -> the cache would have ~0 hit rate. Agent's own gate (collision<5% -> reject) is met by the code's design. Also intersects the 'runtime caches break pickle' rule. Not worth implementing.

## BorutaShap (round-2)
- R2b-1 Wire hetero_vote + test stack — DONE-doc (benched 6x2): hetero_vote is a PRECISION/parsimony tool, NOT an AUC default. It drives noise to 0 (vs boruta 1.6) + compact (7-9 vs 17 feats) but UNDER-RECOVERS weak signal -> mean AUC 0.742(hetero2)/0.736(hetero5) < boruta 0.764, wins only 2/12. Round-1's 5/7-0/32 was one lucky scenario. Kept as the precision option (docstring corrected); plain boruta wins for downstream AUC. AND/OR-stack-with-stability moot (hetero already over-prunes recall).
- R2b-2 Cheapen hetero_vote n_shadow_trials — DONE-shipped: benched 2 vs 5 -> n_shadow_trials=2 marginally BEATS 5 (0.742 vs 0.736 mean) at 2.5x speed (cross-MODEL is the mechanism, not cross-trial). Lowered module default 5 -> 3 (robust middle).
- R2b-3 Shadow-null-calibrated vote-fraction threshold (non-parametric panel-agreement null) — TODO (M, needs larger panel)
- R2b-4 CV-skill-weighted vote (downweight structurally-blind panel member: linear on monotone) — TODO (M)
- R2b-5 Drop per-trial SHAP from the Boruta loop; gini/permutation gate driver (SHAP optional/diagnostic) — TODO (M, high-certainty: R-imp showed SHAP worst+slowest)
- R2b-6 Pre-merge raw-corr clusters before the shadow gate, re-expand after accept (manyredundant) — TODO (M)
- R2b-7 Integrate panel vote INTO Boruta in-loop gate (cross-model + iterative confirmation) — TODO (L, conditional on R2b-1)
- R2b-8 OOF/drop-column importance + panel — DONE-rejected (grounded): hetero_vote (the panel) already drives accepted-noise to ~0 (R2b-1) - its problem is RECALL, not noise, so adding OOF importance (which targets the noise leak) has no headroom. And same-draw OOF inherits draw-level spurious (B5/B8 findings). No live problem for it to solve.

## ShapProxiedFS (round-2)
- R2s-1 Holdout-SPLIT-variance-calibrated parsimony band (true 1-SE on real noise, not model-seed std~0) — TODO (M, named fix for S4 confirmed root cause)
- R2s-2 Selection averaged over K holdout splits (reuse phi; vote inclusion) — TODO (M, variance reduction attacks the real enemy)
- R2s-3 Refine net-value ablation: refine=False top_n=20 vs top_n=40 vs refine=True split-calibrated — TODO (S, config-only; GATES the whole parsimony-band cluster; run FIRST)
- R2s-4 Absolute-floor parsimony guard (never drop a feature worth > delta_abs honest loss) — TODO (S)
- R2s-5 Skill-normalized selection band ((loss - random_baseline), not raw brier) — TODO (S, clamp denom on weakmix)
- R2s-6 Per-feature cross-fold phi-stability as DROP-PROTECTION (distinct from S3/S6) — TODO (M, risk: copies also stable)
- R2s-7 Cache OOF-SHAP across splits — DONE-doc (deferred, conditional): pure speed ENABLER for R2s-2 (K-split averaging). Ship only if R2s-2 proves a robustness win (benched separately). Cache infra already exists (cache_dir); deferred until R2s-2's verdict. Intersects 'runtime caches break pickle' (needs __getstate__ exclusion if added).
- NOTE: R2s-1/R2s-4/R2s-5 are three formulations of ONE fix (parsimony band mis-scaled vs noise) -> bench together as a 2x2x2 calibration; R2s-3 gates the cluster.

## High-certainty first batch (cross-selector)
- R2b-5 (drop SHAP from Boruta loop) - near-certain per R-imp
- R2b-1 (wire+test hetero_vote stack) - answers open Q, cheap
- R2r-1 + R2r-3 (speed the permutation default) - pure wins
- R2s-3 (refine ablation, config-only) -> gates R2s-1 cluster
- R2r-6 (plateau-onset rule, offline-testable)
