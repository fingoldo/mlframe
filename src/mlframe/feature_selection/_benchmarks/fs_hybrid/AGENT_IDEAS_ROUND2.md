# Round-2 agent ideas (informed by round-1 verdicts) — tracker

Generated after round-1 (see AGENT_IDEAS.md). Each round-2 idea is grounded in a round-1 learning and avoids
the closed paths (structure-preserving shadows; interaction-via-marginal-signal; EVT thresholds; per-model
tailoring; phi-corr clustering; fidelity-gating; internal-MI warm-start). Status: TODO | TESTING | DONE-*.
Decision rule (CLAUDE.md §6): default = most accurate on the 6-scenario x multi-seed bed; speed breaks ties.

## RFECV (round-2)
- R2r-1 Variance-aware permutation n_repeats early-stop (full repeats only near the elimination cut) — TODO (S, speed on the shipped permutation default; bit-rank-preserving where settled)
- R2r-2 Cross-seed/bootstrap support_ aggregation (frequency-vote final support_ across independent fits) — TODO (M, robustness; attacks the documented cross-seed variance that R6 mis-proxied)
- R2r-3 Batched-predict / baseline-reuse permutation (bit-identical, restructure compute) — TODO (S, pure speed, zero accuracy risk)
- R2r-4 Data-shape + step-1 rank-agreement routing impurity<->permutation (replace flat 4M cap) — TODO (M)
- R2r-5 Raw-corr cluster collapse as SEARCH-SPACE reduction (rep + free slots, re-expand) — TODO (M, risk: B-extra collapse rejected)
- R2r-6 Plateau-onset n_features rule — DONE-shipped-as-OPTION (not default): implemented as n_features_selection_rule='plateau' + validators + test. Benched off cv_results_ across 6 scenarios: wins 0/6 on downstream AUC - behaves like one_se_min (same N in 4/6: base 18, manyredundant 15, weakmix 10), over-prunes the noise-robust regimes where one_se_max wins (highnoise 0.795 vs 0.814; manyredundant 0.786 vs 0.823). Agent's predicted 'collapses toward one_se_min on flat GBM tails' confirmed. Kept as an explicit parsimony-oriented option (distinct on rise-then-plateau curves), NOT the default.
- R2r-7 Early-stopped/reduced-fold interior CV with calibration revert — TODO (M, speed)
- R2r-8 Permutation memoisation across RFE steps (exact subset+seed key) — TODO (likely-disappointment: MBH revisits N not exact subsets)

## BorutaShap (round-2)
- R2b-1 Wire hetero_vote into roster + test AND/OR stack with cross-subsample stability (the open Q) — TODO (S, HIGHEST; likely ships hetero_vote as cheaper replacement for 10x stability)
- R2b-2 Cheapen hetero_vote: drop n_shadow_trials 5->1-2 (cross-MODEL is the mechanism, not cross-trial) — TODO (S, speed)
- R2b-3 Shadow-null-calibrated vote-fraction threshold (non-parametric panel-agreement null) — TODO (M, needs larger panel)
- R2b-4 CV-skill-weighted vote (downweight structurally-blind panel member: linear on monotone) — TODO (M)
- R2b-5 Drop per-trial SHAP from the Boruta loop; gini/permutation gate driver (SHAP optional/diagnostic) — TODO (M, high-certainty: R-imp showed SHAP worst+slowest)
- R2b-6 Pre-merge raw-corr clusters before the shadow gate, re-expand after accept (manyredundant) — TODO (M)
- R2b-7 Integrate panel vote INTO Boruta in-loop gate (cross-model + iterative confirmation) — TODO (L, conditional on R2b-1)
- R2b-8 OOF/drop-column importance + panel (the surviving B5 angle) — TODO (likely-disappointment: same-draw)

## ShapProxiedFS (round-2)
- R2s-1 Holdout-SPLIT-variance-calibrated parsimony band (true 1-SE on real noise, not model-seed std~0) — TODO (M, named fix for S4 confirmed root cause)
- R2s-2 Selection averaged over K holdout splits (reuse phi; vote inclusion) — TODO (M, variance reduction attacks the real enemy)
- R2s-3 Refine net-value ablation: refine=False top_n=20 vs top_n=40 vs refine=True split-calibrated — TODO (S, config-only; GATES the whole parsimony-band cluster; run FIRST)
- R2s-4 Absolute-floor parsimony guard (never drop a feature worth > delta_abs honest loss) — TODO (S)
- R2s-5 Skill-normalized selection band ((loss - random_baseline), not raw brier) — TODO (S, clamp denom on weakmix)
- R2s-6 Per-feature cross-fold phi-stability as DROP-PROTECTION (distinct from S3/S6) — TODO (M, risk: copies also stable)
- R2s-7 Cache OOF-SHAP across splits + UCB early-stop revalidation (enabler for R2s-2) — TODO (S, speed enabler)
- NOTE: R2s-1/R2s-4/R2s-5 are three formulations of ONE fix (parsimony band mis-scaled vs noise) -> bench together as a 2x2x2 calibration; R2s-3 gates the cluster.

## High-certainty first batch (cross-selector)
- R2b-5 (drop SHAP from Boruta loop) - near-certain per R-imp
- R2b-1 (wire+test hetero_vote stack) - answers open Q, cheap
- R2r-1 + R2r-3 (speed the permutation default) - pure wins
- R2s-3 (refine ablation, config-only) -> gates R2s-1 cluster
- R2r-6 (plateau-onset rule, offline-testable)
