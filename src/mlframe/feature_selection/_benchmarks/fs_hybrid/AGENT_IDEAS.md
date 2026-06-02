# Agent-brainstorm FS improvement ideas — implementation/test tracker

Status legend: TODO | TESTING | DONE-shipped (win, default/opt-in) | DONE-rejected (tested, did not help) | DONE-doc (documented finding)
Every idea must reach a terminal status (shipped / rejected / doc) — the final STRICT CHECK verifies none is left TODO/TESTING.
Decision rule (CLAUDE.md §6): default = most accurate on the wide bench (6 scenarios x seeds); speed breaks ties.

## RFECV
- R1 SHAP-interaction / H-stat synergy bundling (both-or-neither at search time) — TODO
- R2 Paired re-add rescue pass at finalize (2-in-as-unit) — TODO
- R3 OOF permutation as default tree importance — DONE-shipped-as-DEFAULT: shootout (6 scen x 2 seeds) permutation wins 10/12 lgbm cells, best mean AUC 0.7954, cleanest (2.5 vs 6.2 noise). 'auto' now routes to permutation when held-out target present + cells<=4M (cost gate), else impurity. SHAP tested = WORST (0.786) + slowest (153s) -> not default.
- R4 Auto conditional-permutation on correlated clusters — DONE-rejected (wide bench would-be; base: hurts, keeps 5 noise, lower AUC)
- R5 Per-downstream-model-class selection tailoring — TODO
- R6 Stability-gated acquisition (penalize unstable N in MBH target) — TODO
- R7 Multi-objective Pareto front (AUC, N, stability) + pareto_front_ — DONE-doc: shipped pareto_front_/pareto_knee_ as READ-ONLY DIAGNOSTICS + test. Benched as a SELECTION RULE -> REJECTED: knee lost 0/6 scenarios on downstream AUC (over-prunes to ~4-8 feats); argmax/one_se_max won 3/3 (validates current recall default). Knee is a parsimony-only diagnostic, not an accuracy default.
- R8 Warm-start MBH surrogate with MI/interaction ranking; MI computed OR injected externally (e.g. MRMR) — TODO (user favorite)
- R-imp Importance shootout impurity/permutation/shap across scenarios -> set best default + profile — TESTING (be00hj5j4)

## BorutaShap
- B1 Model-X knockoff shadows (preserve covariance) — TODO
- B2 Per-cluster / conditional permutation shadows — TODO
- B3 SHAP interaction-value importance (catch inf_4) — TODO
- B4 Gumbel/EVT shadow-max threshold — TODO (agent flagged likely-partial)
- B5 Independent-draw generalization gate — TODO
- B6 SPRT per-feature early-stop — TODO (note: coarse early-term when no tentatives already DONE-shipped 5dca23ea)
- B7 Heterogeneous base-model importance ensemble — TODO (agent flagged high-effort/uncertain)
- B8 Permutation / drop-column (OOB) importance metric — TODO
- B-extra cross-subsample stability gate (intersection) — DONE-shipped c4193af0 (opt-in)
- B-extra sklearn clone-ability — DONE-shipped fdd8028c
- B-extra redundancy-collapse post-accept — DONE-rejected (drops clean signal)

## ShapProxiedFS
- S1 Downstream-model honest loss (separate honest model from proxy model) — TODO (model-param proxy = high-variance wash; do proper separate-model version)
- S2 Interaction-deficit as bias-corrector regression feature — DONE-rejected (cheap-test falsified, s2_interaction_deficit_test.py): agent predicted corr(interaction_mass, honest-proxy) NEGATIVE (proxy under-credits synergy); measured POSITIVE +0.49 xor2 / +0.33 base. TreeSHAP folds interaction into MAIN-effect phi, so the additive proxy already absorbs synergy and OVER-credits interaction subsets -> the "down-correct to recover synergy" premise is backwards for a TreeSHAP proxy. (User favorite, but the falsifiable test the user liked did its job.)
- S3 Fidelity-adaptive search depth / refine aggressiveness — TODO
- S4 Variance-calibrated parsimony_tol (from holdout-loss std) — DONE-doc: cheap-test CONFIRMS premise — fixed parsimony_tol=0.02 exceeds the drop-one brier contribution of 6/7 informative features (median ~0.013 < 0.02) on base + manyredundant, so refine over-prunes real signal. Documented on the param as the over-pruning dial (lower to ~0.005 or refine=False); the shipped within_cluster_refine=False (335ec1e4) already remedies it in the benchmark. Full variance-calibration left as a future refinement (LightGBM holdout brier is near-deterministic here, so a z*std tol would collapse to ~0 = keep-all; needs the holdout-SPLIT std, not model-seed std).
- S5 Membership->loss surrogate GBM on trust-guard anchors — TODO (agent flagged sample-starved)
- S6 SHAP-vector (attribution) clustering instead of raw-corr — TODO
- S7 Permutation-shortcut refine (speed) — TODO (agent flagged may worsen over-pruning)
- S8 Pair-aware anchor sampling — TODO
- S-extra within_cluster_refine over-pruning caveat — DONE-doc 335ec1e4 (benchmark config refine off)
