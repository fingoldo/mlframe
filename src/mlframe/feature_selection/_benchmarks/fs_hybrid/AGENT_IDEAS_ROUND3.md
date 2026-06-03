# Round-3 agent ideas — tracker (5 agents: HybridSelector, RFECV, BorutaShap, ShapProxiedFS, MRMR)

Generated knowing the full round-1/2 history + the big-benchmark results. UNANIMOUS convergence: feature
engineering / interaction recovery is the dominant lever (mrmr_fe 0.835 vs best pure-selection 0.79), and it is
the #1 pick of every agent. The bed's pure-interaction operands (inf_4*inf_5, inf_6^2) have ~0 marginal, so no
SELECTION method recovers them — only FE creates the term, or an interaction-aware objective values it.
Status: TODO | TESTING | DONE-*. Decision rule (CLAUDE.md §6): most accurate first, fastest breaks ties.

## HybridSelector
- H3-1 **FE-as-shared-substrate** (TOP): run the MRMR member with fe_max_steps=1, build X_aug=[raw|engineered] once
  (MRMR.transform replays recipes leakage-free), run shap/boruta/vote on X_aug. Expect auc_mean 0.786 -> ~0.83
  while keeping recall 1.0. Test: hybrid fe on vs off on make_dataset 3 seeds; KILL if auc<0.81 or recall<0.90. TODO
- H3-2 Per-member realized-subset OOF-AUC vote weight (NOT the rejected standalone-skill weight): weight each
  member's vote by its selected subset's honest OOF AUC (mrmr_fe~0.835 vs shap~0.75 provably differ). Test: <2/12
  cells change -> KILL. TODO
- H3-3 FE-protective cluster rep: never collapse an engineered term into its raw operand at rep-selection. TODO
- H3-4 Add RFECV as a 4th member on X_aug (measured best pure-selection AUC, currently absent). TODO
- H3-5 Honest internal-validation auto-picker for (vote,expand) per dataset. TODO
- H3-6 Decouple vote-unit from emitted-feature in FE-aug clusters (emit engineered, vote on raw). TODO
- H3-7 Prescreen must whitelist engineered columns (correctness prereq for H3-1). TODO
- H3-8 Recompute shared permutation-FI on X_aug (honest FI for engineered cols; subsumes H3-7/part of H3-3). TODO

## RFECV
- R3-1 **Post-selection pairwise-product augmentation of SURVIVORS** (TOP): after MBH picks support_, form pairwise
  products/squares of top survivors, keep a product only if CV score rises >1 SE. Avoids the closed interaction-
  RANKING failure (operates on the ~13 survivors where both operands usually present, gated by honest CV). Test on
  xor2/base: is true inf_a*inf_b among top CV-improving survivor-products? KILL if survivors lack both operands. TODO
- R3-2 Audit adapter's forced one_se_min vs library one_se_max on the 3-model auc_mean (recall recovery). TODO
- R3-3 Held-out-permutation shadow-null noise floor on final support (precision; risk = fi_guard fate). TODO
- R3-4 Stability-selection equal-N re-ranker (CV-gated swaps, not N-reselection). TODO
- R3-5 Two-resolution N grid: densely eval integer N inside the 1-SE band (curve resolution, not a new rule). TODO
- R3-6 Cluster-representative elimination credit (de-dilute split FI across copies; ports R2b-6 win to RFECV vote). TODO
- R3-7 Bench n_stability_elbow_ (stability x score) as the N-rule (cheapest; likely no-op if stability~1 on small p). TODO

## BorutaShap
- B3-1 **Promote cluster-premerge into the class** (TOP; coordination-flagged): R2b-6 confirmed win (7/12, +recall,
  -noise, faster) lives only in the hybrid. Add premerge_clusters= option. Test = round2_boruta_driver bench vs class. TODO
- B3-2 Flip BorutaSel adapter default to held-out permutation driver (noise 1.3->~0.3; fs_selectors-only, safe now). TODO
- B3-3 Enable cross-subsample stability intersection in the adapter default (noise->0; risk recall like fi_guard). TODO
- B3-4 FE-augmented Boruta cascade MRMRSel(fe=True)->BorutaSel (fs_selectors-only, safe now; close FE gap). TODO
- B3-5 Trial-progressive shadow percentile (90->99 ramp); risk = shares B4 EVT same-draw ceiling. TODO
- B3-6 Asymmetric acceptance p-value (stricter to ACCEPT, lenient to reject); risk = same-draw ceiling (B5). TODO
- (rejected by the agent itself: pruning the dead IsolationForest sample path = speed/hygiene only, no AUC.)

## ShapProxiedFS
- S3-1 **Turn ON interaction_aware in the AUC callers** (TOP; one-line, already-wired path, never switched on):
  additivity coalition base+sum Phi_ik values the pure-interaction operands the additive proxy misses (the reason
  recall stalls at 5/8). NOT the closed interaction-RANKING (S2/B3/R1) -- it is the coalition OBJECTIVE feeding honest
  revalidation. Test base/xor2: {inf_4,inf_5} recovered + AUC up; KILL if absent or reval discards. TODO
- S3-2 Auto-calibrate parsimony_tol from holdout-SPLIT variance (the named R2s-1 fix, split-std not model-seed-std). TODO
- S3-3 Interaction-augmented prefilter so operands survive to the SHAP stage (prereq for S3-1; check report.prefilter). TODO
- S3-4 Raise n_revalidation_models 2->5 with adaptive early-stop (stabler winner -> recover dropped base). TODO
- S3-5 FE-augment candidate columns (pairwise products of top-SHAP units before search). TODO
- S3-6 Paired-seed greedy restart from top-|Phi_ik| pairs inside the interaction objective. TODO
- S3-7 Interaction-mass corrector feature (HIGH RISK: S2 measured corr +0.49 -> may be pre-refuted; test corr first). TODO

## MRMR (new in scope)
- M3-1 **Relevance-conditioned DCD pruning** (TOP; diagnostic-first): DCD prunes by unsupervised SU>tau, may drop a
  true base with distinct conditional signal. Add CMIM guard: prune only if SU>tau AND I(c;y|anchor)<eps. Diagnostic:
  log (member,SU,I(member;y|anchor)) on make_dataset; KILL if only red_* copies pruned (recall loss is elsewhere). TODO
- M3-2 Turn the run_additional_rfecv rescue ON for FE config + let it re-include DCD-dropped true bases. TODO
- M3-3 Richer FE preset/budget (minimal->medium, pairs 10->25) calibration. TODO
- M3-4 FI-guided synergy bootstrap: seed FE pair-pool with external permutation-FI/shadow (the open R8 external form). TODO
- M3-5 JMIM aggregator for the post-FE re-selection (preserve synergy operands; risk = readmit copies). TODO
- M3-6 Calibrate min_relevance_gain_relative_to_first (0.05->0.02) recall dial (scope per-caller like parsimony_tol). TODO
- M3-7 Finer quantization / Miller-Madow on the FE pair joint-MI gate (de-bias the 2-D baseline). TODO

## Cross-cutting reads
- FE / interaction recovery is the #1 lever for ALL five. The cheapest highest-value first moves: S3-1 (one-line flip),
  M3-1 diagnostic (does DCD drop true bases?), R3-1 survivor-product test (are both operands in survivors?), then the
  big build H3-1 (hybrid FE). B3-2/B3-4 are fs_selectors-only and safe to bench without touching concurrent classes.
- Coordination flags: B3-1/B3-5/B3-6 touch boruta_shap.py; M3-1/M3-5 touch the MRMR core; both files may be concurrently
  edited -> diagnostic/bench-via-kwargs first, edit only after a lock check.
