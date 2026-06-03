# Round-3 agent ideas — tracker (5 agents: HybridSelector, RFECV, BorutaShap, ShapProxiedFS, MRMR)

Generated knowing the full round-1/2 history + the big-benchmark results. UNANIMOUS convergence: feature
engineering / interaction recovery is the dominant lever (mrmr_fe 0.835 vs best pure-selection 0.79), and it is
the #1 pick of every agent. The bed's pure-interaction operands (inf_4*inf_5, inf_6^2) have ~0 marginal, so no
SELECTION method recovers them — only FE creates the term, or an interaction-aware objective values it.
Status: TODO | TESTING | DONE-*. Decision rule (CLAUDE.md §6): most accurate first, fastest breaks ties.

## HybridSelector
- H3-1 **FE-as-shared-substrate** (TOP) — DONE-SHIPPED (huge win): implemented use_fe (MRMR member engineers, X_aug=
  [raw|engineered] built once via leakage-free recipe replay, all members + vote + transform run on X_aug). Benched
  3 seeds on make_dataset: auc_mean 0.786 -> 0.830 (+0.046), logit 0.7515 -> 0.8486 (+0.097) -- it MATCHES the best
  whole-bed strategy mrmr_fe (0.831). FE driver gini vs permutation TIED (0.8288 vs 0.8299) so gini (2x faster) is
  the default. Caveat: with FE the hybrid converges to mrmr_fe behaviour (base_recall 0.96->0.58, the engineered
  features replace raw bases; mrmr_fe makes the same trade) and costs ~2-4x mrmr_fe for the same AUC -- so FE-hybrid
  ties rather than beats mrmr_fe ON THIS BED; its distinct value (recall champion 0.96) is the use_fe=False mode.
  Shipped use_fe=True default. Note: under FE the shap member can't reuse MRMR's precomputed MI/SU (artifacts cover
  raw only, not eng_N) so it recomputes -- compute-once still holds for FI+clusters.
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
- R3-1 Post-selection pairwise-product augmentation of SURVIVORS — DONE-benched (WIN, partial): survivor-FE auc_mean
  0.7882 -> 0.8035 (+0.015) on make_dataset 3 seeds, with noise staying ~0.3. It FORMED the true inf_4*inf_5 on sd2
  (+0.011 CV gain -> 0.828) via the CV-gated product search, sidestepping the closed interaction-RANKING failure.
  LIMIT: only forms the product when BOTH operands survive RFECV's marginal elimination (sd2); when one operand is
  dropped for ~0 marginal (sd0/sd1) it forms 0 or a near-miss (inf_5*red). So it is a genuine but PARTIAL FE win for
  RFECV -- still below mrmr_fe (0.8365) / the FE-hybrid, which engineer before elimination. Worth shipping as an
  RFECVSel post-process option; the hybrid/mrmr FE already capture the bigger FE benefit.
- R3-2 one_se rule audit — DONE-benched (REJECTED): one_se_min (current, 0.7882) is the BEST rule; one_se_max gets
  recall 1.0 but admits 20.7 noise -> worst AUC 0.7735; argmax 0.7821. The adapter's forced one_se_min is correct.
- R3-3 Held-out-permutation shadow-null noise floor — DONE-doc (REJECTED, grounded): one_se_min already keeps noise
  to ~1.0; a noise floor risks the fi_guard fate (cut recall with noise). The +0.015 from R3-1 (add real interaction)
  dominates the ~0 headroom from removing the last ~1 noise col. Not worth the recall risk.
- R3-4 Stability-selection equal-N re-ranker — DONE-doc (subsumed): R2r-2 already measured bootstrap support
  aggregation AUC-NEUTRAL (stability-only). An equal-N CV-gated swap is a marginal refinement of that AUC-neutral
  lever; R3-1's product augmentation is the RFECV AUC win instead.
- R3-5 Two-resolution N grid — DONE-doc (DEFERRED, small): a curve-resolution refinement worth at most the gap
  between adjacent N inside the 1-SE band; on this bed one_se_min already wins and R3-1 captures the real AUC lever.
  Safe future micro-opt; not benched (low ceiling vs R3-1).
- R3-6 Cluster-representative elimination credit — DONE-doc (DEFERRED, promising-but-redundant-here): ports the R2b-6
  premerge win to RFECV's vote to de-dilute split FI across copies. On make_dataset RFECV already keeps noise low and
  R3-1 supplies the FE win; the premerge mechanism is already validated (R2b-6) + shipped in the hybrid, so RFECV-
  side cluster-credit is a recall lever for redundant data, worth porting if a redundant-heavy regime needs it.
- R3-7 n_stability_elbow_ as the N-rule — DONE-doc (REJECTED, grounded): R3-2 showed one_se_min beats one_se_max/
  argmax; the elbow (stability x score) collapses to ~score-argmax on this bed (stability ~1 across N on small p,
  the documented R6 finding), so it cannot beat the rule that already won. No headroom.

## BorutaShap
- B3-1 **Promote cluster-premerge into the class** (TOP; coordination-flagged): R2b-6 confirmed win (7/12, +recall,
  -noise, faster) lives only in the hybrid. Add premerge_clusters= option. Test = round2_boruta_driver bench vs class. TODO
- B3-2 Flip to held-out permutation driver — DONE-benched (REJECTED inside the hybrid): as the hybrid boruta-member
  driver it was net-negative without FE (noise 2.33->1.33 only, recall 0.96->0.88, auc 0.784->0.779, 2.5x slower) and
  merely TIED gini with FE on (0.8288 vs 0.8299) at 2x cost -> gini stays the hybrid default. (Permutation as the
  STANDALONE BorutaSel adapter default vs gini on the big bench is a separate fs_selectors-only question, still open.)
- B3-3 Enable cross-subsample stability intersection in the adapter default (noise->0; risk recall like fi_guard). TODO
- B3-4 FE-augmented Boruta cascade MRMRSel(fe=True)->BorutaSel (fs_selectors-only, safe now; close FE gap). TODO
- B3-5 Trial-progressive shadow percentile (90->99 ramp); risk = shares B4 EVT same-draw ceiling. TODO
- B3-6 Asymmetric acceptance p-value (stricter to ACCEPT, lenient to reject); risk = same-draw ceiling (B5). TODO
- (rejected by the agent itself: pruning the dead IsolationForest sample path = speed/hygiene only, no AUC.)

## ShapProxiedFS
- S3-1 Turn ON interaction_aware in the AUC callers — DONE-benched (REJECTED as default): a pure flip is a NO-OP
  because the path is gated `phi.shape[1] <= max_interaction_features` (=16) and these cells have >16 SHAP units
  post-prefilter, so it never fires (interaction_on == off in all 4 probe cells). RAISING the gate to 60 to force it
  engages an O(P^2) TreeSHAP interaction tensor that is prohibitively slow (~1.6GB, stuck minutes) on ~40 units, so
  the gate at 16 is correct and interaction_aware is NOT a viable default on wide data. It is only usable post-
  aggressive-clustering (units <=16). FE (H3-1) is the recovery mechanism that actually works here instead.
- S3-2 Auto-calibrate parsimony_tol from holdout-SPLIT variance (the named R2s-1 fix, split-std not model-seed-std). TODO
- S3-3 Interaction-augmented prefilter so operands survive to the SHAP stage (prereq for S3-1; check report.prefilter). TODO
- S3-4 Raise n_revalidation_models 2->5 with adaptive early-stop (stabler winner -> recover dropped base). TODO
- S3-5 FE-augment candidate columns (pairwise products of top-SHAP units before search). TODO
- S3-6 Paired-seed greedy restart from top-|Phi_ik| pairs inside the interaction objective. TODO
- S3-7 Interaction-mass corrector feature (HIGH RISK: S2 measured corr +0.49 -> may be pre-refuted; test corr first). TODO

## MRMR (new in scope)
- M3-1 Relevance-conditioned DCD pruning — DONE-diagnostic (REFUTED): the cheap diagnostic settled it -- dcd_enable=
  False AND cluster_aggregate_enable=False give IDENTICAL base_recall (0.667) to default on make_dataset 3 seeds. So
  DCD pruning / cluster-aggregate are NOT the cause of MRMR's low base_recall; the recall loss is the greedy mRMR
  selection / FE re-selection preferring engineered features over raw bases (same trade mrmr_fe makes for AUC). No fix.
- M3-2 Turn the rescue ON — DONE-benched (REJECTED): run_additional_rfecv_minutes=0.5 recovers ALL raw bases
  (recall 0.667->1.0) but readmits ALL 32 noise columns and LOWERS auc (0.8314->0.8310); also raised a ValueError on
  one seed (engineered name 'add(prewarp(red_0_1),...)' not in list -- a real rescue+FE-name bug, in concurrent MRMR
  territory -> flag to owner). The engineered features already carry the signal, so raw-base recovery doesn't help AUC.
- M3-3 Richer FE preset/budget — DONE-benched (REJECTED, no effect): fe_max_pair_features 10->25 gave IDENTICAL
  selection/AUC on all 3 seeds (the pair budget is not the binding constraint here).
- M3-4 FI-guided synergy bootstrap — SUBSUMED by M3-7 (fe_strict): the spurious noise-product problem the probe
  exposed is fixed more directly by tightening the prevalence gates (M3-7) than by re-seeding the pool. The external-
  FI seeding remains a future option for harder data; not needed once fe_strict cuts the spurious products.
- M3-5 JMIM aggregator — DEFERRED (param not top-level): use_jmim is not an exposed MRMR ctor kwarg (internal); needs
  the MRMR owner to surface it. Risk noted (joint-MI readmits redundant copies). Not benchable via kwargs; coordinate.
- M3-6 Calibrate min_relevance_gain_relative_to_first — DONE-benched (REJECTED): 0.05->0.02 gave ~no AUC change
  (0.8314->0.8326) while EXPLODING the set (n 24->34, engineered 15->25, spurious 1->6.7). Net-negative (more spurious
  for nil AUC). Keep 0.05.
- M3-7 Tighter FE prevalence gates ("fe_strict") — DONE-SHIPPED (WIN): fe_synergy_min_prevalence 1.15->1.5 +
  fe_min_engineered_mi_prevalence 0.90->0.97. Measured on make_dataset 3 seeds: standalone mrmr_fe auc 0.8314->0.8365
  (+0.005, positive every seed), HALVES the engineered set (15->6.7) and cuts spurious noise-products (1.0->0.67).
  Applied in the fs_hybrid MRMRSel wrapper + the HybridSelector's MRMR member (use_fe). RECOMMEND flipping the MRMR
  CLASS defaults too -- coordinate with the MRMR-owning session (concurrent file).

## Cross-cutting reads
- FE / interaction recovery is the #1 lever for ALL five. The cheapest highest-value first moves: S3-1 (one-line flip),
  M3-1 diagnostic (does DCD drop true bases?), R3-1 survivor-product test (are both operands in survivors?), then the
  big build H3-1 (hybrid FE). B3-2/B3-4 are fs_selectors-only and safe to bench without touching concurrent classes.
- Coordination flags: B3-1/B3-5/B3-6 touch boruta_shap.py; M3-1/M3-5 touch the MRMR core; both files may be concurrently
  edited -> diagnostic/bench-via-kwargs first, edit only after a lock check.
