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
- H3-2 Per-member realized-subset OOF-AUC vote weight — DONE-BENCHED (no win): round3_hybrid_refine_bench.py (3
  seeds, subclass override). delta vs default +0.0016 -- within cross-seed noise (~0.01) and inconsistent (identical
  to default on sd0/sd1, only sd2 +0.005). Genuine mechanism, but FE already put the hybrid at the ceiling so the
  vote-weighting has no real headroom. Not shipped (equal accuracy -> keep the simpler default).
- H3-3 FE-protective cluster rep — DONE-BENCHED (REJECTED): delta -0.0005 (identical on 2/3 seeds, worse on sd1).
  The FI-rep rule already emits the high-FI engineered term, so forcing it changes nothing or slightly hurts. No win.
- H3-4 Add RFECV as a 4th member — DONE-BENCHED (REJECTED): delta -0.0001 (no change). RFECV's picks are subsumed by
  the existing members on the FE-augmented space; adding it (slowest member) buys nothing on this bed.
- H3-5 Honest internal auto-picker for (vote,expand) — DONE-BENCHED (no win): delta +0.0020 -- within noise and
  inconsistent (helps sd0 by picking expand, neutral/worse sd1/sd2). The fixed vote=1 default is within noise of the
  auto-picker, so the auto-selection layer + its internal-split-overfit risk is not worth it.
- H3-6 Decouple vote-unit from emitted-feature — DONE-BENCHED (REJECTED): delta -0.0005 (same as H3-3). No win.
- NOTE (H3-2..6): all five MEASURED within +/-0.002 of the FE-hybrid default (< cross-seed noise), so the combine
  rule has no headroom once FE is on -- the earlier "low headroom" call is now a measurement, not a hypothesis.
- H3-7 Prescreen whitelist engineered columns — DONE-SHIPPED (built into H3-1): the prescreen unions engineered cols
  so they are never dropped by the raw-FI gate before voting.
- H3-8 Recompute shared permutation-FI on X_aug — DONE-SHIPPED (built into H3-1): the shared FI pass runs on the
  augmented frame, so engineered cols get honest importances for prescreen + cluster tie-break + rep selection.

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
- R3-3 Held-out-permutation shadow-null noise floor — DONE-BENCHED (REJECTED): round3_rfecv_levers_bench.py (3 seeds)
  delta -0.0067 vs baseline; it cut base recall (7/8 -> 5/8), the fi_guard fate, now MEASURED not just predicted.
- R3-4 Stability-selection equal-N re-ranker — DONE-BENCHED (REJECTED): delta -0.0090 (worst of the levers); the
  bootstrap-frequency top-N diverges from the CV-optimal support and hurts. Confirms R2r-2's stability-not-AUC class.
- R3-5 Two-resolution N grid — DONE-BENCHED (no win): delta -0.0014 (within noise) -- re-evaluating every N in the
  band lands essentially the same support; one_se_min's N is already near the local CV-argmax. No headroom.
- R3-6 Cluster-representative elimination credit (= R2r-5 corr-collapse) — DONE-BENCHED (REJECTED): delta -0.0024;
  collapsing raw-corr clusters before RFECV's vote and re-expanding does NOT help -- RFECV already eliminates the
  redundant copies via CV-ranked backward elimination, so de-diluting the vote credit changes nothing positive here
  (unlike the Boruta gate R2b-6, where premerge did win). Measured, not assumed.
- R3-7 n_stability_elbow_ as the N-rule — DONE-doc (REJECTED, grounded): R3-2 showed one_se_min beats one_se_max/
  argmax; the elbow (stability x score) collapses to ~score-argmax on this bed (stability ~1 across N on small p,
  the documented R6 finding), so it cannot beat the rule that already won. No headroom.

## BorutaShap
- B3-1 Promote cluster-premerge into the class — DONE-doc (validated win, RECOMMEND to owner): R2b-6 already proved
  premerge (7/12 cells, +recall, -noise, faster) and it is SHIPPED inside the HybridSelector. Porting it into the
  BorutaShap class itself (premerge_clusters= option) is a concurrent-file change in the BorutaShap owner's domain;
  recommend they add it. The win is captured where it matters (hybrid); no new bench needed.
- B3-2 Flip to held-out permutation driver — DONE-benched (REJECTED inside the hybrid): as the hybrid boruta-member
  driver it was net-negative without FE (noise 2.33->1.33 only, recall 0.96->0.88, auc 0.784->0.779, 2.5x slower) and
  merely TIED gini with FE on (0.8288 vs 0.8299) at 2x cost -> gini stays the hybrid default. (Permutation as the
  STANDALONE BorutaSel adapter default vs gini on the big bench is a separate fs_selectors-only question, still open.)
- B3-3 Cross-subsample stability intersection default — DONE-doc (DEFERRED, recall-risk): the intersection reliably
  drops the draw-level spurious column (R2-era finding) but at a recall cost (it also drops inconsistently-relevant
  features) -- the same recall-for-precision trade the rejected fi_guard made. It is already an opt-in (boruta_stable
  in the roster); not made the default because the big bench's downstream AUC prefers recall. Available, not default.
- B3-4 FE-augmented Boruta cascade — DONE-BENCHED (WIN, ties the best): boruta_fe = MRMRSel(fe=True)->BorutaSel in
  the final big benchmark (3 seeds) scored auc_mean 0.8379 vs plain boruta 0.7722 = +0.066, TYING mrmr_fe (0.8379)
  and the FE tier. FE fully closes Boruta's gap (same mechanism as the hybrid/mrmr FE wins). Shipped as the boruta_fe
  roster strategy.

## Final big benchmark (run_experiment.py, 3 seeds, 24 strategies) -- round-3 wins confirmed head-to-head
auc_mean leaders, all FE-driven: H2 mrmr_fe->rfecv_logit 0.8380 (n=8) ~ mrmr_fe 0.8379 (fe_strict, n_eng=5) ~
boruta_fe 0.8379 ~ hybrid 0.8367 (FE; best lgbm 0.8289) ~ hybrid_strict 0.8363 ~ H6 0.8361. Best pure-selection
trails at ~0.79 (rfecv_lgbm 0.7918). Confirmed round-3 deltas vs each selector's pre-round-3 baseline: FE-hybrid
0.786->0.837, boruta 0.772->0.838 (boruta_fe), rfecv_lgbm_perm 0.791->0.806 (survivor-FE), mrmr_fe cleaner at equal-
top AUC (fe_strict). hybrid_nofe is the recall champion (base_recall 0.917 vs FE-strategies' ~0.65). ONE-SIZE-FITS-
ALL holds: per-model bests (lgbm hybrid, logit H6, knn mrmr_fe/boruta_fe) are all FE strategies within ~0.02 of the
shared best H2. NET: every FS now has an FE path to ~0.84; the deferred non-FE refinements were all measured non-wins.
- B3-5 Trial-progressive shadow percentile — DONE-doc (REJECTED, grounded): the documented same-draw shadow-leak
  (B4/B5/B8) is structural -- the top spurious column beats even the MAX shadow on that draw, so NO percentile
  schedule (lenient->strict) removes it; it only trades recall for noise. B4 already rejected the static-quantile
  variant for this exact reason; a temporal schedule shares the ceiling. No headroom.
- B3-6 Asymmetric acceptance p-value — DONE-doc (REJECTED, grounded): B5 established the spurious column wins its
  binomial test CONSISTENTLY within a single draw (the structure is in both halves), so a stricter ACCEPT threshold
  does not drop it -- only cross-subsample (B3-3) or an independent draw does. Same same-draw ceiling as B3-5.
- (rejected by the agent itself: pruning the dead IsolationForest sample path = speed/hygiene only, no AUC.)

## ShapProxiedFS
- S3-1 Turn ON interaction_aware in the AUC callers — DONE-benched (REJECTED as default): a pure flip is a NO-OP
  because the path is gated `phi.shape[1] <= max_interaction_features` (=16) and these cells have >16 SHAP units
  post-prefilter, so it never fires (interaction_on == off in all 4 probe cells). RAISING the gate to 60 to force it
  engages an O(P^2) TreeSHAP interaction tensor that is prohibitively slow (~1.6GB, stuck minutes) on ~40 units, so
  the gate at 16 is correct and interaction_aware is NOT a viable default on wide data. It is only usable post-
  aggressive-clustering (units <=16). FE (H3-1) is the recovery mechanism that actually works here instead.
- S3-2 Auto-calibrate parsimony_tol from holdout-SPLIT variance — DONE-BENCHED (REFUTED by the flat tol grid):
  round3_shap_levers_bench.py swept parsimony_tol {0.02,0.01,0.005,0.002} x 3 seeds -> auc_mean spans only 0.7733..
  0.7775 (~0.004, within cross-seed noise; tol_0.005 marginally best). An adaptive split-variance band only
  INTERPOLATES a per-dataset tol within that range, so it is bounded by the grid -> it cannot beat the best fixed tol
  by more than noise. The band family (= R2s-1/4/5) is refuted by measurement, not deferred.
- S3-3 Interaction-augmented prefilter — DONE-BENCHED (not the bottleneck): the prefilter report's 'kept' is a count
  (operands reach selection -- oper 1-2/2 recovered across the grid), so the prefilter is NOT dropping the operands;
  ShapProxiedFS's recall ceiling is the ADDITIVE-proxy main-effect (scores a pure-interaction operand ~0 even when
  present). Prefilter-widening cannot address that; FE (product as a column) is the fix, captured in hybrid/mrmr.
- S3-4 Raise n_revalidation_models 2->5 — DONE-BENCHED (REJECTED, no effect): nreval_5 produced an IDENTICAL result
  to the default (auc 0.7733, same n=7.7, same recall) -- the parsimony winner is already stable on this bed, so the
  extra revalidation models only cost time. No win.
- S3-5 FE-augment ShapProxiedFS candidates — DONE-doc (DOMINATED, captured elsewhere): adding engineered products to
  ShapProxiedFS's candidate pool is the FE lever for the weakest selector, but it duplicates exactly what mrmr_fe and
  the FE-hybrid already deliver (both ~0.83). ShapProxiedFS's distinct value is PRECISION (noise 0, compact); its low
  AUC is inherent to that niche. The FE benefit is shipped where it composes best (hybrid); a standalone shap-FE would
  be dominated by mrmr_fe. Not worth a separate productionised path.
- S3-6 Paired-seed greedy restart inside the interaction objective — DONE-doc (gated out by S3-1): the interaction
  objective it seeds is the same O(P^2) TreeSHAP path S3-1 found prohibitive on wide data; only viable post-
  aggressive-clustering (<=16 units), where the plain greedy already covers the few pairs. No headroom on the bed.
- S3-7 Interaction-mass corrector feature — DONE-doc (REJECTED, pre-refuted by S2): S2 MEASURED corr(within-subset
  interaction mass, honest-proxy residual) = +0.49 on xor2 -- the additive proxy already OVER-credits interaction
  subsets, so a corrector feature on that signal pushes the WRONG way. The agent flagged this as the likely outcome;
  S2's measurement confirms it. No implementation.

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
