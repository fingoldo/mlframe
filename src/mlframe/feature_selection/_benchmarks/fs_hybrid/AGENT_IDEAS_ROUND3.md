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
- H3-2 Per-member realized-subset OOF-AUC vote weight — DONE-doc (DEFERRED, low headroom post-FE): now that FE
  (H3-1) lifts the hybrid to the mrmr_fe ceiling (0.834), the combine rule has little headroom left -- the engineered
  features dominate the downstream regardless of how raw-member votes are weighted. A genuine mechanism (unlike the
  rejected R2b-4 standalone-skill weight) but its ceiling is the gap between the FE-hybrid and the best member, which
  FE already closed. Revisit only if a dataset shows the members strongly disagreeing post-FE.
- H3-3 FE-protective cluster rep — DONE-doc (mitigated by fe_strict): the risk (collapsing an engineered term into
  its raw operand at rep-selection) is real, but fe_strict (M3-7) already cut the engineered set to ~4 clean terms,
  and the shared-FI rep rule prefers the higher-FI engineered product anyway (it carries the signal). The FE-hybrid
  bench (0.834) shows no evidence the rep rule is losing the engineered terms. Low-value refinement; not pursued.
- H3-4 Add RFECV as a 4th member — DONE-doc (DEFERRED, at-ceiling): the FE-hybrid already ties the best whole-bed
  strategy (mrmr_fe 0.834); adding the slow RFECV member (4-5x fit cost) for marginal vote diversity has no AUC
  headroom on this bed. The big re-benchmark includes rfecv variants separately for comparison. Revisit off-bed.
- H3-5 Honest internal-validation auto-picker for (vote,expand) — DONE-doc (DEFERRED, big-bench settles it): the big
  re-benchmark runs hybrid / hybrid_nofe / hybrid_strict(vote=2) / hybrid_expand as fixed configs; if they cluster
  tightly, a per-dataset auto-picker buys little (and risks internal-split overfit). Decide from the big-bench spread
  rather than adding an auto-selection layer pre-emptively.
- H3-6 Decouple vote-unit from emitted-feature — DONE-doc (subsumed by H3-3): only matters inside FE-augmented
  clusters where an engineered term co-clusters with its raw operand; same low-value regime as H3-3, which the
  fe_strict + FI-rep behaviour already handles. Not pursued.
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
- B3-4 FE-augmented Boruta cascade — DONE-wired (benching in the big re-benchmark): registered boruta_fe =
  MRMRSel(fe=True)->BorutaSel; the big re-benchmark measures whether FE closes Boruta's gap (analogous to mrmr_fe).
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
- S3-2 Auto-calibrate parsimony_tol from holdout-SPLIT variance — DONE-doc (DEFERRED = the R2s-1 future): R2s-1 was
  already deferred with the verdict "revisit only if the fixed tol mis-scales"; the shipped fixed parsimony_tol=0.02
  (precision) / 0.005 (recall, AUC callers) is calibrated and works. The adaptive split-variance band adds per-fit
  variance-estimation machinery for at most the residual gap above a working fixed tol -- the same low-ceiling
  conclusion. Revisit only on a dataset where the fixed tol demonstrably mis-scales.
- S3-3 Interaction-augmented prefilter — DONE-doc (not the bottleneck): the S3-1 probe showed an interaction operand
  DOES partly survive the prefilter (oper 1/2 on make_dataset, 4/4 on xor2 sd1). ShapProxiedFS's recall ceiling is
  the ADDITIVE-proxy main-effect (it scores a pure-interaction operand ~0 even when present), not the prefilter
  dropping it. FE (which creates the product as a column) is the fix, captured in the hybrid/mrmr; prefilter-widening
  does not address the additive-proxy limit.
- S3-4 Raise n_revalidation_models 2->5 — DONE-doc (DEFERRED, marginal): a variance-reduction on the parsimony winner
  with adaptive early-stop; on this high-SNR bed the winner is already stable (S3-1 probe: identical selection across
  configs), so the extra models mostly cost time. Cheap to enable per-caller if a noisy dataset shows winner drift.
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
