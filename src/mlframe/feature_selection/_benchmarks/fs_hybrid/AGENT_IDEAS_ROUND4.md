# Round-4 FS ideas — 5 agents, 37 ideas, all dispositioned

## SESSION SCOREBOARD (2026-06-04)
- **SHIPPED #1:** A1-3/A3-1 tree-importance member (co-occurrence-product FE, synergy-gated, default ON) —
  madelon +0.038 (3-seed), synthetics within noise; full tests + cProfile + CHANGELOG; committed.
- **SHIPPED #2:** hybrid `mrmr_synergy_cap=250` default (raises the MRMR member's synergy-bootstrap cap from
  MRMR's default 60) — enables the bootstrap on moderate-width frames → hybrid hard_synth **+0.030** (3-seed,
  all seeds up, ADDITIVE to the tree member), byte-identical no-op on narrow (synth 52) AND very-wide (madelon
  500>250, cost-skipped) frames. The 250 cap is the cost/benefit sweet spot. (round4_hybrid_mrmrcap_bench.py)
- **SHIPPED #3:** A3-5 rich FE operators INTO the tree member (`tree_rich_ops=("mul","absd","sign","rat")` default).
  Agent E proved the operator class (not column count) is the lever; integrated into the hybrid tree member and
  3-seed measured: madelon **+0.020** (variance halved 0.0216→0.0098, recovers bad-draw seeds), hard_synth +0.004,
  synth −0.002 (noise). Each op-column synergy-gated independently. (round4_rich_tree_bench.py)
- **Also recommend to FE owner:** the same operators added to MRMR's FE registry (Agent E's standalone result).
- **A4-2 RFECV noise-floor SHIP (standalone) + A1-2 MRMR-cap raise:** recommend to RFECV / MRMR owners.
- **MEASURED, killed/mixed:** A1-1 JMIM (mixed/mild, opt-in only); A4-1 knockoff-FDR (KILLED, degenerate on
  madelon's collinear probes, raw + premerge both empty); A2-2 residual-relay (regime-specific — hard_synth
  +0.017 recovers real dropped features via residual-MI, madelon −0.0085; future GATED candidate).
- **MEASURED (combine-refinements, all built + benched vs the tree-member hybrid):** A2-3/A5-1 confidence-prior
  REJECTED (−0.002 to −0.0026 all beds; confidence floor cuts recall like fi_guard — agents' thesis falsified);
  A2-1 disagreement-referee REJECTED (±0.002, within noise); A5-3 tentative-rescue regime-specific (hard_synth
  +0.0066, madelon −0.0112) — same recall-add shape as A2-2 residual-relay. CONVERGENT FINDING: a real
  recoverable weak-feature recall gap on split-signal data that recall-adds exploit (+0.006..0.017) but that
  HURTS madelon (signal already captured). The GATED SYNTHESIS (permutation-null residual-MI recall-add) was
  built + 3-seed measured → **KILLED**: the permutation null that protects madelon also admits ~nothing on
  hard_synth (the "real" str_0/sq residual-MI does NOT robustly exceed the null across seeds), so the gate that
  prevents the madelon regression also kills the hard_synth gain (all beds within ±0.004, win=False). The
  recall-add lever does NOT survive principled gating into a clean default win — the earlier ungated +0.017 was
  partly bad-draw recovery. (round4_gated_recall_bench.py) The FE-frontier ideas
  (A1-2 widen-p>60, A3-3 holdout-greedy-FE, A3-5 ratio/threshold ops) are largely SUBSUMED by the shipped tree
  member for the madelon regime but remain open for MRMR-standalone.

Informed by ALL of rounds 1-3 + real-data (madelon) validation. Decision rule: most accurate first, fastest
breaks ties; EVERY idea needs a cheap falsifiable test. Disposition column updated as ideas are measured.
Status legend: BENCH-NOW (zero/near-zero code, run immediately) | BENCH-NEXT (small code, high conviction) |
QUEUE (real code, conviction pending a cheaper test first) | LIKELY-DEAD (near a closed path; low ceiling) |
verdict appended once measured.

## CONVERGENCES (multiple independent agents → highest conviction)
- **Tree split-co-occurrence → FE pair seeding**: A1#3 (`fe_tree_cooccurrence_seed`) + A3#1 (`fe_tree_cooccur_seed`).
  Two agents independently picked this as their #1 for the madelon FE collapse. Cheap supervised operand proposer.
- **Widen the p>60 synergy-bootstrap gate**: A1#2 (`fe_synergy_widepool_topk`) + A3#2 (`fe_synergy_wide_p`).
  The CONCRETE madelon root cause: `fe_synergy_screen_max_features=60` SKIPS the synergy bootstrap when p>60
  (madelon=500), so interaction operands are never paired. A3 traced the exact code (`_mrmr_fe_step.py:231`).
- **Disagreement-as-signal**: A1#8 + A2#1 (`disagreement_referee`) + A2#2 (`residual_relay`) + A3#8 + A5#3
  (`tentative_rescue`). The members fail oppositely (MRMR under, RFECV over); their disagreement IS the signal.
- **Per-feature member confidence the hybrid DISCARDS**: A2#3 (`confidence_prior_vote`) + A5#1 (`stack_confidence`).
  Members compute Boruta hit-rate / RFECV elim-rank / SHAP fidelity / MRMR gain, then the hybrid throws it all
  away at `member_selections_` (name-lists only). Both agents made this their #1.
- **Boruta `tentative` set as a routing trigger**: A2#7 (`tentative_to_cmi`) + A5#3 (`tentative_rescue`).

## AGENT 1 — MRMR real-data robustness (the madelon collapse-to-3)
| # | name | mechanism | disposition |
|---|------|-----------|-------------|
|A1-1|`jmim_synergy_stop`|`redundancy_aggregator='jmim'` — JOINT MI I({X,Z};Y) instead of CMIM conditional-min; credits synergy CMIM discards. **Round-3 "njit-gated" was WRONG** (thread-local read at Python level, passed as njit arg). ZERO code change.|**MEASURED — NOT a default flip.** madelon n=3→3 Δ0.000 (no effect: collapse is FE-pairing not aggregator); synth n=15→27 Δ−0.0016 (mild loss, over-selects); hard_synth n=15→22 Δ+0.0045 (small win, recovers more in the under-selection regime). Mixed/mild → keep as an opt-in for the split-signal regime only, NOT default. (round4_jmim_bench.py)|
|A1-2|`fe_synergy_widepool_topk`|widen the p>60 synergy-bootstrap cap via a cheap top-K prefilter so the all-pairs joint-MI sweep runs on a bounded pool incl. interaction operands.|**MEASURED — recommend to MRMR owner (moderate-p win, NOT madelon).** Raising `fe_synergy_screen_max_features` (default 60): hard_synth (220 cols, default SKIPS bootstrap) 0.7576→**0.8030 (+0.045)** — enables the synergy bootstrap on the wider frame; synth (52≤60) BYTE-IDENTICAL no-op → safe to raise. madelon KILL: cap600 engineers 0 products (5-dim XOR not bilinear; no pair clears prevalence) + costly (O(p²), 73s). madelon collapse is UPSTREAM of FE (greedy under-selects to ≤3, discards even injected tree products → tree-seed into MRMR n=1, 0.578). Recommend: raise default cap to ~200-250 (helps moderate-p, safe on narrow, cost-gate very-wide p). (round4_mrmr_rescue_bench.py)|
|A1-3|`fe_tree_cooccurrence_seed`|one shallow GBM; count feature pairs co-occurring on root-leaf paths (gain-weighted); seed FE pairs from top co-occurring regardless of marginal MI.|**SHIPPED as a HybridSelector tree-importance MEMBER (default ON).** 3-seed standalone: tree_top20+cooccur_fe **0.8401** (0.5s) BEATS production hybrid 0.8054 (+0.035) and mrmr_fe 0.6885 (+0.15). Integrated as a member: contributes co-occurrence PRODUCT columns folded into the shared frame, GATED by a "synergy" rule (FI[a*b] > max(FI[a],FI[b])) so it self-regulates by regime. Gate-binding bench (1 seed): madelon +0.0475, hard_synth +0.0004, synth −0.0029 (gate-independent, ~noise); relmed gate rejected (hard_synth −0.0087). synergy default; tree_top_k=0 (products-only beats +raw-votes). **3-seed confirm: madelon +0.038 (all 3 seeds up), hard_synth −0.0002, synth −0.0022 (both within seed std → noise) → SHIP default-ON.** (round4_tree_seed_bench / _tree_rescue_validate / _hybrid_tree_bench / _hybrid_confirm)|
|A1-4|`underselect_floor_relax`|regime-detect "MRMR under-selected" (small support + wide pool + low-MI tail) → re-run final stretch with relative-to-first floor relaxed. Gated, NOT a blanket flip.|QUEUE (near closed relevance-floor; needs regime gate proof)|
|A1-5|`pair_conditional_admit`|post-stop: for each rejected low-marginal candidate, admit if max_S∈selected I({cand,S};Y)−I(S;Y) clears a perm null. Targeted 1-step JMIM on the tail.|QUEUE (subsumed if A1-1 wins; run A1-1 first)|
|A1-6|`order2_screen_prefilter`|inject highest-joint-MI pairs as order-2 candidates into the SCREENING pool (not just FE), so greedy can pick an interaction as a joint feature on iter 1.|QUEUE (O(p²) cost wall; needs cheap pre-rank = A1-3)|
|A1-7|`operand_mi_preserve_binning`|MDLP collapses pure-interaction operands to 1 bin (no marginal split) → zeroes them before FE. Fall back to quantile binning for 1-bin-collapsed cols in the JOINT sweep only.|QUEUE (possible upstream prereq for A1-2/3; cheap diagnostic first)|
|A1-8|`crossselector_fe_seed`|seed MRMR's FE pairs from {RFECV-kept ∖ MRMR-selected} (disagreement) bounded by member FI.|QUEUE (conv. A3-8; hybrid-coupled)|

## AGENT 2 — cross-selector synergy
| # | name | mechanism | disposition |
|---|------|-----------|-------------|
|A2-1|`disagreement_referee`|partition clusters by member agreement; route ONLY contested clusters (1-of-N) through a held-out forward-AUC referee. Unanimous keep, zero-vote drop.|**MEASURED — REJECTED.** vs tree-member hybrid: hard_synth +0.0022, madelon −0.0026, synth +0.0013 — all within noise; the forward-AUC referee on contested clusters does not beat the binary vote (confirms the H3-2..6 combine-refinement ±0.002 ceiling). (round4_synergy_combine_bench.py)|
|A2-2|`residual_relay`|fit MRMR member → OOF residuals → run Boruta/Shap on MRMR's DROPPED columns vs the residualized target. Boosting logic for FS; covers complementary signal.|**MEASURED — regime-specific, NOT default.** Tested as: hybrid→OOF residual→screen DROPPED cols by MI/corr with residual→add top-k. hard_synth: residual-MI screen recovered REAL dropped features (str_0, sq, red_0_0), +0.0173 within-run (lgbm 0.734→0.783) — the MI variant works, corr variant picks noise. madelon: −0.0085 (residual is noise; tree member already captured signal). Confounded by high hybrid run-to-run variance on hard_synth (0.736 vs 0.78). Mechanism genuine but mixed → FUTURE gated candidate (admit residual features only when their residual-MI clears a signal floor), needs multi-seed + regime gate. (round4_residual_relay_bench.py)|
|A2-3|`confidence_prior_vote`|replace binary vote with summed per-feature confidence (Boruta hits/n, RFECV retention, Shap fidelity, MRMR gain), rank-normalized.|**MEASURED — REJECTED.** Multi-signal per-feature confidence (Boruta accepted/tentative tier + distinct-member votes + normalized FI), single-member clusters gated on the consensus-confidence floor. vs tree-member hybrid: hard_synth −0.0022, madelon −0.0026, synth −0.0004 — NEGATIVE/flat everywhere; the confidence floor cuts recall like fi_guard. The per-feature native confidence did NOT beat the binary vote even on split-signal — the agents' thesis falsified. (round4_synergy_combine_bench.py)|
|A2-4|`prior_protected_rfecv`|warm-start RFECV's elimination with must_include = high-confidence core from Boruta/MRMR; RFECV only trims the tail.|QUEUE (RFECV-member was rejected H3-4; must beat that)|
|A2-5|`fe_firstclass_gated`|make engineered cols CONTESTABLE — Boruta/Shap rank them vs shadows; emit only if ≥2 members confirm OR own hit-rate clears shadow.|QUEUE (conv. A3-4)|
|A2-6|`regime_meta_combiner`|pre-calibrated threshold on cross-selector cardinality divergence routes high-divergence→recall rule, low→consensus.|QUEUE (near H3-5 overfit; pre-calibrated not learned)|
|A2-7|`tentative_to_cmi`|route Boruta `tentative` set to MRMR conditional-MI given the accepted core; admit if it adds new info.|QUEUE (conv. A5-3)|
|A2-8|`xsel_stability_dedup`|cross-member agreement (not FI) picks the rep within a kept redundant cluster.|LIKELY-DEAD (agent's own honest low-conviction; rep-choice ≈ no-op)|

## AGENT 3 — feature-engineering frontier
| # | name | mechanism | disposition |
|---|------|-----------|-------------|
|A3-1|`fe_tree_cooccur_seed`|= A1-3 (independent). Shallow GBM split co-occurrence seeds operand pairs into the existing gated FE search.|**BENCH-NEXT** (agent's #1; conv. A1-3)|
|A3-2|`fe_synergy_wide_p`|= A1-2 (independent). Scale synergy bootstrap past p=60 via top-M prefilter instead of skipping.|**BENCH-NEXT** (conv. A1-2; A3 traced exact bug)|
|A3-3|`fe_holdout_greedy_pairs`|iterative greedy FE accept on HELD-OUT engineered-feature gain (forward selection), replacing one-shot prevalence gate. (The deferred "train-based FE selection" deep rewrite.)|QUEUE|
|A3-4|`fe_xselector_confirm`|gate engineered cols through a cheap confirmer (perm-FI/Boruta-shadow) before promotion; FI-gate is SAFE on engineered cols (no recall obligation).|QUEUE (conv. A2-5)|
|A3-5|`fe_ratio_threshold_ops`|add conditional/threshold/ratio binary operators ((a>t)*b, |a−b|<t, signed-mag) the registry lacks.|**MEASURED — WIN (recommend to FE owner; tree-member follow-up next).** Designed beds CONFIRM the mechanism: products CANNOT linearize |a−b| (logit 0.49→absdiff 0.88) or sign(a)|b| (0.79→0.88). Real beds: products+ALLrich madelon 0.8648 vs products-only 0.8389 (+0.026 mean, +0.054 logit); hard_synth +absdiff logit +0.031. CAPACITY CONTROL decisive: products+96 noise cols DROPS AUC (0.839→0.773), products+96 rich-op cols RAISES it → gain is the OPERATOR CLASS, not column count. Recommend ops (priority): absdiff |a−b|, signed sign(a)|b|, ratio a/(|b|+eps), gated_med (median-threshold), thr_and_med. Use MEDIAN thresholds (threshold-0 weak); gate behind FE prevalence/MI. (fe_richops_bench.py / _control_bench.py)|
|A3-6|`fe_residual_target`|run the FE pair-MI sweep against the residual after linear removal → interaction signal stands out (SNR amplification).|QUEUE|
|A3-7|`fe_gradient_outer_screen`|gradient outer-product (Hessian-diagonal proxy) flags loss-coupled feature pairs to seed FE.|LIKELY-DEAD (per-sample tree grads not first-class; likely dominated by A3-1)|
|A3-8|`fe_seed_from_hybrid_disagreement`|= A1-8. {MRMR-dropped}×{Boruta/Shap-kept} cross-product seeds FE pairs.|QUEUE (conv. A1-8)|

## AGENT 4 — RFECV / BorutaShap / ShapProxiedFS standalone
| # | name | mechanism | disposition |
|---|------|-----------|-------------|
|A4-1|`knockoff_fdr` (RFECV)|post-hoc FDR cut: run knockoffs ONCE on the chosen survivor set, W_j=imp(real)−imp(knockoff), select_features_fdr(q=0.1). Draw-INDEPENDENT null. Plumbing exists, unused.|**MEASURED — KILLED on madelon (flagged degeneracy materialized).** RFECV kept 251 (=all-features 0.689); knockoff W>0 for only 68/251, and select_features_fdr returns EMPTY at q∈{0.05,0.1,0.2} — no Barber-Candes threshold achieves the FDR on madelon's collinear probes (W noise-tail too large). Premerge workaround (corr_thr=0.5) barely collapsed (251→238 reps), still EMPTY. Fought + failed. The tree member is madelon's answer; RFECV-on-madelon is a dead end. (round4_knockoff_fdr_bench.py)|
|A4-2|`cv_curve_noise_floor` (RFECV)|permuted-y reference CV curve as noise yardstick; stop where real curve's rise exceeds the shuffled-y noise envelope.|**MEASURED — SHIP (standalone RFECV), recommend to RFECV owner.** The literal "first-clears-noise" rule is a signal-ONSET detector (over-cuts to N=2); the corrected PLATEAU rule (smallest N past which the remaining climb is within the permuted incremental envelope) cuts madelon 251→**N*=8, lgbm 0.9135** (vs all-features 0.87, RFECV-251 0.689; knn 0.61→0.91). synth guard holds (N*=20, base_recall 0.875, AUC neutral). Needs ≥3 permutations; run on a valid FI ranking (note: prod RFECVSel('lgbm_perm') TIMES OUT on madelon at max_runtime_mins=3 → degenerate curve). (round4_noise_floor_bench.py / _robust.py)|
|A4-3|`recall_rescue_when_untrusted` (Shap)|use existing `proxy_fidelity_score` to switch parsimony_tol 0.02→0.005 + raise top_n ONLY when proxy fidelity is low (interactions missed).|QUEUE (near parsimony-band; binary regime switch not interp)|
|A4-4|`su_seeded_interactions` (Shap)|cheap pairwise-SU screen → top-K synergistic pairs → run interaction objective on ONLY those K (avoids O(P²) tensor).|QUEUE|
|A4-5|`adaptive_n_trials` (Boruta)|stop trials by binomial-confidence convergence; reallocate budget to borderline-tentative features.|QUEUE (speed lever; early-stop may already harvest it)|
|A4-6|`knockoff_margin_elim` (RFECV)|drive elimination by per-fold knockoff W instead of relative FI rank (absolute zero-reference).|QUEUE (p>n/2 singularity + per-fold cost; A4-1 cheaper first)|
|A4-7|`joint_shadow` (Boruta)|second shadow family permuting correlated BLOCKS jointly (preserves within-block covariance) to raise the same-draw leak bar.|LIKELY-DEAD (madelon noise may be independent → no-op; agent's own low conviction)|

## AGENT 5 — HybridSelector meta-selection / combine intelligence
| # | name | mechanism | disposition |
|---|------|-----------|-------------|
|A5-1|`stack_confidence`|per-feature logistic/shallow meta-learner over member confidence channels (Boruta hits, RFECV elim-rank+CV-delta, Shap |OOF-SHAP|, MRMR gain, perm-FI, cluster size); killed on honest test only.|**BENCH-NEXT** (agent's #1; conv. A2-3)|
|A5-2|`cluster_oof_gain`|honest forward selection over CLUSTERS (not features): consensus clusters free, single-member clusters admitted by held-out OOF AUC gain + one-SE guard.|BENCH-NEXT|
|A5-3|`tentative_rescue`|readmit Boruta-tentative/MRMR-margin-dropped features when a DIFFERENT member ranks them confidently (two-channel gate).|**MEASURED — regime-specific, NOT default.** Readmit Boruta-tentative whose cluster a different member also picked OR FI top-quartile. vs tree-member hybrid: hard_synth +0.0066 (recall-add recovers weak features, lgbm 0.768→0.789), madelon −0.0112 (over-adds → dilutes; tree member already captured signal), synth 0.0. Same recall-add pattern as A2-2 residual-relay → future GATED candidate (needs a "signal-already-captured" gate to avoid the madelon regression). (round4_synergy_combine_bench.py)|
|A5-4|`per_model_emit`|emit DIFFERENT feature sets for linear vs tree downstreams (test the one-size-fits-all assumption on split signal).|**MEASURED — NEGATIVE (valuable).** Cross-table (each model × each set, 3-seed) falsifies the premise: a SINGLE set is best for ALL three downstreams on each bed; the best set varies by BED not by FAMILY (lgbm wants the clean set on madelon but full-S on hard_synth). The apparent +0.016 madelon "win" was an artifact of handing each model that bed's good set. NO per-family API. The transferable finding: vote=1 S is over-inclusive of single-vote noise on real data, but data-adaptive noise-cleaning of the SHARED set is the fi_guard/noise-floor family already rejected for the hybrid. (round4_permodel_emit_bench.py)|
|A5-5|`conf_rep`|cluster rep chosen by cross-member confidence rank-agg instead of perm-FI alone.|QUEUE (cheap, low ceiling; ship-if-positive)|
|A5-6|`union_backward`|one honest RFECV `one_se_min` backward pass over the small UNION of member picks (prune, not re-discover).|QUEUE (moderate H3-4 subsumption risk)|
