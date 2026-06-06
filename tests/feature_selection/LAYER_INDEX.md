# MRMR `layerNN` biz_value index

Discoverability map for the ~95 `tests/feature_selection/test_biz_value_mrmr_layer*.py` files. Each layer is a quantitative biz_value contract (real LogReg / MRMR numbers, never xfail). The summary below is the first line of each file's module docstring; open the file for the exact pinned thresholds. Generated from the docstrings, not by hand -- regenerate after adding a layer.

| Layer | Contract / parameter pinned |
| --- | --- |
| layer6 | MRMR contracts: adversarial decoy resistance. |
| layer7 | MRMR contracts: MNAR (Missing Not At Random). |
| layer8 | MRMR contracts: polynomial-expansion explosion. |
| layer9 | MRMR contracts: TIME-SERIES LAGGED FEATURES. |
| layer10 | MRMR contracts: HIGH-CARDINALITY CATEGORICAL. |
| layer11 | MRMR contracts: ANOMALY / OUTLIER RESISTANCE. |
| layer12 | MRMR contracts: CONCEPT DRIFT + StabilityMRMR. |
| layer13 | MRMR contracts: EXTREMELY IMBALANCED BINARY ``y``. |
| layer14 | MRMR contracts: sklearn PIPELINE / CLONE / PICKLE / |
| layer15 | MRMR contracts: CONTINUOUS REGRESSION targets. |
| layer16 | MRMR contracts: MULTICLASS + ORDINAL classification. |
| layer17 | MRMR contracts: TARGET-LEAKAGE DETECTION. |
| layer18 | MRMR contracts: DEGENERATE INPUT FEATURES. |
| layer19 | MRMR contracts: TRAIN/TEST DISTRIBUTION-SHIFT. |
| layer20 | MRMR contracts: HIGH-DIMENSIONAL TEXT/EMBEDDING |
| layer21 | HYBRID ORTHOGONAL-POLYNOMIAL + MI-GREEDY FEATURE ENGINEERING. |
| layer22 | CROSS-BASIS PAIR ORTHOGONAL-POLYNOMIAL FE. |
| layer23 | AUTO-WIRED HYBRID ORTHOGONAL-POLYNOMIAL FE INSIDE MRMR.fit(). |
| layer24 | HYBRID FE LIFT ON PRODUCTION-SHAPED REAL-WORLD SCENARIOS. |
| layer25 | HYBRID FE SCALE + EDGE CASES + PERF GATES. |
| layer26 | GENERIC MI-GREEDY FE CONSTRUCTOR INSIDE MRMR.fit(). |
| layer27 | STRESS-TEST the FE pipelines under hostile production conditions. |
| layer28 | PUSH HYBRID FE THROUGH SKLEARN ECOSYSTEM EDGE CASES. |
| layer29 | VALIDATE HYBRID FE ON sklearn TOY DATASETS (real data). |
| layer30 | HYBRID FE PERF OPTIMIZATION (dedup hotspot). |
| layer31 | HYBRID FE PERF -- replace per-column sklearn MI loop |
| layer32 | SPLINE + FOURIER EXTRA-BASIS FE. |
| layer33 | K-FOLD TARGET ENCODING for categorical features. |
| layer34 | COUNT + FREQUENCY encoding + CAT x NUM residual. |
| layer35 | END-TO-END VALIDATION of ALL 8 FE mechanisms in concert. |
| layer36 | STABILITY-AWARE FE -- bootstrap-aggregate MRMR FE |
| layer37 | MISSING-VALUE-AWARE FE -- surface missingness as |
| layer38 | CROSS-FEATURE RATIO + GROUPED-DELTA + LAGGED-DIFF FE. |
| layer39 | COMPREHENSIVE REGRESSION across all 38 prior layers. |
| layer40 | VERIFY GPU PATH for hybrid FE. |
| layer41 | DCD cluster-membership accessor (self-describing summary). |
| layer42 | DCD cluster_size_threshold default investigation. |
| layer43 | DCD commit_swap recipe wiring + auto swap-method. |
| layer44 | enrich the DCD auto bake-off pool with 4 more aggregators. |
| layer45 | DCD anchor refinement (member-swap branch). |
| layer46 | VI-based DCD distance + ``"auto"`` SU/VI bake-off. |
| layer47 | AUTO-TUNE ``dcd_tau_cluster`` via small SU sweep. |
| layer48 | HIERARCHICAL POST-HOC CLUSTERING over DCD anchors. |
| layer49 | realistic kitchen-sink benchmark for DCD (L41-L48). |
| layer50 | DCD performance budget + bit-equivalence guards. |
| layer51 | BATCHED pairwise-SU dispatch for DCD. |
| layer52 | COMPREHENSIVE regression + state-of-the-union. |
| layer53 | INCREMENTAL / STREAMING ``MRMR.partial_fit`` support. |
| layer54 | FE PROVENANCE TRACKING + HUMAN-READABLE REPORT. |
| layer55 | COMPREHENSIVE REGRESSION + DIFF VS L52 BASELINE. |
| layer56 | TRI-PRODUCT cross-basis ORTHOGONAL-POLYNOMIAL FE. |
| layer57 | ADAPTIVE PER-COLUMN DEGREE selection for the |
| layer58 | CONDITIONAL BASIS ROUTING for the orthogonal- |
| layer59 | DIFF-BASIS FE for highly-correlated source pairs. |
| layer60 | CMI-GREEDY FE CONSTRUCTOR INSIDE MRMR.fit(). |
| layer61 | PER-CLUSTER SHARED-BASIS FE. |
| layer62 | BOOTSTRAP-STABLE MI ranking for hybrid orth-poly FE. |
| layer63 | THREE-GATE + K-fold OOF MI for hybrid orth-poly FE. |
| layer64 | COMPREHENSIVE REGRESSION + STATE-OF-THE-UNION. |
| layer65 | KSG / k-NN MI ranking for hybrid orth-poly FE. |
| layer66 | copula-based MI ranking for hybrid orth-poly FE. |
| layer67 | distance-correlation ranking for hybrid orth-poly FE. |
| layer68 | per-column scorer AUTO-SELECTION for hybrid orth-poly FE. |
| layer69 | ENSEMBLE-OF-SCORERS rank-fusion for hybrid orth-poly FE. |
| layer70 | COMPREHENSIVE 69-LAYER REGRESSION + COMPOSITE ALL-ON. |
| layer71 | HSIC kernel-based ranking for hybrid orth-poly FE. |
| layer72 | JMIM (Bennasar 2015) redundancy-aware ranking for |
| layer73 | Total Correlation (Watanabe 1960) multivariate- |
| layer74 | CMIM (Fleuret 2004) redundancy-aware ranking for |
| layer75 | COMPREHENSIVE 74-LAYER REGRESSION + 8-SCORER COMPARISON BENCHMARK. |
| layer76 | META-SCORER auto-selection that LEARNS from cheap |
| layer77 | 4-WAY (QUADRUPLET) cross-basis ORTHOGONAL-POLY FE. |
| layer78 | ADAPTIVE-ARITY cross-basis ORTHOGONAL-POLY FE. |
| layer79 | COMPREHENSIVE STATE-OF-THE-UNION regression test. |
| layer80 | SEMI-SUPERVISED orth-poly basis-preprocess fitting. |
| layer81 | LASSO (L1) coefficient-based pre-selection as an |
| layer82 | ELASTIC NET (L1 + L2) coefficient-based pre-selection |
| layer83 | 10-MECHANISM x 7-DATASET SHOWDOWN. |
| layer84 | CMIM (Layer 74) profiled + optimized hot-path. |
| layer85 | ``fe_hybrid_orth_default_scorer`` routing flag. |
| layer86 | JMIM (L72) + TC (L73) hot-path optimization. |
| layer87 | grouped multi-stat aggregator with CMI gate. |
| layer88 | per-group histogram + quantile FE with target-aware edges. |
| layer89 | cat x cat synergy cross with interaction-information pre-filter. |
| layer90 | NUMERIC DECOMPOSITION FE with bootstrap-MI gate. |
| layer91 | TWO-TIER IT GATES on the four recipe-emitting FE |
| layer92 | TEMPORAL LEAK-SAFE GROUPED AGGREGATIONS. |
| layer93 | COMPOSITE (multi-column) group-key aggregator. |
| layer94 | cat x cat x cat TRIPLE synergy cross via beam search. |
| layer95 | PERIODIC/MODULAR (PART A) + PER-GROUP DISTRIBUTION- |
| layer96 | O(p^2) -> O(p) + O(p^2)-cheap II pre-filter speedup. |
| layer97 | DEFAULT-ON FLIP of the genuinely-safe MRMR mechanism |
| layer99 | META FE-RECOMMENDER -- ~50 opt-in fe_* flags -> 1 auto knob. |
| layer100 | UNIFY scorer-selection under the Param-Oracle. |
| layer101 | comprehensive full-suite regression + state-of-the-union. |
| layer103 | Param-Oracle <-> kernel_tuning_cache migration POC. |
| layer104 | THREE new recipe-based FE families. |
