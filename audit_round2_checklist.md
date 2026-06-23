# Round-2 audit — master checklist (every finding, every disposition)

Disposition legend: PENDING / RESOLVED (fix+regression test) / DOC (docstring/comment caveat) / FUTURE (out-of-scope, reason) / REJECTED (anti-recommendation, reason).
MRMR excluded entirely. "fix" = code fixed + a regression test that fails pre-fix and passes post-fix, unless a different disposition is stated.

## Numerical stability (NUM)
[x] NUM1 `metrics/_core_precision_mape.py:30` precision hits/allpreds div0 (absent class) — RESOLVED (fix+test)
[x] NUM2 `metrics/_core_precision_mape.py:82` recall hits/supports div0 — RESOLVED (hardening guard, non-regressive, finiteness-pin test)
[x] NUM3 `metrics/_core_precision_mape.py:84` f1 0/0 — RESOLVED (hardening guard, non-regressive, finiteness-pin test)
[x] NUM4 `metrics/regression/_regression_metrics.py:186,197,209,221,232,253` s/wsum all-zero weights → inf/nan — RESOLVED (fix+test)
[x] NUM5 `feature_engineering/hurst.py:187` log(n) at n=1 — REJECTED (unreachable behind existing n<20 guard)
[x] NUM6 `feature_engineering/transformer/class_mahalanobis.py:53` inv→pinv near-singular — RESOLVED (hardening guard, non-regressive, finiteness-pin test)
[x] NUM7 `feature_engineering/transformer/lda_projection.py:66` inv→pinv (try/except only catches exact LinAlgError) — RESOLVED (hardening guard, non-regressive, finiteness-pin test)
[x] NUM8 `feature_engineering/transformer/local_classifier.py:95` solve→lstsq(rcond) — RESOLVED (hardening guard, non-regressive, finiteness-pin test)
[x] NUM9 `feature_engineering/spectral.py:146` log of ≈0 spectrum — RESOLVED (hardening guard, non-regressive, finiteness-pin test)
[x] NUM10 `training/composite/discovery/bayesian.py:110` inv(XtX) near-singular→lstsq/ridge — RESOLVED (fix+test)
[x] NUM11 `training/baselines/_dummy_bootstrap.py:303,305` eps=1e-15 too tight; log1p(-p) — RESOLVED (hardening guard, non-regressive, finiteness-pin test)
[x] NUM12 `models/ensembling/predict.py:218` (also P0) std/mean inf/nan uncertainty — RESOLVED (fix+test)

## Concurrency / global state / reproducibility (CON)
[x] CON1 `_cat_confirm_permutation.py:115` njit conditional shuffle unseeded global RNG (default path) — RESOLVED (fix+test)
[x] CON2 `_cat_confirm_permutation.py:76` njit IPF conditional shuffle unseeded — RESOLVED (fix+test)
[x] CON3 `_cat_confirm_permutation.py:364` serial three-MI njit Y-shuffle unseeded — RESOLVED (fix+test)
[x] CON4 `_cat_target_encoding_and_weighted.py:245` njit group-aware shuffle unseeded — RESOLVED (fix+test)
[x] CON5 `boruta_shap/__init__.py:342,344` RandomForest no random_state (comment claims determinism) — RESOLVED (fix+test)
[x] CON6 `_gpu_resident_select.py:503` _PINNED_D2H_BUF shared pinned buffer race — RESOLVED (fix+test)
[x] CON7 `_gpu_resident_select.py:527,544` operand-table single-slot caches clobber — RESOLVED (fix+test)
[x] CON8 `batch_mi_noise_gate_gpu.py:353` _DY_DEVICE_CACHE single-slot clobber — RESOLVED (fix+test)
[x] CON9 `_gpu_resident_fe.py:254,269` handoff globals interleave — RESOLVED (fix+test)
[x] CON10 `feature_engineering/transformer/_kernels_cupy.py:42` _PINNED_BUFFERS race + unbounded growth — RESOLVED (fix+test)
[x] CON11 `training/_predict_guards.py:104` _CB_VAL_POOL_CACHE id()-key recycle + unbounded — RESOLVED (fix+test)
[x] CON12 `training/feature_drift_report.py:669` _DRIFT_INVARIANT_CACHE unbounded — RESOLVED (verified already-capped + pin test)
[x] CON13 `training/composite/ensemble/__init__.py:255` _OOF_HOLDOUT_CACHE eviction-cap verify — RESOLVED (verified already-capped + pin test)
[x] CON14 `training/utils.py:423` _PD_VIEW_LAST_CACHE id()-key recycle — RESOLVED (fix+test)
[x] CON15 `rfecv/_fit_outer_loop.py:238` Parallel sharedmem mutate without lock (GIL-implicit) — RESOLVED (fix+test)
[x] CON16 `_neural_mi.py:220,327,358` model caches unlocked lazy-init double-load — RESOLVED (fix+test)
[x] CON17 `votenrank/leaderboard/Leaderboard.py:118,164` set-iteration row order hashseed-dependent — RESOLVED (fix+test)
[x] CON18 `calibration/probabilities.py:31,43,71...` global np.random in synthetic-prob helpers — RESOLVED (fix+test)
[x] CON19 `filters/discretization/_discretization_dataset.py:135` unseeded noise (helper) — RESOLVED (fix+test)
[x] CON20 `feature_selection/hybrid_selector.py:558` top-k no explicit tiebreak (latent) — RESOLVED (fix+test)
[x] CON21 `training/baselines/_dummy_baseline_compute.py:69` id()-key cache recycle — RESOLVED (fix+test)

## Memory / large-frame safety (MEM)
[x] MEM1 `estimators/custom.py:220,241` encoder/discretizer transform double-copy (frame wrap + astype) — RESOLVED (fix+test)
[x] MEM2 `wrappers/_helpers_importance.py:259,297` X.copy() inside p×n_repeats loop — RESOLVED (fix+test)
[x] MEM3 `preprocessing/cleaning.py:539` df.copy() exactly when RAM already >50% — RESOLVED (fix+test)
[x] MEM4 `estimators/custom.py:358` MyDecorrelator.transform full frame wrap — RESOLVED (fix+test) (P2)
[x] MEM5 `feature_selection/boruta_shap/_fit_explain.py:375` triple X/y copy retained — DOC (unavoidable shadow copy, documented at site) (P2)

## Serialization / pickle / state (SER)
[x] SER1 `training/neural/ranker.py:844,790` MLPRanker live Trainer, no __getstate__ (P0) — RESOLVED (fix+test)
[x] SER2 `training/neural/recurrent_dataset_helpers.py:118-123` live trainer/model/_prediction_cache no __getstate__ — RESOLVED (fix+test)
[x] SER3 `training/neural/keras_compat.py:120` Keras model_ no __getstate__ — RESOLVED (fix+test)

## Statistical / ML methodology (SA)
[x] SA-P0-1 `filters/_cmi_perm_stop.py:124-129` wrong (marginal) null vs conditional — RESOLVED (fix+test)
[x] SA-P0-2 `wrappers/_knockoffs.py:204-308` W from non-neg importances, no flip-sign symmetry → no FDR — RESOLVED (fix+test)
[x] SA1 `_cmi_perm_stop.py:129` naive p, no (b+1)/(B+1) — RESOLVED (fix+test)
[x] SA2 `filters/fleuret.py:202-275,119` data-dependent stop p, no FWER, no add-one — RESOLVED (fix+test)
[x] SA3 `wrappers/_noise_floor.py:108-134` 95th pct from n_perm=3 — RESOLVED (fix+test)
[x] SA4 `filters/_interaction_information.py:84` MM design-cardinality → manufactured synergy — RESOLVED (fix+test)
[x] SA5 `wrappers/rfecv/_stability_select.py:81-241` cites PFER, never enforces bound — RESOLVED (fix+test)
[x] SA6 `filters/stability.py:53-194` sample_fraction=0.75, FDR-control claim false — RESOLVED (fix+test)
[x] SA7 `boruta_shap/_shadow_stats.py:346-392` binomial p=0.5 vs 1/(m+1) — RESOLVED (fix+test)
[x] SA8 `_univariate_ht.py:463-468` kendall tau-b no-ties variance — RESOLVED (fix+test)
[x] SA9 `_univariate_ht.py:471-486` heterogeneous-n p mixed in one BY family — RESOLVED (fix+test)
[x] SA10 `filters/_chao_shen.py:120-122` CS single-entropy estimator misused for MI — RESOLVED (fix+test)
[x] SA11 `filters/_cat_mm_correction.py:104+` MM per-term doesn't telescope, can flip II sign — RESOLVED (fix+test)
[x] SA12 `filters/_pid_decomposition.py:211-222` PID synergy plug-in bias, no correction — RESOLVED (fix+test)
[x] SA13 `filters/_ksg.py:98 vs 325` open/closed neighbour convention mismatch — RESOLVED (fix+test)
[x] SA14 `filters/_fastmi.py:217-223` asymmetric bias scales don't cancel — RESOLVED (fix+test)
[x] SA15 `info_theory/_entropy_kernels.py:281-284` SU normalizer biased plug-in — RESOLVED (fix+test)
[x] SA16 `metrics/calibration/_calibration_metrics.py:118-164` plug-in ECE default + adaptive grid — RESOLVED (fix+test)
[x] SA17 `_calibration_metrics.py:43-44` calibration_coverage rounding artifact feeds ICE — RESOLVED (fix+test)
[x] SA18 `calibration/policy.py:631-659` held-out point + in-sample bootstrap CI mismatch — RESOLVED (fix+test)
[x] SA19 `reporting/charts/calibration.py:90-150` isotonic substitute on degenerate resamples narrows band — RESOLVED (fix+test)
[x] SA20 `reporting/charts/calibration.py:580-599` DeLong normal CI under-covers near AUC≈1 — RESOLVED (fix+test)
[x] SA21 `reporting/charts/calibration_drift.py:154-160` plug-in ECE on small unequal windows → trend artifact — RESOLVED (fix+test)
[x] SA22 `metrics/_drift.py:39-49` near-constant ref → PSI/KL/JS=0 false no-drift — RESOLVED (fix+test)
[x] SA23 `metrics/quantile.py:341-396` CRPS drops tails — RESOLVED (fix+test)
[x] SA24 `metrics/_fairness_metrics.py:149-150,202` subgroup dict keyed by len(arr) collision — RESOLVED (fix+test)
[x] SA25 `metrics/_ranking_extras.py:256` ERR max_grade per-sample; `:302` P@k /k deflates short queries — RESOLVED (fix+test)
[x] SA26 `training/composite/discovery/_eval_waic.py:138-139` σ² from held-out fold's own residuals — RESOLVED (fix+test)
[x] SA27 `training/composite/discovery/_stability.py`+screen winner's-curse, no honest holdout — FUTURE (needs fresh holdout threaded through whole discovery pipeline; DOC note at site)
[x] SA28 `training/composite/conformal.py:118-181` normalized conformal fits σ̂ on same calib residuals — RESOLVED (fix+test)
[x] SA29 `training/composite/conformal.py` exchangeability unchecked on temporal data (time_ordering not plumbed) — RESOLVED (fix+test)
[x] SA30 `training/baselines/_dummy_baseline_classification.py:51-58` train prevalence scored on val/test — RESOLVED (fix+test)

## Statistical P2 / DOC-class (SAP2)
[x] SAP2-1 three inconsistent ECE binning schemes presented as "the ECE" — DOC (caveat added at site)
[x] SAP2-2 `calibration/policy.py` selection="same_oof" fit+eval-same-rows (default inner_cv) — DOC (caveat added at site)
[x] SAP2-3 `_dummy_bootstrap.py:248` percentile (not BCa) CIs; :246 no add-one; dropped resamples — RESOLVED (fix+test) + DOC
[x] SAP2-4 `evaluation/bootstrap.py` AUC bootstrap ignores tie/paired (DeLong is default) — DOC (caveat added at site)
[x] SAP2-5 `_permutation_null.py:153` maxT floor from K=25 — DOC (caveat added at site)
[x] SAP2-6 `_ksg.py:138` KSG subsample to 50k; `_entropy_kernels.py:218` "scrubs bias" overstatement — DOC (caveat added at site)
[x] SAP2-7 `_eval_waic.py` mislabeled "WAIC"/"Chow"; `_eval_stats.py` BH over correlated p (should be BY) — RESOLVED (fix+test) + DOC
[x] SAP2-8 `_classification_report.py` macro-avg includes absent classes; P/R/F1 pinned 0.5 on rare — DOC (caveat added at site)
[x] SAP2-9 `_drift.py` PSI eps=1e-4 drives value; KL Miller-Madow cross-term heuristic — DOC (caveat added at site)
[x] SAP2-10 `metrics/quantile.py` PIT launders crossings; coverage silent on holdout-only — DOC (caveat added at site)
[x] SAP2-11 `venn_abers.py:51-83` off-grid step-function lookup — DOC (caveat added at site)
[x] SAP2-12 `_cmi_perm_stop.py:113` z_comp modulo collision still returns verdict — RESOLVED (already fixed round-1, warning present)
[x] SAP2-13 `_shap_proxy_calibrate.py:122` non-inversion guard checked in-sample — DOC (caveat added at site)
[x] SAP2-14 `_calibration_gate.py:145` gain-vs-penalty unit mismatch — DOC (caveat added at site)

## Complexity / performance (CPX) — measure-first, gate by bit-identity/selection-equivalence
- [ ] CPX-P0-1 `metrics/_auc_per_group.py:54-62` public fn doesn't dispatch to O(n log n) twin — PENDING
- [ ] CPX-P0-2 `signal/dtw.py:226,229-240` full matrix despite band; per-diagonal kernel launches — PENDING
- [ ] CPX-P0-3 `composite/venn_abers.py:68-78` O(n² log n) PAV refits, claims O(n log n) — PENDING
- [ ] CPX1 `filters/_temporal_agg_fe.py:293,567-651,235-456` O(N²)/O(N·card) entity paths — PENDING
- [ ] CPX2 `feature_engineering/anchor.py:178-195` O(A²) per group → EWMA recurrence — PENDING
- [ ] CPX3 `filters/_target_encoding_fe.py:433-434` per-row dict.get → Series.map — PENDING
- [ ] CPX4 `filters/_ratio_delta_fe.py:138-139,174-175,293-362` in-loop to_numpy + per-row dict.get — PENDING
- [ ] CPX5 `filters/_grouped_quantile_fe.py:418-432` uncapped MDLP refits — PENDING
- [ ] CPX6 `feature_engineering/spatial.py:594-614` per-query bincount loop — PENDING
- [ ] CPX7 `_ksg.py:118-135` _count_within_eps pure-Python (not njit) — PENDING
- [ ] CPX8 `_ksg.py:421-451` ColumnKNNCache built but never consumed (dead) — PENDING
- [ ] CPX9 `_mah.py:354,393`/`_pid_decomposition.py:208`/`_chao_shen.py:152` Python joint-build (not njit) — PENDING
- [ ] CPX10 `_mi_greedy_cmi_fe.py:769` greedy re-renumber per candidate (bypasses hoist helpers) — PENDING
- [ ] CPX11 `_orthogonal_copula_mi_fe.py:208-223` y re-ranked per column (claims once) — PENDING
- [ ] CPX12 `_orthogonal_scorer_auto_fe.py:199`/`_orthogonal_three_gate_mi_fe.py:315` discard batched kernels — PENDING
- [ ] CPX13 `_stability_cluster.py:112-117` O(p²) Python double-loop — PENDING
- [ ] CPX14 `inference/explainability.py:139,156` per-fold SHAP on entire dataset — PENDING
- [ ] CPX15 `models/selection.py:86-97` per-fold re-sort of growing prefix — PENDING
- [ ] CPX16 `models/optimization.py:416,537,554,567` `cand not in ndarray` O(K) in loop — PENDING
- [ ] CPX17 `ensembling/predict.py:115,137,142`+`process_method.py` invariant median recomputed per flavour — PENDING
- [ ] CPX18 `preprocessing/cleaning.py:457` np.unique full sort for a count — PENDING
- [ ] CPX19 `preprocessing/cleaning.py:411-428` 5+ full passes (fuse) — PENDING
- [ ] CPX20 `metrics/_multilabel_extras.py:30-72,146-178` O(n·K²) LRAP/ranking-loss — PENDING
- [ ] CPX21 `metrics/iteration_metrics.py:192-197` per-label Python metric calls — PENDING
- [ ] CPX22 `metrics/rank_correlation.py:108-118` O(n²) tie rescan — PENDING
- [ ] CPX23 `metrics/ranking.py:92-99,356-360` per-query re-sort + invariant recount — PENDING
- [ ] CPX24 `evaluation/bootstrap.py:144,516-520,59` np.delete LOO O(n²); rank thrice; per-resample re-sort — PENDING
- [ ] CPX25 `reporting/charts/slice_finder.py:289`/`error_analysis.py:505`/`drift.py:654` decode/hist/factorize in loop — PENDING
- [ ] CPX26 `composite/compare.py:133-134` (n_boot,n) gather materialized (multi-GB) — PENDING
- [ ] CPX27 `composite/conformal_online.py:218-225` full sort of rolling buffer per step — PENDING
- [ ] CPX28 `composite/cache_store.py:190-193,514-515` rewrites whole LRU JSON + globs per op — PENDING
- [ ] CPX29 `composite/conformal.py:377-384` O(G·n) Mondrian mask per group — PENDING
- [ ] CPX30 `_conformal_finalize.py:202-214` per-alpha recompute of loop-invariant sort/cumsum — PENDING
- [ ] CPX31 `rfecv/_fit_outer_loop.py:152` list(dict.keys()) per outer iter O(steps²) — PENDING
- [ ] CPX32 `rfecv/_fit_init.py:300-329` blake2b(X.tobytes()) doubles peak RAM — PENDING
- [ ] CPX33 `wrappers/_univariate_ht.py:441-457` O(n²) kendall, silent subsample — PENDING
- [ ] CPX34 `wrappers/_helpers_importance.py:251-314` np.delete per feature O(n·p²) + copy in repeats — PENDING
- [ ] CPX35 `_tta.py:45-74`+`_uncertainty_eval.py:29-31` redundant model passes — PENDING
- [ ] CPX36 FE transformers per-feature full-predict loop (`decision_region_depth`,`counterfactual_substitution`,`adversarial_flip`,`fisher_weighted_residual`,`local_classifier`) — PENDING
- [ ] CPX37 FE transformers per-fold rebuild of invariant projection (`residual_attention`,`stacked_attention`,`random_features`,`pred_augmented`) — PENDING
- [ ] CPX38 `rf_proximity.py:90` dense (n_q,n_bank) materialization — PENDING
- [ ] CPX39 `estimators/custom.py:344-353` MyDecorrelator.fit O(p²) iloc double-loop — PENDING
- [ ] CPX-P2 misc cold/gated items (bayesian BOCPD uncap, friend_graph scan, hurst arange realloc, SMOTE loops, quantile insertion sort, fairness rescan, optimization concatenate growth, policy nested resample, quality_gate loops, rfecv inspect.signature per fold, conformal_classification ragged build, _feature_importances V2-V5 already bench-rejected → keep) — PENDING

## Public API consistency / contracts / validation (API)
- [ ] API-P0-1 metrics surface no len(y_true)==len(y_pred) check (numba OOB) — PENDING
- [ ] API-P0-2 `preprocessing/cleaning.py:916-929` apply_features_cleaning mutates despite "MUST NOT CHANGE" docstring — PENDING
- [ ] API1 sibling selectors random_state default differs (0/None/42) — PENDING
- [ ] API2 HybridSelector/optbinning hardcode n_jobs=-1 (oversubscription) — PENDING
- [ ] API3 `evaluation/bootstrap.py:570` auc_ci n_bootstrap=2000 vs 1000 elsewhere — PENDING
- [ ] API4 `models/optimization.py:686/680 vs 156/150` wrapper vs class opposite defaults — PENDING
- [ ] API5 `models/tuning.py:412 vs 583` GPU_ENABLED opposite defaults — PENDING
- [ ] API6 `ensembling/predict.py:46 vs 239` harm vs arithm default blend — PENDING
- [ ] API7 `evaluation/bootstrap.py` auc_ci returns {"auc"} vs {"point"} → KeyError — PENDING
- [ ] API8 `metrics/_ranking_extras.py:42-57` _split_by_group bare empty vs 3-tuple — PENDING
- [ ] API9 `estimators/custom.py:260-264 vs 282-300` Arithm no-clip vs Geom clip — PENDING
- [ ] API10 `calibration/post.py:166-186` clip only in 1D→2D branch — PENDING
- [ ] API11 `estimators/custom.py:619-625` IdentityClassifier multiclass raw features — PENDING
- [ ] API12 `models/ensembling/base.py:483-488` RRF K==1 returns raw score stamped as probs — PENDING
- [ ] API13 `metrics/_core_auc_brier.py:594` fast_brier no prob-range/NaN guard — PENDING
- [ ] API14 `feature_selection/hybrid_selector.py:481-509` 2-value regression sniffs as binary — PENDING
- [ ] API15 `calibration/policy.py:497-512` pick_best_calibrator probs/y dead params — PENDING
- [ ] API16 `estimators/base.py:18,27,52-54` test_size/random_state/stratify dead for non-CatBoost — PENDING
- [ ] API17 `models/optimization.py:163/693` input_dtype accepted never read — PENDING
- [ ] API18 `models/tuning.py:573/459` report/create_study documented persist but pass no-op — PENDING
- [ ] API19 `metrics/_core_auc_brier.py:315 vs 439` fast_aucs ignores sample_weight silently — PENDING
- [ ] API20 `_ranking_extras` group_ids=None single-group vs ranking.py crashes — PENDING
- [ ] API21 `inference/predict.py:205-214` get_models_raw_predictions uses predict (labels) not proba — PENDING
- [ ] API22 `inference/explainability.py:156,166` hardcoded binary [:,1] on multiclass — PENDING
- [ ] API23 `models/optimization.py:379` suggest_candidate None means both unready and exhausted — PENDING
- [ ] API24 `models/tuning.py:337` get_model pops caller's "target" column in place — PENDING
- [ ] API25 `models/tuning.py:48,316` hidden process-global trained_models cache by experiment name — PENDING
- [ ] API26 `models/tuning.py:371` ML-gate cross_validate unseeded vs refit seeded — PENDING
- [ ] API27 `models/ensembling/predict.py:104,300` returns (None,None,None) vs raise — PENDING
- [ ] API28 `estimators/early_stopping.py:107-118` holds out last rows, no shuffle/stratify/seed — PENDING
- [ ] API29 `estimators/custom.py:249-264` no nprobs<=X.shape[1] guard — PENDING
- [ ] API30 `estimators/custom.py:344-358` MyDecorrelator always returns DataFrame; threshold positional — PENDING
- [ ] API31 `calibration/post.py:73-147` BinaryPostCalibrator ClassifierMixin no classes_/predict_proba — PENDING
- [ ] API32 `calibration/quality.py:286-288` swallows exception to None; varying return type — PENDING
- [ ] API33 `feature_selection/mi.py:178 vs 74/227` validates [0,127] only on one kernel — PENDING
- [ ] API34 `feature_selection/general.py:312` run_efs exclude_columns list vs set .update; mutates caller — PENDING
- [ ] API35 `preprocessing/cluster.py:44,80` `if true_labels:` truthiness on ndarray — PENDING
- [ ] API36 `preprocessing/transforms.py:30 vs 233` catboost returns new vs xgboost mutates in place — PENDING
- [ ] API-P2 naming drift (random_state/seed, n_bins/nbins, y_prob/y_proba/y_score), FS return shapes, base.py ddof mismatch, score.py default vs docstring, selection.py lists vs ndarrays — PENDING
