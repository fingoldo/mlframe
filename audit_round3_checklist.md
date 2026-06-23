# Round-3 audit — checklist & dispositions

Legend: PENDING / RESOLVED (fix+test) / DOC / FUTURE / REJECTED. MRMR excluded.

## Security (SEC)
[x] SEC1 `_vendored/infonet/infer.py:53` torch.load(weights_only=False) of a Google-Drive checkpoint — RESOLVED (fix+test)
[x] SEC2 `feature_engineering/transformer/_key_bank.py:173,196` bare pickle.load of cache files (bypasses safe_pickle) — RESOLVED (fix+test)
[x] SEC3 `utils/disk_cache.py:289` safe_load(allow_unverified=True) defeats fail-closed gate — RESOLVED (fix+test)
[x] SEC4 `feature_selection/wrappers/rfecv/__init__.py:730` bare pickle.load of resume checkpoint — RESOLVED (fix+test)
[x] SEC5 `training/io.py:575` dill.load(safe=False) — documented opt-out, keep default — DOC (off-by-default + warns + documented; confirmed) (likely DOC)
[x] SEC6 `utils/safe_pickle.py` SHA-256 sidecar is integrity not authenticity — DOC (threat-model caveat added: sidecar=integrity not authenticity) (DOC, or HMAC)

## Deprecated / forward-compat (DEP)
[x] DEP1 `estimators/custom.py:414` `np.NaN` -> AttributeError on numpy 2 (P0) — RESOLVED (fix+test)
[x] DEP2 `feature_selection/_benchmarks/bench_pr4_methods.py:238` datetime.utcnow() — RESOLVED (fix+test)
[x] DEP3 `feature_selection/_benchmarks/bench_rfecv_vs_sklearn.py:276` datetime.utcnow() — RESOLVED (fix+test)

## Resource leaks (LEAK)
[x] LEAK1 `reporting/diagnostics_dispatch.py:60` plt.close inside if-png, early return bypasses — RESOLVED (fix+test)
[x] LEAK2 `reporting/charts/pdp_2d.py:119,133` fig not closed (via LEAK1) — RESOLVED (fix+test)
[x] LEAK3 `reporting/charts/shap_per_instance.py:210` fig returned, never closed (no guard) — RESOLVED (fix+test)
[x] LEAK4 `feature_engineering/transformer/_key_bank.py:240` orphaned tmp dir on write failure — RESOLVED (fix+test)
- [ ] LEAK-P2 pinned cupy buffers never freed (`_gpu_resident_select.py:984`, `_kernels_cupy.py:46`); plt.show in lib helpers (evaluation/reports.py, boruta_shap/_io_plot.py, preprocessing/cluster.py) — PENDING

## Logging & error quality (LOG)
[x] LOG1 `_feature_engineering_pairs/_pairs_score.py:330` (+:501, _pairs_chunks.py:308) swallow + stale buffer = wrong result — RESOLVED (fix+test)
- [ ] LOG2 `calibration/quality.py:296` logging.exception(e) root-logger + obj-as-message — PENDING
[x] LOG3 `training/core/_phase_recurrent.py:612` error w/o traceback, model dropped — RESOLVED (fix+test)
[x] LOG4 `training/neural/base/_base_predict.py:427` error w/o traceback — RESOLVED (fix+test)
[x] LOG-P2 root-logger cluster: models/tuning.py (~18), preprocessing/transforms.py:130,185, data/synthetic.py:75, utils/text.py:34; print() cleaning.py:572, core/helpers.py — RESOLVED (fix+test)
[x] LOG-P2b raise messages omit offending values: orthogonal.py:199, _stackers.py:77, _param_oracle.py:767, distributional.py:231/quantile.py:245/panel.py:166 — RESOLVED (fix+test)

## Type annotations (TYPE)
[x] TYPE1 `estimators/custom.py:459` soft_winsorize -> None but returns ndarray — RESOLVED (fix+test)
[x] TYPE2 `estimators/custom.py:389` create_dummy_lagged_predictions -> np.ndarray but adaptive_lag UnboundLocalError — RESOLVED (fix+test)
[x] TYPE3 `evaluation/bootstrap.py:597` auc_ci -> dict[str,float] but "method" is str — RESOLVED (fix+test)
[x] TYPE4 `core/helpers.py:115` ensure_no_infinity -> bool but returns DataFrame/None — RESOLVED (fix+test)
- [ ] TYPE5 meta-test fails: 5 unannotated public funcs (ensure_fe_gpu_binning_tuning, clear_resident_codes_handoff, create_redundant_continuous_factor, select_batch_mi_kernel, benjamini_yekutieli_reject) — PENDING
[x] TYPE-P2 non-Optional =None params: custom.py:66,601; core/stats.py:50; core/arrays.py:235 — RESOLVED (fix+test)

## Degenerate-input robustness (EDGE)
- [ ] EDGE1 `metrics/quantile.py:355` quantile_summary no ndim guard -> IndexError — PENDING
- [ ] EDGE2 `metrics/_fairness_metrics.py:108` pd.qcut no duplicates="drop" -> ValueError on ties — PENDING
- [ ] EDGE3 `preprocessing/cleaning.py:476` empty/all-NaN column div0 / opaque — PENDING
- [ ] EDGE4 `calibration/quality.py:200` n_samples<nbins -> NaN report — PENDING
- [ ] EDGE5 `preprocessing/outliers.py:185` njit OOB on train/test feature mismatch — PENDING
- [ ] EDGE6 `training/_split_helpers.py:48` singleton class opaque sklearn error (rare_1pct) — PENDING
- [ ] EDGE-P2 quantile.py:131 pit K==0; scoring.py:104 ndim; calibration/quality.py:270/529 nintervals/empty PIT; core/ewma.py:60 alpha range; hetero_vote.py:138/147 n_feat0/single-class; compare_selectors.py:201 empty; importance.py:175 partial mismatch; optbinning.py:67 dup names; _conformal_split.py:48/107 length/floor; _cv_aggregation.py:238 empty shard; estimators/custom.py:476 ==0 span; custom.py:410 empty y_true; evaluation/reports.py:535 single-class; preprocessing/cluster.py:9 empty — PENDING
