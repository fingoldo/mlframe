# mlframe

[![CI](https://github.com/fingoldo/mlframe/workflows/CI/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/ci.yml)
[![sklearn-matrix](https://github.com/fingoldo/mlframe/workflows/sklearn-matrix/badge.svg)](https://github.com/fingoldo/mlframe/actions/workflows/sklearn-matrix-ci.yml)
[![codecov](https://codecov.io/gh/fingoldo/mlframe/branch/master/graph/badge.svg)](https://codecov.io/gh/fingoldo/mlframe)
[![PyPI](https://img.shields.io/pypi/v/mlframe.svg)](https://pypi.org/project/mlframe/)
[![Python](https://img.shields.io/pypi/pyversions/mlframe.svg)](https://pypi.org/project/mlframe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-grade machine-learning framework for tabular data. One uniform entry point (`train_mlframe_models_suite`) trains, evaluates, calibrates, ensembles, and reports across scikit-learn, CatBoost, LightGBM, XGBoost, HistGradientBoosting, and PyTorch Lightning models on the same dataset, with proper handling of polars/pandas frames, mixed dtypes, text features, ranking and quantile targets, and composite-target stacking.

## Recent changes

2026-05-28 RFECV 5-wave audit overhaul + post-audit ergonomics + auto-tune prelim. After the 5-wave audit, three follow-up batches landed: **(a) Cross-selector contract harness** at `tests/feature_selection/test_fs_selector_contract.py` parametrising 19 invariants over MRMR / RFECV / ShapProxiedFS (53 test cases, 51 passing on day-one; surfaced two real cross-selector gaps closed in this batch: MRMR now validates NaN-in-y; ShapProxiedFS got a sklearn-style `get_feature_names_out` shim). **(b) Grouped pydantic configs** (`SearchConfig` / `FIConfig` / `RobustnessConfig` in `_rfecv_configs.py`, ~40 fields total, ConfigDict(extra="allow") matching the canonical mlframe pattern at `mlframe.training._feature_selection_config`): the constructor accepts them as optional kwargs, only `model_fields_set` (explicitly-set fields) override matching flat kwargs, so default config objects don't clobber explicit values; flat-kwarg API kept verbatim for back-compat. **(c) Auto-tune skeleton** (`_auto_tune.py`): `DataFingerprint.from_xy(X, y)` extracts 12 cheap (X, y) signals; `suggest_configs(fp)` returns the 3 configs from a rule-based table encoding the Wave 1-5 bench learnings (p>>n → prescreen='univariate_ht'; flat curves → one_se_max; high-card int → permutation; ...); `RFECV(auto_tune=True)` applies the suggestion to non-explicit knobs only, stores the decision in `auto_tune_decision_` for inspection. Rule-based body is the stop-gap until the synthetic-bench-trained classifier lands (TODO A in `feature_selection/wrappers/TODO.md`). **(d) Sklearn-coverage gap closures:** 25 new tests at `test_wrappers_sklearn_coverage_gaps.py` covering input types / CV-edge / estimator-edge / transformer-contract / other; all 25 passing including the 4 that started as xfail (dotted `importance_getter` like `regressor_.coef_` / `named_steps.lr.coef_` via `attrgetter` traversal + Pipeline / TransformedTargetRegressor / GridSearchCV automatic unwrap to the final estimator; Pipeline+SimpleImputer NaN passthrough; per-split `cv_results_["split{k}_test_score"]` schema parity). **(e) Three pre-Wave-1 high-priority TODO items closed:** `prescreen='mrmr'` for p>>n problems (uses existing `mlframe.feature_selection.filters.mrmr.MRMR` as pre-pass); `importance_getter='boruta'` (in-tree pure-numpy Gini-shadow trick, no shap dependency); `importance_getter='drop_column'` (oracle baseline, O(p * full_fit_time)). Full wrappers test suite now 26 files / 530+ tests on green. Module sizes: `_helpers.py` 880 LOC, `_rfecv.py` 1018 LOC (next significant change will trigger another sibling-split).

### Earlier 2026-05-28 RFECV audit history

2026-05-29 MRMR MI-estimator expansion (Wave 7). Seven new MI estimators land alongside the legacy plug-in path, all benchmarked head-to-head on a 5472-fold CV grid across 8 distributions x 6 signal types x 3 sample sizes: **Mixed-KSG** (Gao 2017 NeurIPS) + **KSG-LNC** (Gao 2015 AISTATS, canonical NPEET port with low-entropy-skip pre-filter) in [`_ksg.py`](src/mlframe/feature_selection/filters/_ksg.py) with njit digamma + tie-breaking jitter + per-column KDTree cache + cupy fallback; **MINE** (Belghazi 2018 ICML, PyTorch+CUDA Donsker-Varadhan with EMA bias correction + early-stop), **InfoNet** (Hu 2024 ICML, pre-trained transformer with vendored checkpoint at `~/.cache/mlframe/infonet/infonet_pretrained.pt`, ~70 ms/pair after one-time 80s CUDA warmup), **MIST** (Gerasimov 2025, HuggingFace `grgera/MIST` set-transformer with empirical piecewise-linear calibration to nats via Gaussian copula lookup), **MINDE** / **DPMINE** skeletons in [`_neural_mi.py`](src/mlframe/feature_selection/filters/_neural_mi.py); **fastMI** copula+FFT-KDE (Purkayastha-Song 2024 JoMVA, Silverman + MISE LOO-bandwidth) in [`_fastmi.py`](src/mlframe/feature_selection/filters/_fastmi.py); **median** + **GENIE-weighted** ensemble aggregators (Moon 2021 IEEE TIT, bias-cancellation closed-form) in [`_mi_aggregator.py`](src/mlframe/feature_selection/filters/_mi_aggregator.py). Six **Wave-1 P0 fixes** ship behind opt-out flags: Miller-Madow correction inside `_plug_in_mi` (FD no-signal floor 0.077 -> 0.010); MDLP `backend='njit'` flipped to default (138x speedup, identical output); Knuth `edge_type='quantile'`+`m_max_cap=64` flipped to default (+0.056 MI, -76% rt); BB midpoint backtrack + `p0=0.10` + `subsample_threshold=1000` (94.5% rt cut, +0.006 MI). Three structural bugs surfaced + patched: Mixed-KSG tie-jitter for discrete_low_card (0.000 -> 0.50 MI on that distribution), Knuth `M_min=2` for uniform inputs (0.000 -> 0.50), BB median-fallback for uniform (0.000 -> 0.50). New **QS** (Quantile Spacing, Gupta 2021 Entropy) bin chooser. Unified dispatcher [`_mi_dispatch.py`](src/mlframe/feature_selection/filters/_mi_dispatch.py) and standalone estimator modules (`_ksg`, `_neural_mi`, `_fastmi`, `_mi_aggregator`) are accessible for ad-hoc / benchmark use; **the MRMR hot path itself stays exclusively on the integer-bin plug-in MI njit kernel chain** (mi_direct / fleuret / permutation). Alternative MI estimator families are intentionally NOT wired into `MRMR.fit()` — 1M-row stress bench (`bench_1M_stress.py`) confirms only the plug-in path scales to production loads (0.02s/pair, 3MB peak vs Mixed-KSG's ~10s/pair at 100k rows). Mixed-KSG and MIST gained sub-sampling guards (`max_input_n=50000` / `=2000` defaults) so standalone callers don't blow up on million-row inputs. Mega-bench v3 honest leaderboard (MI_mean / no_signal / rt_ms): GENIE 0.481/0.016/9, InfoNet 0.439/0.124/72, Mixed-KSG 0.421/0.012/8, FD_legacy 0.420/0.060/0.4, MDLP 0.395/0.000/7. **Per-signal win analysis (by smallest |error vs Monte-Carlo truth|)**: Mixed-KSG wins 2/5 (threshold deterministic Delta -0.018, xor deterministic Delta -0.029) at 8 ms/pair; GENIE wins 2/5 on noisy continuous (monotone Delta -0.039, sin Delta -0.036) at 9 ms; InfoNet wins 1/5 on linear (Delta +0.012) at 72 ms; MDLP is the only method with TRUE zero no-signal floor. **For MRMR's Family-1 plug-in path the same honest ranking applies**: by combined ``|err on signal| + noise floor``, MDLP wins at 0.107 (clean noise), Sturges 0.135, quantile10 0.139; QS has the BEST signal accuracy (err=0.093) but worst noise floor (0.123, big false-positive risk on no-signal columns). Knuth, Blocks, and MAH/SCI sit at 0.213 / 0.233 / 0.373 respectively and are demoted to research-only with AccuracyWarning. MIST monotonic but over-estimates on binary y by 90-200% even after calibration (use as ranking signal only). MINE under-converges on small CV val folds (~167 samples). See [docs/MRMR_RESEARCH_2026_05_28.md](docs/MRMR_RESEARCH_2026_05_28.md) + [`bench_adaptive_nbins_mega.py`](src/mlframe/feature_selection/_benchmarks/bench_adaptive_nbins_mega.py).

2026-05-28 RFECV 5-wave audit overhaul. The wrapper-style RFECV at `mlframe.feature_selection.wrappers.RFECV` got five waves of ML-correctness fixes plus five literature-track extensions. **Wave 1** (correctness): FI rollback when re-explored same-N subset loses the score gate; ragged-NaN voting table imputation policy `fi_missing_policy='worst'` removes silent bias toward late-surviving features in Borda/Dowdall/Copeland/Minimax; `must_include` overrides leakage exclusion; `feature_groups` overlap raised at __init__; `swap_top_k` auto-disabled when val_cv early-stopping is active; `n_features_selection_rule='auto'` now resolves to `'one_se_max'` (plateau-resistant; pre-fix was inverted multi-vs-singular). **Wave 2** (search): MBH surrogate de-dups duplicate-x rows by max-y; ScipyLocal/ScipyGlobal aliased to Dichotomic; epsilon-greedy kick; `convergence_tol`+`convergence_tol_window` plateau-aware break; `optimizer_target='mean'` aligns optimizer with 1-SE picker semantics; adaptive `init_design_size='auto'` scales seeds by p/budget. **Wave 3** (FI semantics): exponential decay via `fi_decay_rate`; `coef_scale_source='train'` (no test-std leakage); multiclass coef collapse via `max(|coef|)`; multi-estimator+AM/GM auto-falls-back to Borda; lexicographic tie-breaker in voting; CPI tree grown by `min_samples_leaf` not `max_depth=5`; rank-based aggregators skip pre-ranking; NaN-score FI runs dropped from voting. **Wave 4** (robustness): high-card int-as-ID detector; near-zero variance filter; minority-class floor warn; knockoff strict lam_min; estimator type-family assert; knockoff numeric-only check; CPI broader except; stability_selection n>=20 floor; NaT in DatetimeIndex warn; `must_exclude_strict=True` raises on typos; smoothing-on-sparse-N warn; no-improve counter only on `was_stored`; `n_features_one_se_(direction='min'|'max')` parameterised. **Wave 5** (literature): `stability_vs_n_curve_()` + `n_stability_elbow_()` from existing FI dict (Nogueira-Brown 2018); TreeSHAP-based knockoff W-statistic via `knockoff_importance(w_statistic='shap'|'auto')` (KOBT, Jiang-Cheng-Zhao 2021); `importance_getter='boruta_shap'|'powershap'` via optional `BorutaShap`/`arfs`/`powershap` deps; `importance_getter='shap_oof'` alias documenting that the standard 'shap' path is already SHAP-OOF; `prescreen='univariate_ht'`/`'tsfresh'` univariate Mann-Whitney/Kendall/chi-sq + BY-FDR via tsfresh.feature_selection.relevance backend; multi-output y rejected with how-to-loop hint. **Module split:** knockoff helpers carved to sibling `_knockoffs.py` to keep `_helpers.py` under the 1k-LOC soft limit. **Test coverage:** 5 new regression files (`test_wrappers_wave1_fixes.py` ... `test_wrappers_wave5_fixes.py`, 66 new tests). Full wrappers suite: 449 passed. Every algorithm change exposes an opt-out knob; explicit bugs/leaks are unconditional fixes. **Next:** automatic per-task parameter tuning via synthetic-bench-driven defaults (see [feature_selection/wrappers/TODO.md](src/mlframe/feature_selection/wrappers/TODO.md#a-auto-parameter-tuning-via-synthetic-benchmarks-planned)) - inspects (X, y) distributions + col-MI to pick the winning knob-combo from a calibrated table per data-fingerprint.

2026-05-28 ML-metrics audit batch (Tier 1 + Tier 2) + per-target-type fused single-pass blocks. Every standard ML metric missing from `train_mlframe_models_suite` reporting is now computed: binary picks up KS, MCC, Cohen-kappa, balanced accuracy, G-mean, BSS, Gini, F-beta family, specificity / NPV / FPR / FNR, Spiegelhalter Z, Lift@k, Hosmer-Lemeshow chi2 + p-value, Accuracy Ratio. Multiclass picks up top-1 / top-3 / top-5 accuracy, Gorodkin multiclass MCC, ordinal RPS, AND a post-loop **macro / weighted aggregation** across per-class scalars so two multiclass models can be compared by a single number per metric. Regression picks up RMSLE, MAPE-mean (vs the existing maximum-APE), SMAPE, MdAPE, wMAPE, MASE (with `mase_seasonality` knob on `ReportingConfig` + `mase_naive_mae` plumbing), MBE, CV(RMSE), NSE, Explained Variance, Huber, Pearson / Spearman / Kendall / Concordance Index, Poisson / Gamma / Tweedie deviance for non-Gaussian targets. Multilabel picks up LRAP, coverage error, ranking loss, one-error, macro / micro / weighted F1 AND per-label AUC + macro / weighted reductions. LTR picks up DCG@k, ERR, Hit@k, Precision@k. New `quantile.crps_from_quantiles` integrates the proper-scoring-rule CRPS from a discrete quantile grid (trapezoidal). New `_drift.py` module: PSI, KL, JS (with pooled-sample binning for symmetry), Wasserstein-1d, KS-distribution distance. **Fused single-pass blocks** (numba two-pass kernels): `fast_regression_metrics_block_extended` lands 12 metrics in 2 passes (5.8-10.5x vs separate calls at N=10k-5M); `fast_binary_confusion_metrics_block` 11 metrics (6.6-8.9x); `fast_binary_probability_metrics_block` 6 metrics (2.0-2.9x); `fast_multilabel_classification_metrics_block` 12 metrics (3.8-4.1x at K=20); `fast_multiclass_confusion_metrics_block`. **Title tokens**: KS / MCC / BSS appended to the binary calibration chart title defaults (Gini intentionally NOT a token: =2\*AUC-1, redundant); regression chart title switches to a token-based template with default MAE / RMSE / MaxError / R2 / RMSLE / Spearman / MBE. Registry direction table extended to cover every new metric for canonical higher-vs-lower-is-better lookups. See [CHANGELOG.md](CHANGELOG.md#2026-05-28-ml-metrics-audit) for the full list + measured speedups.

2026-05-20 composite-discovery + polynom-FE parallel/GPU expansion. Two new parallel-dispatch refactors land in ``CompositeTargetDiscovery``: Phase B ``_tiny_model_rerank`` now runs the per-spec loop through ``joblib.Parallel(backend="threading")`` (new ``tiny_rerank_n_jobs`` knob; 0 = auto, 1 = serial, default 1 for back-compat); Phase A flattens the previously-nested ``(base, transform)`` loops into a single parallel dispatch so ``discovery_n_jobs`` saturates even when a base has fewer eligible transforms than CPU cores. Per-base setup (``base_train`` extraction, ``x_remaining_matrix``, pre-binning, ``mi_y_for_base``) stays serial because it writes ``_auto_base_pool`` and is cheap relative to MI compute. End-to-end speedup on n=200k synthetic discovery bench: **1.55-1.85x** at ``tiny_rerank_n_jobs=4 + discovery_n_jobs=4``. Bit-for-bit equivalence vs serial pinned by tests. **Plug-in MI batch path picks up a cupy port** (``_plugin_mi_classif_batch_cuda`` via ``cp.argsort`` -> rank-to-bin -> single ``cp.bincount`` across flat ``(col, bin, class)`` indices); auto-routed by ``plugin_mi_classif_batch_dispatch`` through the per-host ``pyutilz.system.kernel_tuning_cache`` (same infrastructure powering ``joint_hist_batched`` since WAVE 4; 21 measured ``(n, k)`` regions persist in the cache JSON, first-run sweep ~30s, subsequent lookups ~1ms). Measured **2.17x at n=200k k=20**, **3.20x at n=1M k=20**, **3.06x at n=1M k=50**; max numerical diff vs njit is ~1e-15 (fp64 round-off only). Suite-level ``PipelineCache`` HIT integration test guards against content-key regression for shared-requirement strategy pairs. **Same-day follow-up wave** lands four polynom-FE micro-optimisations: NEW-A pre-computes ``best_trivial_pair`` ONCE per pair (was once per ``fe_smart_polynom_iters`` restart; **1.58x** on 5-restart loop, 617ms saved per pair); NEW-B flips the basis-matrix GEMV gate ON by default after a hardware-current re-measure (**1.13-1.19x** vs Horner at 1500-element inner CMA scale; previous 2026-05-18 "zero speedup" verdict was hardware-specific); NEW-D adds CMA-ES plateau early-stop via ``early_stop_no_improve_gens`` translating the existing Optuna-trial budget into CMA generations (**5.29x** on XOR-like target, identical or marginally higher MI); NEW-E investigation found 91% of plug-in MI time at n=1500 is ``np.argsort`` inside numba (not the histogram math), invalidating the user-proposed "Cython 20% over numba" premise; ``plugin_mi_classif_fast`` helpers added that hoist argsort to pure numpy then dispatch histogram math to njit (**2.51x at k=1** but slower than ``njit_par`` at k>=5 which is the hot path; helpers exposed but not auto-routed). See [CHANGELOG.md](CHANGELOG.md#2026-05-20).

2026-05-19 MRMR GPU stack closure -- WAVE 1-5 cumulative **2.37x speedup** on ``MRMR.fit`` at N=1M, p=30, fe_npermutations=50 (min-of-3, varied seeds, cold ``_FIT_CACHE``; 23.75s baseline -> 10.01s on GTX 1050 Ti cc 6.1). Five layered waves: (1) joblib threading + njit ``batch_pair_mi_prange``; (2) CUDA RawKernels for joint-histogram (shared-mem at ``joint_size<=4096``, global-atomic above); (3) cupy streams + per-thread CUDA device pin for multi-GPU joblib safety; (4) persistent per-host ``kernel_tuning_cache`` (pyutilz schema-v1 JSON at ``~/.pyutilz/kernel_tuning/<hw_fingerprint>.json``, cross-process safe via filelock + merge-on-write, auto-tune sweep on first run, online relearn behind ``MLFRAME_KTC_ONLINE_LEARN=1``, CLI under ``python -m mlframe.feature_selection._benchmarks.kernel_tuning_cache.cli``); (5) FE transformer unary cupy elementwise wire (1/4 shipped; 2-4/4 measurement-rejected -- ensembling already inherits fit-time device, composite_estimator H2D=18ms eats the 1ms GPU FMA win, votenrank is sub-1% fit-time). Cross-HW fallback (`_FALLBACK_BY_CC` covers cc 5/6/7/8/9; joint_size>4096 forces global regardless of cc). 21 utility ``@njit`` kernels also gained ``cache=True`` for cold-import amortisation. See [CHANGELOG.md](CHANGELOG.md#2026-05-19) + [docs/WAVE5_GPU_ROADMAP.md](docs/WAVE5_GPU_ROADMAP.md).

2026-07-27 transformer FE subpackage grew to **86 mechanisms across 86 iterations**, with 6 standing records on the four breakthrough-candidate datasets (kin8nm regression / abalone regression / mammography 1.3%-positive binary / diabetes balanced binary). All mechanisms are stateless, OOF-disciplined, reproducible from seed; downstream consumers are CatBoost / LightGBM / XGBoost (single-iter to 300-iter, depth 3-6, default sklearn-API hyperparams).

**Iter records (lift over raw baseline; same KFold seed / harness across all iters)**:

| Dataset | Metric | Lift | Iter | Mechanism |
|---|---|---|---|---|
| **abalone** | XGB R² | **+4.05%** | iter 61 | multi-temp residual-bands + cdist |
| **abalone** | CB R² | **+3.84%** | iter 69 | baseline-disagreement-as-feature + cdist |
| **abalone** | LGB R² | **+3.19%** | iter 72 | local density gradient ‖∇log p̂(x)‖ alone (pure-X, no baseline) |
| **mammography** | LGB AUC | **+14.46%** | iter 66 | class-balanced hard-row anchors + RFF |
| **kin8nm** | LGB R² | +11.91% (marginal) | iter 68 | multi-baseline (LGB d=3 + LGB d=5 + Ridge) hard rows + RFF |
| **diabetes** | CB PR_AUC | **+6.75%** (marginal) | iter 77 | local curvature via K=40 NN quadratic fit alone |

Also iter 77 hit **ALL-5-metrics × ALL-3-boostings positive on diabetes** (AUC + Brier + PR_AUC + LogLoss + Accuracy), only the 2nd such achievement (1st was iter 17 rfprox+multitemp).

**The recipe that worked**: small per-fold-trained baselines (50-iter LGB d=3 + LGB d=5 + Ridge / LogReg) produce residuals / predictions / disagreement statistics; those become bands (iter 60-63 residual-quintile-bands), anchors (iter 65 hard-rows by `|residual|`, iter 66 class-balanced hardest pos+neg, iter 68 multi-baseline-z-residual), or direct features (iter 69 3-baseline preds + std + range + pair-diffs = 8 features). Standard Kaggle terminology: **OOF predictions as level-1 features / stacked generalization** (Wolpert 1992).

**Pure-input-X geometric mechanisms (iter 72-86) dominated the final session** (2/15 ideas from a 3-agent synthesis set records): density gradient + local curvature on kNN neighborhoods are the structurally cleanest signals, completely orthogonal to residual-based mechanisms. Hybrid features (per-row baseline surprise, pairwise KL divergence, local intrinsic dimension via PCA spectrum, robustness budget under noise injection, prediction-quintile bands, counterfactual feature substitution, adversarial flip distance, gradient direction agreement via Jacobian cosine, Fisher-weighted residual bands, predictive info delta, decision region depth via isotropic probes, IB-quantized baseline codes, geodesic distance via kNN-graph Dijkstra, persistence diagram via gudhi RipsComplex) gave modest signal but no records.

Honest negatives documented in `RESULTS.md`:
- Iter 71 NN-target-mean in OOF embedding (Home Credit 1st-place pattern): K=200/500 optimized for 300k-row competitions catastrophically over-smoothed on diabetes 614-row train fold (small-N is a different regime).
- Iter 62 signed-residual bands: heavy-tailed regression residuals (abalone) collapsed Q1 band centroid into outlier region; -22 to -32% R² standalone.
- Iter 64 prediction-quintile bands: ŷ is a function of X, so bands carry signal partially redundant with the downstream booster's own splits — residuals beat predictions.

See `src/mlframe/feature_engineering/transformer/RESULTS.md` for the full 86-iter narrative.

2026-05-18 audit-fixes wave (`audit-fixes-2026-05-16`, since merged to master). Closes the honest-gaps list from the TVT production log analysis - 21 distinct items across critical bugs, perf, observability, tests, and bench scripts (T-IDs ``T1#1/3/4/5/6/7/9 + T2#8/10/11/12/13/16 + T3#14/18/19/20/22/23/24/25``). Highlights: Pack #10 NameError fix, Pack G universal watchdog (now catches multiplicative-transform divergence; +72.6% measured overhead so a new `enable_wrap_pass_watchdog` config knob lets production opt out after staging-side verification), parallel composite-candidate evaluation infrastructure (`discovery_n_jobs > 1`, MEASURED ~1.05x at n_jobs=8 on n=200k — joblib threading DOES dispatch correctly but the parallel block covers only ~20% of total compute; the remaining 81% sits in serial `_tiny_model_rerank` LightGBM training, which is a separate follow-up to parallelize; the originally-claimed "5-10x" was speculative inherited language and is NOT validated by measurement), Hermite EngineeredRecipe + predict-time replay (closes the "88-min Optuna found but transform couldn't reproduce" gap), `recover_composite_y_scale_metrics` lazy-recovery helper for `skip_wrap_pass_predict=True` callers, `use_stacked_discovery_residual` flag wiring `fit_stacked_on_residual` into the suite, MRMR identity-cache thread-safety, opt-in y-fingerprint cache mode. New benchmarks: `profiling/bench_stacked_discovery_default_flip.py` (verdict: keep False, even on true residual-of-residual non-linear synthetic), `profiling/bench_parallel_discovery_speedup.py` (1.05x measured), `profiling/bench_parallel_discovery_diag.py` (root-cause: 81% serial tail in `_tiny_model_rerank`), `profiling/bench_pack_g_watchdog_overhead.py` (+72.6%), `profiling/bench_hermite_ram_and_df_mutation.py` (pandas vs polars memory equivalent).

2026-05-17 new `mlframe.feature_engineering.transformer` subpackage. Three frozen "transformer-style" FE blocks for tabular models (CatBoost / LightGBM / XGBoost / linear): Random Fourier Features (`compute_rff_features`), sinusoidal positional encoding (`compute_positional_encoding` + `positions_within_group`), and multi-head softmax-weighted kNN-target-encoding over random subspaces (`compute_row_attention`). The pipeline scales to 1-10M rows by 10k-20k cols via a mandatory random-projection front-end and an hnswlib ANN backend; GPU (cupy) accelerates the two stages where it actually wins on this scale (RFF matmul with pinned-memory streaming, row-attention stage-4 fused RawKernel). All other stages (projection in Mode A, ANN build/query, standardisation, positional encoding) are CPU only by design — H2D bandwidth at d=20k dominates GPU compute and cuVS has no Windows wheel. Honest framing: without backprop, "attention" here is (a) random-projection nonlinear feature maps (RFF), (b) softmax-weighted kNN-target-encoding (row-attn) — useful techniques for trees / linear, but the "transformer" name is structural, not algorithmic. Strict OOF discipline on train (mirrors `bruteforce._kfold_target_encode`); train-only key-bank for val / test / OOS / holdout; `seed` is a required positional argument so derived-from-data seeds can't silently leak. Three biz_value harnesses with hard-pass lift thresholds (RFF >= 0.30 R² lift for Ridge; row-attention >= 0.03 AUC lift vs raw AND >= 0.01 lift vs plain kNN-target-encoding) gate the block's "is this theatre?" question. New extras `mlframe[transformer]`, `mlframe[transformer_ann]` (hnswlib), `mlframe[transformer_gpu]` (cupy-cuda12x), `mlframe[transformer_full]`.

2026-05-16 multi-agent audit overhaul of `train_mlframe_models_suite` (branch `audit-fixes-2026-05-16`, 83 commits). User-facing highlights:

- **Honest stacking and calibration.** Level-1 stacking now consumes K-fold OOF predictions instead of in-sample train_preds; val rows are reserved for early stopping only; opt-in `TrainingSplitConfig.calib_size` carves a dedicated calibration slice so `post_calibrate_model` never fits on test rows.
- **Cross-target cache correctness.** y content + target name are now folded into every FS cache key (MRMR `_FIT_CACHE`, target-encoder `InMemoryKey`, RFECV dedup / skip_retraining, PipelineCache digest, `composite_cache` data_signature including row-order). Long-path-safe cache dirs and atomic-write `fsync` before `os.replace`.
- **New ensembling.** Per-quantile blending preserves the K-quantile dimension; RRF + Borda rank-fusion wired as a `LearningToRankConfig.ltr_ensemble_method` and a `score_ensemble` flavour; recurrent models (LSTM / GRU / Transformer) are now consumed by `score_ensemble`.
- **Polars-fastpath speedups.** Drift-snapshot lazy plan 3.56x; per-col auto-detect single-pass 2.74x; per-group baseline polars dispatch 2.57x; polars-native NaN-guard 7.32x; content-fingerprint point-sample 762x at 100k rows / 2740x at 1M; LightGBM shim now uses an Arrow split-blocks bridge for polars X.
- **`sample_weight` plumbed end-to-end.** MRMR (row-resample), RFECV (CV fold + scorer), Ridge / NNLS composite stack solvers (NNLS uses sqrt-weight row scaling), and `LeakageSafeEncoder` all accept weights; default-None preserves byte-for-byte legacy. Gated by `FeatureSelectionConfig.use_sample_weights_in_fs` (default OFF) to preserve FS cache reuse across weight-only variations.
- **Predict-path hardening.** PySR equation column names content-hashed and persisted; datetime methods persisted at suite level with FTE-emitted skip; `_apply_nan_guard` runs polars-native impute + scale; `predict_from_models` skips unfitted placeholder pre_pipelines and resolves the iter#80 lgb+hgb mixed-mode regression.
- **`BorutaSHAP` wired** into `_build_pre_pipelines` behind `FeatureSelectionConfig.use_boruta_shap` (accepts polars; warning filter scoped to `fit`).
- **Precompute bundle.** New `TrainMlframeSuitePrecomputed` dataclass + `precompute_all` one-shot helper lets long-running services hand pre-baked stats into `train_mlframe_models_suite(precomputed=...)` and skip the inline compute.

New / changed config knobs (see CHANGELOG.md for full descriptions):

| Config class                  | Field                                      | Default                | Purpose                                                                       |
| ----------------------------- | ------------------------------------------ | ---------------------- | ----------------------------------------------------------------------------- |
| `FeatureSelectionConfig`      | `use_sample_weights_in_fs`                 | `False`                | Opt-in so FS cache reuses across weight-only variations                       |
| `FeatureSelectionConfig`      | `use_boruta_shap` + `boruta_shap_kwargs`   | `False` / `None`       | Wire BorutaSHAP into `_build_pre_pipelines`                                   |
| `FeatureSelectionConfig`      | `rfecv_leakage_corr_threshold`             | (previously hardcoded) | Expose RFECV leakage corr cutoff                                              |
| `FeatureSelectionConfig`      | `rfecv_mbh_adaptive_threshold`             | (previously hardcoded) | Expose RFECV MBH adaptive-surrogate threshold                                 |
| `FeatureTypesConfig`          | `cat_text_cardinality_threshold_pct`       | size-aware             | Replaces absolute threshold; scales with row count                            |
| `TrainingSplitConfig`         | `calib_size`                               | `0.0` (off)            | Opt-in calibration slice; keeps `post_calibrate_model` off test rows          |
| `EnsembleConfig`              | `diversity_corr_warn_threshold`            | -                      | Near-duplicate member WARN threshold (no drop)                                |
| `LearningToRankConfig`        | `ltr_ensemble_method`                      | `"mean"`               | One of `mean` / `rrf` / `borda`                                               |
| `OutlierDetectionConfig`      | `apply_to_feature_selection`               | -                      | Gate outlier mask propagation into FS fits                                    |
| MRMR `__init__`               | `runtime_max_mins` default                 | `300`                  | Raised from the previous lower value                                          |

Known follow-ups not landed in this branch are listed at the bottom of the CHANGELOG section.

## Installation

```bash
pip install mlframe                       # core: numpy + pandas + scipy + sklearn + pyutilz
pip install mlframe[boosting]             # catboost + lightgbm + xgboost
pip install mlframe[calibration]          # shap + venn-abers + netcal + betacal + pycalib
pip install mlframe[neural]               # torch + pytorch-lightning
pip install mlframe[automl]                # flaml (HPO)
pip install mlframe[feature_engineering]    # pysr (symbolic regression) + optbinning (WoE / monotonic binning)
pip install mlframe[sampling]             # imbalanced-learn + category-encoders + iterative-stratification
pip install mlframe[polars_ext]            # polars-talib + polars-ds (polars itself is a core dep)
pip install mlframe[viz]                  # matplotlib + plotly + seaborn + altair + hvplot
pip install mlframe[mlflow]               # mlflow experiment tracking
pip install mlframe[db]                   # sqlalchemy + psycopg2 + duckdb + pymongo + zstandard
pip install mlframe[signal]               # antropy + astropy + pywavelets + ruptures
pip install mlframe[unsupervised]         # hdbscan + umap-learn
pip install mlframe[stats]                # statsmodels
pip install mlframe[gpu]                  # cupy + gpu-info (match your CUDA build)
pip install mlframe[transformer]          # transformer-style FE: numba-only CPU path (RFF + PE + row-attention without ANN)
pip install mlframe[transformer_ann]      # adds hnswlib so compute_row_attention can use approximate-NN at N >= 500k
pip install mlframe[transformer_gpu]      # adds cupy-cuda12x for the two GPU stages (RFF matmul + row-attention stage 4)
pip install mlframe[transformer_full]     # transformer + ann + gpu (recommended for production boxes)
pip install mlframe[all]                  # all runtime extras above
pip install mlframe[dev]                  # pytest + coverage + ruff + black + mypy + bandit + pre-commit
```

Requires Python 3.9+. Tested on 3.9 through 3.13.

For development from source:

```bash
git clone https://github.com/fingoldo/mlframe.git
cd mlframe
pip install -e ".[boosting,calibration,viz,dev]"
pre-commit install
pytest
```

`mlframe` depends on [`pyutilz`](https://github.com/fingoldo/pyutilz) (sibling utility library: parallel execution, pandas/polars helpers, hardware introspection).

## Modules

| Sub-package                       | Purpose                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| `mlframe.training`                | End-to-end training pipeline: `train_mlframe_models_suite`, per-model strategies, configs, feature handling, dummy baselines, AutoML, neural nets |
| `mlframe.feature_engineering`     | Numerical / time-series / financial / categorical / Hurst / MPS features, brute-force PySR search |
| `mlframe.feature_selection`       | MRMR (multiple variants), RFECV, Boruta-SHAP, mutual-information, optbinning filters and wrappers |
| `mlframe.metrics`                 | ICE, ECE, Brier decomp (REL/RES/UNC), CMAEW, KS, MCC, Cohen kappa, balanced accuracy, G-mean, BSS, Gini, F-beta, specificity / NPV, Spiegelhalter Z, Lift@k, Hosmer-Lemeshow, Accuracy Ratio, top-k accuracy, RPS, multiclass MCC, RMSLE, MAPE / SMAPE / MdAPE / wMAPE / MASE, MBE, CV(RMSE), NSE, Explained Variance, Huber, Pearson / Spearman / Kendall / C-index, Poisson / Gamma / Tweedie deviance, LRAP, coverage / ranking / one-error, per-label F1 + AUC (macro / micro / weighted), DCG@k / ERR / Hit@k / Precision@k, CRPS-from-quantiles, PSI / KL / JS / Wasserstein / KS-distribution, calibration + quantile + ranking plots |
| `mlframe.evaluation`              | Performance reporting across cv folds + holdout sets                                     |
| `mlframe.calibration`             | Calibration-quality diagnostics, isotonic / Platt / beta / Venn-Abers post-hoc calibrators |
| `mlframe.models`                  | Ensembling (stack / blend / vote), hyperparameter optimisation, splitting strategies     |
| `mlframe.estimators`              | sklearn-compatible custom estimators, early-stopping aware wrappers, pipelines           |
| `mlframe.preprocessing`           | NaN cleaning, scaling, outlier handling, clustering for preprocessing                    |
| `mlframe.inference`               | Batch and streaming prediction, SHAP / permutation explainability                        |
| `mlframe.reporting`               | Matplotlib + plotly chart backends; spec-driven panel rendering                          |
| `mlframe.core`                    | numba-accelerated array ops, statistical helpers, EWMA                                   |
| `mlframe.data`                    | Built-in / synthetic dataset generators                                                  |
| `mlframe.testing`                 | Parametric frame generation for property-based tests                                     |
| `mlframe.integrations`            | Optional third-party integrations (MLflow)                                               |
| `mlframe.utils`                   | EDA, experiments, text, miscellaneous helpers                                            |

## Quick examples

**One-call multi-model training and evaluation.** The suite trains every model with its native preprocessing strategy (CatBoost / LightGBM / XGBoost run on raw frames; linear and neural pipelines get encoded + imputed + scaled). It returns per-model predictions, fit-time metrics, calibration diagnostics, and a ready-to-render reporting spec:

```python
from mlframe.training import train_mlframe_models_suite

result = train_mlframe_models_suite(
    df=df,
    target="y",
    models=["cb", "lgb", "xgb", "hgb", "mlp"],
    regression=False,
    cv_folds=5,
    early_stopping_rounds=50,
    use_polars=True,
)

result.models["cb"].metrics["holdout_brier"]
result.models["lgb"].calibration_plot()
result.ensemble("stack").predict_proba(X_new)
```

**Composite-target stacking.** Train K base learners on `y`, fit a meta-learner on out-of-fold residuals to capture targeted error structure. sklearn-compatible (clone, get_params, sample_weight, runtime stats callback), with cross-sklearn-version behaviour pinned by CI:

```python
from mlframe.estimators.custom import CompositeTargetEstimator

est = CompositeTargetEstimator(
    base_estimator="lgb",
    meta_estimator="ridge",
    cv=5,
    target_transform="residual",
)
est.fit(X_train, y_train)
est.feature_importances_       # aggregated across folds
```

**Per-target metric panel.** Computes Brier reliability/resolution/uncertainty decomposition, ICE bands, ECE, CMAEW, with proper handling of class imbalance and small-bin variance:

```python
from mlframe.metrics import compute_calibration_report

report = compute_calibration_report(
    y_true=y_test,
    y_proba=clf.predict_proba(X_test)[:, 1],
    n_bins=15,
    method="quantile",
)
print(report.brier_rel, report.brier_res, report.brier_unc, report.ece)
```

**MRMR / RFECV feature selection.** Several MRMR variants (FCQ, MID, FCD, plus n-way interaction extensions) plus an RFECV wrapper that handles LightGBM / XGBoost / CatBoost feature-importance access correctly:

```python
from mlframe.feature_selection.filters.mrmr import mrmr_classif
from mlframe.feature_selection.wrappers import RFECVCustom

# Filter method (MRMR FCQ variant)
selected = mrmr_classif(X, y, K=30, scheme="fcq")

# Wrapper (RFECV with non-sklearn-native importance fallback)
rfe = RFECVCustom(estimator=LGBMClassifier(), step=0.1, cv=5)
rfe.fit(X, y)
```

**SHAP-proxied feature selection (`ShapProxiedFS`).** Trains one model on all features, computes SHAP values once, then approximates the OOS prediction of a model trained on any feature subset `S` by the coalition value `base + sum_{j in S} phi_j` - so subsets can be *ranked* without retraining (~450x faster per subset than an honest retrain in-repo). The cheap proxy ranking is then honestly re-validated on a disjoint holdout to pick the final subset. Exact numba/CUDA brute force for `n <= ~22`, else beam / greedy / genetic / annealing / gradient ("Schrodinger gates") backends. Ships a proxy-trust guard (measures proxy-vs-honest rank fidelity on your data) and an importance-top-k ablation (verifies it beats plain SHAP importance). Honest about its limit: the proxy under-credits subsets that drop features whose signal correlated survivors could recover (the "<50% coverage" wall), which the trust guard surfaces.

```python
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

sel = ShapProxiedFS(classification=True, metric="brier", optimizer="auto")
sel.fit(X, y)                       # one model + OOF SHAP -> rank subsets -> honest re-validate top-N
print(sel.selected_features_)       # names, in input-column order
report = sel.shap_proxy_report_
print(report["trust"]["spearman"])              # measured proxy fidelity on this data
print(report["importance_ablation"]["proxy_wins"])  # beat SHAP-importance-top-k?
X_sel = sel.transform(X)
```

For wide data (hundreds-to-tens-of-thousands of features) it scales via a native-importance pre-filter + correlated-feature clustering (GPU correlation matmul -> denoised cluster representatives) + SHAP-importance pre-screen, so the exhaustive-approx search runs on a small set of units; the chosen units are expanded and pruned back to compact real columns. The pre-filter (one ranking pass on ALL columns before the expensive SHAP) is the dominant wide-data cost; `prefilter_method` trades speed for interaction-awareness — `"model"` (faithful full booster), `"univariate"` (vectorised O(n*f) ANOVA F-score, ~300x faster but marginal-only), `"fast_model"` (reduced-budget booster, ~5x faster and still interaction-aware), `"gpu_model"` (the full fit on `device="cuda"`, wins at high row counts), `"two_stage"` (cheap O(n*f) F-score funnel to `min(2000, 0.2*n_features)` survivors, then the capped booster on the cohort — measured 4.8x at 6k features with 12/12 informative recovery preserved), and the default `"auto"` keeps the quality-safe `"model"` for moderate widths, `"fast_model"` for mid-wide, and routes very wide data (`n_features >= 8000`) to `"two_stage"` / `"gpu_model"` (per-HW crossover via kernel_tuning_cache). Optional levers (all opt-in): `interaction_aware=True` (SHAP-interaction coalition for XOR/multiplicative signals), `config_jitter`+`uncertainty_penalty` (stabler attributions + penalise unstable subsets), `active_learning=True` (disagreement-driven re-validation), and a bias corrector on by default. `ShapProxiedFS.preflight(X, y)` returns a cheap run/caution/fallback recommendation before a full fit. `_shap_proxy_compose` adds proposal-generator seeding and per-fold stability ensembling.

```python
sel = ShapProxiedFS(classification=True, cluster_features=True, prefilter_top=2000,
                    interaction_aware=True, config_jitter=True, uncertainty_penalty=0.3)
print(ShapProxiedFS.preflight(X, y, classification=True)["recommendation"])  # run / caution / fallback
```

**Friend-graph post-analysis.** After screening, the `MRMR` estimator builds a "friend graph" of the selected features (node = feature sized by entropy, edge = pairwise mutual information, arrow = asymmetric dependency, color = green/unique / red/suspected-sink / yellow/middling). It flags a "universal soldier" feature - one correlated with many genuine predictors but carrying no unique target information - that greedy mRMR can otherwise promote early and let distract strong models. The graph is exposed on the fitted estimator, summarized into the training suite's `feature_selection_report`, and rendered (interactive plotly HTML + static image) through the reporting backends. Diagnostic by default; pruning is opt-in:

```python
from mlframe.feature_selection.filters.mrmr import MRMR

sel = MRMR(build_friend_graph=True, friend_graph_prune=False).fit(X, y)
g = sel.friend_graph_
print(g.suspected_garbage)          # names of suspected redundant sinks
print(g.to_meta()["class_counts"])  # {'green': ..., 'red': ..., 'yellow': ...}

# Opt in to actually drop the suspected sinks from support_ (protects the
# neighbour that carries each removed feature's unique target information):
pruned = MRMR(friend_graph_prune=True).fit(X, y)
```

### Adaptive Feature Engineering & Param-Oracle

**The problem.** `MRMR` accumulated ~50 opt-in `fe_*_enable` / `dcd_*` feature-engineering generators (Layers 33-95), each one a master switch that pays off only on a specific data shape and adds compute everywhere else. A user who has not read 50 docstrings does not know which to flip, so in practice the entire FE zoo stays off and the wins go unrealised. Three layers (98-100) collapse that decision surface into one knob, backed by a meta-learning cache.

**`fe_auto=True` — the 1-knob FE mode.** `MRMR(fe_auto=True).fit(X, y)` cheaply fingerprints `(X, y)` *before* the FE stages run and flips exactly the master `fe_*_enable` flags whose data-shape precondition is met for this fit. It only ADDS generators — a flag you set `True` yourself is never turned off — and the original constructor values are restored after `fit` so semantics stay stable across fits / `clone` / pickle. Default `fe_auto=False` keeps the byte-identical legacy path. The rule-based recommender (`mlframe.feature_selection.filters._meta_fe_recommender.recommend_fe_flags_by_rules`) needs no history; it is the cold-start prior, and maps:

| Data-shape precondition | Master flags enabled |
| --- | --- |
| int-as-cat group column (distinct count 3..500) | `fe_grouped_agg_enable`, `fe_composite_group_agg_enable`, `fe_grouped_quantile_enable` |
| object / category columns | `fe_count_encoding_enable`, `fe_frequency_encoding_enable`, `fe_cat_pair_enable` |
| ... + >= 3 categorical columns | `fe_cat_triple_enable` |
| ... + a continuous numeric column | `fe_cat_num_interaction_enable` |
| time column + entity column | `fe_temporal_agg_enable` |
| any column NaN rate >= 1% | `fe_missingness_indicator_enable` |
| heavy-tailed / skewed continuous | `fe_hybrid_orth_enable` |
| anchored / discrete-pattern numeric | `fe_numeric_decompose_enable`, `fe_modular_enable` |

A clean Gaussian frame (no cats / groups / time / NaNs) returns all-False — no spurious enables. `MRMR.recommend_enabled_fe(X, y)` exposes the same recommendation (its `recommended_enable` list) plus the static flip-safety taxonomy without running a fit.

```python
from mlframe.feature_selection.filters.mrmr import MRMR

# One knob: fingerprint the data, auto-enable the generators that fit its shape.
sel = MRMR(fe_auto=True).fit(X, y)

# Inspect what would be enabled, without fitting:
print(MRMR.recommend_enabled_fe(X, y)["recommended_enable"])
```

**Layer 104 — three more recipe-based FE families (all default OFF).** Each fills a genuine gap in the L21-L103 chain, replays leak-safely from a frozen recipe that reads only `X`, and is byte-identical to the legacy path when disabled.

- **Rare-category indicator + frequency-band** (`fe_rare_category_enable`): a category being RARE is itself predictive (a rare merchant id is a fraud signal). Emits `is_rare_{col}` (fit-time frequency below `fe_rare_category_threshold`, default 1%) and an ordinal `freq_band_{col}` (very-rare / rare / common / dominant). MI-gated against the Layer-91 raw-baseline floor.
- **Cross-feature conditional residual, NUM x NUM** (`fe_conditional_residual_enable`): extends the Layer-34 cat-num residual to `x_i - E[x_i | bin(x_j)]` — how far `x_i` deviates from its conditional expectation given `x_j`'s quantile bin (income high FOR this age bracket). The pair pool is cardinality-bounded by the top raw-MI columns (`fe_conditional_residual_max_pair_cols`) and MI-gated; the recipe stores the `x_j` bin edges + per-bin mean of `x_i`.
- **RankGauss / rank-Gaussianisation** (`fe_rankgauss_enable`): maps each numeric to its rank-based Gaussian quantile, materially helping downstream linear / NN models. RankGauss is strictly MONOTONE, so by the data-processing inequality it CANNOT add mutual information about `y` — it is therefore NOT MI-gated (the value is downstream, not in MI). The pool is bounded by raw marginal MI; the recipe stores the sorted fit values and interpolates each test value's rank at replay.

**`fe_hybrid_orth_default_scorer="auto_oracle"` — unified scorer selection.** The univariate basis-selection stage can route through any of 14 scorer modes. `"auto_oracle"` (Layer 100, `_oracle_scorer_select.OracleScorerSelector`) unifies the two prior adaptive paths under the Param-Oracle: the Layer-76 fingerprint rule cascade is the cold-start prior, the expensive Layer-68 all-scorers bake-off runs ONCE (`benchmark_all_scorers`) to populate the oracle, and once a fingerprint bucket has confident history the learned-best scorer over `{plug_in, ksg, copula, dcor, hsic, jmim, tc, cmim}` wins. The legacy `"auto"` (L68) and `"meta"` (L76) values keep working unchanged; `"auto_oracle"` is the new learning path.

**Param-Oracle (`mlframe.utils._param_oracle.ParamOracle`) — the meta-learning cache.** Almost every hot decision in this codebase (which MI scorer, which CUDA kernel variant, which FE recipe) has an *input-dependent* optimum — a function of the shape and statistics of the data, not a constant you can hardcode. The Param-Oracle generalises that into a small learning-to-optimize loop:

- **fingerprint -> best-param.** `default_fingerprint(args, kwargs)` returns a dict of scalars only — `n`, `p`, `dtype_kind`, `sparsity`, `mean_abs_skew`, `mean_kurtosis`, `cardinality_mean`, `mean_abs_corr` — computed from the first array/DataFrame-like argument. `recommend(fp, fn_name=...)` resolves a new fingerprint via exact log-bucket match -> k-NN in fingerprint space -> global best -> caller default, gated by a `min_observations` confidence threshold.
- **stat-only persistence.** The on-disk parquet store NEVER holds raw arrays — only scalar fingerprint stats, the param combo, and the scalar objective. Tiny, privacy-safe, portable.
- **sibling of `pyutilz.system.kernel_tuning_cache`, not a replacement.** It deliberately *reuses* that cache's proven patterns — the same per-host `hw_fingerprint()` key, the same `$PYUTILZ_KERNEL_CACHE_DIR` / `~/.pyutilz` layout convention (under a `param_oracle/` subdir), log-scale size bucketing, and filelock-guarded atomic merge-on-write — and does NOT modify it. Where it differs: the kernel cache stores a hand-emitted region table of discrete choices; the Param-Oracle stores raw observations and learns the mapping via continuous k-NN over a richer statistical fingerprint.
- **3 modes.** `"benchmark"` sweeps every combo of `param_space` and records each; `"inference"` only recommends, never sweeps; `"hybrid"` is epsilon-greedy — exploit the recommendation with probability `1 - epsilon`, explore a random combo otherwise, recording every call.
- **consumer roadmap.** FE-flag recommendation (Layer 99, done), scorer selection (Layer 100, done), and HPO warm-start (planned: seed hyper-parameter search from the per-fingerprint best instead of a cold start).

Decorate a function so each call is governed by the oracle's mode (the instance itself is the decorator; it records the timed objective and learns over runs):

```python
from mlframe.utils._param_oracle import ParamOracle

oracle = ParamOracle(
    "my_kernel.parquet",                  # bare name -> resolved under the param_oracle store dir
    param_space={"variant": ["njit", "cuda"], "block": [128, 256]},
    mode="hybrid",                        # epsilon-greedy explore/exploit
    minimize="elapsed_s",
    epsilon=0.1,
)

@oracle
def my_kernel(X, variant="njit", block=128):
    ...                                   # heavy compute; oracle picks variant+block by data fingerprint
    return result
```

**Worked example — the full auto pipeline.** Combine both knobs so MRMR fingerprints the data, auto-enables the matching FE generators, and routes scorer choice through the learning oracle:

```python
sel = MRMR(fe_auto=True, fe_hybrid_orth_default_scorer="auto_oracle").fit(X, y)
print(sel.support_)                                   # selected-feature mask
print(MRMR.recommend_enabled_fe(X, y)["recommended_enable"])  # which generators fired
```

**Time-series and financial feature engineering on Polars.** Windowed aggregation, ACF, Hurst exponent, TA-Lib indicators, market-wide rolling features. Most extraction paths run Polars-native without copying to pandas:

```python
from mlframe.feature_engineering.timeseries import create_aggregated_features
from mlframe.feature_engineering.financial import compute_market_features

agg = create_aggregated_features(
    df, value_col="price", time_col="ts",
    windows=[5, 15, 60],
    aggs=["mean", "std", "min", "max", "acf1"],
)

market = compute_market_features(df, ohlc_cols=("open", "high", "low", "close"))
```

**Post-hoc probability calibration.** Compare Venn-Abers, isotonic, Platt, beta, and per-class isotonic on the same out-of-fold preds; pick the calibrator that minimises Brier reliability without inflating reduction:

```python
from mlframe.calibration.post import select_best_calibrator

best, scores = select_best_calibrator(
    y_true=y_val,
    y_proba=oof_proba,
    candidates=["isotonic", "platt", "beta", "venn_abers"],
    objective="brier_rel",
)
calibrated = best.transform(test_proba)
```

**Batch inference with adaptive worker count.** Worker count adapts to available RAM; results stream back with proper exception propagation:

```python
from mlframe.inference.predict import batch_predict

preds = batch_predict(
    model=clf,
    X=large_frame,
    batch_size=10_000,
    n_workers="auto",
)
```

## Caching strategy

mlframe runs two distinct caching layers in the training suite. They deliberately use different key strategies because the lifetime of the cached value, the cost of computing the cache key, and the failure mode of a stale hit are all different. Mixing the strategies (or "unifying" them under one keying scheme) was attempted once and reverted: it either made the cheap layer expensive or made the expensive layer unsafe.

**Layer 1: `_PRE_PIPELINE_CACHE` (content-keyed, `_pipeline_helpers.py`).** Caches the output of `(SimpleImputer + StandardScaler + feature selectors).fit_transform(train_df, val_df)` so consecutive models in the same suite (CatBoost, LightGBM, XGBoost, MLP, linear) that share the same pre-pipeline structure reuse the fitted transforms instead of refitting. Keys are derived from a content fingerprint of `train_df`, `val_df`, the pipeline structural signature, the training target, the target name, and (optionally) sample weights - because two consecutive cache lookups inside one suite call see the SAME Python objects (same `id()`), but the value differs across targets, so id-keying would alias entries across targets and content-keying is the only safe option. The cost of the content hash is amortised across the ~50s pre-pipeline fit that a hit skips, so the upfront fingerprint pass pays for itself.

**Layer 2: `FeatureCache.InMemoryKey` (id-keyed, `feature_handling/fingerprint.py`).** Caches per-column intermediate stats (MRMR scores, target-encoder folds, RFECV ranks) WITHIN a single `train_mlframe_models_suite` call. Keys are tuples of `(session_id, id(train_df), id(train_idx), column, params_canonical_hash, provider_signature)` - `id()` is safe here because the suite holds strong refs to `train_df` and `train_idx` for its entire lifetime, so Python cannot recycle the id under us, and the per-call `session_id` prevents cross-call collisions. The hits land in the inner per-column loops where a content hash per column per lookup would dominate the work the cache is supposed to skip; id keys are O(1) tuple equality and the cache is the inner-loop fastpath. Cross-session reuse is intentionally NOT supported by `InMemoryKey`; persistent cross-session caching is provided by a separate `DiskKey` (content-fingerprint-derived, computed once at suite start in a background thread).

The asymmetry is therefore: layer 1 spans MODEL boundaries inside one suite call (must survive `id()` recycling between consecutive model fits because the loop body deletes intermediate frames, so content-keying is mandatory); layer 2 spans only INNER-LOOP boundaries inside one suite call where the suite-level strong refs guarantee `id()` stability, so id-keying is cheap and safe.

## Design notes

- **Modular, opt-in extras.** Core install pulls `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow`, `joblib`, `tqdm`, `pyutilz`, `pydantic`. Heavy stacks (CatBoost, PyTorch, MLflow, SHAP, plotly) ship as extras; nothing ImportError-s on `import mlframe` because optional deps are lazy-imported at call site.
- **Polars-native where it matters.** Tree models that accept Arrow-backed frames (CatBoost, HGB, XGBoost auto-cast) skip the polars-to-pandas round-trip; non-native models get a zero-copy Arrow view via `pyarrow.Table.to_pandas(zero_copy_only=True)`.
- **sklearn-version pinning.** A dedicated [sklearn-matrix CI workflow](.github/workflows/sklearn-matrix-ci.yml) tests the composite-target wrapper surface against scikit-learn 1.5, 1.6, 1.7, and 1.8 on every PR; attribute-delegation breakage shows up before users hit it.
- **Fuzz-tested.** ~150 pairwise + ~400 3-wise (IPOG-covering) parameter combos run per release, hitting axes the unit tests don't reach. Combo regressions get permanent sensors so they don't recur.

## Roadmap

Operations that do not fit the current pipeline-slot abstractions are parked here pending refactor:

- **Row-wise transformations** (per-sample normalization). `Normalizer(norm="l2"/"l1"/"max")` projects each *sample* onto a unit hypersphere; appropriate for text/embedding similarity but it silently breaks tree-based models (CatBoost / LightGBM / XGBoost / HGB / RF) that rely on absolute feature magnitudes. Was previously routed through `PreprocessingExtensionsConfig.scaler="Normalizer_l2"`; removed 2026-05-15. A dedicated `row_transform` pipeline slot is planned so row-wise ops have an unambiguous home that cannot be confused with column scalers.

## Testing

```bash
pytest                                    # full suite
pytest -m fast                            # representative-subset run (<15s)
pytest -m "not slow and not gpu"          # CI default
pytest --cov=src/mlframe --cov-report=html
```

Markers: `slow`, `integration`, `gpu`, `multigpu`, `benchmark`, `windows_only`, `linux_only`, `fast`.

## Contributing

Pull requests welcome. Code style: `black` + `ruff`, line length 160. Every new feature ships with: a unit test, a quantitative business-value test, a representative `@pytest.mark.fast` subset, and a `cProfile` hotspot check. See [CLAUDE.md](CLAUDE.md) for project conventions and the fuzz / combo test philosophy.

## License

MIT, see [LICENSE](LICENSE).
