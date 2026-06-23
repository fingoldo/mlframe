# Round-4 audit — checklist & dispositions

Legend: PENDING / RESOLVED (fix+test) / DOC / FUTURE / REJECTED. MRMR excluded. Tests parametrized where feasible.

## Docs (DOC*)
[x] DOC1 README.md:138 composite quickstart uses feature_importances_ on Ridge (has coef_) -> AttributeError — RESOLVED (fix+test)
[x] DOC2 src/mlframe/__init__.py:11 docstring imports pick_best_calibrator from calibration.quality (lives in .policy) -> ImportError — RESOLVED (fix+test)
[x] DOC3 CONTRIBUTING.md:66 dead path mlframe.training.composite_estimator — RESOLVED (fix+test)

## Packaging (PKG)
[x] PKG1 metrics/core.py:19 unconditional plotly import in advertised public module (plotly only in viz extra) -> fresh core install ImportError — RESOLVED (fix+test)
[x] PKG2 pyproject.toml:85 pyutilz core dep not on PyPI -> install unresolvable — DOC/FUTURE (pyutilz external; release workflow already gates upload) (likely DOC/FUTURE, external)
[x] PKG3 __init__.py:193 import-time CUDA env mutation + cupy probe-kernel — RESOLVED (fix+test)
[x] PKG4 pyproject.toml dup deps (category-encoders in sampling, matplotlib in viz); numba floor; numpy<2 py3.9 marker only in [tool.uv] — RESOLVED (fix+test)
[x] PKG5 .github/workflows/docs.yml builds without installing pkg — RESOLVED (fix+test)

## sklearn compliance (SK)
[x] SK1 estimators/early_stopping.py fit mutates self.base_model in place (no clone) -> shared across CV folds — RESOLVED (fix+test)
[x] SK2 boruta_shap/__init__.py fit reassigns self.model (None->RF); _rng/train_or_test derived in __init__ — RESOLVED (fix+test)
[x] SK3 training/composite/ranking.py CompositeRankEstimator learned attrs no _ suffix; group required positionally; no check_is_fitted — RESOLVED (fix+test)
[x] SK4 training/neural/recurrent_dataset_helpers.py RecurrentClassifierWrapper no classes_; _RecurrentWrapperBase mutates shared config — RESOLVED (fix+test)
[x] SK5 training/neural/keras_compat.py KerasCompatibleMLP raises in __init__; sets non-signature attr — RESOLVED (fix+test)
[x] SK6 training/strategies/base.py _NumericOnlyTransformer transforms param in __init__; missing allow_nan tags (_Float32/_InfToNaN) — RESOLVED (fix+test)
[x] SK-P2 composite NotFittedError cluster + n_features_in_ gaps + bagging init-validation + quantile_wrapper alphas verbatim + extremes clone + allow_nan tags — RESOLVED (fix+test)

## Property/metamorphic (PROP)
[x] PROP1 metrics/classification/_classification_calibration.py:101 accuracy_ratio not row-perm-invariant + violates 2*AUC-1 on ties (tie-fold the CAP cumsum) — RESOLVED (fix+test)
[x] PROP2 calibration/quality.py:547 MM-corrected ECI goes negative on calibrated input (clamp >=0) — RESOLVED (fix+test)

## Encoding (ENC)
[x] ENC1 models/ensembling/__init__.py:88 to_csv no encoding=utf-8 — RESOLVED (fix+test)
[x] ENC2 training/core/_main_train_suite_phases.py:473 to_csv no encoding=utf-8 — RESOLVED (fix+test)
[x] ENC3 feature_selection/boruta_shap/_io_plot.py:43 to_csv no encoding=utf-8 — RESOLVED (fix+test)
[x] ENC-P2 locking.py / infonet open() without encoding (ASCII content) — DOC (ASCII-only content; consistency note)

## Test-suite health (TST)
[x] TST1 test_meta/test_no_inspect_getsource.py RED: 3 round-3 tests use inspect.getsource (calibration/test_regression_quality_unbound_and_guard.py, inference/test_regression_explainability_nclasses_unbound.py, training/test_regression_temporal_audit_pandas_fallback_coerce.py) -> convert to behavioral — RESOLVED (fix+test)
[x] TST2 unseeded global-RNG cluster (test_ensembling.py ~40, financial/numerical/hurst/mps/timeseries/custom) -> local default_rng; likely root of precision_at_k random-order flake — RESOLVED-partial (global-RNG seeded across the cluster; residual precision_at_k intermittency is numba parallel-reduction bit-identity, FUTURE)
[x] TST3 test_metrics.py:398 wall-clock assert no margin -> perf_speedup_floor — RESOLVED (fix+test)
[x] TST4 test_security_rce.py:26,61 pytest.raises(Exception) too broad -> narrow to UnpicklingError + message — RESOLVED (fix+test)
[x] TST5 reporting/test_feature_drift_report.py:178,340,402 skip masks unfinished drift -> wire or xfail(strict) — RESOLVED (fix+test)
[x] TST6 test_meta/test_config_docstring_drift.py:134 + test_todo_hygiene.py:78 only warn (never red) -> assert — RESOLVED (fix+test)
[x] TST7 tests/signal/ shadows stdlib signal -> rename — RESOLVED (fix+test)
[x] TST-P2 weak/no-assert smoke clusters (validation_records, real_datasets biz_val, training no-assert); broad pytest.raises(Exception) ~11 sites; skipif(True) MIST — FUTURE (weak/no-assert smoke clusters + broad pytest.raises; large mechanical pass, tracked)
