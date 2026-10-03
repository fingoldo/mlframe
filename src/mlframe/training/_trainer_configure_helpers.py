"""Helpers carved out of ``_trainer_configure`` to keep that module under its size budget."""
from __future__ import annotations

import logging
import threading
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    pass

import numpy as np
import pandas as pd

from pyutilz.system import compute_total_gpus_ram

from ._gpu_contention import gpu_fits_now

# Heavy optional deps: defer failures to first actual use so `import mlframe.training` stays cheap and does not crash when a given backend is not installed.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from sklearn.metrics import (
    make_scorer,
)
from mlframe.training._polars_native_support import catboost_polars_fastpath_broken

# Optional model backends: lazy/tolerant of missing deps.
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostRegressor = CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    from .lgb_shim import lgb_default_n_jobs
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore[assignment,misc]

    def lgb_default_n_jobs(requested: "int | None") -> int:
        """Fallback used only when lightgbm itself is not installed (never actually called)."""
        return -1 if requested is None else requested
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore[assignment,misc]

from mlframe.training.cb.shared import (
    cached_gpu_info as _cached_gpu_info,
)
from ._eval_helpers import (  # noqa: F401  -- carved helpers
    _align_xgb_cat_categories, _append_split_rate_suffix,
    _compute_split_metrics, _decategorise_float_cat_columns,
    _filter_categorical_features, run_confidence_analysis,
)
from ._data_helpers import (  # noqa: F401  -- carved helpers
    _setup_eval_set, _setup_early_stopping_callback,
)
from ._model_factories import (
    GPU_VRAM_SAFE_FREE_LIMIT_GB, GPU_VRAM_SAFE_SATURATION_LIMIT,
)

# Optional model backends mirrored from parent: defaults to None when the
# library is not installed; downstream branches gate on that.
try:
    from ngboost import NGBClassifier, NGBRegressor
except ImportError:
    NGBClassifier = None
    NGBRegressor = None


logger = logging.getLogger("mlframe.training.trainer")


# A5#4/#16 session-level memo for ``get_training_configs``. The function is called
# twice per ``select_target`` invocation (CPU + GPU), and ``select_target`` runs
# once per (target, pre_pipeline, model) in the suite -- ten targets x three
# pre_pipelines x five models = 300 calls to ``get_training_configs`` per suite.
# Most of those calls share identical ``config_params`` content because the
# suite-level ``hyperparams_config.model_dump()`` is computed once and never
# changes between targets; only ``subgroups`` may differ when per-target OD
# filtering changes ``train_idx`` / ``val_idx``.
#
# The cache is capped via FIFO eviction at 16 entries to bound memory; in
# practice 2-4 entries cover the canonical suite (CPU + GPU x at most a few
# distinct ``subgroups`` shapes). Falls through to a direct call when the
# kwargs contain unhashable values (callable scorers, large polars-categorical
# dicts) so the contract stays "memo when safe; direct otherwise".
_GTC_CACHE_MAX = 16
_GTC_CACHE: "dict[tuple, Any]" = {}
# Guards _GTC_CACHE and its pin dict together: concurrent fits could otherwise evict a key between another
# thread's insert into one dict and the other, leaving a pin without its entry or an entry without its pin.
_GTC_CACHE_LOCK = threading.Lock()
# Pins the exact ``subgroups`` object whose id() was folded into a live cache key, keyed by that same
# key. CPython can reuse a freed object's id() for a brand-new, unrelated object -- without this pin, a
# short-lived ``indexed_subgroups`` dict from one target being garbage-collected and a DIFFERENT
# target's freshly-built dict landing at the same id() would produce a stale cache HIT mixing one
# target's fairness-subgroup-derived calibration metric config into another target's result. Holding a
# strong reference here for as long as the cache entry lives guarantees id() cannot be reused by
# anything else while that entry is still a valid hit; evicted together with its cache entry below.
_GTC_CACHE_SUBGROUPS_PIN: "dict[tuple, Any]" = {}


logger = logging.getLogger("mlframe.training.trainer")


def _configure_training_common_params_none(common_params, config_params, fairness_features, cb_fit_params, target_type, prefer_calibrated_classifiers, multilabel_dispatch_config, n_classes, mlframe_models):
    """Block of configure_training_params starting at ``if common_params is None:``."""
    if common_params is None:
        common_params = {}
    if config_params is None:
        config_params = {}
    if fairness_features is None:
        fairness_features = []
    if cb_fit_params is None:
        cb_fit_params = {}

    # Multilabel + post-hoc calibration safety gate. ``CalibratedClassifierCV`` is single-output only; combining it with a MULTILABEL target silently fails inside the wrapper (label-list shape mismatch deep in sklearn). Honour ``MultilabelDispatchConfig.allow_uncalibrated_multi``: when False (default, strict), refuse the combo loudly so the misconfiguration is visible at config time; when True, drop the calibration request with a warning and continue. No-op when target is not multilabel or no MultilabelDispatchConfig was supplied.
    prefer_calibrated_classifiers = _configure_training_multilabel_post_hoc_calibration(target_type, prefer_calibrated_classifiers, multilabel_dispatch_config)

    # Route target_type / n_classes into get_training_configs so per-strategy classification dispatch (CB MultiLogloss, XGB multi:softprob+num_class, LGB multiclass+num_class) gets injected. Without this, multilabel targets reach CB without loss_function set and CB's _get_loss_function_for_train tries len(set(label)) on the 2-D ndarray and crashes with TypeError: unhashable type: 'numpy.ndarray'.
    if target_type is not None and "target_type" not in config_params:
        config_params["target_type"] = target_type
    if n_classes is not None and "n_classes" not in config_params:
        config_params["n_classes"] = n_classes
    # Thread mlframe_models -> get_training_configs so the MLP config block (and its ~14s pytorch / lightning import on first call) is skipped when no neural model is requested.
    if mlframe_models is not None and "enabled_models" not in config_params:
        config_params["enabled_models"] = list(mlframe_models)
    return cb_fit_params, common_params, config_params, fairness_features, prefer_calibrated_classifiers


def _configure_training_ngb_params_none(_should_create_model, configs, use_regression, target_type, config_params, metamodel_func, train_target, common_params, target, train_idx, mlframe_models_is_default_allowlist):
    """Block of configure_training_params starting at ``ngb_params = None``."""
    ngb_params = None
    ngb_params = _configure_training_should_create_model_ngb(_should_create_model, configs, use_regression, target_type, config_params, metamodel_func, ngb_params)

    # Auto-detect a genuine point-mass/zero-inflated regression target so gated_outlier can default-ON
    # WITHOUT running on every ordinary regression target (blanket inclusion in the default mlframe_models
    # list would fit a classifier gate against a near-empty or fully-degenerate positive class on plain
    # continuous targets, which is both wasted compute and a real crash risk -- CalibratedClassifierCV /
    # LogisticRegression can fail outright when one class has too few rows for the requested CV folds).
    # Threshold (>=5% of train rows share the single most common value) mirrors the "point mass" framing in
    # ``GatedOutlierEstimator``'s own docstring (a degenerate value, e.g. exact 0.0, dominating a meaningful
    # slice of rows) while staying well below the near-100% share a normal continuous target's mode would show.
    # Gated on ``mlframe_models_is_default_allowlist`` (True only when the caller left the top-level
    # ``mlframe_models`` argument at its ``None`` default): a caller who passes an EXPLICIT allowlist has
    # already made a deliberate model-set decision, and silently appending an extra model to that list would
    # violate the documented "mlframe_models filters which models train" contract (regression-tested by
    # ``tests/training/test_gated_outlier_registry_key.py::test_existing_keys_unperturbed_by_new_registry_entry``).
    # ``train_target``/``common_params["train_target"]``/``config_params["train_target"]`` are all still None
    # at this point in the real suite call path (the OD-filtered ``od_common_params["train_target"]`` entry
    # gets populated AFTER this call, not before -- confirmed by direct tracing). The one value that IS
    # reliably available here is the full ``target`` series plus ``train_idx`` (both real parameters of this
    # function, threaded straight from ``_train_eval_select_target.py``'s own locals) -- slice them ourselves
    # rather than depending on a dict key that isn't populated yet.
    _gated_outlier_train_target = train_target
    if _gated_outlier_train_target is None:
        _gated_outlier_train_target = common_params.get("train_target") if common_params else None
    if _gated_outlier_train_target is None:
        _gated_outlier_train_target = (config_params or {}).get("train_target")
    _gated_outlier_train_target = _configure_training_gated_outlier_train_target(_gated_outlier_train_target, target, train_idx)
    _auto_detected_point_mass = False
    _auto_detected_point_mass = _configure_training_mlframe_models_default_allowlist(mlframe_models_is_default_allowlist, use_regression, _gated_outlier_train_target, _auto_detected_point_mass)

    gated_outlier_params = None
    if (_should_create_model("gated_outlier") or _auto_detected_point_mass) and use_regression and LGBMRegressor is not None:
        # Registry entry for GatedOutlierEstimator (classifier gate + regression blend for zero-inflated
        # targets). Regression-only (RegressorMixin) -- silently unavailable for classification targets or
        # when lightgbm isn't installed, same guard pattern as the lgb/xgb blocks above. Classifier defaults
        # to LogisticRegression internally (see GatedOutlierEstimator.fit). point_mass_value=0.0 is the class
        # default (matches the common "no purchase"/exact-zero degenerate case) -- callers needing a
        # different point mass should pass a configured instance directly via the generic estimator-instance
        # path instead of this key. Default-included (on top of the fixed ``mlframe_models`` allowlist) only
        # when ``_auto_detected_point_mass`` fires above -- see that block's docstring for why blanket
        # inclusion is unsafe.
        #
        # NOT ``configs.LGB_GENERAL_PARAMS``: that config bakes in early-stopping params which require an
        # eval_set, but ``GatedOutlierEstimator.fit`` calls ``self.regressor_.fit(reg_X, reg_y)`` with no
        # eval_set (it fits standalone on the non-point-mass training rows) -- LightGBM raises
        # "For early stopping, at least one dataset and eval metric is required for evaluation" (confirmed by
        # a live suite run). Use the same standalone-safe LGBM default as the E3 distribution-driven
        # composite base (``_estimator_dispatch._default_base_estimator``) instead.
        from mlframe.training.composite.gated_outlier import GatedOutlierEstimator

        _gated_outlier_est = GatedOutlierEstimator(regressor=LGBMRegressor(n_estimators=200, num_leaves=31, verbose=-1, random_state=0, n_jobs=lgb_default_n_jobs(-1)))
        gated_outlier_params = dict(model=metamodel_func(_gated_outlier_est))

    # ``BaggedCompositeEstimator`` registry entry: bootstrap-bagged variance reduction over a plain GBDT
    # regressor. Unlike the ``CompositeTargetEstimator``-family estimators audited above (distributional /
    # quantile / GLM / survival / orthogonal / panel / ranking / ...), this wrapper takes NO dataset-specific
    # ``base_column``/``base_columns``/``event`` argument -- it bags directly over ``(X, y)``, so it is
    # generically instantiable the same way ``gated_outlier`` is. ``base_estimator`` itself has no safe
    # generic default inside the class (``fit`` raises ``ValueError`` on ``None``), so -- same pattern as
    # ``gated_outlier`` above -- the registry supplies the suite's own standalone-safe LGBM default
    # (``configs.LGB_GENERAL_PARAMS`` bakes in early-stopping params requiring an ``eval_set`` that bagging's
    # bootstrap-resample ``fit`` calls never provide). Explicit-allowlist-only (no auto-detection heuristic):
    # unlike ``gated_outlier``'s cheap point-mass classifier gate, bagging is a real Nx compute multiplier
    # (``n_estimators=10`` full refits by default) with no generic trigger signal analogous to "target has a
    # point mass" -- blanket-including it would silently 10x every caller's regression fit time.
    bagging_params = None
    if _should_create_model("bagging") and use_regression and LGBMRegressor is not None:
        from mlframe.training.composite.bagging import BaggedCompositeEstimator

        _bagging_est = BaggedCompositeEstimator(base_estimator=LGBMRegressor(n_estimators=200, num_leaves=31, verbose=-1, random_state=0, n_jobs=lgb_default_n_jobs(-1)))
        bagging_params = dict(model=metamodel_func(_bagging_est))

    # ``CompositeClassificationEstimator`` registry entry: base-margin (init-score) residual composite for
    # classification. Also genuinely generic -- when ``base_margin_column`` is left unset (the default), it
    # auto-fits a ``LogisticRegression`` base-margin model directly on ``X`` (no dataset-specific column
    # needed), then boosts the inner GBDT on the residual log-odds via its native init-score hook. Only
    # ``base_estimator`` has no in-class default (``fit`` raises on ``None``), so the registry supplies a
    # standalone-safe LGBM classifier default, same rationale as ``gated_outlier``/``bagging`` above.
    # Explicit-allowlist-only: no generic auto-detection trigger (unlike gated_outlier's point-mass signal) --
    # every classification target benefits from *some* base margin, so there is no principled "only some
    # targets need this" heuristic to gate a default-ON auto-append on.
    composite_classification_params = None
    if _should_create_model("composite_classification") and not use_regression and LGBMClassifier is not None:
        from mlframe.training.composite.classification import CompositeClassificationEstimator

        _composite_classification_est = CompositeClassificationEstimator(base_estimator=LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1, random_state=0, n_jobs=lgb_default_n_jobs(-1)))
        composite_classification_params = dict(model=metamodel_func(_composite_classification_est))
    return bagging_params, composite_classification_params, gated_outlier_params, ngb_params


def _configure_training_xgb_rfecv_rfecv(use_regression, metamodel_func, configs, prefer_calibrated_classifiers, cat_features, rfecv_scoring, rfecv_params, cb_params, lgb_params, xgb_params, hgb_params, mlp_params, ngb_params, gated_outlier_params, bagging_params, composite_classification_params):
    """Block of configure_training_params starting at ``xgb_rfecv = RFECV(``."""
    from mlframe.training.trainer import RFECV

    xgb_rfecv = RFECV(
        estimator=(
            metamodel_func(XGBRegressor(**configs.XGB_GENERAL_PARAMS))
            if use_regression
            else XGBClassifier(**(configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.XGB_GENERAL_CLASSIF))
        ),
        fit_params=dict(verbose=False),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    # Build models_params dict, only including models that were created
    models_params = {}
    if cb_params is not None:
        models_params["cb"] = cb_params
    if lgb_params is not None:
        models_params["lgb"] = lgb_params
    if xgb_params is not None:
        models_params["xgb"] = xgb_params
    if hgb_params is not None:
        models_params["hgb"] = hgb_params
    if mlp_params is not None:
        models_params["mlp"] = mlp_params
    if ngb_params is not None:
        models_params["ngb"] = ngb_params
    if gated_outlier_params is not None:
        models_params["gated_outlier"] = gated_outlier_params
    if bagging_params is not None:
        models_params["bagging"] = bagging_params
    if composite_classification_params is not None:
        models_params["composite_classification"] = composite_classification_params
    return models_params, xgb_rfecv


def _configure_training_multilabel_post_hoc_calibration(target_type, prefer_calibrated_classifiers, multilabel_dispatch_config):
    """Block of configure_training_params starting at ``if target_type is not None and prefer_calibrated_classifiers and multi``."""
    from mlframe.training.trainer import TargetTypes

    if target_type is not None and prefer_calibrated_classifiers and multilabel_dispatch_config is not None:
        if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
            if not multilabel_dispatch_config.allow_uncalibrated_multi:
                raise NotImplementedError(
                    "prefer_calibrated_classifiers=True is incompatible with "
                    "MULTILABEL_CLASSIFICATION (CalibratedClassifierCV is "
                    "single-output only). Set MultilabelDispatchConfig."
                    "allow_uncalibrated_multi=True to drop calibration with a "
                    "warning instead of raising."
                )
            logger.warning(
                "Multilabel target + prefer_calibrated_classifiers=True; "
                "dropping calibration (MultilabelDispatchConfig."
                "allow_uncalibrated_multi=True). Trained models will be "
                "uncalibrated."
            )
            prefer_calibrated_classifiers = False
    return prefer_calibrated_classifiers


def _configure_training_use_regression_2(use_regression, config_params, target, target_type):
    """Block of configure_training_params starting at ``if not use_regression:``."""
    if not use_regression:
        if "catboost_custom_classif_metrics" not in config_params:
            # Multi-output safe label count: 2-D multilabel uses n_columns;
            # 1-D binary/multiclass uses unique value count.
            target_arr = np.asarray(target) if target is not None else None
            # Multilabel detection: explicit 2-D, OR 1-D object dtype where
            # each cell is itself an array (the polars ``pl.List(pl.Int8)``
            # roundtrip lands here). Without the second clause,
            # ``np.unique(target_arr)`` raised ``truth value of array
            # ambiguous`` on the per-cell-array comparison (cb / pandas / multilabel target).
            _is_object_of_arrays = False
            if target_arr is not None and target_arr.dtype == object and target_arr.ndim == 1 and target_arr.shape[0] > 0:
                _first = target_arr[0]
                _is_object_of_arrays = hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
            if target_arr is not None and target_arr.ndim == 2:
                nlabels = target_arr.shape[1] + 1  # treat as ">2" -> multiclass-style metrics
            elif _is_object_of_arrays:
                try:
                    assert target_arr is not None
                    _first = target_arr[0]
                    nlabels = (len(_first) if hasattr(_first, "__len__") else int(np.asarray(_first).size)) + 1
                except Exception as e:
                    logger.debug("inferring nlabels from target_arr failed, defaulting to 3: %s", e)
                    nlabels = 3
            elif target_arr is not None:
                nlabels = len(np.unique(target_arr))
            else:
                nlabels = 2
            # When multilabel: AUC is incompatible with MultiLogloss (CB rejects
            # it at fit time). Skip the AUC/PRAUC defaults and let the per-strategy
            # multilabel dispatch in helpers.py pick a compatible eval_metric.
            if target_type is not None and getattr(target_type, "name", None) == "MULTILABEL_CLASSIFICATION":
                catboost_custom_classif_metrics = []
            elif nlabels > 2:
                catboost_custom_classif_metrics = ["AUC", "PRAUC:hints=skip_train~true"]
            else:
                catboost_custom_classif_metrics = ["AUC", "PRAUC:hints=skip_train~true", "BrierScore"]
            config_params["catboost_custom_classif_metrics"] = catboost_custom_classif_metrics


def _configure_training_subgroups_none_fairness_features(subgroups, fairness_features, df, train_df, cont_nbins, fairness_min_pop_cat_thresh):
    """Block of configure_training_params starting at ``if subgroups is None and fairness_features:``."""
    from mlframe.training.trainer import create_fairness_subgroups

    if subgroups is None and fairness_features:
        for next_df in (df, train_df):
            if next_df is not None:
                subgroups = create_fairness_subgroups(
                    next_df,
                    features=fairness_features,
                    cont_nbins=cont_nbins,
                    min_pop_cat_thresh=fairness_min_pop_cat_thresh,
                )
                break
    return subgroups


def _configure_training_val_df_size_bytes(val_df_size_bytes, val_df):
    """Block of configure_training_params starting at ``if val_df_size_bytes is not None:``."""
    from mlframe.training.trainer import get_df_memory_consumption

    if val_df_size_bytes is not None:
        val_df_size = float(val_df_size_bytes)
    elif val_df is not None:
        val_df_size = get_df_memory_consumption(val_df, deep=False)
    else:
        val_df_size = 0
    return val_df_size


def _configure_training_prefer_gpu_configs_cb(prefer_gpu_configs, cb_task_type, _no_gpu_model_needed, data_size_gb, cb_devices):
    """Block of configure_training_params starting at ``if not prefer_gpu_configs or cb_task_type == "CPU" or _no_gpu_model_ne``."""
    from mlframe.training.trainer import parse_catboost_devices

    if not prefer_gpu_configs or cb_task_type == "CPU" or _no_gpu_model_needed:
        all_gpus: list = []
        data_fits_gpu_ram = False
        data_fits_cb_gpu_ram = False
    else:
        all_gpus = _cached_gpu_info()
        single_gpu_limits = compute_total_gpus_ram(all_gpus)
        data_fits_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < single_gpu_limits.get("gpu_max_ram_total", 0)
        if cb_devices:
            multi_gpu_limits = compute_total_gpus_ram(parse_catboost_devices(cb_devices, all_gpus=all_gpus))
            data_fits_cb_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < multi_gpu_limits.get("gpus_ram_total", 0)
        else:
            data_fits_cb_gpu_ram = data_fits_gpu_ram
        data_fits_gpu_ram, data_fits_cb_gpu_ram = gpu_fits_now(data_fits_gpu_ram, data_fits_cb_gpu_ram, data_size_gb, cb_devices)
    return data_fits_cb_gpu_ram, data_fits_gpu_ram


def _configure_training_lazy_model_creation_only(_should_create_model, use_regression, metamodel_func, cb_configs, prefer_calibrated_classifiers, verbose, cat_features, text_features, embedding_features, cb_fit_params, cb_params):
    """Block of configure_training_params starting at ``if _should_create_model("cb"):``."""
    if _should_create_model("cb"):
        if use_regression:
            _cb_model = metamodel_func(CatBoostRegressor(**cb_configs.CB_REGR))
        else:
            _cb_classif_params = cb_configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else cb_configs.CB_CLASSIF
            _cb_model = CatBoostClassifier(**_cb_classif_params)
        # Pre-set the polars-fastpath sticky flag when THIS CatBoost build actually needs it. ``_predict_with_fallback`` lazily flips the attribute to True after the FIRST dispatch miss, so the short-circuit would otherwise fire only from the SECOND predict onward -- and in a suite each weight-schema iteration calls ``sklearn.clone()``, which strips non-param attrs and hands every fresh instance a blank flag.
        # The value comes from the installed build's own probe rather than a constant: some CB 1.2.x builds have dispatch gaps on a nullable-Categorical / Enum schema and some do not, and pre-setting it on a build that works costs a polars->pandas conversion on EVERY predict for nothing (a production run logged 52 of them while the probe answered that CatBoost accepts polars). Set on the base instance so ``clone()`` carries the param-equivalent state forward; ``train_eval.py:process_model`` re-asserts it around its clone call.
        try:
            _cb_model._mlframe_polars_fastpath_broken = catboost_polars_fastpath_broken()  # readers use getattr(..., False)
        except Exception as e:  # nosec B110 - non-trivial body
            # CB Python class is permissive about attributes; slot-only forks could refuse - degrade to "pay first-call retry".
            logger.debug("setting _mlframe_polars_fastpath_broken failed: %s", e)
        cb_params = dict(
            model=_cb_model,
            fit_params=dict(
                plot=verbose,
                cat_features=cat_features,
                **({"text_features": text_features} if text_features else {}),
                **({"embedding_features": embedding_features} if embedding_features else {}),
                **cb_fit_params,
            ),
        )
    return cb_params


def _configure_training_should_create_model_mlp(_should_create_model, train_df, train_target, configs, config_params, use_regression, metamodel_func, target_type, mlp_params):
    """Block of configure_training_params starting at ``if _should_create_model("mlp"):``."""
    from mlframe.training.trainer import _configure_mlp_params

    if _should_create_model("mlp"):
        # Pass training rowcount so _configure_mlp_params can auto-reduce
        # network depth on small datasets where a 4-layer LeakyReLU MLP
        # over-fits the few-thousand-row train split and catastrophically
        # extrapolates on the small test split (regression-collapse-sensor
        # documented this mode for 6k-row mixed-scale features).
        _n_train_for_mlp = None
        try:
            if train_df is not None:
                _n_train_for_mlp = len(train_df)
            elif train_target is not None:
                _n_train_for_mlp = len(train_target)
        except (TypeError, ValueError):
            _n_train_for_mlp = None
        mlp_params = _configure_mlp_params(
            configs=configs,
            config_params=config_params,
            use_regression=use_regression,
            metamodel_func=metamodel_func,
            target_type=target_type,
            n_train=_n_train_for_mlp,
        )
    return mlp_params


def _configure_training_should_create_model_ngb(_should_create_model, configs, use_regression, target_type, config_params, metamodel_func, ngb_params):
    """Block of configure_training_params starting at ``if _should_create_model("ngb"):``."""
    from mlframe.training.trainer import TargetTypes

    if _should_create_model("ngb"):
        # Target-type-aware Dist for NGBClassifier. Default ``Dist=Bernoulli`` (binary only) crashes on K>2 with ``IndexError: index out of bounds``; for multiclass we need ``Dist=k_categorical(K)``. NGBoost has no native multilabel / ranker, so those target types fall through to the default (likely with a downstream error if reached - they should be filtered earlier when the suite checks per-strategy multilabel / ranking flags).
        ngb_init_kwargs = dict(configs.NGB_GENERAL_PARAMS)
        if not use_regression and target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
            try:
                from ngboost.distns import k_categorical

                # n_classes pulled from the actual y - NGB needs the exact K to size the categorical Dist's internal parameter array. Fall back to inspecting train_target via config_params (where train_target lives at this call layer).
                _train_target = config_params.get("train_target")
                if _train_target is not None:
                    _y = np.asarray(_train_target).ravel()
                    _K = int(_y.max()) + 1 if len(_y) else 2
                else:
                    _K = max(2, int(config_params.get("n_classes", 2)))
                ngb_init_kwargs["Dist"] = k_categorical(_K)
            except ImportError:
                pass  # ngboost.distns missing -> default Dist crashes loudly downstream

        ngb_params = dict(
            model=(
                metamodel_func(
                    (NGBRegressor(**ngb_init_kwargs) if use_regression else NGBClassifier(**ngb_init_kwargs)),
                )
            ),
            fit_params=({} if config_params.get("early_stopping_rounds") is None else dict(early_stopping_rounds=config_params.get("early_stopping_rounds"))),
        )
    return ngb_params


def _configure_training_gated_outlier_train_target(_gated_outlier_train_target, target, train_idx):
    """Block of configure_training_params starting at ``if _gated_outlier_train_target is None and target is not None and trai``."""
    if _gated_outlier_train_target is None and target is not None and train_idx is not None:
        try:
            _gated_outlier_train_target = np.asarray(target)[np.asarray(train_idx)]
        except (TypeError, ValueError, IndexError):
            _gated_outlier_train_target = None
    return _gated_outlier_train_target


def _configure_training_mlframe_models_default_allowlist(mlframe_models_is_default_allowlist, use_regression, _gated_outlier_train_target, _auto_detected_point_mass):
    """Block of configure_training_params starting at ``if mlframe_models_is_default_allowlist and use_regression and _gated_o``."""
    if mlframe_models_is_default_allowlist and use_regression and _gated_outlier_train_target is not None:
        try:
            _tt_arr = pd.Series(_gated_outlier_train_target).dropna().to_numpy()
            if _tt_arr.size >= 50:
                _tt_vals, _tt_counts = np.unique(_tt_arr, return_counts=True)
                if _tt_counts.size:
                    _auto_detected_point_mass = bool((_tt_counts.max() / _tt_arr.size) >= 0.05)
        except (TypeError, ValueError):
            _auto_detected_point_mass = False
    return _auto_detected_point_mass


def _configure_training_learning_rate_float_trees(linear_models_needed, config_params, linear_config_excluded_keys, linear_model_config, use_regression, _wrap_for_multilabel_if_needed, metamodel_func, linear_model_params):
    """Block of configure_training_params starting at ``for model_type in linear_models_needed:``."""
    from mlframe.training.trainer import LinearModelConfig, create_linear_model

    for model_type in linear_models_needed:
        # Build config by merging: config_params -> linear_model_config -> model_type
        # This allows config_params_override["iterations"] to work for linear models
        linear_config_kwargs: dict[str, Any] = {"model_type": model_type}
        # Apply config_params first (includes iterations from config_params_override)
        if config_params:
            # Only include keys that LinearModelConfig recognizes
            linear_config_fields = set(LinearModelConfig.model_fields.keys()) - linear_config_excluded_keys
            # Also include 'iterations' which gets mapped to max_iter by the validator
            linear_config_fields.add("iterations")
            linear_config_kwargs.update({key: value for key, value in config_params.items() if key in linear_config_fields})
        # Override with explicit linear_model_config if provided
        if linear_model_config:
            linear_config_kwargs.update(linear_model_config.model_dump(exclude={"model_type"}))
        config = LinearModelConfig(**linear_config_kwargs)
        _linear_est = create_linear_model(model_type, config, use_regression=use_regression)
        # Linear classifiers reject 2-D y -> MultiOutputClassifier wrapper for multilabel.
        from mlframe.training.strategies import LinearModelStrategy

        _linear_est = _wrap_for_multilabel_if_needed(_linear_est, LinearModelStrategy)
        linear_model_params[model_type] = dict(model=metamodel_func(_linear_est))


def _configure_training_use_regression(use_regression, default_regression_scoring, prefer_calibrated_classifiers, configs, rfecv_model_verbose, default_classification_scoring):
    """Block of configure_training_params starting at ``if use_regression:``."""
    if use_regression:
        rfecv_scoring = make_scorer(**default_regression_scoring)
    else:
        if prefer_calibrated_classifiers:

            from mlframe.training._picklable_metrics import VerboseBoundMetric

            rfecv_scoring = make_scorer(
                score_func=VerboseBoundMetric(configs.fs_and_hpt_integral_calibration_error, rfecv_model_verbose),
                response_method="predict_proba",
                greater_is_better=False,
            )
        else:
            rfecv_scoring = make_scorer(**default_classification_scoring)
    return rfecv_scoring


def _configure_training_minimal_linear_params_shape(mlframe_models, metamodel_func, models_params):
    """Block of configure_training_params starting at ``if mlframe_models:``."""
    if mlframe_models:
        for _entry in mlframe_models:
            if isinstance(_entry, str):
                continue
            _est = _entry[1] if (isinstance(_entry, tuple) and len(_entry) == 2) else _entry
            models_params[_entry] = dict(model=metamodel_func(_est))
