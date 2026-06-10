"""Confidence analysis for ``train_and_evaluate_model``.

Carved out of ``_eval_helpers`` to keep that module under the LOC ceiling. Trains a small
CatBoost regressor on the predicted probability of the true class, then surfaces which features
drive prediction confidence via a SHAP beeswarm (or a plain feature-importance bar when SHAP is
off). Best-effort diagnostic: every degenerate input (multilabel target, regression target, row
mismatch, constant targets, zero features) skips with a log line rather than crashing the suite.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment] -- plt-using code paths are guarded; matplotlib-less envs skip the plot branches
import pandas as pd
import polars as pl

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None  # type: ignore[assignment] -- only used when CB is the chosen model; guarded at call site

logger = logging.getLogger(__name__)


def run_confidence_analysis(
    test_df: pd.DataFrame,
    test_target: np.ndarray,
    test_probs: np.ndarray,
    cat_features: list[str] = None,
    text_features: list[str] = None,
    embedding_features: list[str] = None,
    confidence_model_kwargs: dict = None,
    fit_params: dict = None,
    use_shap: bool = None,
    max_features: int = None,
    cmap: str = None,
    alpha: float = None,
    title: str = None,
    ylabel: str = None,
    figsize: tuple[float, float] = (10, 6),
    plot_file: str = "",
    verbose: bool = False,
) -> Any:
    """Analyze which features most affect prediction confidence.

    Styling defaults (``use_shap``/``max_features``/``cmap``/``alpha``/``title``/``ylabel``)
    are single-sourced from ``ConfidenceAnalysisConfig`` field defaults when left ``None`` so
    the two layers can never drift. ``plot_file`` (when set) saves the SHAP beeswarm to disk
    and the figure is always closed afterwards (no pyplot-registry leak in long sessions).
    """
    from .configs import ConfidenceAnalysisConfig
    _caf = ConfidenceAnalysisConfig.model_fields
    if use_shap is None:
        use_shap = _caf["use_shap"].default
    if max_features is None:
        max_features = _caf["max_features"].default
    if cmap is None:
        cmap = _caf["cmap"].default
    if alpha is None:
        alpha = _caf["alpha"].default
    if title is None:
        title = _caf["title"].default
    if ylabel is None:
        ylabel = _caf["ylabel"].default
    if test_df is None:
        return None

    if verbose:
        logger.info("Running confidence analysis...")

    if confidence_model_kwargs is None:
        confidence_model_kwargs = {}

    # Bound the confidence model's
    # iteration budget so the CPU fallback can't spin indefinitely.
    # Without a cap CB defaults to iterations=1000, which on the rich
    # feature schema typical for the suite (50+ cols) translates to
    # 4-10+ minutes per confidence fit on CPU. The confidence regressor
    # is best-effort diagnostic -- even 50 boosting rounds gives a
    # serviceable feature-importance signal. Caller can override.
    #
    # CatBoost rejects ``iterations`` co-present with any of its synonyms
    # (``n_estimators`` / ``num_boost_round`` / ``num_trees``) with
    # "only one of the parameters ... should be initialized". A caller who
    # passes the perfectly-valid ``n_estimators`` would otherwise crash the
    # whole confidence pass the moment our ``setdefault("iterations", ...)``
    # fired. Only inject the cap when NONE of the four synonyms is already
    # present so an explicit caller-supplied budget (under any spelling)
    # wins uncontested.
    _CB_ITER_SYNONYMS = ("iterations", "n_estimators", "num_boost_round", "num_trees")
    if not any(_syn in confidence_model_kwargs for _syn in _CB_ITER_SYNONYMS):
        confidence_model_kwargs["iterations"] = 200
    confidence_model_kwargs.setdefault("early_stopping_rounds", 30)

    # Lazy import: _model_factories <-> _eval_helpers circular load; resolved at call time.
    from ._model_factories import CUDA_IS_AVAILABLE
    confidence_task_type = "GPU" if CUDA_IS_AVAILABLE else "CPU"
    confidence_model = CatBoostRegressor(verbose=0, eval_fraction=0.1, task_type=confidence_task_type, **confidence_model_kwargs)

    fit_params_copy = {}
    if fit_params:
        fit_params_copy = copy.copy(fit_params)
        if "eval_set" in fit_params_copy:
            del fit_params_copy["eval_set"]

    # Drop text / embedding columns from test_df upfront.
    # SHAP's TreeExplainer rebuilds a CatBoost Pool using ONLY
    # ``cat_features`` from the model (no text awareness), so text
    # columns reaching Pool as numeric raise ``Bad value for
    # num_feature ...: Cannot convert '<text>' to float``.
    # Free-text TF-IDF features aren't analysable by
    # SHAP anyway, so dropping is the right scope reduction. Done at
    # the test_df level so the confidence_model is trained on the same
    # schema SHAP will see.
    #
    # Also detect string-typed columns directly: when the surrounding
    # model is XGB/HGB, ``text_features`` doesn't propagate via
    # fit_params (those models don't accept the kwarg), so an
    # auto-promoted text column survives in test_df as ``object`` /
    # ``string`` dtype and chokes the confidence Pool the same way.
    _drop_for_conf = []
    if text_features:
        _drop_for_conf.extend([c for c in text_features if c in test_df.columns])
    if embedding_features:
        _drop_for_conf.extend([c for c in embedding_features if c in test_df.columns])
    if isinstance(test_df, pd.DataFrame):
        for _c in test_df.columns:
            if _c in _drop_for_conf:
                continue
            if cat_features and _c in cat_features:
                continue
            try:
                _dt = test_df[_c].dtype
            except Exception as _e_dt:
                # Pre-fix `continue` silent. Pandas dtype access shouldn't
                # raise on a column known to be in ``test_df.columns``, so
                # this branch firing is itself a signal of upstream
                # corruption (custom Series subclass with broken __dtype__,
                # or column became inaccessible mid-iteration). DEBUG log
                # so operators see the trail when CB Pool later crashes
                # because this col wasn't added to _drop_for_conf.
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "_eval_helpers: pandas dtype access failed for col=%r (%s); "
                    "col will NOT be auto-dropped from confidence analysis. "
                    "If CB Pool crashes downstream, this skipped col may be "
                    "the cause.", _c, _e_dt,
                )
                continue
            # ``_dt is object`` is wrong: np.dtype('O') is NOT the Python
            # type ``object`` (it's a numpy dtype wrapper). Identity check
            # always returned False; the regression test
            # ``test_confidence_analysis_pandas_object_dtype_still_dropped``
            # surfaced this by feeding an explicit object-dtype
            # text column - it bypassed the drop and reached CatBoost Pool
            # which crashed on 'Cannot convert s_0 to float'. Use the dtype
            # kind code instead (``"O"`` for object, ``"U"`` for unicode str).
            if getattr(_dt, "kind", None) in ("O", "U") or str(_dt) in ("object", "string", "string[python]", "string[pyarrow]"):
                _drop_for_conf.append(_c)
    elif isinstance(test_df, pl.DataFrame):
        # Polars-side auto-detect. The
        # earlier pandas-only branch missed:
        #   - pl.Utf8 / pl.String text columns
        #     (CB Pool crashes on text-typed numerics like ``text_0``)
        #   - pl.List / pl.Array embedding columns (e.g. ``emb_0``
        #     surfaces here when fit_params['embedding_features'] is
        #     None because the trailing model is HGB/XGB which doesn't
        #     accept the kwarg -> the explicit-drop list is empty)
        #   - pl.Struct nested types (rare but break CB Pool numerics)
        # All these break CB Pool numeric-feature construction. Drop
        # them unless explicitly listed as cat_features.
        for _c in test_df.columns:
            if _c in _drop_for_conf:
                continue
            if cat_features and _c in cat_features:
                continue
            try:
                _dt = test_df.schema[_c]
            except Exception as _e_dt:
                # Same shape as the pandas branch above: schema lookup
                # shouldn't raise on a known column; DEBUG-log so the
                # trail exists when CB Pool crashes later.
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "_eval_helpers: polars schema lookup failed for col=%r "
                    "(%s); col will NOT be auto-dropped from confidence "
                    "analysis.", _c, _e_dt,
                )
                continue
            _dt_name = str(_dt)
            _is_string = _dt in (pl.Utf8, pl.Object) or _dt_name in ("Utf8", "String", "Object")
            _is_collection = (
                _dt_name.startswith("List(")
                or _dt_name.startswith("Array(")
                or _dt_name.startswith("Struct(")
                or (hasattr(pl, "List") and isinstance(_dt, type(pl.List(pl.Int8))))
            )
            if _is_string or _is_collection:
                _drop_for_conf.append(_c)
    if _drop_for_conf:
        if isinstance(test_df, pd.DataFrame):
            test_df = test_df.drop(columns=_drop_for_conf)
        else:
            test_df = test_df.drop([c for c in _drop_for_conf if c in test_df.columns])
        if cat_features is not None:
            cat_features = [c for c in cat_features if c not in _drop_for_conf]

    # CatBoost's polars Pool path is fragile when a kept column is
    # pl.Categorical or pl.Enum AND has nulls (`null_fraction_cats > 0`)
    # or was upgraded by the upstream pipeline (e.g. via
    # align_polars_categorical_dicts) between fit and confidence-analysis
    # time: it raises either ``No matching signature found`` in
    # _set_features_order_data_polars_categorical_column.process OR a
    # generic ``Error while processing column for feature 'X'``. The
    # CatBoost pandas Pool path handles pd.Categorical / object dtypes
    # with nulls cleanly (the pandas branch above already relies on
    # this). When the polars frame carries Categorical/Enum kept columns,
    # convert the WHOLE frame to a pandas Arrow-bridge view so the
    # confidence regressor exercises the rock-solid pandas path. Tiny
    # cost vs full crash on the entire confidence step.
    if isinstance(test_df, pl.DataFrame):
        # CatBoost's polars Pool path is fragile across several axes when
        # the frame carries categorical-typed columns:
        #   - pl.Categorical / pl.Enum with nulls raises a generic
        #     "Error while processing column" on null cells
        #   - pl.Categorical upgraded between fit and confidence-analysis
        #     time (e.g. via align_polars_categorical_dicts) raises
        #     "TypeError: No matching signature found" in
        #     _set_features_order_data_polars_categorical_column
        #   - pl.Utf8 with nulls raises the same NaN-in-cat error
        # The pandas Pool path handles all three cleanly (Categorical
        # + NaN, object + NaN). Convert to a pandas Arrow-bridge view
        # so the post-conversion fillna gate below has a uniform
        # surface to operate on. The polars fastpath gain on a 5-row
        # confidence analyzer is in the noise.
        from .utils import get_pandas_view_of_polars_df
        test_df = get_pandas_view_of_polars_df(test_df)

    if cat_features is not None:
        fit_params_copy["cat_features"] = cat_features
    elif "cat_features" not in fit_params_copy:
        from ._nan_processing import get_categorical_columns  # lazy import: circular load with .utils
        fit_params_copy["cat_features"] = get_categorical_columns(test_df, include_string=False)

    # CatBoost rejects NaN in cat_feature cells with
    # ``cat_features must be integer or string, real number values and
    # NaN values should be converted to string``. The combo enum's
    # `null_fraction_cats > 0` axis routinely produces null cells in
    # categorical columns; the upstream trainer fills them with a
    # sentinel for the main fit, but the confidence analyzer instantiates
    # a fresh CB pool from the post-pipeline test_df where the nulls
    # remain (and surface as NaN after the polars->pandas Arrow bridge).
    # Fill with the literal string "_NULL_" so CB treats missing as a
    # distinct category rather than crashing. Read cat_features from
    # fit_params_copy (covers both the caller-passed and auto-detected
    # cases) so HGB-side calls (which pass cat_features=None and rely
    # on the auto-detect block above) are not skipped.
    _resolved_cat_features = fit_params_copy.get("cat_features")
    if _resolved_cat_features and isinstance(test_df, pd.DataFrame):
        _cat_in_df = [c for c in _resolved_cat_features if c in test_df.columns]
        if _cat_in_df:
            test_df = test_df.copy()
            for _c in _cat_in_df:
                _col = test_df[_c]
                if _col.isna().any():
                    if isinstance(_col.dtype, pd.CategoricalDtype):
                        if "_NULL_" not in _col.cat.categories:
                            _col = _col.cat.add_categories(["_NULL_"])
                        test_df[_c] = _col.fillna("_NULL_")
                    else:
                        test_df[_c] = _col.fillna("_NULL_").astype(str)

    fit_params_copy["plot"] = False

    # Confidence analysis pulls the predicted probability of the TRUE class
    # for each row (``test_probs[i, test_target[i]]``). That fancy-index
    # is well-defined only for single-label targets where ``test_target``
    # is a 1-D vector of integer class indices. For multilabel
    # classification ``test_target`` is (N, K) binary indicators, and for
    # regression there's no probability concept at all - in both cases
    # ``confidence_targets`` is ill-defined. Skip with an INFO log so the
    # caller's ``include_confidence_analysis=True`` is honoured for the
    # cases where it makes sense and silently no-op for the cases where
    # it cannot. Without this guard a multilabel target would
    # raise IndexError on the (N,) vs (N,K) shape mismatch.
    test_target_arr = np.asarray(test_target)
    if test_target_arr.ndim != 1:
        if verbose:
            logger.info(
                "Confidence analysis skipped: test_target has shape %s "
                "(multilabel / multi-output target). The confidence-target "
                "construction ``test_probs[arange, test_target]`` is only "
                "defined for 1-D class-index targets.",
                test_target_arr.shape,
            )
        return None
    if not np.issubdtype(test_target_arr.dtype, np.integer):
        if verbose:
            logger.info(
                "Confidence analysis skipped: test_target dtype is %s, "
                "not an integer class index (regression target?). "
                "Confidence analysis applies only to classification.",
                test_target_arr.dtype,
            )
        return None
    # When test_df, test_target, and test_probs disagree on
    # row count (an upstream filter dropped rows from one but not the
    # others), the confidence model fits with mismatched lengths and
    # CB raises ``Length of label=N1 and length of data=N2 is
    # different``. Skip with INFO so the suite continues; the
    # confidence pass is best-effort, not a hard contract.
    _n_df = test_df.shape[0] if hasattr(test_df, "shape") else None
    _n_probs = test_probs.shape[0] if hasattr(test_probs, "shape") else None
    _n_target = test_target_arr.shape[0]
    if _n_df is not None and (_n_df != _n_probs or _n_df != _n_target):
        if verbose:
            logger.info(
                "Confidence analysis skipped: row count mismatch test_df=%s, "
                "test_probs=%s, test_target=%s. The three artifacts must come "
                "from the same predict pass; an upstream filter likely sliced "
                "one but not the others.",
                _n_df,
                _n_probs,
                _n_target,
            )
        return None
    confidence_targets = test_probs[np.arange(test_probs.shape[0]), test_target_arr]

    # Degenerate confidence_targets.
    # When all rows in test happen to land in the same predicted-
    # probability bucket (small test sets, severely-miscalibrated /
    # constant-output models), every confidence_target is identical
    # and CB rejects with "All train targets are equal". The
    # confidence regressor has nothing to learn anyway. Skip with a
    # WARN so the operator knows the diagnostic was un-runnable.
    n_unique_conf = int(np.unique(confidence_targets).size)
    if n_unique_conf < 2:
        logger.warning(
            "Confidence analysis skipped: all confidence_targets are "
            "equal (n_unique=%d, value=%s). The confidence regressor "
            "has no signal to learn -- typical for tiny test sets where "
            "all rows share one predicted-prob bucket, or for severely "
            "miscalibrated models emitting a constant probability.",
            n_unique_conf,
            float(confidence_targets[0]) if confidence_targets.size else float("nan"),
        )
        return None

    # 0-feature guard: after the text / embedding / object-dtype column
    # drops above, test_df can be left with no usable feature columns --
    # e.g. an aggressive MRMR / feature-selection pass confirmed 0 predictors,
    # or every surviving column was a text/embedding feature. CatBoost then
    # raises "Input data must have at least one feature" and crashes the whole
    # confidence pass. The confidence regressor is best-effort diagnostic, so
    # skip with a WARN -- mirroring the row-mismatch / all-equal-targets skips
    # above. Surfaced by fuzz (cb_hgb_lgb_mlp combo whose FS dropped all cols).
    _n_conf_features = test_df.shape[1] if hasattr(test_df, "shape") else None
    if _n_conf_features is not None and _n_conf_features == 0:
        logger.warning(
            "Confidence analysis skipped: test_df has 0 feature columns after "
            "dropping text / embedding / object-dtype columns (feature selection "
            "may have confirmed 0 predictors, or all survivors were non-numeric). "
            "The confidence regressor needs at least one feature to fit."
        )
        return None

    from .utils import maybe_clean_ram_adaptive as _maybe_clean_ram  # lazy: utils imports from _nan_processing which transitively cycles back
    _maybe_clean_ram()
    try:
        confidence_model.fit(test_df, confidence_targets, **fit_params_copy)
    except Exception as e:
        # CatBoost reports "Environment for task type [GPU] not found" when the
        # host has a CUDA device (so CUDA_IS_AVAILABLE=True) but no CatBoost
        # GPU runtime. Fall back to CPU -- the confidence model is small and
        # CPU-adequate; this keeps training from aborting on mixed environments.
        if confidence_task_type == "GPU" and "Environment for task type [GPU] not found" in str(e):
            logger.warning("CatBoost GPU environment unavailable for confidence model; falling back to CPU.")
            confidence_model = CatBoostRegressor(verbose=0, eval_fraction=0.1, task_type="CPU", **confidence_model_kwargs)
            confidence_model.fit(test_df, confidence_targets, **fit_params_copy)
        else:
            raise
    _maybe_clean_ram()

    if use_shap:
        try:
            import shap
            import shap.utils.transformers

            shap.utils.transformers.is_transformers_lm = lambda model: False
        except (ImportError, AttributeError):
            pass
        # shap.plots.beeswarm internals do ``features[:, i]``
        # with ``i`` typed as numpy.int64. Polars 1.x rejects numpy
        # integer indices on DataFrame ``__getitem__`` with
        # ``cannot select columns using key of type 'numpy.int64'``.
        # Pass a pandas view so the indexing falls back to pandas's
        # numpy-int-tolerant ``__getitem__``.
        # Use the Arrow-backed split-blocks bridge: SHAP rejects polars frames and
        # the default .to_pandas() consolidates blocks (~32x slower on multi-million-row
        # frames). The view materialises lazily where SHAP indexes column-by-column.
        from .utils import get_pandas_view_of_polars_df as _get_pandas_view
        _test_df_for_shap = _get_pandas_view(test_df) if isinstance(test_df, pl.DataFrame) else test_df
        explainer = shap.TreeExplainer(confidence_model)
        shap_values = explainer(_test_df_for_shap)
        # shap.plots.beeswarm creates its own figure (+ a colorbar axes); snapshot the open
        # figure ids first so we can close EVERY figure it opened, not just plt.gcf().
        _figs_before = set(plt.get_fignums())
        shap.plots.beeswarm(
            shap_values,
            max_display=max_features,
            color=plt.get_cmap(cmap),
            alpha=alpha,
            color_bar_label=ylabel,
            show=False,
        )
        plt.xlabel(title)
        fig = plt.gcf()
        if plot_file:
            import os as _os
            _root, _ext = _os.path.splitext(plot_file)
            _path = plot_file if _ext else (plot_file + ".png")
            try:
                fig.savefig(_path, bbox_inches="tight")
            except Exception as _save_err:
                logger.warning("Confidence beeswarm savefig failed for %s: %s", _path, _save_err)
        # Guard plt.show() against the non-interactive Agg backend (CI / pytest / headless scripts
        # pin Agg); plt.show() on Agg emits the "FigureCanvasAgg is non-interactive" warning and
        # renders nothing. Always close every figure the beeswarm opened so none leak in the registry.
        from mlframe.metrics import show_plots_unless_agg
        from mlframe.metrics.calibration import _close_unless_interactive
        _was_shown = show_plots_unless_agg()
        _new_figs = [plt.figure(_num) for _num in plt.get_fignums() if _num not in _figs_before]
        _close_unless_interactive(_new_figs or fig, was_shown=_was_shown)
    else:
        # Lazy import -- see comment near top of module about cycle.
        from .evaluation import plot_model_feature_importances

        plot_model_feature_importances(
            model=confidence_model,
            columns=list(test_df.columns),
            model_name=title,
            num_factors=max_features,
            figsize=(figsize[0] * 0.7, figsize[1] / 2),
        )

    return confidence_model
