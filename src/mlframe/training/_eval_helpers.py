"""Evaluation helpers extracted from ``trainer.py``.

Post-training evaluation: metric computation, feature importance,
confidence analysis, XGB category alignment, column decategorisation.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

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

# CUDA_IS_AVAILABLE / get_categorical_columns / _maybe_clean_ram all sit in modules that ALSO import from this file -- lazy local imports at call sites.

from sklearn.pipeline import Pipeline

from .utils import log_ram_usage, filter_existing



from ._data_helpers import _prepare_df_for_model

logger = logging.getLogger(__name__)

def _align_xgb_cat_categories(model_type_name, train_df, val_df=None, test_df=None):
    """Make ``pd.CategoricalDtype`` columns use the SAME ``categories``
    list across train / val / test so XGBoost's ``enable_categorical``
    path doesn't reject val/test rows whose category was never seen
    at fit time.

    Background: XGBoost stores the category index from train at fit
    time. If a row in ``val_df`` /
    ``test_df`` carries a category that wasn't in ``train_df``, the
    XGBoost C++ ``cat_container.h:29`` raises
    ``"Found a category not in the training set for the Nth column"``.
    Typical sources: ``weird_cat_content`` axis injecting "" / null-like
    values into a subset of rows that the train/val/test split
    happens not to match across, or imbalanced rare cat levels that
    randomly land in only val/test.

    Fix: compute the UNION of category levels across all three splits
    per column, then re-cast each split's column to that union. XGBoost
    now sees val/test as a subset of the train cat universe -- no
    "unseen category" rejections.

    No-op for non-XGB models. CB / HGB / LGB tolerate unseen
    categories natively.

    Returns ``(train_df, val_df, test_df)`` -- frames are copied
    in-place when an alignment happens, otherwise returned unchanged.
    """
    # Alignment runs for ALL pandas frames regardless of
    # model_type_name. Rationale: XGB-on-multilabel wrappers
    # (MultiOutputClassifier / _ChainEnsemble / ClassifierChain) hide
    # the inner XGB fit behind the wrapper's type-name; trying to
    # detect "is there an XGB downstream" via the outer name is
    # error-prone. The alignment itself is a no-op for clean splits
    # (no mismatched cat levels = no copy, no set_categories call) and
    # harmless for CB/HGB/LGB (they tolerate unseen cats but accept the
    # union dtype identically). Cost: one cat-column scan per frame.
    if os.environ.get("MLFRAME_CAT_DIAG"):
        print(
            f"[CAT-DIAG-ENTER] model={model_type_name}, train_df type={type(train_df).__name__}, is_pandas={isinstance(train_df, pd.DataFrame)}, is_polars={isinstance(train_df, pl.DataFrame)}",
            flush=True,
        )

    # Polars frames: alignment is now done UPSTREAM by
    # ``XGBoostStrategy.prepare_polars_dataframe`` + ``build_polars_enum_map``
    # in ``training.core`` (built once from the train+val union, leak-free).
    # Doing it again here would, for non-XGB models in mixed suites, undo
    # the CB-specific ``Enum->String`` text-feature cast that core.py
    # applies elsewhere -- otherwise CB rejects with
    # ``Unsupported data type Enum(...) for a text feature column``.
    if isinstance(train_df, pl.DataFrame):
        return train_df, val_df, test_df

    if not isinstance(train_df, pd.DataFrame):
        return train_df, val_df, test_df

    cat_cols_to_align = []
    for _col in train_df.columns:
        try:
            _dt = train_df[_col].dtype
        except Exception as _e_dt:
            # Same shape as the per-col dtype-access pattern elsewhere in
            # this module: silent skip means a Categorical that should be
            # aligned across train/val/test silently isn't, then later
            # asserts on category-set equality may misreport.
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "_eval_helpers: train_df dtype access failed for col=%r "
                "(%s); col will NOT be added to cat-alignment list.",
                _col, _e_dt,
            )
            continue
        if isinstance(_dt, pd.CategoricalDtype):
            cat_cols_to_align.append(_col)

    # DIAG: print full cat layout to localise cat-alignment failures.
    if os.environ.get("MLFRAME_CAT_DIAG"):
        try:
            for _df_name, _df in (("train", train_df), ("val", val_df), ("test", test_df)):
                if _df is None:
                    continue
                for _c in cat_cols_to_align:
                    if _c in _df.columns:
                        _cats = list(_df[_c].cat.categories)
                        _vals_unique = sorted(set(str(v) for v in _df[_c].dropna()))
                        print(f"[CAT-DIAG] model={model_type_name} {_df_name}.{_c}: dtype.categories={_cats}, actual_unique={_vals_unique}", flush=True)
        except Exception as _exc:
            print(f"[CAT-DIAG] failed: {_exc}", flush=True)

    if not cat_cols_to_align:
        return train_df, val_df, test_df

    def _ensure_copy(df, flag):
        if df is None:
            return df
        if not getattr(df, flag, False):
            df = df.copy()
            try:
                setattr(df, flag, True)
            except Exception:
                pass
        return df

    for _col in cat_cols_to_align:
        # Union of categories across train + val ONLY (NOT test): test categories must never feed back into
        # train at fit-time, that's the canonical leak. val is fair game because it participates in early-stopping /
        # model selection, which the user has already authorised. test rows that carry an unseen category get
        # cast with the existing union and surface as NaN under pandas' Categorical semantics; downstream
        # NaN-tolerant backends (CB/HGB/LGB) handle the NaN natively. XGBoost's hard-rejection of unseen
        # categories is the only path that requires upfront alignment - and for those, the upstream polars
        # Enum builder (build_polars_enum_map) already does the leak-free train+val union.
        train_cats = list(train_df[_col].cat.categories)
        union_cats = list(train_cats)
        seen = set(train_cats)
        if val_df is not None and isinstance(val_df, pd.DataFrame) and _col in val_df.columns:
            _val_dtype = val_df[_col].dtype
            if isinstance(_val_dtype, pd.CategoricalDtype):
                for _cat in _val_dtype.categories:
                    if _cat not in seen:
                        union_cats.append(_cat)
                        seen.add(_cat)
        if union_cats == train_cats:
            continue  # No new categories -- alignment unnecessary.
        # Re-cast each split's column to the union categories. test_df gets the train+val union as well
        # (so XGB's "unseen category" check has the same vocabulary to compare against). Test rows whose
        # native category isn't in the union get cast to NaN, which is the intended leak-free semantics.
        train_df = _ensure_copy(train_df, "_mlframe_filled")
        train_df[_col] = train_df[_col].cat.set_categories(union_cats)
        if val_df is not None and isinstance(val_df, pd.DataFrame) and _col in val_df.columns:
            if isinstance(val_df[_col].dtype, pd.CategoricalDtype):
                val_df = _ensure_copy(val_df, "_mlframe_filled")
                val_df[_col] = val_df[_col].cat.set_categories(union_cats)
        if test_df is not None and isinstance(test_df, pd.DataFrame) and _col in test_df.columns:
            if isinstance(test_df[_col].dtype, pd.CategoricalDtype):
                test_df = _ensure_copy(test_df, "_mlframe_filled")
                test_df[_col] = test_df[_col].cat.set_categories(union_cats)

    return train_df, val_df, test_df


def _decategorise_float_cat_columns(train_df, val_df=None, test_df=None):
    """Convert ``pd.CategoricalDtype`` columns whose underlying category
    values are floats back to plain numeric float columns.

    Background: after ``CatBoostEncoder`` / target encoders /
    RFECV-driven re-encodings, a column may end up
    as a categorical whose category levels are floats (e.g. target-
    encoded means ``[0.13, 0.42, ...]`` were ``.astype("category")``
    boxed). At that point the column is *semantically numeric* -- the
    "categories" are continuous target encodings -- but the dtype still
    says "categorical". Both downstream backends reject this:
      * XGBoost's columnar reader (``columnar.h:134``):
        ``"Category index from DataFrame has floating point dtype,
        consider using strings or integers instead"``.
      * CatBoost (``_catboost.pyx``): ``"bad object for id: 0.0"`` --
        CB's hash-map lookup of a float as a categorical id fails.
    The proper fix is to honour the semantics: drop the categorical
    wrapper and expose the float values as a regular numeric column.
    Apply uniformly to train + val + test so downstream
    ``enable_categorical`` / ``cat_features`` filters see consistent
    dtypes across the fit / predict boundary.

    Returns ``(train_df, val_df, test_df)`` -- frames are copied when a
    decategorise happens, otherwise returned unchanged.
    """

    def _decat(df):
        if df is None or not isinstance(df, pd.DataFrame):
            return df
        decat_cols = []
        for _col in df.columns:
            try:
                _dt = df[_col].dtype
            except Exception as _e_dt:
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "_eval_helpers: dtype access failed for col=%r (%s) in "
                    "de-cat helper; col not considered for de-cat.",
                    _col, _e_dt,
                )
                continue
            if not isinstance(_dt, pd.CategoricalDtype):
                continue
            try:
                _cat_kind = _dt.categories.dtype.kind
            except Exception as _e_cat:
                # Categorical with broken categories.dtype is the warning
                # signal: silent skip leaves the mis-tagged column as cat,
                # which then surfaces as a CatBoost crash on float
                # category values downstream.
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "_eval_helpers: cat.categories.dtype access failed for "
                    "col=%r (%s); col not considered for de-cat.",
                    _col, _e_cat,
                )
                continue
            # 'f'=float, 'c'=complex. Both signal a target-encoded
            # column that's been mis-tagged as categorical.
            if _cat_kind in ("f", "c"):
                decat_cols.append(_col)
        if not decat_cols:
            return df
        if not getattr(df, "_mlframe_filled", False):
            df = df.copy()
            df._mlframe_filled = True
        for _col in decat_cols:
            # ``.astype(_dt.categories.dtype)`` materialises the
            # underlying floats and drops the CategoricalDtype wrapper.
            _src_dtype = df[_col].dtype.categories.dtype
            df[_col] = df[_col].astype(_src_dtype)
        return df

    if isinstance(train_df, pd.DataFrame):
        train_df = _decat(train_df)
    val_df = _decat(val_df)
    test_df = _decat(test_df)
    return train_df, val_df, test_df


def _filter_categorical_features(fit_params, train_df, val_df=None, test_df=None):
    """Filter cat_features to only include actual categorical columns.

    Uses the UNION of categorical columns across train / val / test
    rather than train alone. Rationale: ``eval_set`` contains val
    (already registered into fit_params at this point). If val has a
    categorical dtype in a column that train doesn't -- e.g. upstream
    pipeline cast train but val slipped through a different code path
    -- then CB raises
    ``column 'X' has dtype 'category' but is not in cat_features``
    because we pruned X out of cat_features.

    Union-based detection keeps every column that IS categorical in
    ANY split, matching how CB / XGB expect cat_features to be
    declared (a superset list is fine; a subset causes the error).
    """
    if "cat_features" not in fit_params:
        return

    cat_columns: set = set()
    if isinstance(train_df, pd.DataFrame):
        from .strategies import PANDAS_CATEGORICAL_SELECT_DTYPES

        for split in (train_df, val_df, test_df):
            if isinstance(split, pd.DataFrame):
                cat_columns.update(
                    split.select_dtypes(list(PANDAS_CATEGORICAL_SELECT_DTYPES)).columns
                )
    elif isinstance(train_df, pl.DataFrame):
        from .strategies import get_polars_cat_columns

        for split in (train_df, val_df, test_df):
            if isinstance(split, pl.DataFrame):
                cat_columns.update(get_polars_cat_columns(split))
    else:
        return

    fit_params["cat_features"] = [col for col in fit_params["cat_features"] if col in cat_columns]


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Metrics and Reporting (imported from evaluation module)
# -----------------------------------------------------------------------------------------------------------------------------------------------------

# Report functions live in ``evaluation.py`` and are imported lazily
# inside the call sites below to break the structural import cycle
# (``evaluation.py`` lazy-imports ``_predict_with_fallback`` /
# ``_PerClassIsotonicCalibrator`` / ``_PostHocMultiCalibratedModel``
# from this module -- making the top-level edge here the cycle's only
# unconditional link). Surfaced by ``test_no_import_cycles``.
# Each of the two callsites does its own lazy import so the runtime
# cost is paid once per fit.


_BTTR_RE = re.compile(r"BTTR=(\d+%)")
# Also match MTRESID= (the composite-target rename
# from MTTR; same numeric format).
_MTTR_RE = re.compile(r"(MTTR|MTRESID)=(-?\d+\.\d+)")
_MLTR_RE = re.compile(r"MLTR=([\d,]+%)")


def _append_split_rate_suffix(model_name: str, *, split_name: str, target) -> str:
    """Splice ``/BTV=X%`` (val) or ``/BTTS=X%`` (test) inline next to
    the existing ``BTTR=`` token so the chart title reads
    ``BTTR/BTV=74%/86%`` (instead of the previous append-at-end form
    that scattered ``BTTR`` and ``BTV`` across the title).

    Tag conventions:

    * binary classification -- ``BTV`` / ``BTTS`` (Binary Target Val / TeSt)
    * regression -- ``MTV`` / ``MTTS`` (Mean Target Val / TeSt)
    * multilabel classification -- ``MLV`` / ``MLTS`` (MultiLabel Val / TeSt)

    Train splits and cases where ``model_name`` doesn't carry a
    ``BTTR=`` / ``MTTR=`` / ``MLTR=`` token (e.g. legacy unit-test-driven
    calls with the older ``BT=`` tag) pass through unchanged.
    """
    if split_name not in ("val", "test") or target is None:
        return model_name

    # Coerce target to a flat numpy array.
    if isinstance(target, (pl.Series, pd.Series)):
        arr = target.to_numpy()
    elif isinstance(target, np.ndarray):
        arr = target
    else:
        try:
            arr = np.asarray(target)
        except Exception:
            return model_name
    if arr.size == 0:
        return model_name

    short = "V" if split_name == "val" else "TS"

    # Splice the per-split
    # value INSIDE the train-rate token so the title reads
    # ``BTTR/BTV=78%/39%`` instead of the previous separated form
    # ``... BTTR=78% ... /BTV=39%`` which spread the two numbers
    # across line breaks.
    bttr_match = _BTTR_RE.search(model_name)
    if bttr_match:
        if arr.ndim != 1:
            return model_name
        rate = float((arr == 1).sum()) / arr.size
        train_pct = bttr_match.group(1)
        return _BTTR_RE.sub(f"BTTR/BT{short}={train_pct}/{rate*100:.0f}%", model_name, count=1)
    mttr_match = _MTTR_RE.search(model_name)
    if mttr_match:
        if arr.ndim != 1:
            return model_name
        _tag = mttr_match.group(1)  # "MTTR" or "MTRESID"
        train_val = mttr_match.group(2)
        # Adaptive format -- 2 d.p. for typical magnitudes (MTV/MTTS for raw TVT ~ 11556), more decimals for tiny magnitudes (composite residual MTV ~ -1.17). The split-suffix carries the same tag prefix as train so composite shows ``MTRESID/MRV=...``, mirroring the existing BTTR/BTV / MTTR/MTV pattern.
        from ._format import format_metric as _fmt

        # Split-suffix: keep the trailing letter pair as-is (BTTR -> BTV, MTTR -> MTV); for MTRESID use MR* (MRV / MRTS) to stay under 8 chars on chart titles.
        if _tag == "MTRESID":
            _split_prefix = "MR"
        else:
            _split_prefix = "MT"
        return _MTTR_RE.sub(
            f"{_tag}/{_split_prefix}{short}={train_val}/{_fmt(float(arr.mean()))}",
            model_name,
            count=1,
        )
    mltr_match = _MLTR_RE.search(model_name)
    if mltr_match:
        if arr.ndim != 2 or arr.shape[0] == 0:
            return model_name
        rates = arr.mean(axis=0)
        summary = ",".join(f"{p*100:.0f}" for p in rates)
        train_summary = mltr_match.group(1)
        return _MLTR_RE.sub(f"MLTR/ML{short}={train_summary}/{summary}%", model_name, count=1)
    return model_name


def _compute_split_metrics(
    split_name: str,
    df,
    target,
    idx,
    model,
    model_type_name: str,
    model_name: str,
    metrics_dict: dict,
    group_ids=None,
    target_label_encoder=None,
    preds=None,
    probs=None,
    figsize=(15, 5),
    nbins: int = 10,
    print_report: bool = True,
    plot_file: str = "",
    show_perf_chart: bool = True,
    show_fi: bool = False,
    fi_kwargs: dict = None,
    subgroups: dict = None,
    custom_ice_metric=None,
    custom_rice_metric=None,
    details: str = "",
    has_other_splits: bool = False,
    n_features: int = None,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    title_metrics_tokens: tuple[str, ...] | None = None,
    plot_outputs: str | None = None,
    plot_dpi: int | None = None,
    multiclass_panels: str | None = None,
    multilabel_panels: str | None = None,
    ltr_panels: str | None = None,
    quantile_panels: str | None = None,
    quantile_alphas: tuple[float, ...] | None = None,
    target_type: str | None = None,
):
    """Unified metrics computation for train/val/test splits."""
    # Derive columns from df if available (for feature importance)
    columns = list(df.columns) if df is not None and hasattr(df, "columns") else []
    # Only skip if no precomputed predictions/probabilities exist and
    # either no feature frame is available or no live model exists to
    # predict from. Ensemble pseudo-model rows have ``model=None`` by
    # design; their metrics must be driven by precomputed preds/probs.
    if preds is None and probs is None and (df is None or model is None):
        return preds, probs, columns

    # Historical 0-row split skip removed. The
    # original empty-split window came from outlier detection (val-side)
    # or splitter edge cases; both are now guarded at the source. If a
    # 0-row split still arrives here, the metrics layer would crash with
    # ``Found empty input array`` from classification_report: a clear
    # signal of an upstream bug rather than silently dropping the split's
    # contribution to the report.

    df_prepared = _prepare_df_for_model(df, model_type_name) if df is not None else None

    effective_show_fi = show_fi and not has_other_splits
    split_plot_file = f"{plot_file}_{split_name}" if plot_file else ""

    # Splice the split-specific target rate into
    # model_name for THIS split's report only. ``select_target`` stamped
    # the train rate as ``BTTR=`` / ``MTTR=`` / ``MLTR=`` on the
    # canonical model_name; here we splice the val/test rate inline
    # via regex so chart titles read e.g. ``BTTR/BTV=74%/86%`` (val)
    # and ``BTTR/BTTS=74%/83%`` (test) -- prior shift between train
    # and val/test is visible in every header.
    augmented_model_name = _append_split_rate_suffix(
        model_name,
        split_name=split_name,
        target=target,
    )

    # Lazy import -- see comment near the top of this module about the
    # ``evaluation`` <-> ``trainer`` import cycle.
    from .evaluation import report_model_perf

    preds, probs = report_model_perf(
        targets=target,
        columns=columns,
        df=df_prepared,
        model_name=augmented_model_name,
        model=model,
        target_label_encoder=target_label_encoder,
        preds=preds,
        probs=probs,
        figsize=figsize,
        report_title=" ".join([split_name.upper(), details]).strip(),
        nbins=nbins,
        print_report=print_report,
        plot_file=split_plot_file,
        show_perf_chart=show_perf_chart,
        show_fi=effective_show_fi,
        fi_kwargs=fi_kwargs if fi_kwargs else {},
        subgroups=subgroups,
        subset_index=idx,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        metrics=metrics_dict,
        group_ids=group_ids[idx] if group_ids is not None and idx is not None else None,
        n_features=n_features,
        show_prob_histogram=show_prob_histogram,
        prob_histogram_yscale=prob_histogram_yscale,
        show_inline_population_labels=show_inline_population_labels,
        title_metrics_tokens=title_metrics_tokens,
        plot_outputs=plot_outputs,
        plot_dpi=plot_dpi,
        multiclass_panels=multiclass_panels,
        multilabel_panels=multilabel_panels,
        ltr_panels=ltr_panels,
        quantile_panels=quantile_panels,
        quantile_alphas=quantile_alphas,
        target_type=target_type,
    )
    return preds, probs, columns


def run_confidence_analysis(
    test_df: pd.DataFrame,
    test_target: np.ndarray,
    test_probs: np.ndarray,
    cat_features: list[str] = None,
    text_features: list[str] = None,
    embedding_features: list[str] = None,
    confidence_model_kwargs: dict = None,
    fit_params: dict = None,
    use_shap: bool = True,
    max_features: int = 20,
    cmap: str = "coolwarm",
    alpha: float = 0.5,
    title: str = "Confidence Analysis",
    ylabel: str = "Prediction Confidence",
    figsize: tuple[float, float] = (10, 6),
    verbose: bool = False,
) -> Any:
    """Analyze which features most affect prediction confidence."""
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
    confidence_model_kwargs.setdefault("iterations", 200)
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
        shap.plots.beeswarm(
            shap_values,
            max_display=max_features,
            color=plt.get_cmap(cmap),
            alpha=alpha,
            color_bar_label=ylabel,
            show=False,
        )
        plt.xlabel(title)
        # Guard plt.show() against the non-interactive Agg backend
        # (CI / pytest / headless scripts pin Agg); plt.show() on Agg
        # emits the "FigureCanvasAgg is non-interactive, and thus cannot
        # be shown" UserWarning and renders nothing.
        from mlframe.metrics._calibration_plot import _show_plots_unless_agg
        _show_plots_unless_agg()
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Config Building Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


