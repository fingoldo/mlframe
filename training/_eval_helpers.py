"""Evaluation helpers extracted from ``trainer.py``.

Post-training evaluation: metric computation, feature importance,
confidence analysis, XGB category alignment, column decategorisation.
"""

from __future__ import annotations

import logging
import os
import re
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from sklearn.pipeline import Pipeline

from .utils import log_ram_usage, filter_existing

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from .utils import log_ram_usage

logger = logging.getLogger(__name__)

def _align_xgb_cat_categories(model_type_name, train_df, val_df=None, test_df=None):
    """Make ``pd.CategoricalDtype`` columns use the SAME ``categories``
    list across train / val / test so XGBoost's ``enable_categorical``
    path doesn't reject val/test rows whose category was never seen
    at fit time.

    Background (fuzz seed=2024 c0060 / 2026-04-27): XGBoost stores the
    category index from train at fit time. If a row in ``val_df`` /
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
    # 2026-04-28: alignment runs for ALL pandas frames regardless of
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
    # applies at line ~3466 -- surfaced 2026-04-28 as fuzz c0025
    # (cb_hgb_lgb_xgb / pl_enum) failing with
    # ``Unsupported data type Enum(...) for a text feature column``.
    if isinstance(train_df, pl.DataFrame):
        return train_df, val_df, test_df

    if not isinstance(train_df, pd.DataFrame):
        return train_df, val_df, test_df

    cat_cols_to_align = []
    for _col in train_df.columns:
        try:
            _dt = train_df[_col].dtype
        except Exception:
            continue
        if isinstance(_dt, pd.CategoricalDtype):
            cat_cols_to_align.append(_col)

    # DIAG (2026-04-28): print full cat layout to localise c0060 flake.
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
        # Union of categories across train + val + test, preserving
        # train's order first (keeps the train-fit decision-tree splits
        # using the same integer codes as before).
        train_cats = list(train_df[_col].cat.categories)
        union_cats = list(train_cats)
        seen = set(train_cats)
        for _other_df in (val_df, test_df):
            if _other_df is None or not isinstance(_other_df, pd.DataFrame):
                continue
            if _col not in _other_df.columns:
                continue
            _other_dtype = _other_df[_col].dtype
            if not isinstance(_other_dtype, pd.CategoricalDtype):
                continue
            for _cat in _other_dtype.categories:
                if _cat not in seen:
                    union_cats.append(_cat)
                    seen.add(_cat)
        if union_cats == train_cats:
            continue  # No new categories -- alignment unnecessary.
        # Re-cast each split's column to the union categories.
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

    Background (fuzz c0102 / 2026-04-27): after ``CatBoostEncoder`` /
    target encoders / RFECV-driven re-encodings, a column may end up
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
            except Exception:
                continue
            if not isinstance(_dt, pd.CategoricalDtype):
                continue
            try:
                _cat_kind = _dt.categories.dtype.kind
            except Exception:
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
        from .strategies import PANDAS_CATEGORICAL_DTYPES

        _pd_cats = list(PANDAS_CATEGORICAL_DTYPES)
        for split in (train_df, val_df, test_df):
            if isinstance(split, pd.DataFrame):
                cat_columns.update(split.select_dtypes(_pd_cats).columns)
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
# unconditional link). Surfaced 2026-04-28 by ``test_no_import_cycles``.
# Each of the two callsites does its own lazy import so the runtime
# cost is paid once per fit.


_BTTR_RE = re.compile(r"BTTR=(\d+%)")
# C1 (2026-05-11): also match MTRESID= (the composite-target rename
# from MTTR; same numeric format).
_MTTR_RE = re.compile(r"(MTTR|MTRESID)=(-?\d+\.\d+)")
_MLTR_RE = re.compile(r"MLTR=([\d,]+%)")


def _append_split_rate_suffix(model_name: str, *, split_name: str, target) -> str:
    """Splice ``/BTV=X%`` (val) or ``/BTTS=X%`` (test) inline next to
    the existing ``BTTR=`` token so the chart title reads
    ``BTTR/BTV=74%/86%`` (instead of the previous append-at-end form
    that scattered ``BTTR`` and ``BTV`` across the title).

    Tag conventions:

    * binary classification — ``BTV`` / ``BTTS`` (Binary Target Val / TeSt)
    * regression — ``MTV`` / ``MTTS`` (Mean Target Val / TeSt)
    * multilabel classification — ``MLV`` / ``MLTS`` (MultiLabel Val / TeSt)

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

    # 2026-04-27 Session 7 batch 8 (user feedback): splice the per-split
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
        # 2026-05-11 (user request): adaptive format -- 2 d.p. for typical magnitudes (MTV/MTTS for raw TVT ~ 11556), more decimals for tiny magnitudes (composite residual MTV ~ -1.17). The split-suffix carries the same tag prefix as train so composite shows ``MTRESID/MRV=...``, mirroring the existing BTTR/BTV / MTTR/MTV pattern.
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
    title_metrics_tokens: Optional[Tuple[str, ...]] = None,
    plot_outputs: Optional[str] = None,
    plot_dpi: Optional[int] = None,
    multiclass_panels: Optional[str] = None,
    multilabel_panels: Optional[str] = None,
    ltr_panels: Optional[str] = None,
    quantile_panels: Optional[str] = None,
    quantile_alphas: Optional[Tuple[float, ...]] = None,
    target_type: Optional[str] = None,
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

    # Historical 0-row split skip removed 2026-04-28 (batch 4). The
    # original empty-split window came from outlier detection (val-side)
    # or splitter edge cases; both are now guarded at the source. If a
    # 0-row split still arrives here, the metrics layer would crash with
    # ``Found empty input array`` from classification_report: a clear
    # signal of an upstream bug rather than silently dropping the split's
    # contribution to the report.

    df_prepared = _prepare_df_for_model(df, model_type_name) if df is not None else None

    effective_show_fi = show_fi and not has_other_splits
    split_plot_file = f"{plot_file}_{split_name}" if plot_file else ""

    # 2026-04-26 Session 7: splice the split-specific target rate into
    # model_name for THIS split's report only. ``select_target`` stamped
    # the train rate as ``BTTR=`` / ``MTTR=`` / ``MLTR=`` on the
    # canonical model_name; here we splice the val/test rate inline
    # via regex so chart titles read e.g. ``BTTR/BTV=74%/86%`` (val)
    # and ``BTTR/BTTS=74%/83%`` (test) — prior shift between train
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
    cat_features: List[str] = None,
    text_features: List[str] = None,
    embedding_features: List[str] = None,
    confidence_model_kwargs: dict = None,
    fit_params: dict = None,
    use_shap: bool = True,
    max_features: int = 20,
    cmap: str = "coolwarm",
    alpha: float = 0.5,
    title: str = "Confidence Analysis",
    ylabel: str = "Prediction Confidence",
    figsize: Tuple[float, float] = (10, 6),
    verbose: bool = False,
) -> Any:
    """Analyze which features most affect prediction confidence."""
    if test_df is None:
        return None

    if verbose:
        logger.info("Running confidence analysis...")

    if confidence_model_kwargs is None:
        confidence_model_kwargs = {}

    # 2026-04-26 Session 7 batch 5: bound the confidence model's
    # iteration budget so the CPU fallback can't spin indefinitely.
    # Without a cap CB defaults to iterations=1000, which on the rich
    # feature schema typical for the suite (50+ cols) translates to
    # 4-10+ minutes per confidence fit on CPU. The confidence regressor
    # is best-effort diagnostic — even 50 boosting rounds gives a
    # serviceable feature-importance signal. Caller can override.
    confidence_model_kwargs.setdefault("iterations", 200)
    confidence_model_kwargs.setdefault("early_stopping_rounds", 30)

    confidence_task_type = "GPU" if CUDA_IS_AVAILABLE else "CPU"
    confidence_model = CatBoostRegressor(verbose=0, eval_fraction=0.1, task_type=confidence_task_type, **confidence_model_kwargs)

    fit_params_copy = {}
    if fit_params:
        fit_params_copy = copy.copy(fit_params)
        if "eval_set" in fit_params_copy:
            del fit_params_copy["eval_set"]

    # 2026-04-28: drop text / embedding columns from test_df upfront.
    # SHAP's TreeExplainer rebuilds a CatBoost Pool using ONLY
    # ``cat_features`` from the model (no text awareness), so text
    # columns reaching Pool as numeric raise ``Bad value for
    # num_feature ...: Cannot convert '<text>' to float`` (default-seed
    # c0016 / c0017). Free-text TF-IDF features aren't analysable by
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
            except Exception:
                continue
            if _dt == object or str(_dt) in ("string", "string[python]", "string[pyarrow]"):
                _drop_for_conf.append(_c)
    elif isinstance(test_df, pl.DataFrame):
        # 2026-04-26 Session 7 batch 5: polars-side auto-detect. The
        # earlier pandas-only branch missed:
        #   - pl.Utf8 / pl.String text columns (default-seed c0056:
        #     cb+hgb+xgb pl_nullable n=5000 → CB Pool crash on ``text_0``)
        #   - pl.List / pl.Array embedding columns (same combo, ``emb_0``
        #     surfaces here when fit_params['embedding_features'] is
        #     None because the trailing model is HGB/XGB which doesn't
        #     accept the kwarg → the explicit-drop list is empty)
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
            except Exception:
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

    if cat_features is not None:
        fit_params_copy["cat_features"] = cat_features
    elif "cat_features" not in fit_params_copy:
        fit_params_copy["cat_features"] = get_categorical_columns(test_df, include_string=False)

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
    # it cannot. Surfaced fuzz seed=default c0000 (multilabel target +
    # confidence_analysis_cfg=True): IndexError shape mismatch (N,) (N,K).
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
    # 2026-04-28: when test_df, test_target, and test_probs disagree on
    # row count (an upstream filter dropped rows from one but not the
    # others), the confidence model fits with mismatched lengths and
    # CB raises ``Length of label=N1 and length of data=N2 is
    # different``. Skip with INFO so the suite continues; the
    # confidence pass is best-effort, not a hard contract. Surfaced
    # default-seed c0074 (hgb / binary classification +
    # confidence_analysis_cfg=True).
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

    # 2026-04-26 Session 7 batch 5: degenerate confidence_targets.
    # When all rows in test happen to land in the same predicted-
    # probability bucket (small test sets, severely-miscalibrated /
    # constant-output models), every confidence_target is identical
    # and CB rejects with "All train targets are equal". The
    # confidence regressor has nothing to learn anyway. Skip with a
    # WARN so the operator knows the diagnostic was un-runnable.
    # Surfaced default-seed c0081 (hgb+lgb pandas n=300).
    n_unique_conf = int(np.unique(confidence_targets).size)
    if n_unique_conf < 2:
        logger.warning(
            "Confidence analysis skipped: all confidence_targets are "
            "equal (n_unique=%d, value=%s). The confidence regressor "
            "has no signal to learn — typical for tiny test sets where "
            "all rows share one predicted-prob bucket, or for severely "
            "miscalibrated models emitting a constant probability.",
            n_unique_conf,
            float(confidence_targets[0]) if confidence_targets.size else float("nan"),
        )
        return None

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
        # 2026-04-28: shap.plots.beeswarm internals do ``features[:, i]``
        # with ``i`` typed as numpy.int64. Polars 1.x rejects numpy
        # integer indices on DataFrame ``__getitem__`` with
        # ``cannot select columns using key of type 'numpy.int64'``.
        # Pass a pandas view so the indexing falls back to pandas's
        # numpy-int-tolerant ``__getitem__``. Surfaced default-seed
        # c0074 (hgb / multiclass + confidence_analysis_cfg=True).
        _test_df_for_shap = test_df.to_pandas() if isinstance(test_df, pl.DataFrame) else test_df
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
        plt.show()
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


