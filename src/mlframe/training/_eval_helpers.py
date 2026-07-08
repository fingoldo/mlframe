"""Evaluation helpers extracted from ``trainer.py``.

Post-training evaluation: metric computation, feature importance,
confidence analysis, XGB category alignment, column decategorisation.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]  # plt-using code paths are guarded; matplotlib-less envs skip the plot branches
import pandas as pd
import polars as pl

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None  # type: ignore[assignment]  # only used when CB is the chosen model; guarded at call site

# CUDA_IS_AVAILABLE / get_categorical_columns / _maybe_clean_ram all sit in modules that ALSO import from this file -- lazy local imports at call sites.


from pyutilz.system import ensure_dir_exists


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
        logger.debug(
            "[CAT-DIAG-ENTER] model=%s, train_df type=%s, is_pandas=%s, is_polars=%s",
            model_type_name, type(train_df).__name__,
            isinstance(train_df, pd.DataFrame), isinstance(train_df, pl.DataFrame),
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
        except (KeyError, AttributeError, ValueError) as _e_dt:
            # Narrowed from a bare ``except Exception``: only the expected
            # column-access / dtype-resolution failures are tolerated. A skip
            # here means a Categorical that should be aligned across
            # train/val/test silently isn't, so later asserts on category-set
            # equality may misreport -- log a WARNING (not a silent debug) so
            # the gap is visible. Any other exception now propagates instead of
            # masking a genuine bug.
            logger.warning(
                "_eval_helpers: dtype access failed for col=%r (%s: %s); col "
                "will NOT be added to cat-alignment list, so train/val/test "
                "category alignment may be incomplete for it.",
                _col, type(_e_dt).__name__, _e_dt,
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
                        logger.debug("[CAT-DIAG] model=%s %s.%s: dtype.categories=%s, actual_unique=%s", model_type_name, _df_name, _c, _cats, _vals_unique)
        except Exception as _exc:
            logger.debug("[CAT-DIAG] failed: %s", _exc)

    if not cat_cols_to_align:
        return train_df, val_df, test_df

    def _ensure_copy(df, flag):
        if df is None:
            return df
        if not getattr(df, flag, False):
            # Shallow copy: only cat_cols_to_align columns are reassigned by the caller; deep-copying a 100+ GB frame to realign a few cat columns OOMs. ``deep=False`` shares untouched buffers, caller frame unmutated.
            df = df.copy(deep=False)
            try:
                setattr(df, flag, True)
            except Exception as e:
                logger.debug("swallowed exception in _eval_helpers.py: %s", e)
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
                    "_eval_helpers: dtype access failed for col=%r (%s) in " "de-cat helper; col not considered for de-cat.",
                    _col,
                    _e_dt,
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
                    "_eval_helpers: cat.categories.dtype access failed for " "col=%r (%s); col not considered for de-cat.",
                    _col,
                    _e_cat,
                )
                continue
            # 'f'=float, 'c'=complex. Both signal a target-encoded
            # column that's been mis-tagged as categorical.
            if _cat_kind in ("f", "c"):
                decat_cols.append(_col)
        if not decat_cols:
            return df
        if not getattr(df, "_mlframe_filled", False):
            # Shallow copy: only decat_cols are recast below; deep-copying a 100+ GB frame to decat a few columns OOMs. ``deep=False`` shares untouched buffers, caller frame unmutated.
            df = df.copy(deep=False)
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
                cat_columns.update(split.select_dtypes(list(PANDAS_CATEGORICAL_SELECT_DTYPES)).columns)
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
        # Adaptive format -- 2 d.p. for typical magnitudes (MTV/MTTS for raw targets in the thousands), more decimals for tiny magnitudes (composite residual MTV ~ -1.17). The split-suffix carries the same tag prefix as train so composite shows ``MTRESID/MRV=...``, mirroring the existing BTTR/BTV / MTTR/MTV pattern.
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
    fi_kwargs: dict | None = None,
    subgroups: dict | None = None,
    custom_ice_metric=None,
    custom_rice_metric=None,
    details: str = "",
    has_other_splits: bool = False,
    n_features: int | None = None,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    title_metrics_tokens: tuple[str, ...] | None = None,
    plot_outputs: str | None = None,
    plot_dpi: int | None = None,
    binary_panels: str | None = None,
    multiclass_panels: str | None = None,
    multilabel_panels: str | None = None,
    ltr_panels: str | None = None,
    quantile_panels: str | None = None,
    quantile_alphas: tuple[float, ...] | None = None,
    target_type: str | None = None,
    y_train_envelope_stats: Any = None,
    reporting_config: Any = None,
    split_timestamps=None,
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

    # Feature selection (MRMR / RFECV) can remove every feature, leaving a 0-column frame. Calling model.predict on it
    # crashes deep in the backend (xgboost data.py "list index out of range" building a DMatrix from 0 columns). There is
    # nothing to predict from; skip exactly like the model-None case (the trainer already logged the empty-FS warning).
    if preds is None and probs is None and df is not None and hasattr(df, "shape") and len(getattr(df, "shape", ())) > 1 and df.shape[1] == 0:
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
    # plot_file ending in os.sep is a chart directory -> join keeps a clean `<dir>/val_perfplot.png`; otherwise it is a
    # filename prefix ending in the model-type name -> underscore-join keeps the per-model file flat
    # (`<prefix>_val_perfplot.png`), the layout chart-artifact consumers expect (each model/ensemble-method prefix is unique).
    if plot_file:
        if plot_file.endswith(os.sep):
            split_plot_file = os.path.join(plot_file, split_name)
        else:
            split_plot_file = f"{plot_file}_{split_name}"
        ensure_dir_exists(os.path.dirname(split_plot_file) + os.sep)
    else:
        split_plot_file = ""

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
        binary_panels=binary_panels,
        multiclass_panels=multiclass_panels,
        multilabel_panels=multilabel_panels,
        ltr_panels=ltr_panels,
        quantile_panels=quantile_panels,
        quantile_alphas=quantile_alphas,
        target_type=target_type,
        y_train_envelope_stats=y_train_envelope_stats,
        reporting_config=reporting_config,
    )

    _render_split_diagnostics(
        split_name=split_name,
        df=df_prepared,
        target=target,
        preds=preds,
        probs=probs,
        target_type=target_type,
        plot_outputs=plot_outputs,
        split_plot_file=split_plot_file,
        metrics_dict=metrics_dict,
        columns=columns,
        subgroups=subgroups,
        idx=idx,
        split_timestamps=split_timestamps,
        reporting_config=reporting_config,
    )
    return preds, probs, columns


def _render_split_diagnostics(
    *,
    split_name: str,
    df,
    target,
    preds,
    probs,
    target_type: str | None,
    plot_outputs: str | None,
    split_plot_file: str,
    metrics_dict: dict,
    columns,
    subgroups,
    idx,
    split_timestamps,
    reporting_config=None,
):
    """Render the per-split error-analysis + temporal-drift diagnostics default-ON and attach the worst-K table.

    The diagnostics render only when a feature frame + chart output path + DSL are present (the same render gate the
    rest of the suite honours). Failures are swallowed inside the orchestrator (additive panels never abort a run).
    Large-n safety lives in the orchestrator (worst-error-preserving subsample, column views, capped adversarial fit).
    """
    if df is None or not split_plot_file or not plot_outputs:
        return
    tt = (target_type or "").lower()
    # Regression vs classification gate; the error-analysis builders take "regression" / "classification".
    if "regress" in tt:
        task = "regression"
    elif tt in ("binary_classification", "multiclass_classification", "multilabel_classification"):
        task = "classification"
    else:
        # Unknown target_type: infer from probs presence (classification has probs).
        task = "classification" if probs is not None else "regression"

    # Multilabel / multiclass y is not 1-D pred-vs-actual error analysis material; skip those (their own panels cover them).
    y_arr = np.asarray(target)
    if y_arr.ndim != 1:
        return
    # The per-row error signal the builders score: hard predicted class by default, but for BINARY classification the
    # positive-class probability is a far richer worst-K / weak-segment signal (log-loss vs coarse 0/1 incorrectness),
    # so prefer probs[:, 1] when present. Regression / multiclass keep the point prediction.
    y_pred = np.asarray(preds).ravel() if preds is not None else None
    if task == "classification" and tt == "binary_classification" and probs is not None:
        probs_arr = np.asarray(probs)
        if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
            y_pred = probs_arr[:, 1]
        elif probs_arr.ndim == 1:
            y_pred = probs_arr
    if y_pred is None or np.ndim(y_pred) != 1 or len(y_pred) != len(y_arr):
        return
    y_pred = np.asarray(y_pred).ravel()

    feature_names = list(columns) if columns else None
    fi = None
    if isinstance(metrics_dict, dict):
        _fi = metrics_dict.get("feature_importances")
        if isinstance(_fi, dict) and feature_names:
            fi = [float(_fi.get(c, 0.0)) for c in feature_names]

    # Resolve subgroups to this split's row masks (subgroups are full-length boolean arrays keyed by name).
    split_subgroups = None
    if subgroups and idx is not None:
        try:
            split_subgroups = {k: np.asarray(v)[idx] for k, v in subgroups.items()}
        except Exception:
            split_subgroups = None

    split_ts = None
    if split_timestamps is not None and idx is not None:
        try:
            split_ts = np.asarray(split_timestamps)[idx]
        except Exception:
            split_ts = None

    from mlframe.reporting.diagnostics_dispatch import (
        render_split_error_diagnostics, render_target_drift_diagnostics,
    )

    res = render_split_error_diagnostics(
        df=df, y_true=y_arr, y_pred=y_pred, task=task,
        plot_outputs=plot_outputs, base_path=split_plot_file,
        metrics_dict=metrics_dict, feature_names=feature_names,
        feature_importances=fi, subgroups=split_subgroups,
        timestamps=split_ts,
    )
    if isinstance(metrics_dict, dict) and res.get("worst_k_table") is not None:
        metrics_dict["worst_k_table"] = res["worst_k_table"]
        metrics_dict["worst_k_indices"] = res["worst_k_indices"]

    # Temporal per-split panels (residual-vs-time / metric-over-time) only when timestamps cover this split.
    if split_ts is not None and task == "regression":
        # MSE-over-time for regression; roc_auc would need a binary target.
        render_target_drift_diagnostics(
            train_frame=None, test_frame=None, y_true=y_arr, y_pred=y_pred,
            timestamps=split_ts, task=task, plot_outputs=plot_outputs,
            base_path=split_plot_file, metrics_dict=metrics_dict,
            feature_names=feature_names, metric="mse",
            cusum_drift=getattr(reporting_config, "cusum_drift", True),
        )
    elif split_ts is not None and task == "classification" and probs is not None:
        probs_arr = np.asarray(probs)
        _score = probs_arr[:, 1] if (probs_arr.ndim == 2 and probs_arr.shape[1] == 2) else None
        if _score is not None and len(_score) == len(y_arr):
            render_target_drift_diagnostics(
                train_frame=None, test_frame=None, y_true=y_arr, y_pred=_score,
                timestamps=split_ts, task="classification", plot_outputs=plot_outputs,
                base_path=split_plot_file, metrics_dict=metrics_dict,
                feature_names=feature_names, metric="roc_auc",
            )


from ._confidence_analysis import run_confidence_analysis  # noqa: F401
