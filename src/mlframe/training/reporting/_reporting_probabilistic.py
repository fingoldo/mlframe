"""report_probabilistic_model_perf -- moved out of _reporting.py.

Wave 97 (2026-05-21): the ~520-line ``report_probabilistic_model_perf``
function lives here so its parent module stays below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the symbol is
re-exported from ``_reporting`` so existing
``from mlframe.training._reporting import report_probabilistic_model_perf``
imports continue to work.

The function lazy-imports helpers from ``_reporting`` (``_canonical_multilabel_y``,
``_maybe_display``) inside the body to avoid the circular load with
that module's own top-level imports.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    from sklearn.metrics import classification_report
except ImportError:
    classification_report = None  # type: ignore[assignment]

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from mlframe.metrics.core import compute_fairness_metrics, fast_calibration_report, fast_roc_auc
from pyutilz.pythonlib import get_human_readable_set_size

from ..phases import phase
# Wave 97 (2026-05-21): _canonical_multilabel_y / _maybe_display + the
# DEFAULT_* constants all live in ``_reporting``; that module imports us
# from its bottom (after the helpers + constants are bound at module top),
# so by the time Python resolves these names ``_reporting`` is partially
# loaded and the symbols are already there. No circular-load failure,
# AND a single source of truth (no constant duplication across siblings).
from ._reporting import (  # noqa: E402
    _canonical_multilabel_y,
    _labels_are_arange,
    _maybe_display,
    _style_with_caption,
    DEFAULT_PLOT_SAMPLE_SIZE,
    DEFAULT_REPORT_NDIGITS,
    DEFAULT_CALIB_REPORT_NDIGITS,
    DEFAULT_NBINS,
    DEFAULT_FIGSIZE,
    DEFAULT_RANDOM_SEED,
)

if TYPE_CHECKING:
    from ..configs import MultilabelDispatchConfig

logger = logging.getLogger(__name__)


def _slugify_class(name: str) -> str:
    """Filesystem-safe class-name slug for per-class plot filenames."""
    try:
        from pyutilz.strings import slugify
        slug = slugify(name)
    except Exception:
        slug = ""
    if not slug:
        slug = "".join(ch if ch.isalnum() else "-" for ch in name).strip("-")
    return slug or "class"


def report_probabilistic_model_perf(
    targets: np.ndarray | pd.Series,
    columns: Sequence[str],
    model_name: str,
    model: ClassifierMixin | None,
    subgroups: dict[str, np.ndarray] | None = None,
    subset_index: np.ndarray | None = None,
    report_ndigits: int = DEFAULT_REPORT_NDIGITS,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    report_title: str = "",
    use_weights: bool = True,
    calib_report_ndigits: int = DEFAULT_CALIB_REPORT_NDIGITS,
    verbose: bool = False,
    classes: Sequence | None = None,
    preds: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    df: pd.DataFrame | None = None,
    target_label_encoder: LabelEncoder | None = None,
    nbins: int = DEFAULT_NBINS,
    print_report: bool = True,
    show_perf_chart: bool = True,
    plot_file: str = "",
    plot_outputs: str | None = None,
    custom_ice_metric: Callable | None = None,
    custom_rice_metric: Callable | None = None,
    metrics: dict[str, Any] | None = None,
    group_ids: np.ndarray | None = None,
    n_features: int | None = None,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    title_metrics_tokens: tuple[str, ...] | None = None,
    multilabel_dispatch_config: MultilabelDispatchConfig | None = None,
    plot_dpi: int | None = None,
    calibration_binning: str | None = None,
    reliability_show_ci: bool | None = None,
    reliability_smoothed: bool = True,
    fairness_calibration_charts: bool = True,
    calibration_by_feature_charts: bool = True,
    calibration_heatmap_2d_charts: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a detailed performance report for probabilistic classification models.

    Computes and displays classification metrics including ROC AUC, PR AUC,
    calibration metrics, Brier loss, log loss, and fairness analysis.

    Parameters
    ----------
    targets : np.ndarray or pd.Series
        True target labels.
    columns : Sequence[str]
        Feature column names.
    model_name : str
        Name of the model for display.
    model : ClassifierMixin or None
        Trained classification model. Can be None if probs are provided.
    subgroups : dict, optional
        Dictionary mapping subgroup names to boolean arrays for fairness analysis.
    subset_index : np.ndarray, optional
        Indices to subset the data for fairness analysis.
    report_ndigits : int, default=4
        Number of decimal digits for metric reporting.
    figsize : tuple, default=(15, 5)
        Figure size for plots.
    report_title : str, default=""
        Title prefix for reports.
    use_weights : bool, default=True
        Whether to use weighted calibration metrics.
    calib_report_ndigits : int, default=2
        Decimal digits for calibration metrics.
    verbose : bool, default=False
        Enable verbose output.
    classes : Sequence, optional
        Class labels. If None, inferred from model or targets.
    preds : np.ndarray, optional
        Pre-computed class predictions.
    probs : np.ndarray, optional
        Pre-computed class probabilities. If None, generated from model.
    df : pd.DataFrame, optional
        Feature DataFrame for generating predictions.
    target_label_encoder : LabelEncoder, optional
        Encoder for converting numeric labels to string names.
    nbins : int, default=10
        Number of bins for calibration analysis.
    print_report : bool, default=True
        Whether to print the report.
    show_perf_chart : bool, default=True
        Whether to display calibration and performance charts.
    plot_file : str, default=""
        Base path for saving plots.
    custom_ice_metric : Callable, optional
        Custom integral calibration error metric function.
    custom_rice_metric : Callable, optional
        Custom robust integral calibration error metric function.
    metrics : dict, optional
        Dictionary to store computed metrics (modified in-place).
    group_ids : np.ndarray, optional
        Group identifiers for grouped calibration analysis.

    Returns
    -------
    tuple
        (preds, probs) - class predictions and probability arrays.
    """
    if probs is None:
        # Lazy import avoids circular: trainer.py already imports from
        # evaluation.py at module level.
        from mlframe.training.trainer import _predict_with_fallback
        try:
            # _predict_with_fallback handles the CatBoost Polars-fastpath
            # dispatcher miss ("No matching signature found") symmetrically
            # with fit's fallback. Any OTHER error (model has no
            # predict_proba, returns NotImplemented, or a non-CB TypeError)
            # bubbles to the outer except and hits the predict() fallback
            # path below -- with the same Polars fallback wrapping so we
            # don't retry into the same dispatcher miss.
            probs = _predict_with_fallback(model, df, method="predict_proba")
        except (AttributeError, TypeError, NotImplementedError):
            logger.warning("predict_proba not available for %s, using predict() instead", type(model).__name__, exc_info=True)
            preds_fallback = _predict_with_fallback(model, df, method="predict")

            if hasattr(model, "classes_"):
                n_classes = len(model.classes_)
                # Wave 24 P2 fix (2026-05-20): pre-fix
                # ``np.searchsorted(classes_, preds_fallback)`` had two
                # latent bugs: (a) sort-contract on classes_ was assumed
                # but not asserted; (b) any preds_fallback value NOT in
                # classes_ returned index == n_classes which IndexError'd
                # on the subsequent ``probs[..., class_indices] = 1.0``.
                # Use a dict lookup with explicit fallback to the first
                # class for unseen predictions; WARN-log unseen counts.
                _class_to_idx = {c: i for i, c in enumerate(model.classes_)}
                _unseen = 0
                _class_indices_list = []
                for _p in preds_fallback:
                    if _p in _class_to_idx:
                        _class_indices_list.append(_class_to_idx[_p])
                    else:
                        _class_indices_list.append(0)
                        _unseen += 1
                class_indices = np.asarray(_class_indices_list, dtype=np.int64)
                if _unseen > 0:
                    logger.warning(
                        "report_perf: %d/%d predict() outputs were NOT in "
                        "model.classes_=%r; mapping them to class-0 for "
                        "the proba-fallback one-hot encoding. The model's "
                        "predict() returned values outside the training "
                        "label set -- check for a buggy estimator.",
                        _unseen, len(preds_fallback), list(model.classes_),
                    )
            else:
                n_classes = len(np.unique(preds_fallback))
                class_indices = preds_fallback.astype(int)

            probs = np.zeros((len(preds_fallback), n_classes))
            probs[np.arange(len(preds_fallback)), class_indices] = 1.0

    if preds is None:
        # Multilabel target -> (N, K) probs, threshold
        # each column independently; do NOT argmax (collapses to single class).
        # Also treat object-dtype-of-arrays as 2-D (the
        # ``pl.List`` -> pandas roundtrip). Without this, preds was
        # computed via argmax (1-D class index) while targets stayed
        # multilabel-indicator (2-D), and ``classification_report`` raised
        # ``mix of multilabel-indicator and multiclass targets``.
        _targets_2d = (
            isinstance(targets, np.ndarray) and targets.ndim == 2
        ) or (
            isinstance(targets, pd.DataFrame)
        ) or (
            isinstance(targets, np.ndarray)
            and targets.dtype == object
            and targets.ndim == 1
            and targets.shape[0] > 0
            and (hasattr(targets[0], "shape") or (
                hasattr(targets[0], "__len__")
                and not isinstance(targets[0], (str, bytes))
            ))
        )
        if _targets_2d:
            # MultiOutputClassifier returns list[(N,2)] for predict_proba -- canonicalize to (N, K).
            from ..helpers import _canonical_predict_proba_shape, _predict_from_probs
            from ..configs import TargetTypes
            probs = _canonical_predict_proba_shape(probs)
            # Honour MultilabelDispatchConfig.per_label_thresholds when
            # supplied: per-column decision threshold tuned for label
            # imbalance (defaults to 0.5 across all labels otherwise).
            # ``_predict_from_probs`` already broadcasts a scalar 0.5 vs
            # accepts a (K,) vector -- same downstream shape (N, K).
            _per_label_thr = (
                multilabel_dispatch_config.per_label_thresholds
                if (multilabel_dispatch_config is not None
                    and multilabel_dispatch_config.per_label_thresholds is not None)
                else 0.5
            )
            preds = _predict_from_probs(
                probs, TargetTypes.MULTILABEL_CLASSIFICATION, threshold=_per_label_thr,
            )
        elif probs.shape[1] == 2:
            # For binary classification, use threshold=0.5 on class 1 probability
            # This ensures consistency with calibration metrics in fast_calibration_report
            classes_ = model.classes_ if (model is not None and hasattr(model, "classes_")) else np.array([0, 1])
            preds = np.where(probs[:, 1] >= 0.5, classes_[1], classes_[0])
        else:
            # Wave 21 P2: nan-safe argmax.
            from ...utils.nan_safe import argmax_classes_safe
            preds = argmax_classes_safe(probs, context="_reporting.report_perf")
            if model is not None and hasattr(model, "classes_"):
                preds = model.classes_[preds]

    if isinstance(targets, pd.Series):
        targets = targets.values

    brs = []
    calibs = []
    pr_aucs = []
    roc_aucs = []
    integral_errors = []
    log_losses = []
    robust_integral_errors = []

    # Detect multilabel from 2-D target shape. Each
    # column is an independent binary label; the per-class loop below uses
    # the column directly instead of `targets == class_name` (which would
    # broadcast a 2-D bool against a 1-D y_score and crash).
    # Also detect object-dtype-of-arrays (the polars
    # ``pl.List(pl.Int8)`` -> pandas object roundtrip), stack to 2-D so
    # ``targets_arr[:, class_id]`` works in the multilabel branch.
    # Extracted to ``_canonical_multilabel_y`` helper so the
    # new ``mlframe.training.dummy_baselines`` module can reuse the same
    # canonicalization logic without duplication.
    targets_arr = _canonical_multilabel_y(targets)
    targets = targets_arr  # rebind so downstream uses the stacked form
    is_multilabel = targets_arr.ndim == 2

    # Single full-(N,K) ICE call. return_per_class=True surfaces the per-class ICE vector the
    # batched kernel already computes, so the per-class loop INDEXes it instead of recomputing
    # each 1-D column (bit-identical); a metric without the kwarg falls back to the scalar form.
    integral_error = 0.0
    _per_class_ice: dict | None = None
    if custom_ice_metric:
        try:
            _full_res = custom_ice_metric(y_true=targets, y_score=probs, return_per_class=True)
        except TypeError:
            _full_res = custom_ice_metric(y_true=targets, y_score=probs)
        if isinstance(_full_res, tuple) and len(_full_res) == 2 and isinstance(_full_res[1], dict):
            integral_error, _per_class_ice = _full_res
        else:
            integral_error = _full_res
    robust_integral_error = None
    if custom_rice_metric and custom_rice_metric != custom_ice_metric:
        robust_integral_error = custom_rice_metric(y_true=targets, y_score=probs)

    if not classes:
        if is_multilabel:
            # K independent labels named 0..K-1
            classes = list(range(targets_arr.shape[1]))
        elif model is not None:
            if hasattr(model, "classes_"):
                classes = model.classes_
            else:
                classes = np.unique(targets)
        elif target_label_encoder:
            classes = np.arange(len(target_label_encoder.classes_)).tolist()
        else:
            classes = np.unique(targets)

    if _per_class_ice is not None and not is_multilabel and not _labels_are_arange(classes, probs):
        _per_class_ice = None  # non-0-indexed labels -> kernel column index != class label

    # GPU batch-AUC fastpath: when the suite has many classes (multiclass /
    # multilabel) and the row count is large enough, compute all K
    # (roc_auc, pr_auc) pairs in ONE batched GPU call instead of K serial
    # ``fast_aucs_per_group_optimized`` calls inside the per-class loop.
    # Auto-dispatched by ``compute_batch_aucs``: GPU when cupy + CUDA
    # visible AND N >= threshold, otherwise CPU loop (no behavior change).
    # Only valid when group_ids is None (per-group AUCs need the full
    # function path). Empirical wins documented in ``bench_gpu_metrics.py``:
    # at N=1M K=20 PR AUC alone, GPU = 170 ms vs CPU loop = 2016 ms.
    _precomputed_aucs_per_class: list[tuple[float, float] | None] | None = None
    if group_ids is None and len(classes) >= 2:
        try:
            from mlframe.metrics.core import compute_batch_aucs
            # Build (N, K) score matrix and (N, K)/(N,) label matrix once.
            if is_multilabel:
                _y_true_NK = targets_arr  # already (N, K) binary
                _y_score_NK = probs       # (N, K)
            elif len(classes) == 2:
                # Binary: only class_id=1 is reported (loop skips id=0).
                # Single column, no batching benefit, but the dispatcher
                # auto-falls-back to CPU at small M anyway.
                _y_true_NK = (targets == classes[1]).astype(np.int8)[:, None]
                _y_score_NK = probs[:, [1]]
            else:
                # Multiclass: K columns, one-vs-rest.
                # bench-attempt-rejected (_benchmarks/bench_report_one_hot.py): an
                # arange-scatter one-hot loses here -- raw report labels need a
                # label->col map first; with it, scatter=0.35x col_stack at n=100k/K=5.
                _y_true_NK = np.column_stack(
                    [(targets == c).astype(np.int8) for c in classes]
                )
                _y_score_NK = probs
            roc_batch, pr_batch = compute_batch_aucs(_y_true_NK, _y_score_NK)
            _precomputed_aucs_per_class = [
                (float(roc_batch[j]), float(pr_batch[j])) for j in range(_y_score_NK.shape[1])
            ]
        except (KeyboardInterrupt, MemoryError, SystemExit):
            # Operator cancellation / true OOM MUST propagate -- the
            # previous ``except Exception`` swallowed KI, leaving the
            # suite running in a half-state with no way to interrupt it.
            raise
        except Exception as e:
            # Any other failure -> fall back to per-class fast_aucs path.
            # Broad-except is kept here because the fast path goes through
            # numba which raises numba.errors.TypingError on shape/dtype
            # mismatches, and that's a legitimate fall-back trigger; but
            # KI / MemoryError / SystemExit are re-raised above.
            logger.debug("compute_batch_aucs precompute failed (%s); using per-class path.", e)
            _precomputed_aucs_per_class = None

    # DSL render spec for the reliability diagram. Default ON when the caller
    # supplies plot_outputs (e.g. "png,html" from ReportingConfig.plot_outputs);
    # routes every class's chart through build_calibration_spec so plotly HTML is
    # produced for the single most important classification chart, not just PNG.
    _plot_outputs_dsl = plot_outputs if plot_outputs else None

    true_classes = []
    for class_id, class_name in enumerate(classes):
        if str(class_name).isnumeric() and target_label_encoder:
            str_class_name = str(target_label_encoder.classes_[class_name])
        else:
            str_class_name = str(class_name)
        true_classes.append(str_class_name)

        # Multilabel: never skip class_id=0; every column is an independent label.
        if not is_multilabel and len(classes) == 2 and class_id == 0:
            continue

        if is_multilabel:
            y_true = targets_arr[:, class_id]
        else:
            y_true = (targets == class_name)
        y_score = probs[:, class_id]
        if isinstance(y_true, pl.Series):
            y_true = y_true.to_numpy()

        title = report_title + " " + model_name
        if len(classes) != 2:
            title += "-" + str_class_name

        # Reuse the per-class ICE the batched kernel already produced in the single full-(N,K)
        # call (keyed by class_id); bit-identical. Recompute only when it's unavailable.
        if _per_class_ice is not None and class_id in _per_class_ice:
            class_integral_error = _per_class_ice[class_id]
        else:
            class_integral_error = custom_ice_metric(y_true=y_true, y_score=y_score) if custom_ice_metric else 0.0
        n_cols = n_features if n_features is not None else (len(columns) if columns is not None and len(columns) > 0 else 0)
        nfeatures = f"{n_cols:_}F/" if n_cols > 0 else ""
        title += f" [{nfeatures}{get_human_readable_set_size(len(y_true))} rows]"
        if custom_rice_metric and custom_rice_metric != custom_ice_metric:
            class_robust_integral_error = custom_rice_metric(y_true=y_true, y_score=y_score)
            title += f", RICE={class_robust_integral_error:.{calib_report_ndigits}f}"

        # Per-class plot path: every class gets a distinct filename. A bare
        # ``_perfplot.png`` was reused inside this loop so only the LAST class's
        # chart survived on disk for multiclass / multilabel runs. The class id
        # guarantees uniqueness even when two labels slugify to the same string;
        # the slug keeps the filename human-readable. ``base_path`` mirrors the
        # same per-class suffix so the DSL render path writes distinct files too.
        _class_perfplot = ""
        _class_base_path = ""
        if plot_file:
            _slug = _slugify_class(str_class_name)
            _suffix = f"_perfplot_c{class_id}_{_slug}" if len(classes) != 2 else "_perfplot"
            _class_perfplot = f"{plot_file}{_suffix}.png"
            _class_base_path = f"{plot_file}{_suffix}"

        # Build kwargs for fast_calibration_report. title_metrics_tokens is the
        # post-validation tuple from ReportingConfig - if None, the function's
        # own DEFAULT_TITLE_METRICS_TOKENS applies.
        _fcr_kwargs = dict(
            y_true=y_true,
            y_pred=y_score,
            use_weights=use_weights,
            nbins=nbins,
            group_ids=group_ids,
            title=title,
            figsize=figsize,
            # NOTE: plot_file and show_perf_chart are intentionally independent.
            # `plot_file` (derived from `data_dir`) controls whether plots are SAVED
            # to disk. `show_perf_chart` controls only interactive DISPLAY (plt.show).
            # Saving plots even when show_perf_chart=False is deliberate - users get
            # artifacts on disk without GUI popups. The Agg save-only fastpath in
            # show_calibration_plot handles this case without Qt overhead.
            plot_file=_class_perfplot,
            show_plots=show_perf_chart,
            ndigits=calib_report_ndigits,
            verbose=verbose,
            show_prob_histogram=show_prob_histogram,
            prob_histogram_yscale=prob_histogram_yscale,
            show_inline_population_labels=show_inline_population_labels,
            dpi=plot_dpi,
            # Smoothed isotonic reliability overlay: fast_calibration_report forwards the per-row (y_pred, y_true)
            # views it already holds as raw_probs/raw_labels, so the overlay is default-ON in suite reliability diagrams.
            reliability_smoothed=reliability_smoothed,
        )
        # Thread the DSL render path: when ReportingConfig.plot_outputs is set,
        # fast_calibration_report routes the reliability diagram through
        # build_calibration_spec (matplotlib PNG + plotly HTML + any future
        # backend) instead of the matplotlib-only legacy plotter.
        if _plot_outputs_dsl and _class_base_path:
            _fcr_kwargs["plot_outputs"] = _plot_outputs_dsl
            _fcr_kwargs["base_path"] = _class_base_path
        if title_metrics_tokens is not None:
            _fcr_kwargs["title_metrics_tokens"] = title_metrics_tokens
        # calibration binning strategy (auto/uniform/quantile) from ReportingConfig; default "auto" already picks
        # quantile under rare-event base rates. reliability_show_ci toggles the Wilson-CI band on the reliability
        # diagram and reaches the chart via fast_calibration_report -> build_calibration_spec(show_wilson_ci=...).
        if calibration_binning:
            _fcr_kwargs["binning_strategy"] = calibration_binning
        if reliability_show_ci is not None:
            _fcr_kwargs["reliability_show_ci"] = reliability_show_ci

        # Inject precomputed (roc, pr) for THIS class id when the batched
        # GPU/CPU fastpath ran above. fast_calibration_report skips its
        # internal ``fast_aucs_per_group_optimized`` call when this is set.
        if _precomputed_aucs_per_class is not None:
            # Index alignment:
            #  - multilabel: matrix has K columns, class_id 0..K-1
            #  - binary: matrix has 1 column; we get here only for class_id=1
            #  - multiclass: matrix has K columns, class_id 0..K-1
            if not is_multilabel and len(classes) == 2:
                # Single-column matrix indexed at 0
                _fcr_kwargs["_precomputed_aucs"] = _precomputed_aucs_per_class[0]
            else:
                _fcr_kwargs["_precomputed_aucs"] = _precomputed_aucs_per_class[class_id]

        with phase("fast_calibration_report", class_id=str_class_name, n_rows=len(y_true)):
            (
                brier_loss, calibration_mae, calibration_std, calibration_coverage,
                ece, brier_reliability, brier_resolution, brier_uncertainty,
                roc_auc, pr_auc, ice, ll, precision, recall, f1,
                metrics_string, _fig,
            ) = fast_calibration_report(**_fcr_kwargs)

        if print_report:
            calibs.append(
                f"\t{str_class_name}: MAE{'W' if use_weights else ''}={calibration_mae * 100:.{calib_report_ndigits}f}%, STD={calibration_std * 100:.{calib_report_ndigits}f}%, COV={calibration_coverage * 100:.0f}%"
            )
            pr_aucs.append(f"{str_class_name}={'N/A' if np.isnan(pr_auc) else f'{pr_auc:.{report_ndigits}f}'}")
            roc_aucs.append(f"{str_class_name}={'N/A' if np.isnan(roc_auc) else f'{roc_auc:.{report_ndigits}f}'}")
            brs.append(f"{str_class_name}={brier_loss * 100:.{report_ndigits}f}%")
            integral_errors.append(f"{str_class_name}={ice:.{report_ndigits}f}")
            if ll is None:
                log_losses.append(f"{str_class_name}=None")
            else:
                log_losses.append(f"{str_class_name}={ll:.{report_ndigits}f}")
            if custom_rice_metric and custom_rice_metric != custom_ice_metric:
                robust_integral_errors.append(f"{str_class_name}={class_robust_integral_error:.{report_ndigits}f}")

        if metrics is not None:
            class_metrics = dict(
                roc_auc=roc_auc,
                pr_auc=pr_auc,
                calibration_mae=calibration_mae,
                calibration_std=calibration_std,
                brier_loss=brier_loss,
                ece=ece,
                brier_reliability=brier_reliability,
                brier_resolution=brier_resolution,
                brier_uncertainty=brier_uncertainty,
                log_loss=ll,
                ice=ice,
                class_integral_error=class_integral_error,
                precision=precision,
                recall=recall,
                f1=f1,
            )
            if custom_rice_metric and custom_rice_metric != custom_ice_metric:
                class_metrics["class_robust_integral_error"] = class_robust_integral_error

            # 2026-05-28 audit: extend per-class metrics with the
            # confusion-derived and probability-derived blocks. Both
            # are fused single-pass kernels so the cost is dominated by
            # ONE walk over (y_true, y_pred) and ONE over (y_true, y_score).
            # Failures are narrow-catch and warn-loud rather than silently
            # dropping the new fields.
            try:
                from mlframe.metrics.core import (
                    fast_binary_confusion_metrics_block,
                    fast_binary_probability_metrics_block,
                    ks_statistic,
                    lift_at_k,
                )
                # Re-derive the hard prediction from y_score for this
                # class. The threshold of 0.5 matches the historical
                # ``fast_calibration_report`` convention used to compute
                # the existing precision/recall/f1 row above.
                _y_score_arr = np.asarray(y_score, dtype=np.float64)
                _y_pred_thr = (_y_score_arr >= 0.5).astype(np.int64)
                _y_true_arr = np.asarray(y_true).astype(np.int64, copy=False)

                _cm_block = fast_binary_confusion_metrics_block(_y_true_arr, _y_pred_thr)
                # Avoid name collision: keep historical precision/recall/f1
                # but stamp the rest of the confusion-derived block.
                for _k in (
                    "accuracy", "balanced_accuracy", "MCC", "Cohen_kappa",
                    "F0_5", "F2", "specificity", "NPV", "FPR", "FNR", "G_mean",
                ):
                    class_metrics[_k] = _cm_block[_k]

                _pb_block = fast_binary_probability_metrics_block(_y_true_arr, _y_score_arr)
                # Brier / log_loss already present above; new bits:
                class_metrics["base_rate"] = _pb_block["base_rate"]
                class_metrics["BSS"] = _pb_block["BSS"]
                class_metrics["Spiegelhalter_Z"] = _pb_block["Spiegelhalter_Z"]
                class_metrics["Spiegelhalter_p"] = _pb_block["Spiegelhalter_p"]

                # KS + Gini + Lift@10% are not in the confusion/probability
                # blocks (they need sorted scores OR a closed-form on AUC).
                class_metrics["KS"] = ks_statistic(_y_true_arr, _y_score_arr)
                if np.isfinite(roc_auc):
                    class_metrics["Gini"] = 2.0 * roc_auc - 1.0
                class_metrics["lift_at_10pct"] = lift_at_k(_y_true_arr, _y_score_arr, k_pct=10.0)

                # Tier 2 additions (2026-05-28): Hosmer-Lemeshow calibration
                # chi-square + Accuracy Ratio. HL adds an actionable p-value
                # for "model is miscalibrated" complementing Spiegelhalter Z;
                # AR is the credit-risk convention name for 2*AUC-1.
                try:
                    from mlframe.metrics.core import hosmer_lemeshow_test, accuracy_ratio
                    hl_chi2, hl_p, hl_dof = hosmer_lemeshow_test(_y_true_arr, _y_score_arr, n_groups=10)
                    class_metrics["HL_chi2"] = hl_chi2
                    class_metrics["HL_p"] = hl_p
                    class_metrics["HL_dof"] = hl_dof
                    class_metrics["AccuracyRatio"] = accuracy_ratio(_y_true_arr, _y_score_arr)
                except (ValueError, TypeError) as _hl_err:
                    logger.debug("Tier 2 calibration extras skipped: %s", _hl_err)
            except (ValueError, TypeError, FloatingPointError, ZeroDivisionError) as _ext_err:
                logger.warning(
                    "extended classification metrics failed for class %s: %s. "
                    "Continuing with the historical metric set only.",
                    str_class_name, _ext_err,
                )

            metrics.update({class_id: class_metrics})

    # 2026-05-28 audit batch: post-loop macro / weighted aggregation across
    # classes. The per-class loop above stamped each class's KS / MCC / F1 /
    # BSS / HL / AccuracyRatio / ROC_AUC / log_loss / ... but provided no
    # single scalar to compare two multiclass models. We compute:
    #   macro_<m>    = mean of class-m across classes (equal weight)
    #   weighted_<m> = mean weighted by class true-support (prevalence)
    # for every scalar emitted under per-class dicts. NaN-safe: a class
    # whose metric is NaN (e.g. AUC on a single-class slice) is dropped
    # from the macro mean and its support excluded from the weighted denom.
    # Skipped entirely on binary (single positive class, aggregation
    # collapses to the per-class value itself - no new information).
    if metrics is not None and is_multilabel is False and len(classes) > 2:
        _per_class_blocks = [
            (cid, metrics[cid]) for cid in metrics
            if isinstance(cid, (int, np.integer)) and isinstance(metrics[cid], dict)
        ]
        if _per_class_blocks:
            # Class supports for weighted-mean (true positives per class).
            try:
                _yt_all = np.asarray(targets).astype(np.int64, copy=False)
            except (TypeError, ValueError):
                _yt_all = None
            _supports = {}
            if _yt_all is not None and _yt_all.ndim == 1:
                for cid, _ in _per_class_blocks:
                    # ``cid`` is the per-class block's ENUMERATE position
                    # (0..K-1), but ``_yt_all`` holds the RAW target labels --
                    # which are not label-encoded to 0..K-1. Counting
                    # ``_yt_all == cid`` mis-weights any non-0-indexed integer
                    # multiclass target (e.g. labels [1, 2, 3]): supports shift
                    # by one and the highest label's count is lost, silently
                    # corrupting every weighted_* aggregate. Count against the
                    # actual class label at that position.
                    _label = classes[cid] if classes is not None and cid < len(classes) else cid
                    _supports[cid] = int(np.sum(_yt_all == _label))
            # Aggregate every numeric key shared across per-class blocks.
            _all_keys = set()
            for _, blk in _per_class_blocks:
                for k, v in blk.items():
                    if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                        _all_keys.add(k)
            for key in _all_keys:
                vals = []
                wts = []
                for cid, blk in _per_class_blocks:
                    v = blk.get(key)
                    if v is None:
                        continue
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(fv):
                        continue
                    vals.append(fv)
                    wts.append(_supports.get(cid, 1))
                if not vals:
                    metrics[f"macro_{key}"] = float("nan")
                    metrics[f"weighted_{key}"] = float("nan")
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                w = np.asarray(wts, dtype=np.float64)
                metrics[f"macro_{key}"] = float(arr.mean())
                w_total = w.sum()
                metrics[f"weighted_{key}"] = (
                    float((arr * w).sum() / w_total) if w_total > 0 else float(arr.mean())
                )

    if print_report and logger.isEnabledFor(logging.INFO):
        # Logger.isEnabledFor gate: when verbose=0 / file handler filters out
        # INFO, the multilabel branch below would still pay sklearn's
        # ``classification_report`` cost (~45 ms/call x 62 calls = 2.94s on
        # fuzz combo c0140) and then immediately drop the formatted text in
        # logger.info(). Skipping the whole block when no handler will accept
        # INFO recovers the full 2.94s on the multilabel-suite path.
        # Route through logger so file handlers (e.g.
        # pyutilz.logginglib.init_logging) capture the report block.
        # See sibling fix in report_regression_model_perf at line 659.
        # Replace sklearn's ``classification_report`` with the
        # njit-backed ``format_classification_report``. cProfile traced
        # ~90ms (55 %) of the warm-path
        # ``report_probabilistic_model_perf`` to sklearn's
        # ``precision_recall_fscore_support`` + ``multilabel_confusion_matrix``
        # path, which is overkill for single-label classification. The
        # njit version computes the same numbers in ~1ms warm and formats
        # to the identical text shape.
        _cls_report_text = ""
        try:
            from mlframe.metrics.core import format_classification_report
            _y_true = np.asarray(targets).astype(np.int64) if not is_multilabel else None
            _y_pred = np.asarray(preds).astype(np.int64) if not is_multilabel else None
            if (
                _y_true is not None and _y_pred is not None
                and _y_true.ndim == 1 and _y_pred.ndim == 1
                and len(_y_true) == len(_y_pred)
            ):
                # Remap raw integer labels to positions 0..K-1 against ``classes`` so the table carries one row PER class
                # in label order with correct macro/weighted averages. Inferring nclasses from ``max(label)+1`` instead
                # injects phantom 0-support rows for non-0-indexed labels (e.g. [1,2,3] -> a spurious class-0 row) which
                # silently drag the macro average below sklearn's. Any label outside ``classes`` -> sklearn fallback.
                _label_to_pos = {int(c): i for i, c in enumerate(classes)} if classes is not None else None
                if _label_to_pos is not None and len(_label_to_pos) == len(classes):
                    _pos_true = np.array([_label_to_pos.get(int(v), -1) for v in _y_true], dtype=np.int64)
                    _pos_pred = np.array([_label_to_pos.get(int(v), -1) for v in _y_pred], dtype=np.int64)
                    if _pos_true.min(initial=0) >= 0 and _pos_pred.min(initial=0) >= 0:
                        _names = [str(tc) for tc in true_classes] if len(true_classes) == len(classes) else None
                        _cls_report_text = format_classification_report(
                            _pos_true, _pos_pred, nclasses=len(classes), digits=report_ndigits,
                            zero_division=0, target_names=_names,
                        )
                    else:
                        _cls_report_text = classification_report(targets, preds, zero_division=0, digits=report_ndigits)
                else:
                    _nclasses = max(int(_y_true.max()) + 1, int(_y_pred.max()) + 1, 2) if len(_y_true) else 2
                    _cls_report_text = format_classification_report(
                        _y_true, _y_pred, nclasses=_nclasses, digits=report_ndigits, zero_division=0,
                    )
            else:
                _cls_report_text = classification_report(targets, preds, zero_division=0, digits=report_ndigits)
        except (ValueError, TypeError, ImportError, AttributeError) as _cls_err:
            # Fall back to sklearn's classification_report when the njit-backed
            # path can't handle the input shape / dtype. Narrow catch leaves
            # programming bugs (KeyboardInterrupt, MemoryError) to propagate.
            logger.debug("fast classification_report fallback: %s", _cls_err)
            _cls_report_text = classification_report(targets, preds, zero_division=0, digits=report_ndigits)
        _report_lines = [
            report_title + " " + model_name,
            _cls_report_text,
            f"ROC AUCs: {', '.join(roc_aucs)}",
            f"PR AUCs: {', '.join(pr_aucs)}",
            f"CALIBRATIONs: \n{', '.join(calibs)}",
            f"BRIER LOSSes: \n\t{', '.join(brs)}",
            f"LOG_LOSSes: \n\t{', '.join(log_losses)}",
            f"ICEs: \n\t{', '.join(integral_errors)}",
        ]
        if custom_ice_metric != custom_rice_metric:
            _report_lines.append(f"RICEs: \n\t{', '.join(robust_integral_errors)}")
        logger.info("\n".join(_report_lines))

        logger.info("TOTAL INTEGRAL ERROR: %.4f", integral_error)
        if robust_integral_error is not None:
            logger.info("TOTAL ROBUST INTEGRAL ERROR: %.4f", robust_integral_error)

        # Pluggable multi-output metrics registry.
        # Dispatches hamming_loss / subset_accuracy / jaccard_score_multilabel
        # (registered in mlframe.training.metrics_registry) when the
        # report-caller context indicates a multilabel target. Additional
        # metrics can be registered externally via
        # ``register_metric(target_type, name, fn)`` -- no code change to
        # this report function required.
        try:
            from ..metrics_registry import iter_extra_metrics
            # Heuristic inference: multilabel if targets is 2-D binary.
            if hasattr(targets, "ndim") and targets.ndim == 2:
                from ..configs import TargetTypes
                extra = list(iter_extra_metrics(
                    TargetTypes.MULTILABEL_CLASSIFICATION, targets, probs, preds,
                ))
                if extra:
                    _ml_lines = ["MULTILABEL METRICS:"]
                    for name, val in extra:
                        try:
                            _ml_lines.append(f"\t{name}={val:.{report_ndigits}f}")
                        except (TypeError, ValueError):
                            # val is non-numeric (str / dict / etc.); format
                            # as-is rather than crashing the whole report.
                            _ml_lines.append(f"\t{name}={val}")
                    logger.info("\n".join(_ml_lines))
        except (ImportError, AttributeError, ValueError, TypeError) as e:
            # Narrow: import failures, missing attributes, sklearn metric input
            # rejection. Anything else (programming bug) propagates so it is
            # diagnosed at the call site instead of silently dropped.
            logger.debug("multilabel metrics registry skipped: %s", e)

    if subgroups:
        subgroups_metrics = {"ICE": custom_ice_metric}
        metrics_higher_is_better = {"ICE": False}

        if probs.shape[1] == 2:
            subgroups_metrics["ROC AUC"] = fast_roc_auc
            metrics_higher_is_better["ROC AUC"] = True

        with phase("compute_fairness_metrics"):
            fairness_report = compute_fairness_metrics(
                subgroups=subgroups,
                subset_index=subset_index,
                y_true=targets,
                y_pred=probs,
                metrics=subgroups_metrics,
                metrics_higher_is_better=metrics_higher_is_better,
            )
        if fairness_report is not None:
            if print_report:
                _maybe_display(_style_with_caption(fairness_report, "ML perf fairness by group"))
            if metrics is not None:
                metrics.update(dict(fairness_report=fairness_report))

        # Per-subgroup reliability + ECE: equal accuracy across groups does not imply equal calibration, so render a
        # calibration-fairness figure per group feature for binary targets. Default-ON when charts are saved AND a
        # plot DSL is active; a no-op for multiclass / when no plot dir is configured.
        if fairness_calibration_charts and plot_file and probs.shape[1] == 2:
            _render_fairness_calibration(
                subgroups=subgroups, subset_index=subset_index, y_true=targets, pos_score=probs[:, 1],
                plot_file=plot_file, plot_outputs=plot_outputs, metrics=metrics,
            )

    # Per-feature calibration: a pooled reliability curve can hide miscalibration that varies with a continuous
    # feature (calibrated for low values, overconfident for high). Render reliability+ECE conditioned on the
    # top-importance feature(s) for binary targets. Default-ON when charts are saved AND a feature frame is present.
    if calibration_by_feature_charts and plot_file and probs is not None and probs.shape[1] == 2 and df is not None:
        _render_calibration_by_feature(
            df=df, columns=columns, model=model, y_true=targets, pos_score=probs[:, 1],
            plot_file=plot_file, plot_outputs=plot_outputs, metrics=metrics,
        )

    # 2D calibration heatmap: a miscalibration pocket may surface only at a joint corner of the TOP-2 features (high f0
    # AND high f1) that either 1D per-feature view averages away. Render the ECE grid for the top-2-importance pair.
    if calibration_heatmap_2d_charts and plot_file and probs is not None and probs.shape[1] == 2 and df is not None:
        _render_calibration_heatmap_2d(
            df=df, columns=columns, model=model, y_true=targets, pos_score=probs[:, 1],
            plot_file=plot_file, plot_outputs=plot_outputs, metrics=metrics,
        )

    return preds, probs


def _render_fairness_calibration(
    *,
    subgroups: dict[str, Any],
    subset_index: np.ndarray | None,
    y_true: np.ndarray | pd.Series,
    pos_score: np.ndarray,
    plot_file: str,
    plot_outputs: str | None,
    metrics: dict[str, Any] | None,
) -> None:
    """Render a per-subgroup calibration-fairness figure per group feature (binary positive-class score).

    For each fairness group feature, slice the per-row group labels for THIS split (``bins.loc[subset_index]``) and
    compose the per-group reliability overlay + per-group ECE bar + max-min disparity gap. Failures are logged and
    skipped per feature so one bad group feature never aborts the report.
    """
    from mlframe.reporting.charts.fairness_calibration import (
        compose_fairness_calibration_figure, compute_subgroup_ece_disparity,
    )

    yt = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    _dsl = plot_outputs if plot_outputs else "matplotlib[png]"
    disparities: dict[str, Any] = {}

    for group_name, group_params in subgroups.items():
        if group_name in ("**ORDER**", "**RANDOM**"):
            continue  # robustness pseudo-groups, not sensitive features
        try:
            bins = group_params.get("bins") if isinstance(group_params, dict) else None
            if bins is None:
                continue
            if subset_index is not None and hasattr(bins, "loc"):
                bins = bins.loc[subset_index]
            labels = bins.to_numpy() if hasattr(bins, "to_numpy") else np.asarray(bins)
            if labels.shape[0] != yt.shape[0]:
                continue
            disparity = compute_subgroup_ece_disparity(yt, pos_score, labels)
            disparities[group_name] = disparity
            spec = compose_fairness_calibration_figure(
                yt, pos_score, labels, title=f"Calibration fairness by {group_name}",
            )
            _slug = _slugify_class(str(group_name))
            base_path = f"{plot_file}_faircal_{_slug}"
            from mlframe.reporting.output import parse_plot_output_dsl
            from mlframe.reporting.renderers import render_and_save
            render_and_save(spec, parse_plot_output_dsl(_dsl), base_path)
        except Exception as e:
            logger.debug("fairness_calibration chart for %r skipped: %s", group_name, e)

    if metrics is not None and disparities:
        metrics.update(dict(fairness_calibration_disparity=disparities))


def _top_importance_features(model: Any, columns: Sequence[str], top_k: int) -> list[str]:
    """Top-k feature names by the model's ``feature_importances_``, aligned to ``columns``. Empty when unavailable."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    imp = np.asarray(imp, dtype=np.float64).ravel()
    cols = list(columns)
    if imp.shape[0] != len(cols) or not np.isfinite(imp).any():
        return []
    order = np.argsort(imp)[::-1][:top_k]
    return [cols[i] for i in order]


def _render_calibration_by_feature(
    *,
    df: pd.DataFrame,
    columns: Sequence[str],
    model: Any,
    y_true: np.ndarray | pd.Series,
    pos_score: np.ndarray,
    plot_file: str,
    plot_outputs: str | None,
    metrics: dict[str, Any] | None,
) -> None:
    """Render per-feature calibration (reliability + ECE by feature quantile bin) for the top-importance features.

    The top-1..2 features by ``model.feature_importances_`` are pulled from ``df`` as continuous columns; each yields
    a small-multiples reliability figure + an ECE-vs-feature-bin line + the max-min "calibration heterogeneity"
    metric. Non-numeric / missing columns and degenerate features are skipped per feature so one bad column never
    aborts the report.
    """
    from mlframe.reporting.charts.calibration_by_feature import (
        compose_calibration_by_feature_figure, compute_calibration_by_feature_heterogeneity,
    )

    feats = _top_importance_features(model, columns, top_k=2)
    if not feats or not hasattr(df, "__getitem__"):
        return
    yt = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    _dsl = plot_outputs if plot_outputs else "matplotlib[png]"
    heterogeneity: dict[str, Any] = {}

    for fname in feats:
        try:
            col = df[fname]
            fv = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            fv = np.asarray(fv, dtype=np.float64).ravel()  # NaN for non-numeric; dropped downstream
            if fv.shape[0] != yt.shape[0]:
                continue
            het = compute_calibration_by_feature_heterogeneity(yt, pos_score, fv)
            heterogeneity[str(fname)] = het
            spec = compose_calibration_by_feature_figure(yt, pos_score, fv, feature_name=str(fname))
            _slug = _slugify_class(str(fname))
            base_path = f"{plot_file}_calibfeat_{_slug}"
            from mlframe.reporting.output import parse_plot_output_dsl
            from mlframe.reporting.renderers import render_and_save
            render_and_save(spec, parse_plot_output_dsl(_dsl), base_path)
        except Exception as e:
            logger.debug("calibration_by_feature chart for %r skipped: %s", fname, e)

    if metrics is not None and heterogeneity:
        metrics.update(dict(calibration_by_feature_heterogeneity=heterogeneity))


def _render_calibration_heatmap_2d(
    *,
    df: pd.DataFrame,
    columns: Sequence[str],
    model: Any,
    y_true: np.ndarray | pd.Series,
    pos_score: np.ndarray,
    plot_file: str,
    plot_outputs: str | None,
    metrics: dict[str, Any] | None,
) -> None:
    """Render a 2D calibration-ECE heatmap over the quantile grid of the top-2-importance feature pair (binary target).

    Both features are quantile-binned; per cell ``|mean(score) - mean(true)|`` is shown on an RdYlGn_r grid, with the
    worst cell + its location as the headline -- a localized over/under-confidence pocket the pooled / 1D views hide.
    Needs >=2 distinct top-importance features; degenerate columns are skipped without aborting the report.
    """
    from mlframe.reporting.charts.calibration_heatmap_2d import (
        compose_calibration_heatmap_2d_figure, compute_calibration_heatmap_2d,
    )

    feats = _top_importance_features(model, columns, top_k=2)
    if len(feats) < 2 or not hasattr(df, "__getitem__"):
        return
    fx_name, fy_name = feats[0], feats[1]
    yt = y_true.to_numpy() if hasattr(y_true, "to_numpy") else np.asarray(y_true)
    try:
        cx, cy = df[fx_name], df[fy_name]
        fx = np.asarray(cx.to_numpy() if hasattr(cx, "to_numpy") else cx, dtype=np.float64).ravel()
        fy = np.asarray(cy.to_numpy() if hasattr(cy, "to_numpy") else cy, dtype=np.float64).ravel()
        if fx.shape[0] != yt.shape[0] or fy.shape[0] != yt.shape[0]:
            return
        res = compute_calibration_heatmap_2d(yt, pos_score, fx, fy)
        if metrics is not None and res.get("worst_cell") is not None:
            metrics.update(dict(calibration_heatmap_2d={
                "worst_ece": res["worst_ece"], "worst_cell": res["worst_cell"],
                "median_cell_ece": res["median_cell_ece"], "traffic_light": res["traffic_light"],
                "feat_x": str(fx_name), "feat_y": str(fy_name),
            }))
        spec = compose_calibration_heatmap_2d_figure(
            yt, pos_score, fx, fy, feat_x_name=str(fx_name), feat_y_name=str(fy_name),
        )
        _dsl = plot_outputs if plot_outputs else "matplotlib[png]"
        base_path = f"{plot_file}_calib2d_{_slugify_class(str(fx_name))}_{_slugify_class(str(fy_name))}"
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
        render_and_save(spec, parse_plot_output_dsl(_dsl), base_path)
    except Exception as e:
        logger.debug("calibration_heatmap_2d chart skipped: %s", e)


