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
from typing import TYPE_CHECKING, Any, Callable, Sequence

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
    classification_report = None

from pyutilz.pythonlib import get_human_readable_set_size
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from mlframe.metrics.core import fast_calibration_report

from ..phases import phase

# Wave 97 (2026-05-21): _canonical_multilabel_y / _maybe_display + the
# DEFAULT_* constants all live in ``_reporting``; that module imports us
# from its bottom (after the helpers + constants are bound at module top),
# so by the time Python resolves these names ``_reporting`` is partially
# loaded and the symbols are already there. No circular-load failure,
# AND a single source of truth (no constant duplication across siblings).
from ._reporting import (
    DEFAULT_CALIB_REPORT_NDIGITS,
    DEFAULT_FIGSIZE,
    DEFAULT_NBINS,
    DEFAULT_REPORT_NDIGITS,
    _canonical_multilabel_y,
    _labels_are_arange,
)

if TYPE_CHECKING:
    from ..configs import MultilabelDispatchConfig

# Share the parent module's logger (logging.getLogger returns the same object for a given name, so this is
# identical to importing it) so the INFO-gate that guards the sklearn classification_report cost is controlled
# from one place. classification_report itself is still accessed through the module below so callers can swap
# the fallback at runtime.

# Per-class keys that are not metrics: a p-value, a degrees-of-freedom count or a base rate has no meaningful mean.
from ._reporting_probabilistic_helpers import (  # noqa: F401  -- carved helpers
    logger,
    _NOT_AGGREGATABLE,
    _aggregate_per_class_metrics,
    _tuned_threshold_kwargs,
    _resolve_f1_opt_threshold,
    _report_probabilist_preds_none,
    _report_probabilist_omits_only_row_logged,
    _report_probabilist_each_column_bit_identical,
    _report_probabilist_elements_case_function_own,
    _report_probabilist_pr_auc_alone_gpu,
    _report_probabilist_dropping_new_fields,
    _report_probabilist_collapses_per_class_value,
    _report_probabilist_metrics_none,
    _report_probabilist_want_cls_log_cls,
    _report_probabilist_identical_text_shape,
    _report_probabilist_report_function_required,
    _report_probabilist_subgroups,
    _predict_probs_if_missing,
)


def _resolve_class_label(class_id: int, class_name: Any, target_label_encoder: Any) -> str:
    """Human-readable label for one report class.

    When the report class is a numeric stand-in and a label encoder is present, look the original label up by ENUMERATE POSITION (``class_id``) in the positionally-ordered ``classes_`` -- indexing by the raw integer label VALUE picks the wrong class (or raises IndexError) whenever labels are not a contiguous 0-based range.
    """
    if str(class_name).isnumeric() and target_label_encoder:
        return str(target_label_encoder.classes_[class_id])
    return str(class_name)


def _slugify_class(name: str) -> str:
    """Filesystem-safe class-name slug for per-class plot filenames."""
    try:
        from pyutilz.strings import slugify
        slug = slugify(name)
    except Exception as e:
        logger.debug("slugify() failed for name=%r: %s", name, e)
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
    classes: Sequence | np.ndarray | None = None,
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
    # Deliberately dual-keyed: per-class blocks are keyed by their int class_id, macro_*/weighted_*
    # aggregates by str metric name. The int/str key TYPE is the discriminator _per_class_blocks
    # below uses to pick out the per-class blocks from the aggregate scalars in the same dict.
    metrics: dict[int | str, Any] | None = None,
    group_ids: np.ndarray | None = None,
    n_features: int | None = None,
    show_prob_histogram: bool = False,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    title_metrics_tokens: tuple[str, ...] | None = None,
    multilabel_dispatch_config: MultilabelDispatchConfig | None = None,
    plot_dpi: int | None = None,
    calibration_binning: str | None = None,
    reliability_show_ci: bool | None = None,
    reliability_smoothed: bool = False,
    fairness_calibration_charts: bool = True,
    calibration_by_feature_charts: bool = True,
    calibration_heatmap_2d_charts: bool = True,
    f1_opt_threshold: float | None = None,
    tune_f1_threshold: bool = False,
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
    y_true: Any = None
    _pos_label: Any = None
    probs = _predict_probs_if_missing(probs, model, df)

    preds, probs = _report_probabilist_preds_none(preds, targets, probs, multilabel_dispatch_config, model)

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
    _per_class_ice, integral_error = _report_probabilist_each_column_bit_identical(custom_ice_metric, targets, probs, _per_class_ice, integral_error)
    robust_integral_error = None
    if custom_rice_metric and custom_rice_metric != custom_ice_metric:
        robust_integral_error = custom_rice_metric(y_true=targets, y_score=probs)

    # Explicit None/empty check instead of `if not classes:` -- the bare-truthiness form crashes with
    # "truth value of an array with more than one element is ambiguous" for any ndarray `classes` with
    # >=2 elements, a case the function's own type hint (Sequence | np.ndarray | None) documents as supported.
    classes = _report_probabilist_elements_case_function_own(classes, is_multilabel, targets_arr, model, targets, target_label_encoder)

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
    _precomputed_aucs_per_class = _report_probabilist_pr_auc_alone_gpu(group_ids, classes, is_multilabel, targets_arr, probs, targets, _precomputed_aucs_per_class)

    # DSL render spec for the reliability diagram. Default ON when the caller
    # supplies plot_outputs (e.g. "png,html" from ReportingConfig.plot_outputs);
    # routes every class's chart through build_calibration_spec so plotly HTML is
    # produced for the single most important classification chart, not just PNG.
    _plot_outputs_dsl = plot_outputs if plot_outputs else None

    true_classes = []
    for class_id, class_name in enumerate(classes):
        str_class_name = _resolve_class_label(class_id, class_name, target_label_encoder)
        true_classes.append(str_class_name)

        # Multilabel: never skip class_id=0; every column is an independent label.
        if not is_multilabel and len(classes) == 2 and class_id == 0:
            continue

        if is_multilabel:
            y_true = targets_arr[:, class_id]
        else:
            y_true = targets == class_name
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
        _fcr_kwargs: dict[str, Any] = dict(
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
        _fcr_kwargs.update(_tuned_threshold_kwargs(len(classes) == 2 and class_id == 1, y_true, y_score, f1_opt_threshold, tune_f1_threshold, metrics))
        # calibration binning strategy (auto/uniform/quantile) from ReportingConfig; default "auto" already picks
        # quantile under rare-event base rates. reliability_show_ci toggles the Wilson-CI band on the reliability
        # diagram and reaches the chart via fast_calibration_report -> build_calibration_spec(show_wilson_ci=...).
        if calibration_binning:
            _fcr_kwargs["binning_strategy"] = calibration_binning
        if reliability_show_ci is not None:
            _fcr_kwargs["reliability_show_ci"] = reliability_show_ci

        # Inject precomputed (roc, pr) for THIS class id when the batched GPU/CPU fastpath ran above; fast_calibration_report then skips its own
        # ``fast_aucs_per_group_optimized`` call. Multilabel and multiclass matrices have K columns indexed by class_id; the binary matrix has ONE
        # column (we only get here for class_id=1), indexed at 0.
        if _precomputed_aucs_per_class is not None:
            _fcr_kwargs["_precomputed_aucs"] = _precomputed_aucs_per_class[0 if (not is_multilabel and len(classes) == 2) else class_id]

        with phase("fast_calibration_report", class_id=str_class_name, n_rows=len(y_true)):
            (
                brier_loss, calibration_mae, calibration_std, calibration_coverage,
                ece, brier_reliability, brier_resolution, brier_uncertainty,
                roc_auc, pr_auc, ice, ll, precision, recall, f1,
                _metrics_string, _fig,
            ) = fast_calibration_report(**_fcr_kwargs)

        if print_report:
            # A partial-coverage calibration figure describes only the covered slice: a constant dummy baseline
            # reported "MAEW=0.00%, COV=10%", which reads as perfect calibration unless the caveat travels along.
            _cov_note = "" if calibration_coverage >= 0.999 else " (over the covered slice only)"
            calibs.append(
                f"\t{str_class_name}: MAE{'W' if use_weights else ''}={calibration_mae * 100:.{calib_report_ndigits}f}%, STD={calibration_std * 100:.{calib_report_ndigits}f}%, COV={calibration_coverage * 100:.0f}%{_cov_note}"
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
            _report_probabilist_dropping_new_fields(y_score, y_true, class_metrics, roc_auc, str_class_name)

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
    _report_probabilist_collapses_per_class_value(metrics, is_multilabel, classes, targets)

    # Registered single-label classification scalars (quadratic_weighted_kappa / weighted_kappa /
    # exploss from metrics_registry). Mirrors the multilabel dispatch below, but lands the values in
    # the metrics dict so the suite reports them automatically for binary/multiclass targets. Each
    # metric is isolated by iter_extra_metrics' own narrow try/except, so a degenerate class slice
    # omits only that row (logged) rather than poisoning the report.
    _report_probabilist_omits_only_row_logged(is_multilabel, probs, print_report, metrics, targets, preds, report_ndigits)

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
        _cls_report_text = _report_probabilist_identical_text_shape(is_multilabel, targets, preds, classes, true_classes, report_ndigits, _cls_report_text)
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
        if custom_ice_metric != custom_rice_metric and robust_integral_errors:  # no header over an empty body (no per-class RICE values)
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
        _report_probabilist_report_function_required(targets, probs, preds, report_ndigits)

    # Binary positive-class indicator (0/1) for the fairness + calibration-chart paths.
    # Those consumers (fast_roc_auc, ECE binning) assume y_true is a 0/1 indicator;
    # the raw ``targets`` may carry non-0/1 binary labels (e.g. {1,2} or strings),
    # which silently inverts / NaNs the AUC and corrupts ECE base rates. Map once to
    # the positive class (column 1 of probs == classes[1] by sklearn convention).
    _y_true_pos_bin = None
    if probs is not None and probs.shape[1] == 2:
        _pos_label = classes[1] if classes is not None and len(classes) > 1 else 1
        _y_true_pos_bin = (np.asarray(targets) == _pos_label).astype(np.int8)

    _report_probabilist_subgroups(subgroups, custom_ice_metric, probs, _pos_label, subset_index, targets, print_report, metrics, fairness_calibration_charts, plot_file, _y_true_pos_bin, plot_outputs)

    # Per-feature calibration: a pooled reliability curve can hide miscalibration that varies with a continuous
    # feature (calibrated for low values, overconfident for high). Render reliability+ECE conditioned on the
    # top-importance feature(s) for binary targets. Default-ON when charts are saved AND a feature frame is present.
    if calibration_by_feature_charts and plot_file and probs is not None and probs.shape[1] == 2 and df is not None:
        _render_calibration_by_feature(
            df=df, columns=columns, model=model, y_true=_y_true_pos_bin, pos_score=probs[:, 1],
            plot_file=plot_file, plot_outputs=plot_outputs, metrics=metrics,
        )

    # 2D calibration heatmap: a miscalibration pocket may surface only at a joint corner of the TOP-2 features (high f0
    # AND high f1) that either 1D per-feature view averages away. Render the ECE grid for the top-2-importance pair.
    if calibration_heatmap_2d_charts and plot_file and probs is not None and probs.shape[1] == 2 and df is not None:
        _render_calibration_heatmap_2d(
            df=df, columns=columns, model=model, y_true=_y_true_pos_bin, pos_score=probs[:, 1],
            plot_file=plot_file, plot_outputs=plot_outputs, metrics=metrics,
        )

    return preds, probs


# calibration/fairness render helpers carved to _reporting_probabilistic_calib.py (1k-LOC ceiling).
from ._reporting_probabilistic_calib import (  # noqa: F401
    _render_calibration_by_feature,
    _render_calibration_heatmap_2d,
    _render_fairness_calibration,
    _top_importance_features,
)
