"""Stages of ``report_probabilistic_model_perf``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from mlframe.training.reporting._reporting_probabilistic import _report_probabilistic_step1_produced_single_most
from mlframe.training.reporting._reporting_probabilistic import _report_probabilistic_step2_omits_only_row
from mlframe.training.reporting._reporting_probabilistic_calib import _render_calibration_by_feature
from mlframe.training.reporting._reporting_probabilistic_calib import _render_calibration_heatmap_2d
from mlframe.training.reporting._reporting_probabilistic_helpers import _report_probabilist_collapses_per_class_value
from mlframe.training.reporting._reporting_probabilistic_helpers import _report_probabilist_omits_only_row_logged
from mlframe.training.reporting._reporting_probabilistic_helpers import _report_probabilist_subgroups
# --- end imports ---


def _report_probabilistic_plot_charts(probs, classes, st, targets, subgroups, custom_ice_metric, subset_index, print_report, metrics, fairness_calibration_charts, plot_file, plot_outputs, calibration_by_feature_charts, df, columns, model, calibration_heatmap_2d_charts):
    """Stage of ``report_probabilistic_model_perf``."""
    if probs is not None and probs.shape[1] == 2:
        st._pos_label = classes[1] if classes is not None and len(classes) > 1 else 1
        st._y_true_pos_bin = (np.asarray(targets) == st._pos_label).astype(np.int8)

    _report_probabilist_subgroups(
        subgroups,
        custom_ice_metric,
        probs,
        st._pos_label,
        subset_index,
        targets,
        print_report,
        metrics,
        fairness_calibration_charts,
        plot_file,
        st._y_true_pos_bin,
        plot_outputs,
    )

    # Per-feature calibration: a pooled reliability curve can hide miscalibration that varies with a continuous
    # feature (calibrated for low values, overconfident for high). Render reliability+ECE conditioned on the
    # top-importance feature(s) for binary targets. Default-ON when charts are saved AND a feature frame is present.
    if calibration_by_feature_charts and plot_file and probs is not None and probs.shape[1] == 2 and df is not None:
        _render_calibration_by_feature(
            df=df,
            columns=columns,
            model=model,
            y_true=st._y_true_pos_bin,
            pos_score=probs[:, 1],
            plot_file=plot_file,
            plot_outputs=plot_outputs,
            metrics=metrics,
        )

    # 2D calibration heatmap: a miscalibration pocket may surface only at a joint corner of the TOP-2 features (high f0
    # AND high f1) that either 1D per-feature view averages away. Render the ECE grid for the top-2-importance pair.
    if calibration_heatmap_2d_charts and plot_file and probs is not None and probs.shape[1] == 2 and df is not None:
        _render_calibration_heatmap_2d(
            df=df,
            columns=columns,
            model=model,
            y_true=st._y_true_pos_bin,
            pos_score=probs[:, 1],
            plot_file=plot_file,
            plot_outputs=plot_outputs,
            metrics=metrics,
        )


def _report_probabilistic_class_stages(st, classes, target_label_encoder, targets, probs, report_title, model_name, custom_ice_metric, n_features, columns, custom_rice_metric, calib_report_ndigits, plot_file, use_weights, nbins, group_ids, figsize, show_perf_chart, verbose, show_prob_histogram, prob_histogram_yscale, show_inline_population_labels, plot_dpi, reliability_smoothed, title_metrics_tokens, f1_opt_threshold, tune_f1_threshold, metrics, calibration_binning, reliability_show_ci, print_report, calibs, pr_aucs, report_ndigits, roc_aucs, brs, integral_errors, log_losses, robust_integral_errors, preds):
    """Stage of ``report_probabilistic_model_perf``."""
    st.true_classes = []
    _report_probabilistic_step1_produced_single_most(
        classes,
        target_label_encoder,
        st,
        targets,
        probs,
        report_title,
        model_name,
        custom_ice_metric,
        n_features,
        columns,
        custom_rice_metric,
        calib_report_ndigits,
        plot_file,
        use_weights,
        nbins,
        group_ids,
        figsize,
        show_perf_chart,
        verbose,
        show_prob_histogram,
        prob_histogram_yscale,
        show_inline_population_labels,
        plot_dpi,
        reliability_smoothed,
        title_metrics_tokens,
        f1_opt_threshold,
        tune_f1_threshold,
        metrics,
        calibration_binning,
        reliability_show_ci,
        print_report,
        calibs,
        pr_aucs,
        report_ndigits,
        roc_aucs,
        brs,
        integral_errors,
        log_losses,
        robust_integral_errors,
    )

    # audit batch: post-loop macro / weighted aggregation across
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
    _report_probabilist_collapses_per_class_value(metrics, st.is_multilabel, classes, targets)

    # Registered single-label classification scalars (quadratic_weighted_kappa / weighted_kappa /
    # exploss from metrics_registry). Mirrors the multilabel dispatch below, but lands the values in
    # the metrics dict so the suite reports them automatically for binary/multiclass targets. Each
    # metric is isolated by iter_extra_metrics' own narrow try/except, so a degenerate class slice
    # omits only that row (logged) rather than poisoning the report.
    _report_probabilist_omits_only_row_logged(st.is_multilabel, probs, print_report, metrics, targets, preds, report_ndigits)

    _report_probabilistic_step2_omits_only_row(
        print_report,
        st,
        targets,
        preds,
        classes,
        report_ndigits,
        report_title,
        model_name,
        roc_aucs,
        pr_aucs,
        calibs,
        brs,
        log_losses,
        integral_errors,
        custom_ice_metric,
        custom_rice_metric,
        robust_integral_errors,
        probs,
    )

    # Binary positive-class indicator (0/1) for the fairness + calibration-chart paths.
    # Those consumers (fast_roc_auc, ECE binning) assume y_true is a 0/1 indicator;
    # the raw ``targets`` may carry non-0/1 binary labels (e.g. {1,2} or strings),
    # which silently inverts / NaNs the AUC and corrupts ECE base rates. Map once to
    # the positive class (column 1 of probs == classes[1] by sklearn convention).
