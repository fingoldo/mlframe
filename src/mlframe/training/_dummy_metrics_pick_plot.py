"""Metrics-table compute + strongest-pick + overlay-plot for ``dummy_baselines``.

Split out of ``dummy_baselines.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the moved entries so historical
``from mlframe.training.dummy_baselines import plot_best_dummy_baseline_overlay``
imports continue to resolve.

What lives here:
  - ``_compute_metrics_table`` (per-baseline metrics on val + test)
  - ``_pick_strongest`` (target-type-specific strongest-baseline selector)
  - ``plot_best_dummy_baseline_overlay`` (single-figure overlay for the
    strongest baseline)
  - ``_safe_metric_for_title`` (title helper)
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    roc_auc_score,
)

from ._dummy_baseline_compute import _safe_metric
from ._dummy_report_type import BaselineReport
# ``_has_signal`` lives in ``dummy_baselines.py``; imported lazily inside the
# function bodies that use it to dodge the
# ``dummy_baselines -> _dummy_metrics_pick_plot -> dummy_baselines`` import
# cycle.
from ._dummy_numba_kernels import _NUMBA_AVAILABLE
if _NUMBA_AVAILABLE:
    from ._dummy_numba_kernels import _numba_macro_log_loss, _numba_micro_log_loss

logger = logging.getLogger(__name__)

# iter431: bind math.isfinite for per-row scalar checks below. np.isfinite
# on a Python float pays the full numpy dispatcher (~1us / call) while
# math.isfinite is a C float-only check (~0.13us / call, 7.5x). The
# baseline-metrics loops iterate alphas/labels/baselines and aggregate
# across baselines (~50 scalar checks per fit); same dispatcher pattern
# that iter417 fixed in bootstrap.
_isfinite = math.isfinite


def _compute_metrics_table(
    target_type: str,
    val_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    group_ids_val: Any = None,
    group_ids_test: Any = None,
    extras: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Build the per-baseline x per-split metrics DataFrame (D1, D5)."""
    rows: list[dict[str, Any]] = []
    baseline_names = sorted(set(val_preds.keys()) | set(test_preds.keys()))

    if target_type == "quantile_regression" and (extras or {}).get("quantile_alphas"):
        # Per-alpha pinball-loss table. Predictions are 2D (N, K). Headline =
        # mean pinball over non-boundary alpha (alpha in [0.05, 0.95]).
        alphas = list(extras["quantile_alphas"])
        primary_metric = "val_pinball_mean"
        non_boundary_idx = [i for i, a in enumerate(alphas) if 0.05 <= a <= 0.95]
        for name in baseline_names:
            row: dict[str, Any] = {"baseline": name}
            vp = val_preds.get(name)
            tp = test_preds.get(name)
            for split_name, y, p in [("val", val_y, vp), ("test", test_y, tp)]:
                pinball_per_a: list[float] = []
                if p is not None and y is not None and len(p) == len(y) and p.ndim == 2 and p.shape[1] == len(alphas):
                    for j, a in enumerate(alphas):
                        v = _safe_metric(mean_pinball_loss, y, p[:, j], alpha=a)
                        row[f"{split_name}_pinball@{a:.3f}"] = v
                        if _isfinite(v):
                            pinball_per_a.append(v if j in non_boundary_idx else float("nan"))
                    if non_boundary_idx:
                        non_boundary_vals = [
                            row[f"{split_name}_pinball@{alphas[j]:.3f}"]
                            for j in non_boundary_idx
                            if _isfinite(row.get(f"{split_name}_pinball@{alphas[j]:.3f}", float("nan")))
                        ]
                        row[f"{split_name}_pinball_mean"] = (
                            float(np.mean(non_boundary_vals)) if non_boundary_vals else float("nan")
                        )
                    else:
                        row[f"{split_name}_pinball_mean"] = float("nan")
                else:
                    for a in alphas:
                        row[f"{split_name}_pinball@{a:.3f}"] = float("nan")
                    row[f"{split_name}_pinball_mean"] = float("nan")
            row["failed"] = not (
                _isfinite(row.get("val_pinball_mean", float("nan")))
                or _isfinite(row.get("test_pinball_mean", float("nan")))
            )
            rows.append(row)
    elif target_type in ("regression", "quantile_regression"):
        primary_metric = "val_RMSE"
        for name in baseline_names:
            row: dict[str, Any] = {"baseline": name}
            vp = val_preds.get(name)
            tp = test_preds.get(name)
            if vp is not None and val_y is not None and len(vp) == len(val_y):
                row["val_RMSE"] = _safe_metric(
                    lambda y, p: np.sqrt(mean_squared_error(y, p)), val_y, vp,
                )
                row["val_MAE"] = _safe_metric(mean_absolute_error, val_y, vp)
            else:
                row["val_RMSE"] = float("nan")
                row["val_MAE"] = float("nan")
            if tp is not None and test_y is not None and len(tp) == len(test_y):
                row["test_RMSE"] = _safe_metric(
                    lambda y, p: np.sqrt(mean_squared_error(y, p)), test_y, tp,
                )
                row["test_MAE"] = _safe_metric(mean_absolute_error, test_y, tp)
            else:
                row["test_RMSE"] = float("nan")
                row["test_MAE"] = float("nan")
            row["failed"] = not (
                _isfinite(row["val_RMSE"]) or _isfinite(row["test_RMSE"])
            )
            rows.append(row)

    elif target_type in ("binary_classification", "multiclass_classification"):
        # log_loss is headline; AUC secondary.
        primary_metric = "val_log_loss"
        n_classes = (extras or {}).get("n_classes", 2)
        labels = np.arange(n_classes)
        for name in baseline_names:
            row = {"baseline": name}
            vp = val_preds.get(name)
            tp = test_preds.get(name)
            for split_name, y, p in [("val", val_y, vp), ("test", test_y, tp)]:
                if p is not None and y is not None and len(p) == len(y) and p.ndim == 2:
                    row[f"{split_name}_log_loss"] = _safe_metric(
                        log_loss, y, p, labels=labels,
                    )
                    if target_type == "binary_classification":
                        row[f"{split_name}_AUC"] = _safe_metric(
                            roc_auc_score, y, p[:, 1],
                        )
                    else:
                        row[f"{split_name}_AUC_macro"] = _safe_metric(
                            roc_auc_score, y, p,
                            multi_class="ovr", average="macro", labels=labels,
                        )
                else:
                    row[f"{split_name}_log_loss"] = float("nan")
                    auc_key = f"{split_name}_AUC" if target_type == "binary_classification" else f"{split_name}_AUC_macro"
                    row[auc_key] = float("nan")
            row["failed"] = not (
                _isfinite(row.get("val_log_loss", float("nan")))
                or _isfinite(row.get("test_log_loss", float("nan")))
            )
            rows.append(row)

    elif target_type == "multilabel_classification":
        primary_metric = "val_log_loss_macro"
        for name in baseline_names:
            row = {"baseline": name}
            vp = val_preds.get(name)
            tp = test_preds.get(name)
            for split_name, y, p in [("val", val_y, vp), ("test", test_y, tp)]:
                if p is not None and y is not None and y.ndim == 2 and p.ndim == 2:
                    K = y.shape[1]
                    n = y.shape[0]
                    if _NUMBA_AVAILABLE and n > 0 and K > 0:
                        # Numba kernel: ~57x faster than per-label sklearn loop.
                        try:
                            y_int = np.ascontiguousarray(y, dtype=np.int64)
                            p_arr = np.ascontiguousarray(p, dtype=np.float64)
                            macro = float(_numba_macro_log_loss(y_int, p_arr, n, K))
                            micro = float(_numba_micro_log_loss(y_int, p_arr, n, K))
                        except (TypeError, ValueError, FloatingPointError, RuntimeError) as _ll_exc:
                            # Numba kernel rejects non-contiguous / wrong-dtype input
                            # with TypeError; ValueError on shape mismatch. Fall back
                            # to NaN so the metric row still emits.
                            logger.debug(
                                "[dummy-baselines] numba log_loss kernel fallback: %s: %s",
                                type(_ll_exc).__name__, _ll_exc,
                            )
                            macro = float("nan")
                            micro = float("nan")
                        row[f"{split_name}_log_loss_macro"] = macro
                        row[f"{split_name}_log_loss_micro"] = micro
                    else:
                        # Fallback: sklearn per-label loop.
                        label_lls: list[float] = []
                        for k in range(K):
                            if len(np.unique(y[:, k])) >= 2:
                                ll = _safe_metric(log_loss, y[:, k], p[:, k], labels=[0, 1])
                                if _isfinite(ll):
                                    label_lls.append(ll)
                        row[f"{split_name}_log_loss_macro"] = (
                            float(np.mean(label_lls)) if label_lls else float("nan")
                        )
                        row[f"{split_name}_log_loss_micro"] = _safe_metric(
                            log_loss, y.ravel(), p.ravel(), labels=[0, 1],
                        )
                else:
                    row[f"{split_name}_log_loss_macro"] = float("nan")
                    row[f"{split_name}_log_loss_micro"] = float("nan")
            row["failed"] = not (
                _isfinite(row["val_log_loss_macro"]) or _isfinite(row["test_log_loss_macro"])
            )
            rows.append(row)

    elif target_type == "learning_to_rank":
        primary_metric = "val_NDCG@10"
        from mlframe.metrics.ranking import compute_ranking_summary
        for name in baseline_names:
            row = {"baseline": name}
            for split_name, y, p, g in [
                ("val", val_y, val_preds.get(name), group_ids_val),
                ("test", test_y, test_preds.get(name), group_ids_test),
            ]:
                if p is not None and y is not None and g is not None and len(p) == len(y):
                    try:
                        summary = compute_ranking_summary(
                            np.asarray(y), np.asarray(p), np.asarray(g), eval_at=(1, 5, 10),
                        )
                        for k in (1, 5, 10):
                            row[f"{split_name}_NDCG@{k}"] = summary.get(f"ndcg@{k}", float("nan"))
                        row[f"{split_name}_MAP@10"] = summary.get("map@10", float("nan"))
                        row[f"{split_name}_MRR"] = summary.get("mrr", float("nan"))
                    except Exception:
                        for k in (1, 5, 10):
                            row[f"{split_name}_NDCG@{k}"] = float("nan")
                        row[f"{split_name}_MAP@10"] = float("nan")
                        row[f"{split_name}_MRR"] = float("nan")
                else:
                    for k in (1, 5, 10):
                        row[f"{split_name}_NDCG@{k}"] = float("nan")
                    row[f"{split_name}_MAP@10"] = float("nan")
                    row[f"{split_name}_MRR"] = float("nan")
            row["failed"] = not (
                _isfinite(row.get("val_NDCG@10", float("nan")))
                or _isfinite(row.get("test_NDCG@10", float("nan")))
            )
            rows.append(row)

    else:
        return pd.DataFrame(), ""

    table = pd.DataFrame(rows).set_index("baseline")
    return table, primary_metric


def _pick_strongest(
    target_type: str,
    table: pd.DataFrame,
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    primary_metric: str,
    extras: dict[str, Any],
    config: Any,
) -> tuple[str | None, int | None]:
    """Pick strongest baseline with non-degeneracy + paired-bootstrap gates (D2)."""
    if table.empty or not primary_metric:
        return None, None

    excluded = set(extras.get("strongest_pick_excluded", []))
    eligible = table.drop(index=[b for b in excluded if b in table.index], errors="ignore")
    if eligible.empty:
        return None, None

    from .dummy_baselines import _has_signal  # lazy: import-cycle, see module top
    # Non-degeneracy gate on reference split
    val_ok, val_reason = _has_signal(target_type, val_y) if val_y is not None else (False, "val=None")
    test_metric_name = primary_metric.replace("val_", "test_")

    if val_ok and primary_metric in eligible.columns:
        ref_metric = primary_metric
    elif test_metric_name in eligible.columns:
        test_ok, test_reason = _has_signal(target_type, test_y) if test_y is not None else (False, "test=None")
        if test_ok:
            ref_metric = test_metric_name
        else:
            logger.info(
                "[dummy-baselines] strongest=None (val: %s; test: %s)",
                val_reason, test_reason,
            )
            return None, None
    else:
        return None, None

    # Wave 20 fix: delegate metric direction to the central registry
    # dispatcher instead of an ad-hoc whitelist that missed common
    # higher-is-better classification metrics (val_AUC, val_F1,
    # val_accuracy, val_R2, val_AP, val_precision, val_recall) -- those
    # would otherwise pick the WORST baseline as strongest via idxmin.
    from .metrics_registry import metric_name_higher_is_better as _mhb
    _direction = _mhb(primary_metric)
    if _direction is None:
        # Unknown metric: WARN and default to "minimize" (the prior
        # behaviour for unknowns); the operator should register the
        # metric to get the right direction.
        logger.warning(
            "_pick_strongest: cannot determine direction for primary_metric=%r; "
            "defaulting to minimize=True. Register via metrics_registry to fix.",
            primary_metric,
        )
        minimize = True
    else:
        minimize = not _direction
    metric_col = eligible[ref_metric].dropna()
    if metric_col.empty:
        return None, None
    # Deterministic tiebreaker: when two baselines share the optimum metric
    # value, pick the alphabetically first name. ``idxmin/idxmax`` resolve
    # ties by first-occurrence in DataFrame order, which is sensitive to the
    # insertion order chosen by upstream dispatchers; alphabetical ordering
    # is reproducible across reruns and across dispatcher refactors.
    if minimize:
        best = float(metric_col.min())
        tied = sorted(metric_col.index[metric_col == best].tolist())
        strongest = tied[0] if tied else metric_col.idxmin()
    else:
        best = float(metric_col.max())
        tied = sorted(metric_col.index[metric_col == best].tolist())
        strongest = tied[0] if tied else metric_col.idxmax()

    # Determine ts_period if strongest is a TS baseline
    ts_period_used = None
    if "(ts" in str(strongest):
        m = re.search(r"_p(\d+)", str(strongest))
        if m:
            ts_period_used = int(m.group(1))
        elif "rolling_mean_w" in str(strongest):
            m2 = re.search(r"_w(\d+)", str(strongest))
            if m2:
                ts_period_used = int(m2.group(1))

    return strongest, ts_period_used


# ``_save_overlay_plot`` REMOVED. The
# standard ``report_regression_model_perf`` / ``report_probabilistic_model_perf``
# pipelines already render per-model scatter / residual / calibration
# charts; the dummy_baselines side rendering its own PNG was redundant
# noise on disk. The dummy_baselines TABLE (val/test metric grid +
# strongest verdict line + paired-bootstrap CI) remains the actionable
# artifact. To re-enable a baseline-overlay PNG in the future, the
# call site at ``compute_dummy_baselines`` should be the single place
# to add it back, gated behind a config flag (default off).


def plot_best_dummy_baseline_overlay(
    report: BaselineReport,
    *,
    val_y: np.ndarray | None = None,
    test_y: np.ndarray | None = None,
    save_path: str | None = None,
    show: bool = True,
    figsize: tuple[float, float] = (12, 4.5),
) -> Any | None:
    """Pre-training overlay for the strongest dummy baseline.

    Renders, in one figure, the visual floor your trained models will
    be measured against:

    * **Left** (regression / quantile): predictions-vs-actual scatter
      for val + test with the diagonal y=x reference.
    * **Right** (regression / quantile): residual histogram (val + test
      overlaid).
    * **Classification** falls back to a class-prior bar (one panel)
      since the canonical "scatter" is meaningless for class labels.

    The strongest baseline's val/test predictions are pulled from
    ``report.extras["strongest_val_preds"]`` / ``["strongest_test_preds"]``
    (populated by ``compute_dummy_baselines``).

    Renders inline in Jupyter via ``IPython.display`` (works regardless
    of matplotlib backend) and saves PNG to ``save_path`` if given.
    Returns the ``matplotlib.figure.Figure`` so the caller can take
    further action; returns ``None`` when the report has no plottable
    content (no strongest baseline, no val/test y, etc.).

    User-facing rationale: this chart fires BEFORE any model trains so
    you can eyeball the no-model floor before sinking compute into
    XGB/CB/LGB.
    """
    import matplotlib.pyplot as _plt
    if report.strongest is None:
        logger.info(
            "[dummy-baselines] target='%s' no strongest baseline -- "
            "overlay plot skipped.", report.target_name,
        )
        return None
    sv = report.extras.get("strongest_val_preds")
    st = report.extras.get("strongest_test_preds")
    if sv is None and st is None:
        logger.info(
            "[dummy-baselines] target='%s' no strongest preds in "
            "extras -- overlay plot skipped.", report.target_name,
        )
        return None
    is_regression = report.target_type in (
        "regression", "quantile_regression",
    )

    # Sample big arrays to keep render fast on millions of rows.
    rng = np.random.default_rng(0)
    plot_sample = 20_000

    def _maybe_subsample(y, p):
        if y is None or p is None:
            return None, None
        n = min(len(y), len(p))
        if n > plot_sample:
            idx = rng.choice(n, size=plot_sample, replace=False)
            return np.asarray(y)[idx], np.asarray(p)[idx]
        return np.asarray(y[:n]), np.asarray(p[:n])

    val_y_s, val_p_s = _maybe_subsample(val_y, sv)
    test_y_s, test_p_s = _maybe_subsample(test_y, st)

    if is_regression:
        fig, (ax_scatter, ax_resid) = _plt.subplots(
            1, 2, figsize=figsize, constrained_layout=True,
        )
        # Left: predictions vs actual scatter.
        lo, hi = None, None
        if val_y_s is not None:
            ax_scatter.scatter(
                val_y_s, val_p_s, s=4, alpha=0.35,
                color="tab:blue", label=f"val (n={len(val_y_s)})",
            )
            lo = float(np.nanmin(val_y_s))
            hi = float(np.nanmax(val_y_s))
        if test_y_s is not None:
            ax_scatter.scatter(
                test_y_s, test_p_s, s=4, alpha=0.35,
                color="tab:orange", label=f"test (n={len(test_y_s)})",
            )
            if lo is None:
                lo, hi = float(np.nanmin(test_y_s)), float(np.nanmax(test_y_s))
            else:
                lo = min(lo, float(np.nanmin(test_y_s)))
                hi = max(hi, float(np.nanmax(test_y_s)))
        if lo is not None and hi is not None:
            ax_scatter.plot(
                [lo, hi], [lo, hi], color="black",
                linestyle="--", linewidth=1.0, label="y = y_hat",
            )
        ax_scatter.set_xlabel("y_true")
        ax_scatter.set_ylabel("y_hat (strongest baseline)")
        ax_scatter.set_title(
            f"Baseline floor: {report.strongest}\n"
            f"({report.primary_metric}={_safe_metric_for_title(report)})"
        )
        ax_scatter.legend(loc="best", fontsize=9)
        ax_scatter.grid(alpha=0.3)

        # Right: residual histogram.
        for label, y_s, p_s, color in [
            ("val",  val_y_s,  val_p_s,  "tab:blue"),
            ("test", test_y_s, test_p_s, "tab:orange"),
        ]:
            if y_s is None or p_s is None:
                continue
            res = p_s - y_s
            res = res[np.isfinite(res)]
            if res.size == 0:
                continue
            ax_resid.hist(
                res, bins=40, alpha=0.5, color=color,
                label=f"{label} (n={res.size}, mean={res.mean():+.3g}, "
                      f"std={res.std():.3g})",
            )
        ax_resid.axvline(0, color="black", linestyle="--", linewidth=1.0)
        ax_resid.set_xlabel("y_hat - y_true (baseline residual)")
        ax_resid.set_ylabel("count")
        ax_resid.set_title("Baseline residuals (pre-training floor)")
        ax_resid.legend(loc="best", fontsize=9)
        ax_resid.grid(alpha=0.3)
    else:
        # Classification: single-panel bar of class-prior probabilities
        # for the strongest baseline (typically ``majority`` /
        # ``stratified_random``).
        fig, ax = _plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]),
                                constrained_layout=True)
        # Pull P(y=k) from the strongest baseline's val predictions.
        if sv is not None and sv.ndim == 1:
            # Single-label hard-prediction baseline (e.g. majority).
            uniq, counts = np.unique(sv, return_counts=True)
            ax.bar(range(len(uniq)),
                   counts / counts.sum(), color="tab:blue")
            ax.set_xticks(range(len(uniq)))
            ax.set_xticklabels([str(u) for u in uniq])
            ax.set_ylabel("P(predicted class)")
            ax.set_xlabel("class")
        elif sv is not None and sv.ndim == 2:
            # Probabilistic baseline (stratified_random_proba etc.).
            avg_p = sv.mean(axis=0)
            ax.bar(range(len(avg_p)), avg_p, color="tab:blue")
            ax.set_xticks(range(len(avg_p)))
            ax.set_ylabel("mean P(class) on val")
            ax.set_xlabel("class")
        ax.set_title(
            f"Baseline floor: {report.strongest}\n"
            f"({report.primary_metric}={_safe_metric_for_title(report)})"
        )
        ax.grid(axis="y", alpha=0.3)

    # Suptitle with target name.
    fig.suptitle(
        f"Dummy-baseline floor: target='{report.target_name}' "
        f"(target_type={report.target_type})",
        fontsize=11, y=1.02,
    )

    if save_path:
        try:
            fig.savefig(save_path, bbox_inches="tight")
            logger.info(
                "[dummy-baselines] target='%s' baseline-overlay plot "
                "saved: %s", report.target_name, save_path,
            )
        except Exception as _save_err:
            logger.warning(
                "[dummy-baselines] target='%s' baseline-overlay save "
                "failed: %s", report.target_name, _save_err,
            )

    # Inline display in Jupyter (mirrors the fix shipped in
    # feature_importance.plot_feature_importance -- use IPython.display
    # so the chart shows up even when the global matplotlib backend is
    # Agg).
    if show:
        try:
            _in_kernel = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            _in_kernel = False
        if _in_kernel:
            try:
                from IPython.display import display as _ipy_display
                _ipy_display(fig)
                # Close right after inline display: IPython has already
                # serialised the figure to the kernel display channel,
                # so the pyplot-registry reference is dead weight.
                # Leaving it alive causes the inline backend's end-of-
                # cell auto-flush to re-render the figure (the "толпа
                # графиков" double-render seen 2026-05-26).
                _plt.close(fig)
            except Exception:
                _plt.ion()
                _plt.show()
        else:
            # In a script/CI: close after save to avoid figure leak.
            _plt.close(fig)
    return fig


def _safe_metric_for_title(report: BaselineReport) -> str:
    """Pull the strongest baseline's primary metric value as a short
    string for the chart title. Falls back to '?' on lookup error."""
    try:
        col = report.primary_metric
        val = report.table.loc[report.strongest, col]
        if isinstance(val, float) and _isfinite(val):
            return f"{val:.4g}"
        return "?"
    except Exception:
        return "?"

