"""Helpers carved out of ``_reporting_probabilistic`` to keep that module under its size budget."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    from sklearn.metrics import classification_report
except ImportError:
    classification_report = None


from mlframe.metrics.core import compute_fairness_metrics, fast_roc_auc

from ..phases import phase

# Wave 97 (2026-05-21): _canonical_multilabel_y / _maybe_display + the
# DEFAULT_* constants all live in ``_reporting``; that module imports us
# from its bottom (after the helpers + constants are bound at module top),
# so by the time Python resolves these names ``_reporting`` is partially
# loaded and the symbols are already there. No circular-load failure,
# AND a single source of truth (no constant duplication across siblings).
from ._reporting import (
    _maybe_display,
    _style_with_caption,
)

if TYPE_CHECKING:
    pass

# Share the parent module's logger (logging.getLogger returns the same object for a given name, so this is
# identical to importing it) so the INFO-gate that guards the sklearn classification_report cost is controlled
# from one place. classification_report itself is still accessed through the module below so callers can swap
# the fallback at runtime.
from . import _reporting as _reporting_mod
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(_reporting_mod.__name__)

# Per-class keys that are not metrics: a p-value, a degrees-of-freedom count or a base rate has no meaningful mean.
_NOT_AGGREGATABLE = re.compile(r"(^|_)(p|pval|pvalue|p_value|dof|df)$|^base_rate$|^n_|support", re.IGNORECASE)


logger = logging.getLogger(_reporting_mod.__name__)


_NOT_AGGREGATABLE = re.compile(r"(^|_)(p|pval|pvalue|p_value|dof|df)$|^base_rate$|^n_|support", re.IGNORECASE)


def _aggregate_per_class_metrics(metrics: dict, per_class_blocks: list, supports: dict) -> None:
    """Write ``macro_<key>`` and, when real class supports exist, ``weighted_<key>`` for every per-class METRIC key.

    Only metric keys: a mean of p-values, degrees of freedom or base rates is not a statistic (a "macro_HL_p" is not a
    p-value), so ``_NOT_AGGREGATABLE`` keys stay per-class. ``weighted_*`` needs a support for every class; without
    them it used to be the macro mean under another name, so it is simply not emitted.
    """
    all_keys = set()
    for _, blk in per_class_blocks:
        for k, v in blk.items():
            if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and not _NOT_AGGREGATABLE.search(str(k)):
                all_keys.add(k)
    have_supports = len(supports) == len(per_class_blocks) and sum(supports.values()) > 0
    for key in all_keys:
        vals, wts = [], []
        for cid, blk in per_class_blocks:
            try:
                fv = float(blk.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                vals.append(fv)
                wts.append(supports.get(cid, 0))
        arr, w = np.asarray(vals, dtype=np.float64), np.asarray(wts, dtype=np.float64)
        metrics[f"macro_{key}"] = float(arr.mean()) if vals else float("nan")
        if have_supports:
            metrics[f"weighted_{key}"] = float((arr * w).sum() / w.sum()) if vals and w.sum() > 0 else float("nan")


def _tuned_threshold_kwargs(*args: Any) -> dict[str, float]:
    """``{"tuned_threshold": thr}`` when ``_resolve_f1_opt_threshold`` yields a threshold, else ``{}`` (same arguments)."""
    thr = _resolve_f1_opt_threshold(*args)
    return {} if thr is None else {"tuned_threshold": thr}


def _resolve_f1_opt_threshold(is_binary_positive: bool, y_true: Any, y_score: Any, given: float | None, tune_here: bool, metrics: Any) -> float | None:
    """F1-optimal decision threshold for the title's tuned block: the one handed in (the val-tuned one, for test), else tuned on this split when asked.

    A threshold tuned here is recorded in ``metrics['f1_opt_threshold']`` so the test report can reuse it unchanged -- the test number must
    not be optimised on test labels.
    """
    if not is_binary_positive:
        return None
    if given is not None:
        return float(given)
    if not tune_here:
        return None
    from mlframe.metrics.classification import optimal_threshold

    thr, _ = optimal_threshold(np.asarray(y_true), np.asarray(y_score), metric="f1")
    if not np.isfinite(thr):
        return None
    if isinstance(metrics, dict):
        metrics["f1_opt_threshold"] = float(thr)
    return float(thr)


def _report_probabilist_preds_none(preds, targets, probs, multilabel_dispatch_config, model):
    """Block of report_probabilistic_model_perf starting at ``if preds is None:``."""
    if preds is None:
        # Multilabel target -> (N, K) probs, threshold
        # each column independently; do NOT argmax (collapses to single class).
        # Also treat object-dtype-of-arrays as 2-D (the
        # ``pl.List`` -> pandas roundtrip). Without this, preds was
        # computed via argmax (1-D class index) while targets stayed
        # multilabel-indicator (2-D), and ``classification_report`` raised
        # ``mix of multilabel-indicator and multiclass targets``.
        _targets_2d = (
            (isinstance(targets, np.ndarray) and targets.ndim == 2)
            or (isinstance(targets, pd.DataFrame))
            or (
                isinstance(targets, np.ndarray)
                and targets.dtype == object
                and targets.ndim == 1
                and targets.shape[0] > 0
                and (hasattr(targets[0], "shape") or (hasattr(targets[0], "__len__") and not isinstance(targets[0], (str, bytes))))
            )
        )
        if _targets_2d:
            # MultiOutputClassifier returns list[(N,2)] for predict_proba -- canonicalize to (N, K).
            from mlframe.training.configs import TargetTypes
            from mlframe.training.helpers import _canonical_predict_proba_shape, _predict_from_probs
            probs = _canonical_predict_proba_shape(probs)
            # Honour MultilabelDispatchConfig.per_label_thresholds when
            # supplied: per-column decision threshold tuned for label
            # imbalance (defaults to 0.5 across all labels otherwise).
            # ``_predict_from_probs`` already broadcasts a scalar 0.5 vs
            # accepts a (K,) vector -- same downstream shape (N, K).
            _per_label_thr = (
                multilabel_dispatch_config.per_label_thresholds
                if (multilabel_dispatch_config is not None and multilabel_dispatch_config.per_label_thresholds is not None)
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
            from mlframe.utils.nan_safe import argmax_classes_safe
            preds = argmax_classes_safe(probs, context="_reporting.report_perf")
            if model is not None and hasattr(model, "classes_"):
                preds = model.classes_[preds]
    return preds, probs


def _report_probabilist_omits_only_row_logged(is_multilabel, probs, print_report, metrics, targets, preds, report_ndigits):
    """Block of report_probabilistic_model_perf starting at ``if not is_multilabel and probs is not None and getattr(probs, "ndim", ``."""
    if not is_multilabel and probs is not None and getattr(probs, "ndim", 0) == 2:
        _want_cls_log = print_report and logger.isEnabledFor(logging.INFO)
        if metrics is not None or _want_cls_log:
            try:
                from mlframe.training.configs import TargetTypes
                from mlframe.training.metrics_registry import iter_extra_metrics

                _cls_tt = TargetTypes.BINARY_CLASSIFICATION if probs.shape[1] == 2 else TargetTypes.MULTICLASS_CLASSIFICATION
                _cls_extra = list(iter_extra_metrics(_cls_tt, targets, probs, preds))
                _report_probabilist_metrics_none(metrics, _cls_extra)
                _report_probabilist_want_cls_log_cls(_want_cls_log, _cls_extra, report_ndigits)
            except (ImportError, AttributeError, ValueError, TypeError) as e:
                logger.debug("classification metrics registry skipped: %s", e)


def _report_probabilist_each_column_bit_identical(custom_ice_metric, targets, probs, _per_class_ice, integral_error):
    """Block of report_probabilistic_model_perf starting at ``if custom_ice_metric:``."""
    if custom_ice_metric:
        try:
            _full_res = custom_ice_metric(y_true=targets, y_score=probs, return_per_class=True)
        except TypeError:
            _full_res = custom_ice_metric(y_true=targets, y_score=probs)
        if isinstance(_full_res, tuple) and len(_full_res) == 2 and isinstance(_full_res[1], dict):
            integral_error, _per_class_ice = _full_res
        else:
            integral_error = _full_res
    return _per_class_ice, integral_error


def _report_probabilist_elements_case_function_own(classes, is_multilabel, targets_arr, model, targets, target_label_encoder):
    """Block of report_probabilistic_model_perf starting at ``if classes is None or len(classes) == 0:``."""
    if classes is None or len(classes) == 0:
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
    return classes


def _report_probabilist_pr_auc_alone_gpu(group_ids, classes, is_multilabel, targets_arr, probs, targets, _precomputed_aucs_per_class):
    """Block of report_probabilistic_model_perf starting at ``if group_ids is None and len(classes) >= 2:``."""
    if group_ids is None and len(classes) >= 2:
        try:
            from mlframe.metrics.core import compute_batch_aucs
            # Build (N, K) score matrix and (N, K)/(N,) label matrix once.
            if is_multilabel:
                _y_true_NK = targets_arr  # already (N, K) binary
                _y_score_NK = probs  # (N, K)
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
                _y_true_NK = np.column_stack([(targets == c).astype(np.int8) for c in classes])
                _y_score_NK = probs
            roc_batch, pr_batch = compute_batch_aucs(_y_true_NK, _y_score_NK)
            _precomputed_aucs_per_class = [(float(roc_batch[j]), float(pr_batch[j])) for j in range(_y_score_NK.shape[1])]
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
    return _precomputed_aucs_per_class


def _report_probabilist_dropping_new_fields(y_score, y_true, class_metrics, roc_auc, str_class_name):
    """Block of report_probabilistic_model_perf starting at ``try:``."""
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
            from mlframe.metrics.core import accuracy_ratio, hosmer_lemeshow_test
            hl_chi2, hl_p, hl_dof = hosmer_lemeshow_test(_y_true_arr, _y_score_arr, n_groups=10)
            class_metrics["HL_chi2"] = hl_chi2
            class_metrics["HL_p"] = hl_p
            class_metrics["HL_dof"] = hl_dof
            class_metrics["AccuracyRatio"] = accuracy_ratio(_y_true_arr, _y_score_arr)
        except (ValueError, TypeError) as _hl_err:
            logger.debug("Tier 2 calibration extras skipped: %s", _hl_err)
    except (ValueError, TypeError, FloatingPointError, ZeroDivisionError) as _ext_err:
        log_throttle(
            logger,
            "reporting_probabilistic_extended_metrics_failed",
            logging.WARNING,
            "extended classification metrics failed for class %s: %s. " "Continuing with the historical metric set only.",
            str_class_name,
            _ext_err,
        )


def _report_probabilist_collapses_per_class_value(metrics, is_multilabel, classes, targets):
    """Block of report_probabilistic_model_perf starting at ``if metrics is not None and is_multilabel is False and len(classes) > 2``."""
    if metrics is not None and is_multilabel is False and len(classes) > 2:
        _per_class_blocks = [(cid, metrics[cid]) for cid in metrics if isinstance(cid, (int, np.integer)) and isinstance(metrics[cid], dict)]
        if _per_class_blocks:
            # Class supports, with no int cast: string labels failed it and left weighted_* a silent copy of macro_*.
            _yt_all = np.asarray(targets) if targets is not None else None
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
            _aggregate_per_class_metrics(metrics, _per_class_blocks, _supports)


def _report_probabilist_metrics_none(metrics, _cls_extra):
    """Block of report_probabilistic_model_perf starting at ``if metrics is not None:``."""
    if metrics is not None:
        for _name, _val in _cls_extra:
            metrics[_name] = _val


def _report_probabilist_want_cls_log_cls(_want_cls_log, _cls_extra, report_ndigits):
    """Block of report_probabilistic_model_perf starting at ``if _want_cls_log and _cls_extra:``."""
    if _want_cls_log and _cls_extra:
        _cls_lines = ["CLASSIFICATION METRICS:"]
        for _name, _val in _cls_extra:
            try:
                _cls_lines.append(f"\t{_name}={_val:.{report_ndigits}f}")
            except (TypeError, ValueError):  # noqa: PERF203 - per-iteration fault isolation is intentional, not a hoisting candidate
                _cls_lines.append(f"\t{_name}={_val}")
        logger.info("\n".join(_cls_lines))


def _report_probabilist_identical_text_shape(is_multilabel, targets, preds, classes, true_classes, report_ndigits, _cls_report_text):
    """Block of report_probabilistic_model_perf starting at ``try:``."""
    try:
        from mlframe.metrics.core import format_classification_report
        _y_true = np.asarray(targets).astype(np.int64) if not is_multilabel else None
        _y_pred = np.asarray(preds).astype(np.int64) if not is_multilabel else None
        if _y_true is not None and _y_pred is not None and _y_true.ndim == 1 and _y_pred.ndim == 1 and len(_y_true) == len(_y_pred):
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
                    _nclasses = max(int(_y_true.max()) + 1, int(_y_pred.max()) + 1, 2) if len(_y_true) else 2
                    _cls_report_text = format_classification_report(
                        _y_true, _y_pred, nclasses=_nclasses, digits=report_ndigits, zero_division=0,
                    )
            else:
                _nclasses = max(int(_y_true.max()) + 1, int(_y_pred.max()) + 1, 2) if len(_y_true) else 2
                _cls_report_text = format_classification_report(
                    _y_true, _y_pred, nclasses=_nclasses, digits=report_ndigits, zero_division=0,
                )
        else:
            _cls_report_text = _reporting_mod.classification_report(targets, preds, zero_division=0, digits=report_ndigits)
    except (ValueError, TypeError, ImportError, AttributeError) as _cls_err:
        # Fall back to sklearn's classification_report when the njit-backed
        # path can't handle the input shape / dtype. Narrow catch leaves
        # programming bugs (KeyboardInterrupt, MemoryError) to propagate.
        logger.debug("fast classification_report fallback: %s", _cls_err)
        _cls_report_text = _reporting_mod.classification_report(targets, preds, zero_division=0, digits=report_ndigits)
    return _cls_report_text


def _report_probabilist_report_function_required(targets, probs, preds, report_ndigits):
    """Block of report_probabilistic_model_perf starting at ``try:``."""
    try:
        from mlframe.training.metrics_registry import iter_extra_metrics
        # Heuristic inference: multilabel if targets is 2-D binary.
        if hasattr(targets, "ndim") and targets.ndim == 2:
            from mlframe.training.configs import TargetTypes
            extra = list(iter_extra_metrics(
                TargetTypes.MULTILABEL_CLASSIFICATION, targets, probs, preds,
            ))
            if extra:
                _ml_lines = ["MULTILABEL METRICS:"]
                for name, val in extra:
                    try:
                        _ml_lines.append(f"\t{name}={val:.{report_ndigits}f}")
                    except (TypeError, ValueError):  # noqa: PERF203 - per-iteration fault isolation is intentional, not a hoisting candidate
                        # val is non-numeric (str / dict / etc.); format
                        # as-is rather than crashing the whole report.
                        _ml_lines.append(f"\t{name}={val}")
                logger.info("\n".join(_ml_lines))
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        # Narrow: import failures, missing attributes, sklearn metric input
        # rejection. Anything else (programming bug) propagates so it is
        # diagnosed at the call site instead of silently dropped.
        logger.debug("multilabel metrics registry skipped: %s", e)


def _report_probabilist_subgroups(subgroups, custom_ice_metric, probs, _pos_label, subset_index, targets, print_report, metrics, fairness_calibration_charts, plot_file, _y_true_pos_bin, plot_outputs):
    """Block of report_probabilistic_model_perf starting at ``if subgroups:``."""
    from ._reporting_probabilistic_calib import _render_fairness_calibration  # lazy: that module imports this package's facade

    if subgroups:
        subgroups_metrics = {"ICE": custom_ice_metric}
        metrics_higher_is_better = {"ICE": False}

        if probs.shape[1] == 2:

            def _fair_roc_auc(y_true, y_pred):
                """Per-subgroup ROC-AUC binarized against ``_pos_label``, extracting the positive-class column from a 2-D probability matrix."""
                y_bin = (np.asarray(y_true) == _pos_label).astype(np.int8)
                y_score = y_pred[:, 1] if getattr(y_pred, "ndim", 1) == 2 else y_pred
                return fast_roc_auc(y_bin, y_score)

            subgroups_metrics["ROC AUC"] = _fair_roc_auc
            metrics_higher_is_better["ROC AUC"] = True

        with phase("compute_fairness_metrics"):
            fairness_report = compute_fairness_metrics(
                subgroups=subgroups,
                subset_index=subset_index,  # compute_fairness_metrics handles None internally (see its subset_index is None branch)
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
                subgroups=subgroups, subset_index=subset_index, y_true=_y_true_pos_bin, pos_score=probs[:, 1],
                plot_file=plot_file, plot_outputs=plot_outputs, metrics=metrics,
            )


def _predict_probs_if_missing(probs, model, df):
    """Compute class probabilities from the model when the caller did not supply them."""
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
            probs = np.asarray(_predict_with_fallback(model, df, method="predict_proba"))
        except (AttributeError, TypeError, NotImplementedError):
            logger.warning("predict_proba not available for %s, using predict() instead", type(model).__name__, exc_info=True)
            preds_fallback = np.asarray(_predict_with_fallback(model, df, method="predict"))

            if model is not None and hasattr(model, "classes_"):
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
    return probs
