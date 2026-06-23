"""Classification + calibration reports + ICE batch kernel for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import format_classification_report`` (and
the other moved names) imports continue to work.

What lives here:
  - ``format_classification_report``
  - ``_compute_pr_recall_f1_metrics_seq`` / ``_par`` / ``compute_pr_recall_f1_metrics``
  - ``fast_calibration_report`` (heavy calibration / ICE report)
  - ``_batch_per_class_ice_kernel`` (njit kernel)
  - ``fast_ice_only``
  - ``predictions_time_instability``
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import numba

from .._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD  # noqa: F401
from ..calibration._calibration_plot import (  # noqa: F401
    DEFAULT_TITLE_METRICS_TOKENS,
    calibration_binning,
    fast_calibration_binning,
    render_title_metric_token,
    show_calibration_plot,
)
from .._auc_per_group import (  # noqa: F401
    fast_aucs_per_group_optimized,
    compute_mean_aucs_per_group,
)
from ..calibration._calibration_metrics import (  # noqa: F401
    calibration_metrics_from_freqs,
    compute_brier_decomposition_debiased,
    compute_ece_and_brier_decomposition,
    compute_ece_brier_full_and_debiased,
    compute_ece_debiased,
    integral_calibration_error_from_metrics,
)
from .._log_loss_and_separation import fast_log_loss  # noqa: F401
# ``fast_brier_score_loss`` and ``fast_classification_report`` still live in
# core.py; we import lazily inside the function bodies to dodge the
# core <-> _classification_report import cycle that the eager form would
# trigger (core imports from this module to re-export, so a top-level
# ``from .core import ...`` here would deadlock during module init).

logger = logging.getLogger(__name__)


def format_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nclasses: int = 2,
    digits: int = 4,
    target_names: Optional[Sequence] = None,
    zero_division: int = 0,
) -> str:
    """Drop-in replacement for ``sklearn.metrics.classification_report``.

    Computes precision / recall / f1 / support per class plus accuracy /
    macro-avg / weighted-avg via the @njit ``fast_classification_report``
    kernel and formats the result as the same fixed-width text block
    sklearn produces. Used by ``evaluation.py`` instead of sklearn's
    Python-side ``precision_recall_fscore_support`` + multilabel
    confusion-matrix machinery, which dominated 90 ms of every
    ``report_probabilistic_model_perf`` warm call (cProfile of fuzz
    combo c0014: 4 calls * 22ms each, 55 % of the warm 164ms suite cost
    after the GPU-probe cache landed).

    The numerics match sklearn's ``classification_report`` for the
    common single-label classification path; weighted/macro avg
    formulas mirror sklearn's exactly. The helper drops support for
    sklearn's multilabel-indicator input (use sklearn directly for that)
    and ``output_dict=True`` (use ``fast_classification_report`` for the
    raw arrays).

    MACRO-AVG CAVEAT (matches sklearn): macro avg is the UNWEIGHTED mean of the per-class
    precision / recall / f1 over ALL ``nclasses``, INCLUDING any class with support 0 in
    ``y_true``. An absent class contributes its ``zero_division`` value (0 by default) to the
    macro mean, which DEFLATES macro precision/recall/f1 on rare-event / sparse-label targets
    where some classes never appear in this split. If you want the mean over only the classes
    actually present, filter ``target_names`` to the observed labels before averaging, or read the
    per-class rows directly. (weighted avg is support-weighted, so absent classes get weight 0 and
    do not move it.)
    """
    from ..core import fast_classification_report  # lazy: see import-cycle note at module top
    hits, misses, accuracy, balanced_accuracy, supports, precisions, recalls, f1s, macro_averages, weighted_averages = (
        fast_classification_report(y_true, y_pred, nclasses=nclasses, zero_division=zero_division)
    )
    if target_names is None:
        target_names = [str(i) for i in range(nclasses)]

    n_total = int(supports.sum())
    label_width = max(len("weighted avg"), max((len(str(t)) for t in target_names), default=1))
    head = " " * (label_width + 2)
    head += f"{'precision':>{digits + 5}} {'recall':>{digits + 5}} {'f1-score':>{digits + 5}} {'support':>{digits + 6}}"
    lines = [head, ""]
    for i, name in enumerate(target_names):
        lines.append(
            f"{str(name):>{label_width}}  "
            f"{precisions[i]:>{digits + 5}.{digits}f} "
            f"{recalls[i]:>{digits + 5}.{digits}f} "
            f"{f1s[i]:>{digits + 5}.{digits}f} "
            f"{int(supports[i]):>{digits + 6}}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':>{label_width}}  "
        f"{'':>{digits + 5}} "
        f"{'':>{digits + 5}} "
        f"{accuracy:>{digits + 5}.{digits}f} "
        f"{n_total:>{digits + 6}}"
    )
    lines.append(
        f"{'macro avg':>{label_width}}  "
        f"{macro_averages[0]:>{digits + 5}.{digits}f} "
        f"{macro_averages[1]:>{digits + 5}.{digits}f} "
        f"{macro_averages[2]:>{digits + 5}.{digits}f} "
        f"{n_total:>{digits + 6}}"
    )
    lines.append(
        f"{'weighted avg':>{label_width}}  "
        f"{weighted_averages[0]:>{digits + 5}.{digits}f} "
        f"{weighted_averages[1]:>{digits + 5}.{digits}f} "
        f"{weighted_averages[2]:>{digits + 5}.{digits}f} "
        f"{n_total:>{digits + 6}}"
    )
    return "\n".join(lines) + "\n"


@numba.njit(**NUMBA_NJIT_PARAMS)
def _compute_pr_recall_f1_metrics_seq(y_true, y_pred):
    TP = 0
    FP = 0
    FN = 0

    # Calculate TP, FP, FN
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _compute_pr_recall_f1_metrics_par(y_true, y_pred):
    """Parallel TP/FP/FN counters (numba auto-detects ``+=`` reductions).
    ~6× faster than seq at N=10M. Public wrapper auto-selects."""
    n = len(y_true)
    TP = 0
    FP = 0
    FN = 0
    for i in numba.prange(n):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_pr_recall_f1_metrics(y_true, y_pred):
    """Precision/Recall/F1 from binary predictions, auto seq/par.

    Sequential below N=100k (avoids thread spawn overhead); parallel
    above (~6× speedup at N=10M).
    """
    if len(y_true) >= _PARALLEL_REDUCTION_THRESHOLD:
        return _compute_pr_recall_f1_metrics_par(y_true, y_pred)
    return _compute_pr_recall_f1_metrics_seq(y_true, y_pred)


def fast_calibration_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int = 10,
    ece_debiased: bool = True,
    brier_debiased: bool = True,
    show_plots: bool = True,
    #
    title_metrics_tokens: Sequence[str] = DEFAULT_TITLE_METRICS_TOKENS,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    binning_strategy: str = "auto",
    reliability_show_ci: bool = True,
    reliability_smoothed: bool = True,
    #
    plot_file: str = "",
    plot_outputs: Optional[str] = None,
    base_path: Optional[str] = None,
    figsize: tuple = (15, 6),
    ndigits: int = 3,
    backend: str = "matplotlib",
    title: str = "",
    use_weights: bool = True,
    verbose: bool = False,
    group_ids: np.ndarray = None,
    binary_threshold: float = 0.5,
    _precomputed_aucs: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    **ice_kwargs,
):
    """Bins predictions, then computes regresison-like error metrics between desired and real binned probs.
    Input arrays y_true and y_pred are 1d.

    ``_precomputed_aucs`` is an internal escape hatch for callers that
    have already batch-computed (roc_auc, pr_auc) on GPU across multiple
    classes (see ``compute_batch_aucs``). When supplied AND
    ``group_ids is None``, the per-call ``fast_aucs_per_group_optimized``
    is skipped and the precomputed values are used. Reserved for use by
    the multiclass dispatcher in ``report_probabilistic_model_perf`` —
    other callers should let this default to None.

    Title composition is controlled by ``title_metrics_tokens`` (an ordered tuple
    of token names). ECE and Brier decomposition (REL/RES/UNC) are always
    computed and returned regardless of which tokens render. The 9 historical
    ``show_*_in_title`` booleans were collapsed into this one parameter so
    callers get explicit control over both metric selection AND order.
    Validation lives in ReportingConfig (training/configs.py); see the
    ``TITLE_METRIC_TOKENS`` frozenset for the complete grammar.

    ``reliability_show_ci`` (default on) renders the per-bin Wilson CI band on the
    reliability diagram; set False to suppress it (the toggle reaches the chart via
    ``show_calibration_plot`` -> ``build_calibration_spec(show_wilson_ci=...)``).

    ``reliability_smoothed`` (default on) overlays a binning-free smoothed isotonic reliability curve. The raw per-row
    ``(y_pred, y_true)`` arrays this function already holds (post finite-mask) are forwarded as views to
    ``show_calibration_plot`` -> ``build_calibration_spec(raw_probs=, raw_labels=)``; the smoother subsamples internally
    so no extra full-n copy is made. The overlay rides the DSL render path only; the legacy inline path is unaffected.
    """

    from ..core import fast_brier_score_loss  # lazy: import-cycle, see module top

    if backend not in ("plotly", "matplotlib"):
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'; got {backend!r}.")

    def _degenerate_result():
        """Empty / all-non-finite input: degenerate metrics, no crash.

        roc_auc is PINNED at 0.5 here (the no-skill / coin-flip AUC) and precision/recall/f1 at 0.0.
        On a rare-event target whose split happens to be single-class (no positives, or no negatives)
        the ranking AUC is genuinely undefined; the 0.5 pin is a neutral placeholder, NOT a measured
        score -- treat a 0.5 from this path as "AUC not computable on this split", not as a real
        random-ranking result.
        """
        return (
            1.0, 1.0, 1.0, 0.0,            # brier, cal_mae, cal_std, cal_cov
            1.0, 1.0, 0.0, 0.0,            # ece, rel, res, unc
            0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,  # roc, pr, ice, ll, precision, recall, f1
            "", None,                      # metrics_string, fig
        )

    if len(y_true) == 0:
        return _degenerate_result()

    # NaN / inf pre-mask. The preds path is NaN-guarded upstream (argmax_classes_safe)
    # but the probs path is not, and floor(nan) under numba (boundscheck off) yields an
    # undefined index -> out-of-bounds write in the binning kernel (silent corruption).
    # Drop non-finite scores here (cheap vectorized check, house "fastmath + Python NaN-gate"
    # pattern) so the whole report runs on finite data; log the dropped count once.
    _y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    _finite_mask = np.isfinite(_y_pred_arr)
    if not _finite_mask.all():
        _n_dropped = int((~_finite_mask).sum())
        logger.warning(
            "fast_calibration_report: dropped %d/%d non-finite probabilities before binning.",
            _n_dropped, len(_y_pred_arr),
        )
        y_pred = _y_pred_arr[_finite_mask]
        y_true = np.asarray(y_true)[_finite_mask]
        if len(y_true) == 0:
            logger.warning("fast_calibration_report: all probabilities were non-finite; skipping calibration.")
            return _degenerate_result()

    brier_loss = fast_brier_score_loss(y_true=y_true, y_prob=y_pred)

    freqs_predicted, freqs_true, hits = calibration_binning(
        y_true=y_true, y_pred=y_pred, nbins=nbins, strategy=binning_strategy,
    )
    if verbose:
        print("freqs_predicted", freqs_predicted)
        print("freqs_true", freqs_true)
    min_hits, max_hits = (np.min(hits), np.max(hits)) if len(hits) > 0 else (0, 0)
    calibration_mae, calibration_std, calibration_coverage = calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights
    )

    # Always compute ECE + Brier decomposition. Same data-adaptive bin grid as
    # CMAEW (kernel re-bins internally so it can capture per-bin pred_means,
    # which fast_calibration_binning doesn't expose). Cost is one short pass.
    #
    # Headline ECE/Brier default to the bias-corrected estimators (ece_debiased / brier_debiased): the plug-in
    # binned ECE/REL positively overstate miscalibration by the per-bin Bernoulli sampling noise (a perfectly
    # calibrated model reports a spurious positive value growing with nbins). compute_ece_debiased /
    # compute_brier_decomposition_debiased subtract that noise floor (Kumar et al. NeurIPS 2019; Broecker 2009);
    # bench_ece_debiased / bench_brier_decomp_debiased show they more than halve the bias-vs-truth on calibrated
    # scenarios. brier debiasing subtracts Var(acc_b) from BOTH REL and RES, so REL-RES, UNC and BinnedBrier are
    # all preserved exactly (Murphy identity still holds). Set ece_debiased / brier_debiased False for the legacy
    # plug-in values.
    #
    # When both debiased flags are on (the default), the plug-in decomposition + both debiased estimators share
    # the IDENTICAL counts/pred_sum/true_sum binning histogram -- so compute_ece_brier_full_and_debiased bins ONCE
    # and emits every reduction (bit-identical to three separate kernels by construction; ~2.7-3x faster, see
    # bench_fused_ece_brier_calibration). The split-kernel path stays for the opt-out combinations.
    if ece_debiased and brier_debiased:
        (
            _ece_plugin, _rel_plugin, _res_plugin, brier_uncertainty, _brier_binned_plugin,
            ece, brier_reliability, brier_resolution, _brier_binned,
        ) = compute_ece_brier_full_and_debiased(y_true=y_true, y_pred=y_pred, nbins=nbins)
    else:
        ece, brier_reliability, brier_resolution, brier_uncertainty, _brier_binned = compute_ece_and_brier_decomposition(
            y_true=y_true, y_pred=y_pred, nbins=nbins,
        )
        if ece_debiased:
            ece = compute_ece_debiased(y_true=y_true, y_pred=y_pred, nbins=nbins)
        if brier_debiased:
            brier_reliability, brier_resolution, brier_uncertainty, _brier_binned = compute_brier_decomposition_debiased(
                y_true=y_true, y_pred=y_pred, nbins=nbins,
            )

    _auc_desc_order = None
    _ks_fused = None
    if _precomputed_aucs is not None and group_ids is None:
        roc_auc, pr_auc = _precomputed_aucs
        group_aucs = {}
    else:
        roc_auc, pr_auc, group_aucs, _auc_desc_order, _ks_fused = fast_aucs_per_group_optimized(
            y_true=y_true, y_score=y_pred, group_ids=group_ids, return_order=True, return_ks=True
        )
    mean_group_roc_auc, mean_group_pr_auc = compute_mean_aucs_per_group(group_aucs) if group_aucs else (None, None)

    ice = integral_calibration_error_from_metrics(
        calibration_mae=calibration_mae,
        calibration_std=calibration_std,
        calibration_coverage=calibration_coverage,
        brier_loss=brier_loss,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        **ice_kwargs,
    )

    # Use fast numba version (returns nan for single-class data)
    ll = fast_log_loss(y_true, y_pred)
    if np.isnan(ll):
        ll = None

    _y_pred_thr = y_pred >= binary_threshold

    # KS / MCC / BSS title tokens. KS is one argsort+scan (reuses the AUC desc order when present);
    # MCC + the precision/recall/f1 family both derive from one shared binary confusion-counts pass.
    # KS/MCC/BSS stay gated by a narrow try-block so degenerate inputs emit N/A and the render survives.
    ks_val = float("nan")
    mcc_val = float("nan")
    bss_val = float("nan")
    # Single binary confusion-counts pass over (y_true, _y_pred_thr) feeds BOTH the
    # precision/recall/f1 family and MCC -- previously two independent full-n scans
    # over the identical arrays. P/R/F1/MCC are pure closed forms on the same
    # (TP, FP, TN, FN), so deriving them is bit-identical by construction.
    from ._classification_extras import (
        ks_statistic, brier_skill_score_from_brier,
        _confusion_counts_binary_dispatch,
        precision_recall_f1_from_counts, matthews_corrcoef_from_counts,
    )
    _yt_int = np.asarray(y_true).astype(np.int64, copy=False)
    _tp, _fp, _tn, _fn = _confusion_counts_binary_dispatch(_yt_int, _y_pred_thr)
    precision, recall, f1 = precision_recall_f1_from_counts(_tp, _fp, _fn)
    try:
        # iter86: KS was folded into the desc-order AUC walk (fast_numba_aucs_with_ks) -- reuse it. Fall back to a standalone
        # ks_statistic only on the precomputed-AUC path (no fused KS available) where _auc_desc_order is None too.
        if _ks_fused is not None:
            ks_val = _ks_fused
        else:
            ks_val = ks_statistic(_yt_int, y_pred, desc_order=_auc_desc_order)
        mcc_val = matthews_corrcoef_from_counts(_tp, _fp, _tn, _fn)
        bss_val = brier_skill_score_from_brier(brier_loss, _yt_int)
    except (ValueError, TypeError, FloatingPointError) as _ext_err:
        logger.debug("title-token extras (KS/MCC/BSS) skipped: %s", _ext_err)

    fragments = []
    for token in title_metrics_tokens:
        rendered = render_title_metric_token(
            token,
            ndigits=ndigits,
            ice=ice,
            brier_loss=brier_loss,
            ece=ece,
            brier_reliability=brier_reliability,
            brier_resolution=brier_resolution,
            brier_uncertainty=brier_uncertainty,
            calibration_mae=calibration_mae,
            calibration_std=calibration_std,
            use_weights=use_weights,
            calibration_coverage=calibration_coverage,
            nbins=nbins,
            ll=ll,
            max_hits=int(max_hits),
            min_hits=int(min_hits),
            roc_auc=roc_auc,
            mean_group_roc_auc=mean_group_roc_auc,
            pr_auc=pr_auc,
            mean_group_pr_auc=mean_group_pr_auc,
            precision=precision,
            recall=recall,
            f1=f1,
            ks=ks_val, mcc=mcc_val, bss=bss_val,
        )
        if rendered:
            fragments.append(rendered)

    # 2026-04-27 Session 7 batch 8 (user feedback): insert a hard line
    # break after the ``LL=`` fragment so the metrics-string doesn't
    # render as one ~200-char wall. Two-line layout reads naturally:
    # line 1 = calibration / loss family (ICE / BR / ECE / CMAEW / LL),
    # line 2 = ranking / classification family (ROC / PR / PR / RE / F1).
    metrics_string = ""
    for i, frag in enumerate(fragments):
        sep = ", "
        if i == 0:
            sep = ""
        elif fragments[i - 1].startswith("LL="):
            sep = "\n"
        metrics_string += sep + frag

    fig = None

    _dsl_render = bool(plot_outputs and base_path)
    if plot_file or show_plots or _dsl_render:

        plot_title = metrics_string

        if title:
            plot_title = title.strip() + "\n" + plot_title

        fig = show_calibration_plot(
            freqs_predicted=freqs_predicted,
            freqs_true=freqs_true,
            hits=hits,
            plot_title=plot_title,
            show_plots=show_plots,
            plot_file=plot_file,
            plot_outputs=plot_outputs,
            base_path=base_path,
            figsize=figsize,
            backend=backend,
            show_prob_histogram=show_prob_histogram,
            prob_histogram_yscale=prob_histogram_yscale,
            show_inline_population_labels=show_inline_population_labels,
            show_wilson_ci=reliability_show_ci,
            reliability_smoothed=reliability_smoothed,
            raw_probs=y_pred,
            raw_labels=y_true,
            dpi=dpi,
        )

    return (
        brier_loss, calibration_mae, calibration_std, calibration_coverage,
        ece, brier_reliability, brier_resolution, brier_uncertainty,
        roc_auc, pr_auc, ice, ll, precision, recall, f1,
        metrics_string, fig,
    )


@numba.njit(fastmath=False, cache=True, nogil=True, parallel=True)
def _batch_per_class_ice_kernel(
    y_true_NK: np.ndarray,
    y_pred_NK: np.ndarray,
    nbins: int,
    use_weights: bool,
    mae_weight: float,
    std_weight: float,
    brier_loss_weight: float,
    roc_auc_weight: float,
    pr_auc_weight: float,
    min_roc_auc: float,
    roc_auc_penalty: float,
) -> np.ndarray:
    """Batched per-class ICE: one numba dispatch, prange over K.

    Inlines the work of ``fast_ice_only`` (Brier + calibration binning +
    AUC + ICE combination) so the Python ``for class_id in range(K)``
    loop in ``compute_probabilistic_multiclass_error`` collapses to a
    single Python->numba transition. On 1M-row multiclass workloads
    this drops the Python-glue overhead from ~10-20 ms per call * K
    classes to ~10-20 ms total per call.

    Inputs:
        y_true_NK : (N, K) int8 — per-class indicator matrix
        y_pred_NK : (N, K) float64 — per-class predicted probability

    Returns ice_per_class : (K,) float64.

    Bit-exact equivalent of looping ``fast_ice_only`` per class
    (verified against the legacy form in
    ``bench_compute_multiclass_error.py``).

    bench-attempt-rejected (2026-05-21, c0146 / iter133): fusing the
    Brier + min/max passes (3 N-passes -> 2) saved only 1.04x at
    N=1M/K=3, 1.01-1.02x smaller. Argsort + AUC walk dominates the
    kernel; pre-argsort pass fusion is below the measurable speedup
    floor. Bench: profiling/bench_batch_ice_kernel_pass_fusion.py.
    """
    N = y_true_NK.shape[0]
    K = y_true_NK.shape[1]
    ice_per_class = np.empty(K, dtype=np.float64)

    for k in numba.prange(K):
        y_t = y_true_NK[:, k]
        y_p = y_pred_NK[:, k]

        # ---- Brier loss (mean squared error vs indicator) ----
        s = 0.0
        for i in range(N):
            d = float(y_t[i]) - y_p[i]
            s += d * d
        brier = s / N if N > 0 else 1.0

        # ---- Calibration binning (uniform-strategy, fixed nbins) ----
        # Replicates fast_calibration_binning + calibration_metrics_from_freqs
        # logic inline so the kernel stays single-entry.
        min_val = 1.0
        max_val = 0.0
        for i in range(N):
            v = y_p[i]
            if v > max_val:
                max_val = v
            if v < min_val:
                min_val = v
        span = max_val - min_val
        pockets_pred = np.zeros(nbins, dtype=np.int64)
        pockets_true = np.zeros(nbins, dtype=np.int64)
        if span > 0:
            multiplier = (nbins - 1) / span
            for i in range(N):
                ind = int(np.floor((y_p[i] - min_val) * multiplier))
                pockets_pred[ind] += 1
                pockets_true[ind] += y_t[i]
        else:
            for i in range(N):
                pockets_pred[0] += 1
                pockets_true[0] += y_t[i]

        # Collapse to non-empty bins
        n_nonempty = 0
        for b in range(nbins):
            if pockets_pred[b] > 0:
                n_nonempty += 1
        freqs_pred = np.empty(n_nonempty, dtype=np.float64)
        freqs_true = np.empty(n_nonempty, dtype=np.float64)
        hits = np.empty(n_nonempty, dtype=np.int64)
        ptr = 0
        for b in range(nbins):
            if pockets_pred[b] > 0:
                freqs_pred[ptr] = min_val + (b + 0.5) * span / nbins
                freqs_true[ptr] = pockets_true[b] / pockets_pred[b]
                hits[ptr] = pockets_pred[b]
                ptr += 1

        # ---- Calibration MAE / std / coverage ----
        # (calibration_metrics_from_freqs inlined with power-weighting on)
        if n_nonempty > 0:
            # Compute weights (power_weighting alpha=0.8 default of use_weights)
            if use_weights:
                weights = np.empty(n_nonempty, dtype=np.float64)
                for b in range(n_nonempty):
                    weights[b] = hits[b] ** 0.8
                w_sum = 0.0
                for b in range(n_nonempty):
                    w_sum += weights[b]
                if w_sum > 0:
                    for b in range(n_nonempty):
                        weights[b] /= w_sum
                # Weighted MAE
                cal_mae = 0.0
                for b in range(n_nonempty):
                    cal_mae += abs(freqs_pred[b] - freqs_true[b]) * weights[b]
                # Weighted std around weighted-mean MAE
                cal_var = 0.0
                for b in range(n_nonempty):
                    d = abs(freqs_pred[b] - freqs_true[b]) - cal_mae
                    cal_var += d * d * weights[b]
                cal_std = np.sqrt(cal_var)
            else:
                # Unweighted
                cal_mae = 0.0
                for b in range(n_nonempty):
                    cal_mae += abs(freqs_pred[b] - freqs_true[b])
                cal_mae /= n_nonempty
                cal_var = 0.0
                for b in range(n_nonempty):
                    d = abs(freqs_pred[b] - freqs_true[b]) - cal_mae
                    cal_var += d * d
                cal_std = np.sqrt(cal_var / n_nonempty)
        else:
            cal_mae = 1.0
            cal_std = 1.0

        # Coverage: number of distinct rounded freqs_pred values / nbins.
        # Use the same _round_prec rule as the standalone helper.
        _round_prec = max(1, int(np.ceil(np.log10(max(nbins, 2)))))
        # Round and count unique (numba can't use set; sort + unique-counting works).
        if n_nonempty > 0:
            rounded = np.empty(n_nonempty, dtype=np.float64)
            scale = 10.0 ** _round_prec
            for b in range(n_nonempty):
                rounded[b] = np.round(freqs_pred[b] * scale) / scale
            rounded = np.sort(rounded)
            n_unique = 1
            for b in range(1, n_nonempty):
                if rounded[b] != rounded[b - 1]:
                    n_unique += 1
            cal_cov = n_unique / nbins
        else:
            cal_cov = 0.0

        # ---- ROC AUC + PR AUC (fast_numba_aucs body inline) ----
        # Descending-sort argsort on y_p, then walk through (y_t, y_p)
        # in score-desc order, accumulating TP / FP / current_precision /
        # current_recall as in sklearn.average_precision_score.
        # numba 0.65 @njit np.argsort accepts kind="mergesort" (same stable
        # algorithm) but rejects the "stable" alias -- UnboundLocalError in
        # _sort_dispatch. Tie determinism preserved.
        desc_idx = np.argsort(-y_p, kind="mergesort")
        y_t_sorted = y_t[desc_idx]
        y_p_sorted = y_p[desc_idx]
        total_pos = 0
        for i in range(N):
            total_pos += y_t_sorted[i]
        total_neg = N - total_pos
        if total_pos == 0 or total_neg == 0:
            roc_auc = np.nan
            pr_auc = np.nan
        else:
            last_fps = 0
            last_tps = 0
            tps = 0
            fps = 0
            roc_acc = 0.0
            pr_acc = 0.0
            prev_recall = 0.0
            for i in range(N):
                yi = y_t_sorted[i]
                tps += yi
                fps += 1 - yi
                if i == N - 1 or y_p_sorted[i + 1] != y_p_sorted[i]:
                    delta_fps = fps - last_fps
                    sum_tps = last_tps + tps
                    roc_acc += delta_fps * sum_tps
                    last_fps = fps
                    last_tps = tps
                    current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
                    current_recall = tps / total_pos
                    delta_recall = current_recall - prev_recall
                    pr_acc += delta_recall * current_precision
                    prev_recall = current_recall
            denom_roc = tps * fps * 2
            if denom_roc > 0:
                roc_auc = roc_acc / denom_roc
            else:
                roc_auc = np.nan
            pr_auc = pr_acc

        # ---- Combine into ICE (integral_calibration_error_from_metrics body) ----
        base_loss = (
            brier * brier_loss_weight
            + cal_mae * mae_weight
            + cal_std * std_weight
        )
        roc_term = 0.0 if np.isnan(roc_auc) else np.abs(roc_auc - 0.5) * roc_auc_weight
        pr_term = 0.0 if np.isnan(pr_auc) else pr_auc * pr_auc_weight
        ice = base_loss - roc_term - pr_term
        threshold_width = min_roc_auc - 0.5
        if threshold_width > 0.0 and not np.isnan(roc_auc):
            deficit = threshold_width - np.abs(roc_auc - 0.5)
            if deficit > 0.0:
                ice += (deficit / threshold_width) * roc_auc_penalty

        ice_per_class[k] = ice

    return ice_per_class


def fast_ice_only(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int = 10,
    use_weights: bool = True,
    **ice_kwargs,
) -> float:
    """Compute only the ICE scalar from y_true/y_pred, skipping the
    log_loss / precision-recall-f1 / title / plotting work that
    ``fast_calibration_report`` does for its reporting callers.

    Bit-exact equivalent of ``fast_calibration_report(...)[6]``. Used by
    the fairness fan-out hot path — verified 1.1-1.7x faster per call
    (bench_ice_only.py, 2026-04-19) with ICE drift < 1e-9.
    """
    from ..core import fast_brier_score_loss  # lazy: import-cycle, see module top
    if len(y_true) == 0:
        return 1.0
    brier_loss = fast_brier_score_loss(y_true=y_true, y_prob=y_pred)
    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    cal_mae, cal_std, cal_cov = calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights,
    )
    roc_auc, pr_auc, _ = fast_aucs_per_group_optimized(y_true=y_true, y_score=y_pred, group_ids=None)
    return integral_calibration_error_from_metrics(
        calibration_mae=cal_mae, calibration_std=cal_std, calibration_coverage=cal_cov,
        brier_loss=brier_loss, roc_auc=roc_auc, pr_auc=pr_auc, **ice_kwargs,
    )


def predictions_time_instability(preds: pd.Series) -> float:
    """Computes how stable are true values or predictions over time.
    It's hard to use predictions that change upside down from point to point.
    For binary classification instability ranges from 0 to 1, for regression from 0 to any value depending on the target stats.
    """
    return np.abs(np.diff(preds)).mean()
