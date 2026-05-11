# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging
import sys

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import numba
from math import floor
from scipy.special import expit
import matplotlib
from matplotlib import pyplot as plt
import numpy as np, pandas as pd, polars as pl
from sklearn.metrics import log_loss, average_precision_score
from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_image

from collections import defaultdict
from pyutilz.pythonlib import sort_dict_by_value
from mlframe.stats import get_tukey_fences_multiplier_for_quantile

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


def _assert_numba_nogil_active() -> bool:
    """Verify JIT-compiled kernels actually released the GIL (nogil=True took effect).

    Numba silently retains the GIL when a kernel references Python objects it
    cannot prove safe — the compile succeeds (nopython mode) but parallelism
    under ThreadPoolExecutor is lost without any warning. Inspect one hot
    kernel's compiled signatures after prewarm; if the flag didn't stick,
    log a warning so the degradation is observable.

    Called once at end of prewarm_numba_cache(). Cheap (dict attribute read).
    Returns True iff every compiled signature on the canary kernel reports
    `release_gil`.
    """
    try:
        # cb_logits_to_probs_binary is forward-declared; pick any @njit symbol
        # that definitely compiled during prewarm.
        canary = globals().get("fast_roc_auc")
        if canary is None or not hasattr(canary, "signatures"):
            return True  # nothing to check — don't spam warnings
        overloads = getattr(canary, "overloads", {})
        for sig, compile_result in overloads.items():
            # numba CompileResult has `.targetctx` / `.fndesc`; the nogil flag
            # lives under `compile_result.type_annotation.ir`. Safer: check
            # `compile_result.fndesc.release_gil` when present, else trust
            # the NUMBA_NJIT_PARAMS.
            fndesc = getattr(compile_result, "fndesc", None)
            if fndesc is not None and hasattr(fndesc, "release_gil"):
                if not fndesc.release_gil:
                    logger.warning(
                        "numba JIT: nogil=True requested but kernel retained GIL "
                        f"({canary.__name__}, sig={sig}). ThreadPoolExecutor parallelism "
                        "over metrics will silently degrade to sequential."
                    )
                    return False
        return True
    except Exception as e:
        logger.debug("_assert_numba_nogil_active: inspection failed", exc_info=e)
        return True


def prewarm_numba_cache():
    """Pre-warm Numba JIT cache to avoid compilation overhead during profiling.

    Calls all @njit functions with small dummy data to trigger JIT compilation
    before timing-sensitive operations. Warms up both float32 and float64 paths.
    """
    # 2026-05-08 perf: kick the loky/wmic physical-core-count probe in a
    # background thread BEFORE numba JIT. The probe is a Windows wmic
    # subprocess (~1.5s wall) that loky caches per-process. Running it
    # in parallel with the JIT compile (which is THIS-process CPU work,
    # not subprocess) overlaps the wait, so the suite never pays the
    # 1.5s when it later asks for cpu_count via joblib. Daemon thread
    # so it dies with the interpreter if the user interrupts. No-op if
    # joblib isn't installed (it is, on every supported config).
    try:
        import threading

        def _kick_cpu_count():
            try:
                from joblib.parallel import cpu_count as _cc
                _cc()
            except Exception:
                pass

        threading.Thread(target=_kick_cpu_count, daemon=True).start()
    except Exception:
        pass

    # Warm up with both float32 and float64 (Numba compiles for each type separately)
    for dtype in [np.float32, np.float64]:
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=dtype)
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5], dtype=dtype)

        # Core AUC functions
        _ = fast_roc_auc(y_true, y_pred)
        _ = fast_aucs(y_true, y_pred)

        # Calibration functions
        _ = fast_calibration_binning(y_true, y_pred, nbins=10)
        _ = fast_calibration_metrics(y_true, y_pred, nbins=10)

        # Scoring functions
        _ = brier_score_loss(y_true, y_pred)
        _ = fast_brier_score_loss(y_true, y_pred)
        _ = fast_log_loss(y_true, y_pred)
        _ = maximum_absolute_percentage_error(y_true, y_pred)
        _ = probability_separation_score(y_true, y_pred)

        # calibration_metrics_from_freqs: consumes the output of
        # fast_calibration_binning. Prewarm with matching dtype.
        freqs_p, freqs_t, hits = fast_calibration_binning(y_true, y_pred, nbins=10)
        _ = calibration_metrics_from_freqs(
            freqs_predicted=freqs_p, freqs_true=freqs_t, hits=hits,
            nbins=10, array_size=len(y_true), use_weights=True,
        )

    # Classification functions need integer class labels
    for dtype in [np.int32, np.int64]:
        y_true_int = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=dtype)
        y_pred_int = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=dtype)
        _ = fast_classification_report(y_true_int, y_pred_int, nclasses=2)
        _ = fast_precision(y_true_int, y_pred_int, nclasses=2)
        _ = compute_pr_recall_f1_metrics(y_true_int, y_pred_int)

    # ICE metric (float parameters)
    _ = integral_calibration_error_from_metrics(0.01, 0.01, 0.9, 0.25, 0.7, 0.7)

    # 2026-05-08: prewarm calibration-report inner kernels that the
    # original list missed. Each costs ~1-2s of JIT compile on first
    # call from a fresh process; without prewarm, they all hit on the
    # first ``fast_calibration_report`` invocation (typically the val
    # split of model 1, contributing ~5-10s of cold-start latency to
    # every suite run). Confirmed via cProfile of c0044 (4 models, 80k
    # rows) where numba compile time was 11.7s before this addition.
    #
    # 2026-05-08 cont'd: cover the dominant suite-runtime dtype combo.
    # In the per-class loop of report_probabilistic_model_perf, y_true
    # arrives as numpy bool (``targets == class_name``) and y_pred as
    # float64 (predict_proba column). We only prewarm THIS exact combo
    # -- prewarming additional dtype-pairs costs ~5-10s each in JIT
    # compile time (verified empirically) and the suite's hot path is
    # always (bool, float64) per the per-class slicing convention. Other
    # callers that need different dtypes (regression, multilabel) hit
    # the same cold-compile cost on first call but the calibration
    # report path -- which dominates the 80s c0044 profile -- is now
    # warm before entering the suite.
    _yt_bool = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)
    _yp_f64 = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5], dtype=np.float64)
    try:
        _ = compute_ece_and_brier_decomposition(_yt_bool, _yp_f64, nbins=10)
        _ = fast_aucs_per_group_optimized(y_true=_yt_bool, y_score=_yp_f64, group_ids=None)
        _ = fast_log_loss(_yt_bool, _yp_f64)
        _ = fast_ice_only(_yt_bool, _yp_f64, nbins=10, use_weights=True)
    except Exception:
        pass
    # format_classification_report: njit-backed, ~3s cold.
    _yti = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    _ypi = np.array([0, 1, 0, 0, 1, 1], dtype=np.int64)
    try:
        _ = format_classification_report(_yti, _ypi, nclasses=2)
    except Exception:
        pass

    # CatBoost logits to probs conversion
    logits_binary = np.array([-1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5, -1.5, 0.25, -0.25], dtype=np.float64)
    _ = cb_logits_to_probs_binary(logits_binary)

    logits_multi = np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0], [0.0, 1.0, -1.0]], dtype=np.float64)
    _ = cb_logits_to_probs_multiclass(logits_multi)

    # 2026-05-09: prewarm parallel-numba variants. The public
    # ``fast_brier_score_loss`` / ``fast_log_loss`` /
    # ``compute_pr_recall_f1_metrics`` / ``subset_accuracy`` /
    # ``jaccard_score_multilabel`` wrappers dispatch to ``_par`` kernels
    # above the per-metric size threshold. Each parallel variant is a
    # SEPARATE numba compilation that ``cache=True`` writes to disk
    # only after a successful run; the ``parallel=True`` IR adds
    # ~1-3s per kernel on first call from a fresh process. Prewarm
    # them all here so the suite never pays JIT cost during a real run.
    try:
        _yt_f64 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
        _yp_f64 = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5], dtype=np.float64)
        _ = _fast_brier_score_loss_par(_yt_f64, _yp_f64)
        _ = _fast_log_loss_binary_par(_yt_f64, _yp_f64, 1e-15)
        _yt_i64 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        _yp_i64 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int64)
        _ = _compute_pr_recall_f1_metrics_par(_yt_i64, _yp_i64)
        _ml_yt = np.zeros((10, 3), dtype=np.uint8); _ml_yt[:5, 0] = 1
        _ml_yp = np.zeros((10, 3), dtype=np.uint8); _ml_yp[:5, 0] = 1
        _ = _fast_subset_accuracy_par(_ml_yt, _ml_yp)
        _ = _fast_jaccard_score_par(_ml_yt, _ml_yp)

        # Round 2 par variants (cb_logits + max_abs_pct + probability_sep).
        _logits_b = np.array([-1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5, -1.5, 0.25, -0.25], dtype=np.float64)
        _ = _cb_logits_to_probs_binary_par(_logits_b)
        _logits_mc = np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0], [0.0, 1.0, -1.0]], dtype=np.float64)
        _ = _cb_logits_to_probs_multiclass_par(_logits_mc)
        _ = _max_abs_pct_error_kernel_par(_yt_f64, _yp_f64)
        _yt_i64_psep = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        _ = _probability_separation_score_par(_yt_i64_psep, _yp_f64, 1, 0.5)

        # Round 3: regression-metric par variants. Each kernel adds
        # ~1-1.5s of cold JIT compile if hit during a real run; prewarm
        # them here so the regression report path is warm by the time
        # the suite needs it.
        _reg_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
        _reg_p = _reg_y + 0.05
        _reg_w = np.ones_like(_reg_y)
        _ = _fast_mae_seq(_reg_y, _reg_p)
        _ = _fast_mae_par(_reg_y, _reg_p)
        _ = _fast_mae_weighted_seq(_reg_y, _reg_p, _reg_w)
        _ = _fast_mae_weighted_par(_reg_y, _reg_p, _reg_w)
        _ = _fast_mse_seq(_reg_y, _reg_p)
        _ = _fast_mse_par(_reg_y, _reg_p)
        _ = _fast_mse_weighted_seq(_reg_y, _reg_p, _reg_w)
        _ = _fast_mse_weighted_par(_reg_y, _reg_p, _reg_w)
        _ = _fast_max_error_seq(_reg_y, _reg_p)
        _ = _fast_r2_score_seq(_reg_y, _reg_p)
        _ = _fast_r2_score_par(_reg_y, _reg_p)
        _ = _fast_r2_score_weighted_seq(_reg_y, _reg_p, _reg_w)
        _ = _fast_r2_score_weighted_par(_reg_y, _reg_p, _reg_w)
        _ = _fast_r2_variance_seq(_reg_y)
    except Exception:
        # Parallel kernels cache the same way njit kernels do; a bad
        # cache or a numba-runtime hiccup is non-fatal — the seq path
        # still works.
        pass

    # 2026-04-25 Session 6 polish: prewarm multilabel kernels too. Without
    # this the first multilabel-aware report path call eats a 1-3s JIT
    # compile budget per kernel × per dtype, which surprises new users
    # running the report on small data and shows up in time-sensitive
    # benchmarks.
    yt_ml = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    yp_ml = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1]], dtype=np.uint8)
    _ = _fast_hamming_loss_seq(yt_ml, yp_ml)
    _ = _fast_hamming_loss_par(yt_ml, yp_ml)
    _ = _fast_subset_accuracy_seq(yt_ml, yp_ml)
    _ = _fast_jaccard_score_seq(yt_ml, yp_ml)
    # Bitmap variant takes packed uint64 + K — prewarm for K<=64 path.
    yt_packed = np.array([0b011, 0b101, 0b110, 0b001], dtype=np.uint64)
    yp_packed = np.array([0b110, 0b101, 0b100, 0b011], dtype=np.uint64)
    _ = _fast_jaccard_bitmap_seq(yt_packed, yp_packed, 3)

    # Audit hook: verify nogil=True actually stuck. Silent fallback would
    # make parallel val/test metric evaluation secretly sequential.
    _assert_numba_nogil_active()

    # 2026-05-11: also warm feature_selection numba kernels. Without
    # this, the first MRMR.fit call pays ~60s of cumulative JIT
    # compile (verified on 1M-row regression+MRMR c0089 profile).
    # Hooked here so any caller of ``prewarm_numba_cache`` (incl. the
    # profile harness in profiling/profile_one_combo.py) gets BOTH
    # metrics AND FS kernels warmed in one shot. Lazy import keeps
    # this module's import cost unchanged.
    try:
        from mlframe.feature_selection.filters._prewarm import (
            prewarm_fs_numba_cache,
        )
        prewarm_fs_numba_cache()
    except Exception:
        # Non-fatal: FS kernels will JIT-compile on first real call
        # (current behaviour). Caller bears the cold-start cost then.
        pass

    # 2026-05-11: also warm dummy_baselines kernels. The suite already
    # calls ``_warmup_numba_kernels`` early in
    # ``train_mlframe_models_suite``, but that lands INSIDE the suite
    # wall-time (verified on c0089 W11 profile: 17.7 s inside the
    # suite call). Pre-warming here shifts the cost to the same
    # prewarm phase as metrics + FS kernels -- before any
    # user-visible timer starts. The suite's own call becomes
    # near-zero after this completes (numba's process-wide JIT cache
    # carries over).
    try:
        from mlframe.training.dummy_baselines import _warmup_numba_kernels
        _warmup_numba_kernels()
    except Exception:
        # Non-fatal: kernels will JIT lazily on first real call.
        pass

    # 2026-05-11 Wave 19: prewarm-import the heavy neural-net stack.
    # ``mlframe.lightninglib`` / ``mlframe.training.neural`` pulls in
    # PyTorch Lightning (lightning.fabric, pytorch_lightning, etc.) which
    # is a 275 s cold-import on Windows (verified 2026-05-11 on c0082
    # MLP-only regression: ``configure_training_params`` cumtime 275.77 s
    # of which ~260 s is ``lightning.fabric.__init__``). The cost lands
    # INSIDE the suite call (in ``select_target`` -> ``get_training_configs``)
    # because the import is deferred until ``mlp`` is in the model list.
    # Pre-importing here shifts that cost out of the user-visible suite
    # timer. Cheaply skipped via try/except for users who don't have
    # lightning installed (or don't want to pay the 275 s prewarm).
    #
    # Triggered ONLY when lightning is already discoverable; otherwise
    # the import attempt would itself be a 5-10 s "ModuleNotFoundError"
    # walk through sys.path. Use ``importlib.util.find_spec`` for the
    # cheap availability check.
    try:
        import importlib.util as _ilu
        if _ilu.find_spec("lightning") is not None:
            try:
                import lightning.fabric  # noqa: F401 - import-for-side-effects
            except Exception:
                pass
            try:
                import mlframe.lightninglib  # noqa: F401
            except Exception:
                pass
    except Exception:
        pass

    # 2026-05-11 Wave 18: warm cupy GPU AUC kernels.
    # ``compute_batch_aucs`` dispatches to ``gpu_multiple_roc_auc_scores``
    # / ``gpu_multiple_pr_auc_scores`` when N>=100k AND M>=5. cupy compiles
    # CUDA kernels via NVRTC on first call -- ~128s of compile latency per
    # fresh process (measured 2026-05-11 on c0008 1M binary-cls profile:
    # 127.13s in nvrtc.compileProgram). Pre-warming with a tiny synthetic
    # input shifts that cost out of the user-visible suite timer.
    # No-op when cupy isn't installed (`_require_cupy()` raises ImportError).
    try:
        from mlframe.metrics import (
            gpu_multiple_roc_auc_scores, gpu_multiple_pr_auc_scores,
            gpu_multiple_rmse_scores,
        )
        _yt_gpu = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 16, dtype=np.int8)
        _yp_gpu = np.random.RandomState(0).rand(len(_yt_gpu), 3).astype(np.float64)
        _ = gpu_multiple_roc_auc_scores(_yt_gpu, _yp_gpu)
        _ = gpu_multiple_pr_auc_scores(_yt_gpu, _yp_gpu)
        # gpu_multiple_rmse_scores: separate cupy kernel (ReductionKernel
        # for the 2-D-actual fallback or numba.cuda for the 1-D fastpath).
        # Each path's first call compiles a fresh NVRTC kernel; prewarming
        # both signatures avoids the regression-suite cold-compile cost.
        _yt_rmse = _yp_gpu[:, 0]  # 1-D path
        _ = gpu_multiple_rmse_scores(_yt_rmse, _yp_gpu)
    except Exception:
        # cupy unavailable / CUDA missing / etc. -- non-fatal, the CPU
        # path keeps working.
        pass

    # 2026-05-11 Wave 16: warm ``ranking_metrics._summary_batched_kernel``
    # (parallel njit). On LTR combos (c0120 profile: 23.4 s wall, 17.7 s
    # numba JIT) ``compute_ranking_summary`` is called once per dummy
    # baseline (~10 calls per target) and the first call eats the entire
    # JIT-compile budget. Pre-warming shifts the cost out of the user-
    # visible timer. We compile with the canonical dtype combo
    # (sorted_y_true float64, sorted_y_score float64, group_starts int64,
    # eval_ks int64) used by ``compute_ranking_summary`` itself.
    try:
        from mlframe.ranking_metrics import _summary_batched_kernel
        _yt_rank = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 0.0], dtype=np.float64)
        _ys_rank = np.array([0.3, 0.9, 0.5, 0.7, 0.2, 0.1], dtype=np.float64)
        _gs_rank = np.array([0, 3, 6], dtype=np.int64)
        _ks_rank = np.array([1, 5, 10], dtype=np.int64)
        _ = _summary_batched_kernel(_yt_rank, _ys_rank, _gs_rank, _ks_rank)
    except Exception:
        # Non-fatal: kernel will JIT lazily on first ranking-metrics call.
        pass


# ----------------------------------------------------------------------------------------------------------------------------
# CatBoost logits to probabilities conversion
# ----------------------------------------------------------------------------------------------------------------------------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _cb_logits_to_probs_binary_seq(logits: np.ndarray) -> np.ndarray:
    """Sequential variant. Public wrapper auto-dispatches at N>=100k."""
    n = len(logits)
    probs = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        p1 = 1.0 / (1.0 + np.exp(-logits[i]))
        probs[i, 0] = 1.0 - p1
        probs[i, 1] = p1
    return probs


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _cb_logits_to_probs_binary_par(logits: np.ndarray) -> np.ndarray:
    """Parallel sigmoid. ~2.9× faster than seq at N=1M."""
    n = len(logits)
    probs = np.empty((n, 2), dtype=np.float64)
    for i in numba.prange(n):
        p1 = 1.0 / (1.0 + np.exp(-logits[i]))
        probs[i, 0] = 1.0 - p1
        probs[i, 1] = p1
    return probs


def cb_logits_to_probs_binary(logits: np.ndarray) -> np.ndarray:
    """Convert CatBoost binary logits to probabilities, auto seq/par.

    Args:
        logits: 1D array of logits from CatBoost (single class output)

    Returns:
        2D array of shape (n_samples, 2) with probabilities for [class_0, class_1]
    """
    if len(logits) >= _PARALLEL_REDUCTION_THRESHOLD:
        return _cb_logits_to_probs_binary_par(logits)
    return _cb_logits_to_probs_binary_seq(logits)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _cb_logits_to_probs_multiclass_seq(logits_list: np.ndarray) -> np.ndarray:
    """Sequential variant. Public wrapper auto-dispatches at N>=100k."""
    n_classes, n_samples = logits_list.shape
    probs = np.empty((n_samples, n_classes), dtype=np.float64)
    for i in range(n_samples):
        max_logit = logits_list[0, i]
        for c in range(1, n_classes):
            if logits_list[c, i] > max_logit:
                max_logit = logits_list[c, i]
        exp_sum = 0.0
        for c in range(n_classes):
            probs[i, c] = np.exp(logits_list[c, i] - max_logit)
            exp_sum += probs[i, c]
        for c in range(n_classes):
            probs[i, c] /= exp_sum
    return probs


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _cb_logits_to_probs_multiclass_par(logits_list: np.ndarray) -> np.ndarray:
    """Parallel softmax. ~2.3× faster than seq at N=1M (K=5)."""
    n_classes, n_samples = logits_list.shape
    probs = np.empty((n_samples, n_classes), dtype=np.float64)
    for i in numba.prange(n_samples):
        max_logit = logits_list[0, i]
        for c in range(1, n_classes):
            if logits_list[c, i] > max_logit:
                max_logit = logits_list[c, i]
        exp_sum = 0.0
        for c in range(n_classes):
            probs[i, c] = np.exp(logits_list[c, i] - max_logit)
            exp_sum += probs[i, c]
        for c in range(n_classes):
            probs[i, c] /= exp_sum
    return probs


def cb_logits_to_probs_multiclass(logits_list: np.ndarray) -> np.ndarray:
    """Convert CatBoost multiclass logits to probabilities (softmax),
    auto seq/par.

    Args:
        logits_list: 2D array of shape (n_classes, n_samples) with logits

    Returns:
        2D array of shape (n_samples, n_classes) with probabilities
    """
    if logits_list.shape[1] >= _PARALLEL_REDUCTION_THRESHOLD:
        return _cb_logits_to_probs_multiclass_par(logits_list)
    return _cb_logits_to_probs_multiclass_seq(logits_list)


# ----------------------------------------------------------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------------------------------------------------------


def fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    """Compute ROC AUC efficiently using numba.

    Note: np.argsort needs to stay out of njitted func.
    """
    # **kwargs needed for sklearn not to break it by passing unexpected params.
    # We explicitly reject sample_weight here rather than silently ignoring it (previous
    # behavior would produce unweighted results while callers thought they were weighting).
    if "sample_weight" in kwargs and kwargs["sample_weight"] is not None:
        raise NotImplementedError(
            "fast_roc_auc does not support sample_weight; use sklearn.metrics.roc_auc_score"
        )

    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = np.argsort(y_score)[::-1]
    return fast_numba_auc_nonw(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_auc_nonw(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> float:
    """code taken from fastauc lib."""
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    auc = 0

    l = len(y_true) - 1
    for i in range(l + 1):
        tps += y_true[i]
        fps += 1 - y_true[i]
        if i == l or y_score[i + 1] != y_score[i]:
            auc += (fps - last_counted_fps) * (last_counted_tps + tps)
            last_counted_fps = fps
            last_counted_tps = tps
    tmp = tps * fps * 2
    if tmp > 0:
        return auc / tmp
    else:
        # Single-class data: ROC AUC is undefined
        return np.nan


# ----------------------------------------------------------------------------------------------------------------------------
# GPU batch metrics: vectorize across multiple prediction columns
# ----------------------------------------------------------------------------------------------------------------------------
#
# Backend: cupy (default). cupy is OPTIONAL — without it, the GPU helpers
# raise a clear ImportError and callers fall back to CPU (see
# ``compute_batch_aucs`` and ``compute_batch_rmse`` dispatchers).
#
# Why cupy and not numba.cuda? Bench (verified 2026-05-08, GTX 1050 Ti):
#   - RMSE: cupy ReductionKernel = 3.7× over naive cupy expression and
#     within 30% of a hand-written numba.cuda kernel. Reusing cupy
#     primitives (sort/argsort/cumsum) keeps one optional dep.
#   - AUC : the work is ~80% argsort which is a single cupy call. A
#     numba.cuda hybrid (cupy sort + custom reduction) tied cupy at
#     1.02× — the reduction wasn't the bottleneck. So no point adding
#     a second backend dep for a 0% net win.
#
# Crossover with CPU (1M rows, 5 cols, vs CPU/numba per-col loop):
#   RMSE: GPU 4.2ms / CPU 62ms -> 14×
#   ROC AUC: GPU 105ms / CPU 477ms -> 4.6×
#   PR AUC : GPU 93ms  / CPU 435ms -> 4.7×
#   At 1M × 20 cols, PR AUC widens to **11.8×** (170ms vs 2016ms).
#   At 5M × 5, PR AUC = **19.5×** (212ms vs 4141ms).
#
# Below N=100k the CPU/numba path wins (kernel-launch and host->device
# transfer dominate sub-ms workloads). The dispatchers below auto-pick
# the right backend based on N, M, and GPU availability.

# Default crossover thresholds. Tunable via ``mlframe.metrics.set_gpu_thresholds(...)``.
_GPU_BATCH_THRESHOLD_N: int = 100_000
_GPU_BATCH_THRESHOLD_M: int = 1   # M >= 1 works; benefit grows with M.

# GPU availability cache (computed lazily on first dispatch call).
# Sentinels: None = unchecked, True/False = result.
_GPU_AVAILABLE: Optional[bool] = None

# numba.cuda availability cache. Used by the RMSE fast-path -- 25-30%
# faster than cupy ReductionKernel above N=1M (verified empirically on
# GTX 1050 Ti). cupy ReductionKernel stays as the fallback when numba
# CUDA isn't usable (no toolkit, mismatched runtime, etc.).
_NUMBA_CUDA_AVAILABLE: Optional[bool] = None


def set_gpu_thresholds(*, n: Optional[int] = None, m: Optional[int] = None) -> None:
    """Override the (N, M) thresholds that gate GPU dispatch in
    ``compute_batch_aucs`` / ``compute_batch_rmse``. Pass ``None`` to
    leave a threshold unchanged. For tests / benchmarking only —
    production callers should let auto-dispatch handle this."""
    global _GPU_BATCH_THRESHOLD_N, _GPU_BATCH_THRESHOLD_M
    if n is not None:
        _GPU_BATCH_THRESHOLD_N = int(n)
    if m is not None:
        _GPU_BATCH_THRESHOLD_M = int(m)


def is_gpu_metrics_available() -> bool:
    """True iff cupy is importable AND a CUDA device is visible.

    Result is cached after the first call (the answer doesn't change
    within a process). Failures during the import path return False
    silently — the dispatcher uses CPU when GPU is unavailable.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE
    try:
        import cupy as cp  # type: ignore
        # Probe: device count >= 1
        if cp.cuda.runtime.getDeviceCount() >= 1:
            _GPU_AVAILABLE = True
            return True
    except Exception:
        pass
    _GPU_AVAILABLE = False
    return False


def _is_numba_cuda_available() -> bool:
    """Probe numba.cuda once. Cached. Used by the RMSE fast-path which
    runs 25-30% faster than the cupy ReductionKernel fallback at large N."""
    global _NUMBA_CUDA_AVAILABLE
    if _NUMBA_CUDA_AVAILABLE is not None:
        return _NUMBA_CUDA_AVAILABLE
    try:
        from numba import cuda  # type: ignore
        if cuda.is_available():
            _NUMBA_CUDA_AVAILABLE = True
            return True
    except Exception:
        pass
    _NUMBA_CUDA_AVAILABLE = False
    return False


def _require_cupy():
    """Lazy cupy import; raise ImportError with install hint if missing."""
    try:
        import cupy as cp  # type: ignore
        return cp
    except ImportError as e:
        raise ImportError(
            "GPU metrics require cupy. Install for your CUDA version, e.g. "
            "`pip install cupy-cuda12x` (CUDA 12) or `cupy-cuda11x` (CUDA 11)."
        ) from e


# Cached cupy ReductionKernel for fused (y-p)**2 sum-per-col. Built lazily
# on first call. 3.7× faster than ``cp.sqrt(cp.mean((y-p)**2, axis=0))``
# because it avoids materializing the (y-p)**2 intermediate (N×M fp64
# allocations dominate the naive expression). Bit-equivalent to numpy.
# Used as fallback when numba.cuda is unavailable.
_CUPY_SSE_PER_COL = None
_NUMBA_RMSE_KERNEL = None  # cached numba.cuda kernel handle


def _get_cupy_sse_kernel():
    """Build (or return cached) cupy ReductionKernel for SSE per column."""
    global _CUPY_SSE_PER_COL
    if _CUPY_SSE_PER_COL is None:
        cp = _require_cupy()
        _CUPY_SSE_PER_COL = cp.ReductionKernel(
            in_params="float64 y, float64 p",
            out_params="float64 z",
            map_expr="(y - p) * (y - p)",
            reduce_expr="a + b",
            post_map_expr="z = a",
            identity="0.0",
            name="mlframe_sse_per_col",
        )
    return _CUPY_SSE_PER_COL


def _get_numba_rmse_kernel():
    """Build (or return cached) numba.cuda kernel that computes per-block,
    per-column SSE via atomic.add. Final reduction + sqrt happens in cupy.
    25-30% faster than the cupy ReductionKernel at N >= 1M."""
    global _NUMBA_RMSE_KERNEL
    if _NUMBA_RMSE_KERNEL is None:
        from numba import cuda  # type: ignore

        @cuda.jit
        def _rmse_partial_sum(y, p, partial, N, M):
            # y: (N,) (broadcast across M cols), p: (N, M)
            # partial: (grid_x, M) per-block accumulator
            j = cuda.blockIdx.y
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if j >= M or i >= N:
                return
            d = y[i] - p[i, j]
            cuda.atomic.add(partial, (cuda.blockIdx.x, j), d * d)

        _NUMBA_RMSE_KERNEL = _rmse_partial_sum
    return _NUMBA_RMSE_KERNEL


def gpu_multiple_rmse_scores(actual, predicted):
    """Vectorized RMSE across columns on GPU.

    Computes ``sqrt(mean((actual - predicted)**2, axis=0))`` for an
    ``(N,)`` or ``(N, M)`` ``actual`` and an ``(N, M)`` ``predicted``,
    returning ``(M,)`` RMSEs as a cupy array.

    Backend selection (auto):
      - ``numba.cuda`` kernel when ``numba.cuda.is_available()``: fused
        (subtract, square, atomic-add) in one pass; per-block partials
        finalised by cupy.sum. **25-30% faster than the cupy fallback at
        N >= 1M** (verified on GTX 1050 Ti). Tiny fp jitter (~1e-15)
        from non-deterministic atomic-add accumulation order.
      - ``cupy.ReductionKernel`` when only cupy is available: fuses
        subtract+square+reduce into one kernel pass, avoiding the
        ``(N, M)`` fp64 intermediate that the naive expression allocates.
        Bit-equivalent to numpy.

    Use when N >= 100k. Below that the numpy path is faster (host
    overhead). For automatic dispatch (numpy / GPU based on (N, M)
    + availability) see ``compute_batch_rmse``.

    Inputs may be cupy or numpy arrays. Output is a cupy array; call
    ``cp.asnumpy(...)`` to bring back to host.
    """
    cp = _require_cupy()
    actual = cp.asarray(actual, dtype=cp.float64)
    predicted = cp.asarray(predicted, dtype=cp.float64)
    if predicted.ndim == 1:
        predicted = predicted[:, cp.newaxis]

    # numba.cuda fast-path expects 1-D y for broadcast. If user passed
    # 2-D actual, we'd have to write a different kernel. Cheap path:
    # fall back to ReductionKernel for the 2-D case (rare in practice).
    can_use_numba = _is_numba_cuda_available() and actual.ndim == 1
    N = predicted.shape[0]
    M = predicted.shape[1]

    if can_use_numba:
        from numba import cuda  # local import for speed
        kernel = _get_numba_rmse_kernel()
        BLOCK_N = 256
        grid_x = (N + BLOCK_N - 1) // BLOCK_N
        partial = cp.zeros((grid_x, M), dtype=cp.float64)
        kernel[(grid_x, M), BLOCK_N](
            cuda.as_cuda_array(actual),
            cuda.as_cuda_array(predicted),
            cuda.as_cuda_array(partial),
            N, M,
        )
        return cp.sqrt(cp.sum(partial, axis=0) / N)

    # cupy ReductionKernel fallback (also handles 2-D actual case).
    if actual.ndim == 1:
        actual = actual[:, cp.newaxis]
    sse = _get_cupy_sse_kernel()(actual, predicted, axis=0)
    return cp.sqrt(sse / N)


def gpu_multiple_roc_auc_scores(actual, predicted):
    """Vectorized ROC AUC across columns on GPU (cupy), tie-correct.

    Computes ROC AUC via Mann-Whitney U with FRACTIONAL (average) ranks
    on tied scores, matching ``sklearn.metrics.roc_auc_score`` and
    ``mlframe.metrics.fast_roc_auc`` bit-for-bit on continuous data and
    within fp64 noise on tied data.

    Use when N >= 100k AND M >= 5 (roughly: enough rows + enough columns
    to amortise GPU launch). At N=1M, M=20: ~4.5× over a Python loop;
    at N=5M, M=5: ~8×. Below N=100k the per-column numba loop is faster.
    For automatic dispatch see ``compute_batch_aucs``.

    Args:
        actual: ``(N,)`` binary 0/1 labels (numpy or cupy).
        predicted: ``(N, M)`` per-column scores (numpy or cupy). Higher
            score = more likely positive.

    Returns:
        ``(M,)`` cupy array of AUCs. NaN for columns where the label
        column is degenerate (all 0 or all 1).

    Notes:
        The naive ``argsort(argsort(x))+1`` Mann-Whitney trick gives
        strict (not fractional) ranks on tied scores, producing ~1e-5
        error vs sklearn on calibrated probability bins. This impl
        replaces the second argsort with a cumsum-based fractional-rank
        computation: O(N) vs O(N log N), so it's both correct AND
        faster than the naive double-argsort version.
    """
    cp = _require_cupy()
    actual = cp.asarray(actual)
    predicted = cp.asarray(predicted)
    if predicted.ndim == 1:
        predicted = predicted[:, cp.newaxis]

    N, M = predicted.shape
    n_pos = cp.sum(actual)
    n_neg = N - n_pos

    # Per-column fractional ranks. ``order[:, j]`` = ascending sort indices
    # of column j; equal-value runs get the average of the strict ranks
    # they occupy (e.g. three 0.5's at positions 4,5,6 all get rank 5).
    order = cp.argsort(predicted, axis=0)
    ranks = cp.empty((N, M), dtype=cp.float64)
    for j in range(M):  # M is small (typ. 1-100); loop is cheap on GPU
        col = predicted[:, j]
        col_sorted = col[order[:, j]]
        # `is_new[i]` True iff position i starts a new run of equal values.
        is_new = cp.concatenate([
            cp.array([True]),
            col_sorted[1:] != col_sorted[:-1],
        ])
        run_id = cp.cumsum(is_new) - 1
        run_starts = cp.where(is_new)[0]
        run_ends = cp.concatenate([run_starts[1:], cp.array([N])])
        run_sizes = run_ends - run_starts
        # Average strict rank over each run: start_strict_rank + (size+1)/2 - 1
        # = run_starts + (size+1)/2  (since strict ranks are 1-based and
        #   run_starts is 0-based, the offset cancels out).
        avg_rank_per_run = run_starts.astype(cp.float64) + (run_sizes + 1) / 2.0
        avg_ranks_sorted = avg_rank_per_run[run_id]
        col_ranks = cp.empty(N, dtype=cp.float64)
        col_ranks[order[:, j]] = avg_ranks_sorted
        ranks[:, j] = col_ranks

    aucs = (cp.sum(ranks[actual == 1, :], axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return aucs


def gpu_multiple_pr_auc_scores(actual, predicted):
    """Vectorized PR AUC (Average Precision) across columns on GPU.

    Computes Riemann-sum AP matching ``sklearn.metrics.average_precision_score``
    and ``mlframe.metrics.fast_aucs`` bit-for-bit (verified ≤ 5.55e-17 on
    continuous and tied data).

    Algorithm: descending-sort by score, build cumulative-TP / FP arrays,
    detect tie-run boundaries (one threshold per run), sum
    ``(R_n - R_{n-1}) × P_n`` anchored at R_0 = 0.

    Use when N >= 100k AND M >= 5. PR AUC is the most GPU-friendly of the
    three metrics here: at 1M × 20 cols, GPU = 170 ms vs CPU loop = 2016 ms
    (**11.8× speedup**); at 5M × 5 cols, GPU = 213 ms vs CPU = 4141 ms
    (**19.5× speedup**). For automatic dispatch see ``compute_batch_aucs``.

    Args:
        actual: ``(N,)`` binary 0/1 labels.
        predicted: ``(N,)`` or ``(N, M)`` scores.

    Returns:
        ``(M,)`` cupy array of average-precision scores. NaN for
        single-class label columns (sklearn behavior would raise).
    """
    cp = _require_cupy()
    actual = cp.asarray(actual).astype(cp.float64)
    predicted = cp.asarray(predicted)
    if predicted.ndim == 1:
        predicted = predicted[:, cp.newaxis]

    N, M = predicted.shape
    aps = cp.empty(M, dtype=cp.float64)
    total_pos = cp.sum(actual)
    if total_pos == 0 or total_pos == N:
        # Single-class data — AP is undefined for all columns.
        aps.fill(cp.nan)
        return aps

    idx = cp.arange(1, N + 1, dtype=cp.float64)
    for j in range(M):
        col = predicted[:, j]
        order = cp.argsort(col, kind="stable")[::-1]   # descending
        col_sorted = col[order]
        y_sorted = actual[order]

        cumtps = cp.cumsum(y_sorted)
        precision = cumtps / idx
        recall = cumtps / total_pos
        # Threshold boundaries: i is a boundary iff i == N-1 OR
        # col_sorted[i] != col_sorted[i+1].
        is_boundary = cp.concatenate([
            col_sorted[:-1] != col_sorted[1:],
            cp.array([True]),
        ])
        recall_b = recall[is_boundary]
        precision_b = precision[is_boundary]
        # delta_recall[0] anchors at recall=0 (matches sklearn AP definition).
        delta_recall = cp.concatenate([
            cp.array([recall_b[0]]),
            recall_b[1:] - recall_b[:-1],
        ])
        aps[j] = cp.sum(delta_recall * precision_b)
    return aps


# ----------------------------------------------------------------------------------------------------------------------------
# Auto-dispatching wrappers: pick GPU or CPU based on (N, M) + availability
# ----------------------------------------------------------------------------------------------------------------------------


def _normalize_scores_2d(y_score: np.ndarray) -> np.ndarray:
    """Promote ``(N,)`` -> ``(N, 1)``. Pass-through ``(N, M)``."""
    if y_score.ndim == 1:
        return y_score[:, np.newaxis]
    return y_score


def compute_batch_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    force_backend: Optional[str] = None,
) -> np.ndarray:
    """RMSE per column with auto GPU/CPU dispatch.

    Returns an ``(M,)`` numpy array (always host-side; the GPU result
    is copied back via ``cp.asnumpy``) so callers don't need to know
    which backend ran.

    Dispatch:
      - ``force_backend='gpu'``: always GPU (raises if cupy missing).
      - ``force_backend='cpu'``: always numpy reference.
      - ``None`` (default): GPU iff cupy + CUDA visible AND N >=
        ``_GPU_BATCH_THRESHOLD_N``; else CPU.
    """
    yt = np.asarray(y_true)
    yp = _normalize_scores_2d(np.asarray(y_pred))
    N = yp.shape[0]

    use_gpu = _resolve_backend(force_backend, N, yp.shape[1])
    if use_gpu:
        import cupy as cp  # lazy
        out = gpu_multiple_rmse_scores(yt, yp)
        return cp.asnumpy(out)
    # CPU reference
    if yt.ndim == 1:
        yt = yt[:, np.newaxis]
    return np.sqrt(np.mean((yt - yp) ** 2.0, axis=0))


def compute_batch_aucs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    force_backend: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Both ROC AUC and PR AUC per column, auto GPU/CPU dispatch.

    Returns ``(roc_aucs, pr_aucs)`` as host-side numpy arrays of shape
    ``(M,)``. NaNs are returned for label columns where ROC/PR is
    undefined (single-class).

    Dispatch policy: same as ``compute_batch_rmse`` — GPU when
    ``cupy`` is present, a CUDA device is visible, and N exceeds the
    threshold (default 100k). Override via ``force_backend``.
    """
    yt = np.asarray(y_true)
    ys = _normalize_scores_2d(np.asarray(y_score))
    N, M = ys.shape

    use_gpu = _resolve_backend(force_backend, N, M)
    if use_gpu:
        import cupy as cp  # lazy
        roc = cp.asnumpy(gpu_multiple_roc_auc_scores(yt, ys))
        pr = cp.asnumpy(gpu_multiple_pr_auc_scores(yt, ys))
        return roc, pr
    # CPU loop over per-column ``fast_aucs`` (returns roc, pr in one pass).
    roc = np.empty(M, dtype=np.float64)
    pr = np.empty(M, dtype=np.float64)
    for j in range(M):
        roc[j], pr[j] = fast_aucs(yt, ys[:, j])
    return roc, pr


def _resolve_backend(force: Optional[str], N: int, M: int) -> bool:
    """Return True iff the GPU path should be used. ``force`` may be
    ``'gpu'`` / ``'cpu'`` / ``None``."""
    if force == "gpu":
        if not is_gpu_metrics_available():
            raise RuntimeError(
                "compute_batch_*: force_backend='gpu' but no GPU is "
                "available (cupy missing or no CUDA device)."
            )
        return True
    if force == "cpu":
        return False
    if force is not None:
        raise ValueError(f"force_backend must be 'gpu', 'cpu', or None; got {force!r}")
    # Auto: GPU iff available + over thresholds.
    return (
        is_gpu_metrics_available()
        and N >= _GPU_BATCH_THRESHOLD_N
        and M >= _GPU_BATCH_THRESHOLD_M
    )


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_precision(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0):
    # storage inits
    allpreds = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)
    # count stats
    for true_class, predicted_class in zip(y_true, y_pred):
        if 0 <= predicted_class < nclasses:
            allpreds[predicted_class] += 1
            if predicted_class == true_class:
                hits[predicted_class] += 1
    precisions = hits / allpreds
    return precisions[-1]


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_classification_report(y_true: np.ndarray, y_pred: np.ndarray, nclasses: int = 2, zero_division: int = 0):
    """Custom classification report, proof of concept."""

    N_AVG_ARRAYS = 3  # precisions, recalls, f1s

    # storage inits
    weighted_averages = np.empty(N_AVG_ARRAYS, dtype=np.float64)
    macro_averages = np.empty(N_AVG_ARRAYS, dtype=np.float64)
    supports = np.zeros(nclasses, dtype=np.int64)
    allpreds = np.zeros(nclasses, dtype=np.int64)
    misses = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)

    # count stats
    for true_class, predicted_class in zip(y_true, y_pred):
        # Bounds-check both labels: out-of-range values are silently dropped rather than
        # triggering a numba-level buffer overflow / segfault.
        if 0 <= true_class < nclasses:
            supports[true_class] += 1
        if 0 <= predicted_class < nclasses:
            allpreds[predicted_class] += 1
            if predicted_class == true_class:
                hits[predicted_class] += 1
            else:
                misses[predicted_class] += 1

    # main calcs
    accuracy = hits.sum() / len(y_true)

    # Balanced accuracy: classes absent from y_true (supports==0) are EXCLUDED from
    # the mean rather than contributing zero_division. sklearn.metrics.balanced_accuracy_score
    # computes mean recall over present classes only — matching that semantics.
    present_mask = supports > 0
    if present_mask.any():
        per_class_recall = np.empty(nclasses, dtype=np.float64)
        for c in range(nclasses):
            per_class_recall[c] = hits[c] / supports[c] if supports[c] > 0 else 0.0
        balanced_accuracy = per_class_recall[present_mask].mean()
    else:
        balanced_accuracy = 0.0

    recalls = hits / supports
    precisions = hits / allpreds
    f1s = 2 * (precisions * recalls) / (precisions + recalls)

    # Weighted averages must divide by supports.sum() (== number of labeled samples with
    # in-range class ids), NOT len(y_true): out-of-range labels were dropped above, so
    # dividing by the raw length under-reports the weighted mean proportionally to the
    # OOB fraction.
    support_total = supports.sum()
    weight_denom = support_total if support_total > 0 else 1

    # fix nans & compute averages
    i = 0
    for arr in (precisions, recalls, f1s):
        np.nan_to_num(arr, copy=False, nan=zero_division)
        weighted_averages[i] = (arr * supports).sum() / weight_denom
        macro_averages[i] = arr.mean()
        i += 1

    return hits, misses, accuracy, balanced_accuracy, supports, precisions, recalls, f1s, macro_averages, weighted_averages


# ============================================================================
# Multi-label numba metrics — 2026-04-24
# ============================================================================
#
# All three accept either (N, K) binary indicator matrices or (N,) binary
# arrays (auto-reshaped to (N, 1) by the public wrappers).
#
# Sequential variants are the default. Parallel variants (`@njit(parallel=True)`)
# are auto-selected by the public wrapper when ``N * K > 1_000_000`` —
# benchmarked threshold on Win32 where `numba.prange` cold-spawn cost is
# ~40-80ms (rules out small-frame parallelism).
#
# All three follow sklearn semantics:
# - `hamming_loss`: mean fraction of incorrect labels (lower is better)
# - `subset_accuracy`: fraction of samples where ALL labels match (exact match)
# - `jaccard_score_multilabel`: per-row averaged |y_true & y_pred| / |y_true | y_pred|;
#   empty-union row counts as 1.0 (defined as "both empty = perfect" — sklearn
#   `jaccard_score(average='samples')` raises in that case unless `zero_division`
#   is explicit; we pick 1.0 as the well-defined choice).


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_hamming_loss_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sequential mean-mismatch fraction. Both inputs (N, K) uint8."""
    N, K = y_true.shape
    err = 0.0
    for i in range(N):
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                err += 1.0
    return err / (N * K)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_hamming_loss_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Parallel variant. Auto-selected by hamming_loss() when N*K > 1M."""
    N, K = y_true.shape
    err_per_row = np.zeros(N, dtype=np.float64)
    for i in numba.prange(N):
        local = 0.0
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                local += 1.0
        err_per_row[i] = local / K
    return err_per_row.mean()


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_subset_accuracy_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Row-wise all-equal then mean. Both inputs (N, K) uint8."""
    N, K = y_true.shape
    correct = 0.0
    for i in range(N):
        all_eq = True
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                all_eq = False
                break
        if all_eq:
            correct += 1.0
    return correct / N


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_subset_accuracy_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Parallel subset accuracy. ~5.6× faster than seq at N=1M.
    Public ``subset_accuracy`` wrapper auto-selects above N=50k."""
    N, K = y_true.shape
    correct = 0
    for i in numba.prange(N):
        all_eq = True
        for j in range(K):
            if y_true[i, j] != y_pred[i, j]:
                all_eq = False
                break
        if all_eq:
            correct += 1
    return correct / N


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_jaccard_score_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Per-row Jaccard (|A∩B|/|A∪B|), averaged. Empty-union → 1.0."""
    N, K = y_true.shape
    total = 0.0
    for i in range(N):
        intersect = 0.0
        union = 0.0
        for j in range(K):
            t = y_true[i, j]
            p = y_pred[i, j]
            if t == 1 and p == 1:
                intersect += 1.0
            if t == 1 or p == 1:
                union += 1.0
        if union > 0:
            total += intersect / union
        else:
            total += 1.0  # both empty — perfect by definition
    return total / N


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_jaccard_score_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Parallel per-row Jaccard. ~6.2× faster than seq at N=1M.
    Public ``jaccard_score_multilabel`` wrapper auto-selects above N=50k."""
    N, K = y_true.shape
    total = 0.0
    for i in numba.prange(N):
        intersect = 0.0
        union = 0.0
        for j in range(K):
            t = y_true[i, j]
            p = y_pred[i, j]
            if t == 1 and p == 1:
                intersect += 1.0
            if t == 1 or p == 1:
                union += 1.0
        if union > 0:
            total += intersect / union
        else:
            total += 1.0
    return total / N


@numba.njit(**NUMBA_NJIT_PARAMS)
def _popcount64(x: np.uint64) -> np.int64:
    """Population-count for uint64 — Hacker's Delight bit-twiddle.

    Numba doesn't expose `__builtin_popcountll` directly; this 5-instruction
    sequence is ~as fast as the intrinsic on modern x86-64. Used by the
    bitmap-Jaccard fast path for K≤64 multilabel arrays.
    """
    x = x - ((x >> 1) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> 56) & np.int64(0x7F)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_jaccard_bitmap_seq(y_true_packed: np.ndarray, y_pred_packed: np.ndarray, K: int) -> float:
    """Bitmap Jaccard via popcount — ~10-50× faster than elementwise loop on K≤64.

    Inputs are already packed uint64 of shape (N,) — caller's responsibility
    to pack via ``np.packbits`` and view-as-uint64. K is the original number
    of labels (needed for empty-union detection — packed zero means all-zero
    labels, regardless of K).
    """
    N = y_true_packed.shape[0]
    total = 0.0
    for i in range(N):
        t = y_true_packed[i]
        p = y_pred_packed[i]
        intersect = _popcount64(t & p)
        union = _popcount64(t | p)
        if union > 0:
            total += intersect / union
        else:
            total += 1.0  # both empty
    return total / N


def _can_use_bitmap_jaccard(K: int) -> bool:
    """Bitmap Jaccard fits if 16 <= K <= 64 (single uint64 per row).

    K threshold benchmarks (N=200_000, jaccard_score_multilabel, Win32 Anaconda 3.11):
        K=3  : bitmap 22ms,  elem  4ms — bitmap LOSES 5x (pack overhead)
        K=16 : bitmap 21ms,  elem 25ms — bitmap wins 1.2x
        K=32 : bitmap 23ms,  elem 52ms — bitmap wins 2.3x
        K=64 : bitmap 12ms,  elem 101ms — bitmap wins 8.6x

    Cutoff at K=16 (~breakeven) means the elementwise loop wins for the
    common 3-5-label case and bitmap kicks in for tag-cloud cases (K>=16).
    """
    return 16 <= K <= 64


def _pack_for_bitmap(arr: np.ndarray) -> np.ndarray:
    """Pack a (N, K) uint8 binary array into (N,) uint64.

    Handles K not multiple of 8 by zero-padding to next 64-bit boundary.
    Excess bits are zero — they contribute 0 to popcount, so safe.
    """
    N, K = arr.shape
    # Pad to 64 bits per row (K' = ceil(K, 64) but capped at 64).
    if K < 64:
        padded = np.zeros((N, 64), dtype=np.uint8)
        padded[:, :K] = arr
    else:
        padded = arr  # K == 64 exactly
    # packbits packs into uint8s big-endian within each byte; then view as uint64.
    packed_u8 = np.packbits(padded, axis=1)  # (N, 8) uint8
    return packed_u8.view(np.uint64).ravel()  # (N,) uint64


def _coerce_multilabel_array(arr) -> np.ndarray:
    """Single-pass cast to contiguous uint8 (N, K). Reshape (N,) → (N, 1)."""
    a = np.ascontiguousarray(np.asarray(arr), dtype=np.uint8)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"multilabel array must be 1-D or 2-D, got shape {a.shape}")


def _validate_multilabel_pair(y_true, y_pred) -> tuple:
    """Coerce + validate y_true / y_pred shape match BEFORE calling numba.

    Numba @njit kernels do not bounds-check inner loops; passing arrays
    of mismatched second-dimension would silently read garbage memory.
    Public wrappers MUST validate up-front.
    """
    yt = _coerce_multilabel_array(y_true)
    yp = _coerce_multilabel_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true shape {yt.shape} != y_pred shape {yp.shape}; "
            "multilabel metrics require matching shapes."
        )
    return yt, yp


def hamming_loss(y_true, y_pred) -> float:
    """sklearn-compatible Hamming loss for multilabel targets.

    Accepts (N,) binary or (N, K) multilabel. For N*K > 1M, auto-routes
    to the parallel numba variant.

    Same return-value semantics as ``sklearn.metrics.hamming_loss``.
    """
    yt, yp = _validate_multilabel_pair(y_true, y_pred)
    if yt.shape[0] * yt.shape[1] > 1_000_000:
        return _fast_hamming_loss_par(yt, yp)
    return _fast_hamming_loss_seq(yt, yp)


def subset_accuracy(y_true, y_pred) -> float:
    """Subset accuracy (a.k.a. exact-match) for multilabel targets.

    Equivalent to ``sklearn.metrics.accuracy_score(y_true, y_pred)`` on
    multilabel inputs (sklearn does row-wise all-equal under the hood).
    Auto-dispatches to the parallel kernel above N=50k rows
    (~5.6× speedup at 1M rows on an 8-thread runtime).
    """
    yt, yp = _validate_multilabel_pair(y_true, y_pred)
    if yt.shape[0] >= _PARALLEL_MULTILABEL_THRESHOLD:
        return _fast_subset_accuracy_par(yt, yp)
    return _fast_subset_accuracy_seq(yt, yp)


def jaccard_score_multilabel(y_true, y_pred, *, force_elementwise: bool = False) -> float:
    """Per-row averaged Jaccard score for multilabel targets.

    Equivalent to ``sklearn.metrics.jaccard_score(y_true, y_pred, average='samples')``
    with the well-defined choice of 1.0 for empty-union rows
    (sklearn's default raises a ``DivisionWarning`` on those).

    Performance: when ``K ≤ 64`` (the common case in multilabel tagging),
    uses a bitmap-popcount fast path (~10-50× faster than the elementwise
    loop) — these are sequential because the per-row work is tiny on
    bit-packed input. For elementwise mode (``force_elementwise=True``
    or K>64), the parallel kernel kicks in above N=50k rows
    (~6× speedup at 1M rows). Set ``force_elementwise=True`` to bypass
    the bitmap path — useful for benchmarks and verifying numerical
    equivalence between the two paths.
    """
    yt, yp = _validate_multilabel_pair(y_true, y_pred)
    K = yt.shape[1]
    if not force_elementwise and _can_use_bitmap_jaccard(K):
        yt_packed = _pack_for_bitmap(yt)
        yp_packed = _pack_for_bitmap(yp)
        return _fast_jaccard_bitmap_seq(yt_packed, yp_packed, K)
    if yt.shape[0] >= _PARALLEL_MULTILABEL_THRESHOLD:
        return _fast_jaccard_score_par(yt, yp)
    return _fast_jaccard_score_seq(yt, yp)


# Closed set of title-metrics tokens recognised by render_title_metrics() and
# validated by ReportingConfig at construction time. Order in DEFAULT matches
# the historical title layout (ICE first, then BR with decomposition, ECE between
# BR and CMAEW per spec, then LL, ROC_AUC, PR_AUC). Adding a new token requires:
# 1) extending TITLE_METRIC_TOKENS, 2) adding a render_* branch in
# render_title_metric_token, 3) updating ReportingConfig validator allowed-set.
TITLE_METRIC_TOKENS: frozenset = frozenset({
    "ICE", "BR", "BR_DECOMP", "ECE", "CMAEW",
    "COV", "LL", "ROC_AUC", "PR_AUC", "DENS",
})
DEFAULT_TITLE_METRICS_TOKENS: tuple = ("ICE", "BR_DECOMP", "ECE", "CMAEW", "LL", "ROC_AUC", "PR_AUC")


def render_title_metric_token(
    token: str,
    *,
    ndigits: int,
    ice: float,
    brier_loss: float,
    ece: float,
    brier_reliability: float,
    brier_resolution: float,
    brier_uncertainty: float,
    calibration_mae: float,
    calibration_std: float,
    use_weights: bool,
    calibration_coverage: float,
    nbins: int,
    ll: Optional[float],
    max_hits: int,
    min_hits: int,
    roc_auc: float,
    mean_group_roc_auc: Optional[float],
    pr_auc: float,
    mean_group_pr_auc: Optional[float],
    precision: float,
    recall: float,
    f1: float,
) -> str:
    """Render one calibration-report title fragment for a token.

    Returns the empty string when the token has no usable data (e.g. LL with
    single-class y_true). Tokens are validated by ReportingConfig at config
    construction time, so unknown tokens cannot reach this function in practice -
    the final ``return ""`` is a defence-in-depth against bypassed validation.

    Percent-suffixed metrics (BR / BR_DECOMP / ECE / CMAEW / PR / RE / F1)
    render with one fewer decimal than ``ndigits`` -- ``%`` already adds
    two extra characters per metric and the headline still reads cleanly
    at ``9.1%`` instead of ``9.10%``. Bare-scalar metrics (ICE, LL,
    ROC_AUC, PR_AUC) keep ``ndigits`` since precision matters more there
    (single-decimal AUC squashes 0.974 vs 0.976 to "1.0"). User feedback
    2026-05-04. ``COV`` derives its own precision from log10(nbins) and
    is unchanged.
    """
    pct_digits = max(0, ndigits - 1)
    if token == "ICE":
        return f"ICE={ice:.{ndigits}f}"
    if token == "BR":
        return f"BR={brier_loss * 100:.{pct_digits}f}%"
    if token == "BR_DECOMP":
        # 2026-04-27 Session 7 batch 8 (user feedback): compact form
        # of the Brier decomposition. The math is BR = REL - RES + UNC
        # (Murphy 1973), so the most informative compact rendering is
        # the actual signed-sum: ``BR=X%(RL<rel>%+U<unc>%-RS<res>%)`` where
        # RL = ReLiability (calibration error, lower is better),
        # U  = Uncertainty (irreducible noise = base_rate * (1-base_rate)),
        # RS = ReSolution (subtractive: how well bins separate from base
        # rate, higher is better). Reads naturally as the formula with
        # signs preserved, ~30% shorter than the labelled form.
        return (
            f"BR={brier_loss * 100:.{pct_digits}f}%"
            f"(RL{brier_reliability * 100:.{pct_digits}f}%"
            f"+U{brier_uncertainty * 100:.{pct_digits}f}%"
            f"-RS{brier_resolution * 100:.{pct_digits}f}%)"
        )
    if token == "ECE":
        return f"ECE={ece * 100:.{pct_digits}f}%"
    if token == "CMAEW":
        return (
            f"CMAE{'W' if use_weights else ''}="
            f"{calibration_mae * 100:.{pct_digits}f}%"
            f"±{calibration_std * 100:.{pct_digits}f}%"
        )
    if token == "COV":
        # log10(nbins) decides COV's decimal precision; matches pre-template behaviour.
        cov_prec = max(0, int(np.log10(max(nbins, 1))))
        return f"COV={calibration_coverage * 100:.{cov_prec}f}%"
    if token == "LL":
        if ll is None:
            return ""
        return f"LL={ll:.{ndigits}f}"
    if token == "DENS":
        return f"DENS=[{max_hits:_};{min_hits:_}]"
    if token == "ROC_AUC":
        suffix = ""
        if mean_group_roc_auc is not None and not np.isnan(mean_group_roc_auc):
            suffix = f"[{mean_group_roc_auc:.{ndigits}f}]"
        if np.isnan(roc_auc):
            return f"ROC AUC=N/A{suffix}"
        return f"ROC AUC={roc_auc:.{ndigits}f}{suffix}"
    if token == "PR_AUC":
        suffix = ""
        if mean_group_pr_auc is not None and not np.isnan(mean_group_pr_auc):
            suffix = f"[{mean_group_pr_auc:.{ndigits}f}]"
        if np.isnan(pr_auc):
            base = f"PR AUC=N/A{suffix}"
        else:
            base = f"PR AUC={pr_auc:.{ndigits}f}{suffix}"
        return (
            f"{base}, PR={precision * 100:.{pct_digits}f}%,"
            f"RE={recall * 100:.{pct_digits}f}%,F1={f1 * 100:.{pct_digits}f}%"
        )
    return ""


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_calibration_binning(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100):
    """Computes bins of predicted vs actual events frequencies. Corresponds to sklearn's UNIFORM strategy."""

    pockets_predicted = np.zeros(nbins, dtype=np.int64)
    pockets_true = np.zeros(nbins, dtype=np.int64)

    # compute span

    min_val, max_val = 1.0, 0.0
    for predicted_prob in y_pred:
        if predicted_prob > max_val:
            max_val = predicted_prob
        if predicted_prob < min_val:
            min_val = predicted_prob
    span = max_val - min_val

    if span > 0:
        multiplier = (nbins - 1) / span
        for true_class, predicted_prob in zip(y_true, y_pred):
            ind = floor((predicted_prob - min_val) * multiplier)
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class
    else:
        ind = 0
        for true_class, predicted_prob in zip(y_true, y_pred):
            pockets_predicted[ind] += 1
            pockets_true[ind] += true_class

    idx = np.nonzero(pockets_predicted > 0)[0]

    hits = pockets_predicted[idx]
    if len(hits) > 0:
        freqs_predicted, freqs_true = (min_val + (np.arange(nbins)[idx] + 0.5) * span / nbins).astype(np.float64), pockets_true[idx] / pockets_predicted[idx]
    else:
        freqs_predicted, freqs_true = np.array((), dtype=np.float64), np.array((), dtype=np.float64)

    return freqs_predicted, freqs_true, hits


def _close_unless_interactive(figs, was_shown: bool) -> None:
    """Close one or more matplotlib figures unless we're running inside a
    real IPython / Jupyter kernel.

    Parameters
    ----------
    figs : Figure or iterable of Figures
        The figure(s) to potentially close. ``None`` and empty iterables
        are silently ignored.
    was_shown : bool
        Whether the caller already ran ``plt.ion(); plt.show()``. When
        ``False`` we always close (caller decided not to display); when
        ``True`` we close only outside an interactive Python kernel
        (the Jupyter / IPython inline backends register the figure with
        their display hooks during ``plt.show()``, so closing right
        after preserves the inline render).

    Detection: a true IPython kernel sets the ``__IPYTHON__`` builtin;
    the bare REPL sets ``sys.ps1``. The naive ``"IPython" in
    sys.modules`` heuristic is unreliable -- matplotlib + many ML
    libraries drag IPython into ``sys.modules`` as a transitive
    dependency even from plain Python scripts, giving false positives
    that cause figures to leak across the training-suite calls (~12
    cal-plots / fit, tripping the matplotlib ``More than 20 figures
    have been opened`` warning + a real per-fit slowdown).

    Background: 2026-05-09 leak fix on top of show_calibration_plot;
    extracted as a helper after the same pattern was needed in
    feature_importance.py and training/evaluation.py.
    """
    try:
        is_interactive = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        is_interactive = hasattr(sys, "ps1")
    if is_interactive and was_shown:
        return  # let the kernel's inline display keep the rendered figure
    # Either we never showed (no display happened), OR we're in a
    # script / GUI session where leaving figures alive just leaks them.
    if figs is None:
        return
    if hasattr(figs, "savefig"):  # single Figure
        plt.close(figs)
        return
    for f in figs:
        plt.close(f)


def show_calibration_plot(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    show_plots: bool = True,
    plot_file: str = "",
    plot_title: str = "",
    figsize: tuple = (12, 6),
    backend: str = "matplotlib",
    label_freq: str = "Observed Frequency",
    label_perfect: str = "Perfect",
    label_real: str = "Real",
    label_prob: str = "Predicted Probability",
    colorbar_label: str = "Bin population",
    use_size: bool = False,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "linear",
    show_inline_population_labels: bool = True,
    label_histogram: str = "Bin population",
    plot_outputs: Optional[str] = None,
    base_path: Optional[str] = None,
    dpi: Optional[int] = None,
):
    """Plots reliability digaram from the binned predictions.

    With ``show_prob_histogram=True`` (default) a probability-distribution
    histogram is drawn under the reliability scatter, sharing the X axis.
    Histogram bar heights are bin populations (``hits``) and bars are
    coloured by population using the same ``RdYlBu`` colormap as the
    calibration scatter, so the bottom plot reads consistently with the
    top one (a single bar that matches the colorbar tells you "this bin
    holds N samples"). Y-scale defaults to ``linear`` -- the legacy
    ``"auto"`` mode (log iff max/min skew > 100) flipped to log on
    skewed distributions and made empty bins look populated; pass
    ``prob_histogram_yscale="log"`` if you genuinely need log.
    Inline per-bin population annotations (the small text labels next to each
    scatter point) are independently controlled by
    ``show_inline_population_labels`` so users can keep both, drop both, or
    keep only one.
    """

    assert backend in ("plotly", "matplotlib")
    assert prob_histogram_yscale in ("auto", "log", "linear")

    # 2026-05-09: short-circuit when there is NO plot consumer. The
    # ``show_plots=True`` default expresses "render if a human can see
    # it"; in a script / CI / fuzz process (no IPython kernel, no
    # ``sys.ps1``) that's a contradiction -- ``plt.show()`` is a
    # documented no-op (UserWarning: ``FigureCanvasAgg is non-
    # interactive, and thus cannot be shown``) AND nothing is written
    # to disk because ``plot_file`` is empty. The figure render is
    # 130-180 ms / call and dominates the warm wall on plot-heavy
    # multilabel fits (e.g. c0104: 12 cal-plots / fit -> 1.6-2.2 s of
    # pure waste per fit). Caller can opt back in by passing a real
    # ``plot_file`` (saves to disk regardless of session) OR by
    # running inside IPython / Jupyter where the inline display
    # backends will pick up the figure.
    if show_plots and not plot_file:
        try:
            _is_interactive = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            _is_interactive = hasattr(sys, "ps1")
        if not _is_interactive:
            return None

    # 2026-05-08: opt-in DSL render path. When ``plot_outputs`` + ``base_path``
    # are supplied, route through the shared spec pipeline (matplotlib +
    # plotly + any future backends via the same DSL). Default behaviour
    # preserved -- legacy callers see no change.
    if plot_outputs and base_path:
        from mlframe.reporting.charts.calibration import build_calibration_spec
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
        spec = build_calibration_spec(
            freqs_predicted, freqs_true, hits,
            plot_title=plot_title,
            show_prob_histogram=show_prob_histogram,
            show_inline_population_labels=show_inline_population_labels,
            label_freq=label_freq, label_prob=label_prob,
            label_histogram=label_histogram,
            colorbar_label=colorbar_label,
            figsize=figsize,
        )
        render_and_save(spec, parse_plot_output_dsl(plot_outputs), base_path)
        return None

    x_min, x_max = np.min(freqs_predicted), np.max(freqs_predicted)

    # nbins-derived bar width: use the bin centre spacing as the bar width.
    # When all bins have data this matches fast_calibration_binning's geometry;
    # if some bins are empty (sparse hits filter at metrics.py:600) we fall back
    # to the average centre spacing across present bins so bars don't overlap.
    if len(freqs_predicted) > 1:
        _bar_width = float(np.mean(np.diff(np.sort(freqs_predicted))))
    else:
        _bar_width = 0.05

    def _resolve_yscale(hits_arr) -> str:
        """auto -> log iff max/min skew > 100, else linear. Explicit modes pass through."""
        if prob_histogram_yscale != "auto":
            return prob_histogram_yscale
        if len(hits_arr) == 0:
            return "linear"
        max_h = float(np.max(hits_arr))
        min_h = max(float(np.min(hits_arr)), 1.0)
        return "log" if (max_h / min_h) > 100.0 else "linear"

    if backend == "matplotlib":
        # Function to format hits values with B, M, K suffixes
        def format_population(n):
            if n >= 1e9:
                return f"{n/1e9:.1f}B"
            elif n >= 1e6:
                return f"{n/1e6:.1f}M"
            elif n >= 1e3:
                return f"{n/1e3:.1f}K"
            else:
                return f"{n:.0f}"

        def _draw_calibration_axes(ax, fig, draw_xlabel: bool, *, cbar_ax=None):
            """Render the reliability scatter + perfect-calibration line + colorbar on ``ax``.

            ``cbar_ax`` (optional) is the axes list / single ax the
            colorbar attaches to. When the calibration plot stacks
            with a histogram below, pass ``[ax_main, ax_hist]`` so the
            colorbar spans both — otherwise the colorbar steals
            horizontal space from only the calibration axes, making
            the histogram's plot-area visibly wider and breaking the
            shared-X alignment (2026-04-27 user feedback).
            """
            cm = matplotlib.colormaps["RdYlBu"]
            sc = ax.scatter(
                x=freqs_predicted, y=freqs_true, marker="o",
                s=5000 * hits / hits.sum(), c=hits, label=label_freq, cmap=cm,
            )
            ax.plot(
                [min(freqs_predicted), max(freqs_predicted)],
                [min(freqs_predicted), max(freqs_predicted)],
                "g--", label=label_perfect,
            )
            if draw_xlabel:
                ax.set_xlabel(label_prob)
            ax.set_ylabel(label_freq)
            cbar = fig.colorbar(sc, ax=(cbar_ax if cbar_ax is not None else ax))
            cbar.set_label(colorbar_label)
            if show_inline_population_labels:
                vertical_offset = 0.02
                for x, y, hit in zip(freqs_predicted, freqs_true, hits):
                    ax.text(
                        x, y + vertical_offset, format_population(hit),
                        fontsize=8, ha="right", va="bottom",
                    )

        def _draw_histogram_axes(ax):
            """Render the predicted-probability histogram under the calibration axes.

            Bars are coloured by the same ``RdYlBu`` colormap + same
            normalisation as the top calibration scatter, so the colorbar
            reads consistently across both subplots: a tall blue bar in
            the histogram matches the blue scatter bubble at the same X
            (both encode "this bin is populated").
            """
            cm = matplotlib.colormaps["RdYlBu"]
            # Same normalisation as the scatter (which uses ``c=hits`` and
            # auto-normalises across the value range). Reproduce that here:
            _h_min = float(np.min(hits)) if len(hits) else 0.0
            _h_max = float(np.max(hits)) if len(hits) else 1.0
            if _h_max <= _h_min:
                _h_max = _h_min + 1.0
            _bar_colors = cm((hits - _h_min) / (_h_max - _h_min))
            ax.bar(
                freqs_predicted, hits,
                width=_bar_width, align="center",
                color=_bar_colors, edgecolor="white", linewidth=0.5,
            )
            ax.set_xlabel(label_prob)
            ax.set_ylabel(label_histogram)
            ax.set_yscale(_resolve_yscale(hits))

        # Save-only fast path: bypass pyplot + GUI backend (Qt init costs ~1.7s per call).
        # Using Figure + FigureCanvasAgg directly drops this to ~0.2s. Also thread-safe,
        # which matters for parallel val/test evaluation.
        #
        # 2026-05-09: tried extending the guard to also fire for
        # ``show_plots=True`` outside an IPython kernel, hypothesising
        # that the GUI-bound path's pyplot+Qt overhead was wasted in
        # script / CI runs. Two A/B benches on c0104 (5 fits each)
        # showed 2733 +- 236 ms and 3145 +- 723 ms vs the
        # close-fix-only baseline of 2275 +- 185 ms -- a regression,
        # not a win. Revert; keep the original guard. Closing figures
        # in the interactive path (already done above) is enough.
        if plot_file and not show_plots:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            # 2026-05-11: layout was ``"constrained"`` per 2026-04-27 batch 8
            # to handle the multi-axis colorbar (``ax=[ax_main, ax_hist]``)
            # without tight_layout's compat warning. But the Wave 4 1M-row
            # fuzz aggregate showed 146 s across 108 calls (1.35 s / call)
            # in this path -- constrained_layout's iterative bbox solver
            # was a needless ~170 ms / call on the calibration plot
            # geometry.
            # Visual A/B on the actual chart (12x6 figsize, 2-line title,
            # multi-axis colorbar, sharex'd hist below scatter) showed
            # matplotlib's default geometry fits everything with zero
            # clipping; the rightmost scatter dot stays inside the
            # colorbar boundary. Switched to ``layout=None``.
            # Bench: 430 ms -> 257 ms (1.67x, ~170 ms / call saving).
            # See profiling/bench_calibration_layout.py for the A/B and
            # D:/Temp/calib_ab_*.png for visual proof.
            # 2026-05-11: honour ``dpi`` kwarg so ReportingConfig.plot_dpi
            # propagates here; ``None`` defers to matplotlib's default.
            _fig_kwargs = {"figsize": figsize, "layout": None}
            if dpi is not None:
                _fig_kwargs["dpi"] = dpi
            fig = Figure(**_fig_kwargs)
            FigureCanvasAgg(fig)
            if show_prob_histogram:
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_main = fig.add_subplot(gs[0, 0])
                ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_main)
                # Colorbar spans BOTH axes so each subplot loses the
                # same horizontal slice -> X-axes stay aligned via
                # sharex (was: colorbar attached only to ax_main,
                # making ax_hist visually wider — user feedback 2026-04-27).
                _draw_calibration_axes(ax_main, fig, draw_xlabel=False,
                                       cbar_ax=[ax_main, ax_hist])
                _draw_histogram_axes(ax_hist)
                # hide top axes' x tick labels since hist below carries them via sharex
                plt.setp(ax_main.get_xticklabels(), visible=False)
                if plot_title:
                    ax_main.set_title(plot_title)
            else:
                ax = fig.add_subplot(1, 1, 1)
                _draw_calibration_axes(ax, fig, draw_xlabel=True)
                if plot_title:
                    ax.set_title(plot_title)
            # constrained_layout handles spacing automatically — no
            # tight_layout() (which warns + mis-shapes colorbar).
            fig.savefig(plot_file)
            return fig

        # Interactive path (show_plots=True) — keep pyplot so the GUI window is managed.
        # 2026-05-11: layout="constrained" -> layout=None (same rationale as the
        # save-only path above: 1.67x faster, visually equivalent on this
        # 12x6 figsize + multi-axis colorbar + 2-line title geometry).
        if show_prob_histogram:
            _subplots_kwargs = dict(
                nrows=2, ncols=1,
                figsize=figsize,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
                layout=None,
            )
            if dpi is not None:
                _subplots_kwargs["dpi"] = dpi
            fig, (ax_main, ax_hist) = plt.subplots(**_subplots_kwargs)
            # Colorbar spans both subplots — see _draw_calibration_axes
            # docstring for why (X-axis alignment under sharex).
            _draw_calibration_axes(ax_main, fig, draw_xlabel=False,
                                   cbar_ax=[ax_main, ax_hist])
            _draw_histogram_axes(ax_hist)
            plt.setp(ax_main.get_xticklabels(), visible=False)
            if plot_title:
                ax_main.set_title(plot_title)
        else:
            _fig_kwargs2 = {"figsize": figsize, "layout": None}
            if dpi is not None:
                _fig_kwargs2["dpi"] = dpi
            fig = plt.figure(**_fig_kwargs2)
            ax = fig.add_subplot(1, 1, 1)
            _draw_calibration_axes(ax, fig, draw_xlabel=True)
            if plot_title:
                ax.set_title(plot_title)

        # Default geometry fits this layout (verified via visual A/B
        # against constrained_layout, see bench_calibration_layout.py).

        if plot_file:
            fig.savefig(plot_file)

        # 2026-05-09: ``show_plots=True`` previously ran ``plt.ion();
        # plt.show()`` and left the figure open. In an automated /
        # headless / fuzz / CI run (no REPL, no Jupyter kernel),
        # ``plt.show()`` is either a no-op (Agg canvas, with
        # ``UserWarning: FigureCanvasAgg is non-interactive``) or
        # opens an unowned Qt window the script cannot dismiss; in
        # both cases the figure stayed alive in pyplot's registry,
        # accumulating across the 12+ cal-plot calls per multilabel
        # fit and tripping ``More than 20 figures have been opened``
        # plus a real per-fit slowdown (each new ``plt.subplots`` has
        # to track all live figures). Detect interactive consumer
        # (Python REPL / IPython / Jupyter kernel) by ``sys.ps1`` and
        # ``IPython`` in ``sys.modules``; if neither is present, close
        # the figure right after the (no-op or fire-and-forget)
        # ``plt.show()``. Inline-display backends (Jupyter inline
        # ``%matplotlib inline``, ipympl) register the figure with the
        # display hooks during ``plt.show()`` BEFORE we close it, so
        # the rendered output is preserved.
        if show_plots:
            plt.ion()
            plt.show()
        _close_unless_interactive(fig, was_shown=show_plots)

    else:

        df = pd.DataFrame(
            {
                label_prob: freqs_predicted,
                label_freq: freqs_true,
                "NCases": hits,
            }
        )
        hover_data = {label_prob: ":.2%", label_freq: ":.2%", "NCases": True}

        if use_size:
            df["size"] = 5000 * hits / hits.sum()
            hover_data["size"] = False

        if show_prob_histogram:
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.75, 0.25],
                vertical_spacing=0.05,
            )
            calib_row, hist_row = 1, 2
        else:
            fig = go.Figure()
            calib_row = hist_row = None

        marker_dict = {"color": df["NCases"], "colorscale": "RdYlBu", "showscale": True}
        if use_size:
            marker_dict["size"] = df["size"]
        scatter_trace = go.Scatter(
            x=df[label_prob],
            y=df[label_freq],
            mode="markers",
            marker=marker_dict,
            name=label_real,
            hovertemplate=f"{label_prob}: %{{x:.2%}}<br>{label_freq}: %{{y:.2%}}<br>NCases: %{{marker.color}}<extra></extra>",
        )
        perfect_trace = go.Scatter(
            x=[x_min, x_max], y=[x_min, x_max],
            line={"color": "green", "dash": "dash"}, name=label_perfect, mode="lines",
        )
        if show_prob_histogram:
            fig.add_trace(scatter_trace, row=calib_row, col=1)
            fig.add_trace(perfect_trace, row=calib_row, col=1)
            fig.add_trace(
                go.Bar(
                    x=df[label_prob], y=df["NCases"], name=label_histogram,
                    marker={"color": "steelblue"}, showlegend=False,
                ),
                row=hist_row, col=1,
            )
            fig.update_yaxes(title_text=label_freq, row=calib_row, col=1)
            fig.update_yaxes(
                title_text=label_histogram,
                type=_resolve_yscale(hits),
                row=hist_row, col=1,
            )
            fig.update_xaxes(title_text=label_prob, row=hist_row, col=1)
        else:
            fig.add_trace(scatter_trace)
            fig.add_trace(perfect_trace)
        fig.update(layout_coloraxis_showscale=False)

        if plot_title:
            fig.update_layout(title=plot_title)

        if plot_file:
            ext = plot_file.split(".")[-1]
            if not ext:
                ext = "png"
            write_image(fig, file=plot_file, format=ext)

        if show_plots:
            fig.show()
    return fig


@numba.njit(**NUMBA_NJIT_PARAMS)
def _max_abs_pct_error_kernel(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, int]:
    """Returns (max MAPE value, count of y_true==0 entries encountered).

    The zero-count is surfaced so the Python wrapper can emit a warning — silently
    swallowing y_true==0 hides the fact that the epsilon fallback dominates the ratio
    and the "percentage" becomes meaningless.
    """
    epsilon = np.finfo(np.float64).eps
    n_zero = 0
    for i in range(len(y_true)):
        if y_true[i] == 0.0:
            n_zero += 1
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.nanmax(mape), n_zero


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _max_abs_pct_error_kernel_par(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, int]:
    """Parallel variant. ~2.3× faster than seq at N=1M.

    NOTE: ``if err > max_err: max_err = err`` inside ``prange`` is a
    race -- numba auto-detects ``+=`` as a reduction but NOT if-based
    max-update; concurrent threads can drop max-updates. Solution:
    per-thread max array + final reduction outside the prange.
    """
    n = len(y_true)
    epsilon = np.finfo(np.float64).eps
    nthr = numba.get_num_threads()
    per_thread_max = np.zeros(nthr, dtype=np.float64)
    n_zero = 0
    for i in numba.prange(n):
        if y_true[i] == 0.0:
            n_zero += 1
        denom = abs(y_true[i])
        if denom < epsilon:
            denom = epsilon
        err = abs(y_pred[i] - y_true[i]) / denom
        # NaN guard: np.nanmax in seq variant skips NaNs; mirror that.
        if err == err:  # not NaN
            tid = numba.get_thread_id()
            if err > per_thread_max[tid]:
                per_thread_max[tid] = err
    max_err = 0.0
    for t in range(nthr):
        if per_thread_max[t] > max_err:
            max_err = per_thread_max[t]
    return max_err, n_zero


def maximum_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Auto seq/par dispatch. Parallel only wins at large N (race-free
    # max via per-thread accumulator + final reduction; lose-band runs
    # to ~200k due to setup cost).
    if len(y_true) >= 500_000:
        value, n_zero = _max_abs_pct_error_kernel_par(y_true, y_pred)
    else:
        value, n_zero = _max_abs_pct_error_kernel(y_true, y_pred)
    if n_zero > 0:
        logger.warning(
            "maximum_absolute_percentage_error: %d of %d y_true entries are zero; "
            "the epsilon fallback makes those ratios dominate the result.",
            n_zero, len(y_true),
        )
    return value


@numba.njit(**NUMBA_NJIT_PARAMS)
def calibration_metrics_from_freqs(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    nbins: int,
    array_size: int,
    use_weights: bool = True,
    use_log_weighting: bool = False,
    use_sqrt_weighting: bool = False,
    use_power_weighting: bool = True,
):
    # Rounding precision must be >= 1 decimal place even for small nbins (previously
    # int(np.log10(5)) == 0 meant integer-rounding, collapsing all bins together).
    _round_prec = max(1, int(np.ceil(np.log10(max(nbins, 2)))))
    calibration_coverage = len(set(np.round(freqs_predicted, _round_prec))) / nbins
    if len(hits) > 0:
        diffs = np.abs((freqs_predicted - freqs_true))
        if use_weights:

            if use_log_weighting:
                weights = np.log1p(hits)
            elif use_sqrt_weighting:
                weights = np.sqrt(hits)
            elif use_power_weighting:
                alpha = 0.8  # adjust between (0, 1)
                weights = hits**alpha
            else:
                weights = hits.astype(np.float64)

            # Normalize weights to sum to 1. The previous +1e-6 constant was arbitrary and
            # mattered whenever weights.sum() was small (few bins, low hits counts) — it
            # biased the weighted MAE toward zero. Guard against the only legitimate zero
            # case explicitly instead.
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum

            calibration_mae = np.sum(diffs * weights)
            calibration_std = np.sqrt(np.sum(((diffs - calibration_mae) ** 2) * weights))
        else:
            calibration_mae = np.mean(diffs)
            calibration_std = np.sqrt(np.mean(((diffs - calibration_mae) ** 2)))
    else:
        calibration_mae, calibration_std = 1.0, 1.0

    return calibration_mae, calibration_std, calibration_coverage


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_ece_and_brier_decomposition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nbins: int,
):
    """ECE plus Murphy 1973 Brier score decomposition.

    Returns ``(ece, reliability, resolution, uncertainty, brier_binned)``.

    ECE = sum_k (n_k / N) * |p_mean_k - acc_k|
    BinnedBrier = REL - RES + UNC      (exact identity by construction)
        REL = sum_k (n_k / N) * (p_mean_k - acc_k)^2
        RES = sum_k (n_k / N) * (acc_k - base_rate)^2
        UNC = base_rate * (1 - base_rate)
    where p_mean_k is the *mean predicted probability* in bin k (NOT the bin
    centre), acc_k is the observed positive rate in bin k, base_rate is the
    overall positive rate.

    The kernel does its own binning over [min(y_pred), max(y_pred)] using the
    same data-adaptive grid as fast_calibration_binning - so ECE/REL bin
    boundaries match CMAEW exactly. Per-bin p_mean (not bin centre) is used so
    the Murphy identity ``BinnedBrier == REL - RES + UNC`` holds exactly to fp
    precision; this matters for the test asserting the identity, and for users
    checking REL when they care about absolute magnitude. Raw Brier (computed
    by ``fast_brier_score_loss``) differs from BinnedBrier by the within-bin
    variance of predictions; that gap shrinks with finer binning.

    Returns 1.0/1.0/0.0/0.0/1.0 on empty input - mirrors degenerate handling
    elsewhere in the calibration pipeline.
    """
    n = len(y_true)
    if n == 0:
        return 1.0, 1.0, 0.0, 0.0, 1.0

    base_rate = 0.0
    for i in range(n):
        base_rate += y_true[i]
    base_rate /= n

    # Min/max span - same data-adaptive grid as fast_calibration_binning.
    min_val = 1.0
    max_val = 0.0
    for i in range(n):
        p = y_pred[i]
        if p > max_val:
            max_val = p
        if p < min_val:
            min_val = p
    span = max_val - min_val

    pred_sum = np.zeros(nbins, dtype=np.float64)
    true_sum = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    if span > 0:
        multiplier = (nbins - 1) / span
        for i in range(n):
            p = y_pred[i]
            ind = int(floor((p - min_val) * multiplier))
            counts[ind] += 1
            pred_sum[ind] += p
            true_sum[ind] += y_true[i]
    else:
        # All predictions identical - one bin holds everything.
        for i in range(n):
            counts[0] += 1
            pred_sum[0] += y_pred[i]
            true_sum[0] += y_true[i]

    ece = 0.0
    reliability = 0.0
    resolution = 0.0
    inv_n = 1.0 / n
    for k in range(nbins):
        if counts[k] == 0:
            continue
        w = counts[k] * inv_n
        p_mean = pred_sum[k] / counts[k]
        acc = true_sum[k] / counts[k]
        diff = p_mean - acc
        ece += w * abs(diff)
        reliability += w * diff * diff
        resolution += w * (acc - base_rate) ** 2
    uncertainty = base_rate * (1.0 - base_rate)
    brier_binned = reliability - resolution + uncertainty
    return ece, reliability, resolution, uncertainty, brier_binned


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray, nbins: int = 100, use_weights: bool = False, verbose: int = 0):
    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    if verbose:
        print(freqs_predicted, freqs_true)
    return calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights
    )


def fast_aucs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute both ROC AUC and PR AUC efficiently."""
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = np.argsort(y_score)[::-1]
    return fast_numba_aucs(y_true=y_true, y_score=y_score, desc_score_indices=desc_score_indices)


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_aucs(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> tuple[float, float]:
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    total_pos = np.sum(y_true_sorted)
    total_neg = len(y_true_sorted) - total_pos
    if total_pos == 0 or total_neg == 0:
        # Single-class data: both ROC AUC and PR AUC are undefined
        return np.nan, np.nan

    # Variables for ROC AUC
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    roc_auc = 0.0

    # Variables for PR AUC. sklearn.average_precision_score computes
    #   AP = sum_n (R_n - R_{n-1}) * P_n
    # starting from R_0 = 0 (implicit anchor). The previous implementation already matches
    # this; we explicitly document the starting (R=0) anchor here. No behavioral change
    # needed — parity test below verifies |our - sklearn| < 1e-8.
    prev_recall = 0.0
    pr_auc = 0.0

    n = len(y_true_sorted)
    for i in range(n):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            # Update ROC AUC
            delta_fps = fps - last_counted_fps
            sum_tps = last_counted_tps + tps
            roc_auc += delta_fps * sum_tps
            last_counted_fps = fps
            last_counted_tps = tps

            # sklearn AP: sum over thresholds of (R_n - R_{n-1}) * P_n (current precision).
            current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            current_recall = tps / total_pos
            delta_recall = current_recall - prev_recall
            pr_auc += delta_recall * current_precision  # Riemann sum
            prev_recall = current_recall

    # Normalize ROC AUC
    denom_roc = tps * fps * 2
    if denom_roc > 0:
        roc_auc /= denom_roc
    else:
        # Should not reach here due to early return, but handle defensively
        roc_auc = np.nan

    return roc_auc, pr_auc


def fast_aucs_per_group(y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
    """
    Compute overall AUCs and per-group AUCs efficiently.

    Returns:
        - Overall ROC AUC
        - Overall PR AUC
        - Dictionary mapping group_id -> (roc_auc, pr_auc)
    """
    if y_score.ndim == 2:
        y_score = y_score[:, -1]

    # Overall AUCs
    desc_score_indices = np.argsort(y_score)[::-1]
    overall_roc_auc, overall_pr_auc = fast_numba_aucs(y_true, y_score, desc_score_indices)

    # Per-group AUCs
    unique_groups = np.unique(group_ids)
    group_aucs = {}

    for group_id in unique_groups:
        group_mask = group_ids == group_id
        group_y_true = y_true[group_mask]
        group_y_score = y_score[group_mask]

        if len(group_y_true) > 1:  # Need at least 2 samples
            group_desc_indices = np.argsort(group_y_score)[::-1]
            roc_auc, pr_auc = fast_numba_aucs(group_y_true, group_y_score, group_desc_indices)
            group_aucs[int(group_id)] = (roc_auc, pr_auc)
        else:
            group_aucs[int(group_id)] = (0.0, 0.0)

    return overall_roc_auc, overall_pr_auc, group_aucs


def fast_aucs_per_group_optimized(y_true: np.ndarray, y_score: np.ndarray, group_ids: np.ndarray = None) -> Tuple[float, float, Dict[int, Tuple[float, float]]]:
    """
    More memory-efficient version that groups data by group first.
    Better for cases with many groups and reasonable group sizes.

    Upfront filter (2026-04-21): groups with <2 samples OR single-class
    y_true are NaN-bound by the underlying formula. We precompute per-group
    (count, pos_count) once, drop sample rows belonging to doomed groups
    before calling the numba inner loop, and emit the NaNs directly. On
    production workloads where 95 %+ of groups are single-sample
    (fine-grained group_ids), this slashes the per-group sort + iteration
    to only the valid-group subset.
    """
    if y_score.ndim == 2:
        y_score = y_score[:, -1]

    # Overall AUCs
    desc_score_indices = np.argsort(y_score)[::-1]
    overall_roc_auc, overall_pr_auc = fast_numba_aucs(y_true, y_score, desc_score_indices)

    # By group very efficiently
    if group_ids is not None:
        # One pass over (group_id -> sample count, pos_count). np.bincount is
        # ~2-3x faster than np.add.at for the pos-count accumulation.
        unique_groups, inverse, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
        pos_counts = np.bincount(inverse, weights=y_true, minlength=len(unique_groups))
        valid_mask = (counts >= 2) & (pos_counts > 0) & (pos_counts < counts)

        group_aucs: Dict[int, Tuple[float, float]] = {}
        # Emit NaN entries for all doomed groups up front — preserves the
        # full output contract so downstream (compute_mean_aucs_per_group +
        # the >=50 % NaN warning below) sees the same dict keys as before.
        invalid_group_ids = unique_groups[~valid_mask]
        for gid in invalid_group_ids:
            group_aucs[int(gid)] = (np.nan, np.nan)

        if valid_mask.any():
            # Mask samples belonging to valid groups only (typically 5 % of
            # input when single-sample granularity dominates) and pass the
            # subset to the JIT loop.
            sample_valid = valid_mask[inverse]
            sub_y_true = y_true[sample_valid]
            sub_y_score = y_score[sample_valid]
            sub_group_ids = group_ids[sample_valid]

            sort_indices = np.argsort(sub_group_ids)
            sorted_group_ids = sub_group_ids[sort_indices]
            sorted_y_true = sub_y_true[sort_indices]
            sorted_y_score = sub_y_score[sort_indices]

            valid_group_aucs = compute_grouped_group_aucs(sorted_group_ids, sorted_y_true, sorted_y_score)
            group_aucs.update(valid_group_aucs)

        # Observability preserved: log once per call when >=50 % of groups
        # collapsed to NaN, so operators still see "most of my group AUCs
        # are single-sample" without reading every entry.
        if group_aucs:
            n_total = len(group_aucs)
            n_nan_roc = sum(1 for (r, _p) in group_aucs.values() if np.isnan(r))
            if n_nan_roc and n_nan_roc * 2 >= n_total:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "fast_aucs_per_group_optimized: %d / %d groups returned NaN ROC AUC "
                    "(single-class or single-sample). Per-group mean is built on %d "
                    "valid groups; likely causes: target imbalance concentrated in few "
                    "groups, or group_ids granularity too fine (many 1-sample groups).",
                    n_nan_roc, n_total, n_total - n_nan_roc,
                )
    else:
        group_aucs = {}

    return overall_roc_auc, overall_pr_auc, group_aucs


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_grouped_group_aucs(sorted_group_ids: np.ndarray, sorted_y_true: np.ndarray, sorted_y_score: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """
    Compute AUCs for each group from pre-sorted data.

    NOT parallelised. Benched 2026-05-09 against a ``parallel=True`` variant
    that pre-computes group boundaries + writes to per-thread arrays
    (avoiding Dict-write contention). Result: ``par/seq`` ratio 1.77-4.80×
    SLOWER than seq across (n_groups, avg_size) shapes from (100, 10) up
    to (100_000, 10). Reason: per-group work (argsort + AUC scan on 5-50
    elements) is microseconds; numba prange thread-spawn overhead per
    iteration dominates. Bench preserved at
    ``profiling/bench_grouped_aucs_parallel.py``.
    """
    group_aucs = {}
    n = len(sorted_group_ids)

    if n == 0:
        return group_aucs

    start_idx = 0
    current_group = sorted_group_ids[0]

    for i in range(1, n + 1):
        # Check if we've reached end or found a new group
        if i == n or sorted_group_ids[i] != current_group:
            end_idx = i
            group_size = end_idx - start_idx

            if group_size > 1:
                # Extract group data
                group_y_true = sorted_y_true[start_idx:end_idx]
                group_y_score = sorted_y_score[start_idx:end_idx]

                # Sort by score for this group
                group_desc_indices = np.argsort(group_y_score)[::-1]

                # Compute AUCs for this group
                roc_auc, pr_auc = fast_numba_aucs_simple(group_y_true, group_y_score, group_desc_indices)
                group_aucs[int(current_group)] = (roc_auc, pr_auc)
            else:
                # Single-sample group: AUC is mathematically undefined.
                # Return NaN (not 0.0) so compute_mean_aucs_per_group's
                # NaN filter drops it from the mean. Previously (0.0, 0.0)
                # was treated as legitimate data and silently depressed
                # the mean AUC when a fold had many single-sample groups.
                group_aucs[int(current_group)] = (np.nan, np.nan)

            # Move to next group
            if i < n:
                start_idx = i
                current_group = sorted_group_ids[i]

    return group_aucs


@numba.njit(**NUMBA_NJIT_PARAMS)
def fast_numba_aucs_simple(y_true: np.ndarray, y_score: np.ndarray, desc_score_indices: np.ndarray) -> Tuple[float, float]:
    """
    Simplified version of your original function for per-group computation.
    """
    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    total_pos = np.sum(y_true_sorted)
    total_neg = len(y_true_sorted) - total_pos

    if total_pos == 0 or total_neg == 0:
        # Single-class data: both ROC AUC and PR AUC are undefined
        return np.nan, np.nan

    # Variables for ROC AUC
    last_counted_fps = 0
    last_counted_tps = 0
    tps, fps = 0, 0
    roc_auc = 0.0

    # Variables for PR AUC
    prev_recall = 0.0
    pr_auc = 0.0
    n = len(y_true_sorted)

    for i in range(n):
        tps += y_true_sorted[i]
        fps += 1 - y_true_sorted[i]

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            # Update ROC AUC
            delta_fps = fps - last_counted_fps
            sum_tps = last_counted_tps + tps
            roc_auc += delta_fps * sum_tps
            last_counted_fps = fps
            last_counted_tps = tps

            # Update PR AUC
            current_precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
            current_recall = tps / total_pos
            delta_recall = current_recall - prev_recall
            pr_auc += delta_recall * current_precision
            prev_recall = current_recall

    # Normalize ROC AUC
    denom_roc = tps * fps * 2
    if denom_roc > 0:
        roc_auc /= denom_roc
    else:
        # Should not reach here due to early return, but handle defensively
        roc_auc = np.nan

    return roc_auc, pr_auc


def compute_mean_aucs_per_group(group_aucs: dict) -> tuple:

    # Compute mean per-group AUCs, ignoring NaN values
    group_roc_aucs = np.array([aucs[0] for aucs in group_aucs.values()])
    group_pr_aucs = np.array([aucs[1] for aucs in group_aucs.values()])

    # Filter out NaN values for mean calculation
    valid_roc = ~np.isnan(group_roc_aucs)
    valid_pr = ~np.isnan(group_pr_aucs)
    mean_roc_auc = np.mean(group_roc_aucs[valid_roc]) if np.any(valid_roc) else np.nan
    mean_pr_auc = np.mean(group_pr_aucs[valid_pr]) if np.any(valid_pr) else np.nan

    return mean_roc_auc, mean_pr_auc


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
    """
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
    show_plots: bool = True,
    #
    title_metrics_tokens: Sequence[str] = DEFAULT_TITLE_METRICS_TOKENS,
    show_prob_histogram: bool = True,
    prob_histogram_yscale: str = "auto",
    show_inline_population_labels: bool = True,
    #
    plot_file: str = "",
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
    """

    assert backend in ("plotly", "matplotlib")
    if len(y_true) == 0:
        (
            brier_loss,
            calibration_mae,
            calibration_std,
            calibration_coverage,
        ) = (
            1.0,
            1.0,
            1.0,
            0.0,
        )
        roc_auc, pr_auc, ice, ll, precision, recall, f1 = 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0
        ece, brier_reliability, brier_resolution, brier_uncertainty = 1.0, 1.0, 0.0, 0.0
        metrics_string, fig = "", None
        return (
            brier_loss, calibration_mae, calibration_std, calibration_coverage,
            ece, brier_reliability, brier_resolution, brier_uncertainty,
            roc_auc, pr_auc, ice, ll, precision, recall, f1,
            metrics_string, fig,
        )

    brier_loss = fast_brier_score_loss(y_true=y_true, y_prob=y_pred)

    freqs_predicted, freqs_true, hits = fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=nbins)
    if verbose:
        print("freqs_predicted", freqs_predicted)
        print("freqs_true", freqs_true)
    min_hits, max_hits = np.min(hits), np.max(hits)
    calibration_mae, calibration_std, calibration_coverage = calibration_metrics_from_freqs(
        freqs_predicted=freqs_predicted, freqs_true=freqs_true, hits=hits, nbins=nbins, array_size=len(y_true), use_weights=use_weights
    )

    # Always compute ECE + Brier decomposition. Same data-adaptive bin grid as
    # CMAEW (kernel re-bins internally so it can capture per-bin pred_means,
    # which fast_calibration_binning doesn't expose). Cost is one short pass.
    ece, brier_reliability, brier_resolution, brier_uncertainty, _brier_binned = compute_ece_and_brier_decomposition(
        y_true=y_true, y_pred=y_pred, nbins=nbins,
    )

    if _precomputed_aucs is not None and group_ids is None:
        roc_auc, pr_auc = _precomputed_aucs
        group_aucs = {}
    else:
        roc_auc, pr_auc, group_aucs = fast_aucs_per_group_optimized(y_true=y_true, y_score=y_pred, group_ids=group_ids)
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

    precision, recall, f1 = compute_pr_recall_f1_metrics(y_true=y_true, y_pred=y_pred >= binary_threshold)

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

    if plot_file or show_plots:

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
            figsize=figsize,
            backend=backend,
            show_prob_histogram=show_prob_histogram,
            prob_histogram_yscale=prob_histogram_yscale,
            show_inline_population_labels=show_inline_population_labels,
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
        desc_idx = np.argsort(-y_p)
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


# ----------------------------------------------------------------------------------------------------------------------------
# Errors & scorers
# ----------------------------------------------------------------------------------------------------------------------------


def compute_probabilistic_multiclass_error(
    y_true: Union[pd.Series, pd.DataFrame, np.ndarray],
    y_score: Union[pd.Series, pd.DataFrame, np.ndarray, Sequence],
    labels: np.ndarray = None,
    method: str = "multicrit",
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    pr_auc_weight: float = 0.1,
    brier_loss_weight: float = 0.8,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
    use_weighted_calibration: bool = True,
    weight_by_class_npositives: bool = False,
    nbins: int = 10,
    verbose: bool = False,
    ndigits: int = 4,
    multilabel: bool = False,
    **kwargs,  # as scorer can pass kwargs of this kind: {'needs_proba': True, 'needs_threshold': False}
):
    """Given a sequence of per-class probabilities (predicted by some model), and ground truth targets,
    computes weighted sum of per-class errors.
    Supports several error estimation methods: "multicrit", "brier_score", "precision".
    If number of classes is only 2, skips class 0 as it's fully complementary to class 1.

    ``multilabel=True``: y_true is a 2D (n_samples, n_classes) indicator matrix, each column
    treated as an independent binary target. Single-label (``y_true == class_id``) comparison
    is wrong for multilabel data because a sample can carry multiple positives simultaneously.

    NOT threaded across classes. Benched 2026-05-09 against a
    ``ThreadPoolExecutor.map`` variant fanning each class to its own
    thread. Result: ``par/seq`` ratio 2.65-10.65× SLOWER across (N, K)
    shapes from (10k, 3) to (1M, 5). Reason: per-class work is small
    (each inner kernel already releases GIL, but Python-level fanout
    overhead dominates) AND the inner kernels (fast_brier_score_loss,
    fast_ice_only) already auto-dispatch to par at N>=100k, so threading
    over classes layers concurrency on top of per-kernel concurrency
    and produces oversubscription. Bench preserved at
    ``profiling/bench_multiclass_error_parallel.py``.
    """

    assert method in ("multicrit", "brier_score", "precision")

    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.values
    if isinstance(y_score, (pd.Series, pd.DataFrame)):
        y_score = y_score.values
    if labels is not None and isinstance(labels, (pd.Series, pd.DataFrame)):
        labels = labels.values

    if isinstance(y_score, Sequence):
        probs = y_score
    else:
        if len(y_score.shape) == 1:
            y_score = np.vstack([1 - y_score, y_score]).T
        probs = [y_score[:, i] for i in range(y_score.shape[1])]

    # Auto-detect multilabel from shape: a 2D y_true with width matching probs count is
    # an indicator matrix; caller can also set ``multilabel=True`` explicitly.
    # Object-dtype-of-arrays (``pl.List`` -> pandas roundtrip) presents as 1-D
    # but each cell is a per-row label vector - stack to 2-D so the shape
    # check below activates the multilabel branch correctly. Surfaced 3-way
    # fuzz c0000 / c0008 (cb / multilabel target) - without the stack, the
    # ``y_true == class_id`` fall-through raised ``truth value of array
    # ambiguous`` on the cell-array comparison.
    if (
        isinstance(y_true, np.ndarray)
        and y_true.dtype == object
        and y_true.ndim == 1
        and y_true.shape[0] > 0
    ):
        _first = y_true[0]
        if hasattr(_first, "shape") or (
            hasattr(_first, "__len__") and not isinstance(_first, (str, bytes))
        ):
            try:
                y_true = np.stack([np.asarray(c) for c in y_true], axis=0)
            except Exception:
                pass
    if not multilabel and isinstance(y_true, np.ndarray) and y_true.ndim == 2 and y_true.shape[1] == len(probs):
        multilabel = True
        logger.debug("compute_probabilistic_multiclass_error: detected multilabel y_true shape, enabling multilabel mode.")

    total_error = 0.0
    weights_sum = 0

    # 2026-05-11: batched-numba fastpath. When the hot path applies
    # (method='multicrit' AND not verbose AND probs convert to a clean
    # (N, K) float64 stack), process all K classes in one numba dispatch
    # via _batch_per_class_ice_kernel. Eliminates the per-class
    # Python->numba transition overhead that dominated the K-class
    # Python loop (cProfile attributed ~60 ms / call avg, of which
    # ~30 ms was Python glue for K=3 classes). Bit-exact equivalent
    # of the legacy fast_ice_only K-loop -- verified in
    # profiling/bench_compute_multiclass_error.py.
    if method == "multicrit" and not verbose:
        # Build the set of class_ids to evaluate (binary case skips 0).
        _class_ids = [
            c for c in range(len(probs))
            if not (len(probs) == 2 and c == 0 and not multilabel)
        ]
        try:
            # Stack y_pred_NK
            _y_pred_cols = []
            for _c in _class_ids:
                _yp = probs[_c]
                if isinstance(_yp, pl.Series):
                    _yp = _yp.to_numpy()
                elif isinstance(_yp, (pd.Series,)):
                    _yp = _yp.values
                _y_pred_cols.append(np.ascontiguousarray(_yp, dtype=np.float64))
            _y_pred_NK = np.column_stack(_y_pred_cols)
            # Stack y_true_NK (indicator matrix)
            _y_true_cols = []
            for _c in _class_ids:
                if multilabel:
                    _yt = y_true[:, _c]
                elif labels is not None:
                    _yt = y_true == labels[_c]
                else:
                    _yt = y_true == _c
                if isinstance(_yt, pl.Series):
                    _yt = _yt.cast(pl.Int8).to_numpy()
                elif isinstance(_yt, (pd.Series,)):
                    _yt = _yt.values.astype(np.int8)
                else:
                    _yt = np.ascontiguousarray(_yt, dtype=np.int8)
                _y_true_cols.append(_yt)
            _y_true_NK = np.column_stack(_y_true_cols)
            # Single-dispatch batched kernel
            ice_per_class = _batch_per_class_ice_kernel(
                _y_true_NK, _y_pred_NK, nbins,
                bool(use_weighted_calibration),
                float(mae_weight), float(std_weight), float(brier_loss_weight),
                float(roc_auc_weight), float(pr_auc_weight),
                float(min_roc_auc), float(roc_auc_penalty),
            )
            # Reduce with per-class weights
            for _k, _cid in enumerate(_class_ids):
                if weight_by_class_npositives:
                    weight = int(_y_true_NK[:, _k].sum())
                else:
                    weight = 1
                total_error += float(ice_per_class[_k]) * weight
                weights_sum += weight
            if weights_sum > 0:
                total_error /= weights_sum
            else:
                logger.warning(
                    "compute_probabilistic_multiclass_error: sum of per-class weights is 0; returning NaN."
                )
                total_error = float("nan")
            return total_error
        except Exception as _exc:
            # Defensive: any kernel / stacking issue falls through to the
            # legacy per-class Python loop below. Log at DEBUG so the path
            # transition stays visible during dev but doesn't spam INFO.
            logger.debug(
                "_batch_per_class_ice_kernel fastpath failed (%s); falling back to per-class loop.",
                _exc,
            )

    for class_id in range(len(probs)):

        if len(probs) == 2 and class_id == 0 and not multilabel:
            continue

        # Get prediction and ground truth

        y_pred = probs[class_id]
        if multilabel:
            # Indicator column for this class; each row is an independent binary label.
            correct_class = y_true[:, class_id]
        elif labels is not None:
            correct_class = y_true == labels[class_id]
        else:
            correct_class = y_true == class_id

        if isinstance(correct_class, (pd.Series, np.ndarray)):
            correct_class = correct_class.astype(np.int8)
        elif isinstance(correct_class, pl.Series):
            correct_class = correct_class.cast(pl.Int8).to_numpy()

        # Compute class error. When verbose=False (the fairness fan-out
        # hot path), take the ICE-only / brier-only fastpath and skip the
        # log_loss + precision/recall/f1 + title work that
        # fast_calibration_report does for its reporting callers.
        # Bit-exact equivalent — see bench_ice_only.py (2026-04-19).

        if method == "multicrit":
            if verbose:
                brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, *_, metrics_string, fig = fast_calibration_report(
                    y_true=correct_class, y_pred=y_pred, use_weights=use_weighted_calibration, nbins=nbins,
                    show_plots=False, verbose=False,
                    mae_weight=mae_weight, std_weight=std_weight, brier_loss_weight=brier_loss_weight,
                    roc_auc_weight=roc_auc_weight, pr_auc_weight=pr_auc_weight,
                    min_roc_auc=min_roc_auc, roc_auc_penalty=roc_auc_penalty,
                )
                logger.info("\t class_id=%s, %s", class_id, metrics_string)
                class_error = ice
            else:
                class_error = fast_ice_only(
                    y_true=correct_class, y_pred=y_pred, nbins=nbins, use_weights=use_weighted_calibration,
                    mae_weight=mae_weight, std_weight=std_weight, brier_loss_weight=brier_loss_weight,
                    roc_auc_weight=roc_auc_weight, pr_auc_weight=pr_auc_weight,
                    min_roc_auc=min_roc_auc, roc_auc_penalty=roc_auc_penalty,
                )
        elif method == "brier_score":
            # Only brier_loss is used — skip binning/AUC/ICE entirely.
            class_error = fast_brier_score_loss(y_true=correct_class, y_prob=y_pred)
            if verbose:
                logger.info(f"\t class_id={class_id}, brier_loss={class_error:.{ndigits}f}")
        elif method == "precision":
            class_error = fast_precision(y_true=correct_class, y_pred=(y_pred >= 0.5).astype(np.int8), zero_division=0)

        # Assign weights

        if weight_by_class_npositives:
            weight = correct_class.sum()
        else:
            weight = 1

        total_error += class_error * weight
        weights_sum += weight

    # Guard against div-by-zero when every per-class weight was 0 (e.g. weight_by_class_npositives
    # with all-negative y_true, or empty probs). Previously propagated 0/0 → NaN silently.
    if weights_sum > 0:
        total_error /= weights_sum
    else:
        logger.warning("compute_probabilistic_multiclass_error: sum of per-class weights is 0; returning NaN.")
        total_error = float("nan")

    if verbose:
        logger.info(f"method={method}, data size={len(correct_class):_} mean_class_error={total_error:.{ndigits}f}")

    return total_error


class ICE:
    """Custom probabilistic prediction error metric balancing predictive power with calibration.
    Can regularly create a calibration plot.
    """

    def __init__(
        self,
        metric: Callable,
        higher_is_better: bool,
        calibration_plot_period: int = 0,
        max_arr_size: int = 0,
    ) -> None:

        # save params
        store_params_in_object(obj=self, params=get_parent_func_args())

        self.nruns = 0

    def is_max_optimal(self):
        return self.higher_is_better

    def __sklearn_clone__(self):
        """Identity clone for sklearn's ``clone()`` (sklearn >= 1.3).

        ICE has no fit-state worth re-initialising on clone; sharing the
        same instance is harmless. Returning ``self`` here is necessary
        but not sufficient for the CB+ICE clone bind: ``CatBoost``'s
        ``__init__`` always deep-copies ``eval_metric`` internally, so
        the sklearn clone assertion (``new_obj.get_params()[name] is
        param``) still trips. The ``__sklearn_clone__`` patch installed
        below on ``CatBoostClassifier`` / ``CatBoostRegressor`` closes
        that gap by returning a CB instance with explicit identity-shared
        ``eval_metric`` instead of going through the parametric-check
        path.
        """
        return self

    def evaluate(self, approxes, target, weight):
        output_weight = 1  # weight is not used

        # to avoid expensive train set metric evaluation, we simply return 0 for any input larger than max_arr_size
        if self.max_arr_size and len(approxes[0]) > self.max_arr_size:
            return 0, output_weight

        # Convert CatBoost logits to probabilities using numba-optimized functions
        if len(approxes) == 1:
            # Binary classification
            probs_2d = cb_logits_to_probs_binary(approxes[0])
            probs = probs_2d  # Shape: (n_samples, 2)
            class_id = 1
            y_pred = probs_2d[:, 1]  # For plotting
        else:
            # Multiclass: stack approxes into 2D array (n_classes, n_samples)
            logits_2d = np.vstack(approxes)
            probs_2d = cb_logits_to_probs_multiclass(logits_2d)
            probs = probs_2d  # Shape: (n_samples, n_classes)
            class_id = len(approxes) - 1
            y_pred = probs_2d[:, class_id]  # For plotting

        total_error = self.metric(y_true=target, y_score=probs)

        self.nruns += 1

        # Additional visualization of training process (for the last class_id) is possible.

        if self.calibration_plot_period and (self.calibration_plot_period > 0 and (self.nruns % self.calibration_plot_period == 0)):
            y_true = (target == class_id).astype(np.int8)
            brier_loss, calibration_mae, calibration_std, calibration_coverage, roc_auc, pr_auc, ice, ll, *_, metrics_string, fig = fast_calibration_report(
                y_true=y_true,
                y_pred=y_pred,
                title=f"{len(approxes[0]):_} records of class {class_id}, integral error={total_error:.4f}, nruns={self.nruns:_}\r\n",
                use_weights=True,
                verbose=False,
            )
            logger.info(metrics_string)

        return total_error, output_weight

    def get_final_error(self, error, weight):
        return error


# -----------------------------------------------------------------------------
# CatBoost compat: ``__sklearn_clone__`` patch for CB+ICE eval_metric clone bug
# -----------------------------------------------------------------------------
# CatBoost's ``__init__`` deep-copies its ``eval_metric`` argument
# internally, then ``get_params(deep=False)`` returns the deep-copied
# instance. ``sklearn.base.clone()``'s parametric path verifies that
# ``new_object.get_params()[name] is param`` for every parameter — and
# this fails for CB+ICE because ``param`` (post-clone-of-ICE) is the
# original ICE while ``new_object.get_params()['eval_metric']`` is a CB-
# internal deep copy. Identity is destroyed by CB, not by ICE.
#
# Touching ICE.__deepcopy__ to return self DID restore identity for CB's
# internal copy but BROKE pickle: sharing ICE across the pre-pickle
# ``copy.deepcopy(model)`` retains a CatBoost-internal numba JIT
# cyfunction reference (``_cpu_jit_method_wrap.<locals>.new_method``)
# that dill cannot serialize.
#
# The clean fix lives at the CB level: install ``__sklearn_clone__`` on
# ``CatBoostClassifier`` / ``CatBoostRegressor`` that reuses the same
# ``eval_metric`` instance for the cloned estimator, bypassing
# ``_clone_parametrized``'s identity check entirely. ICE retains its
# default deepcopy/pickle behaviour, so save/load is unaffected.
def _install_catboost_sklearn_clone_patch() -> None:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        return

    def _cb_sklearn_clone(self):
        # Reuse the SAME eval_metric instance to bypass CB's internal
        # deep-copy of eval_metric on __init__ that breaks identity.
        params = self.get_params(deep=False)
        eval_metric = getattr(self, "_init_params", {}).get("eval_metric")
        if eval_metric is None:
            eval_metric = params.get("eval_metric")
        cls = type(self)
        # Strip eval_metric from params before re-init so we can attach
        # the original instance after construction (CB's __init__ would
        # deep-copy it otherwise).
        params_no_em = {k: v for k, v in params.items() if k != "eval_metric"}
        new = cls(**params_no_em)
        if eval_metric is not None:
            # Re-attach by direct assignment — CB stores eval_metric in
            # ``_init_params`` and the public param map.
            try:
                new._init_params["eval_metric"] = eval_metric
            except (AttributeError, KeyError):
                pass
            try:
                new.set_params(eval_metric=eval_metric)
            except Exception:
                pass
        return new

    for cls in (CatBoostClassifier, CatBoostRegressor):
        if not hasattr(cls, "__sklearn_clone__"):
            cls.__sklearn_clone__ = _cb_sklearn_clone


_install_catboost_sklearn_clone_patch()


@numba.njit(**NUMBA_NJIT_PARAMS)
def integral_calibration_error_from_metrics(
    calibration_mae: float,
    calibration_std: float,
    calibration_coverage: float,
    brier_loss: float,
    roc_auc: float,
    pr_auc: float,
    mae_weight: float = 3,
    std_weight: float = 2,
    roc_auc_weight: float = 1.5,
    pr_auc_weight: float = 0.1,
    brier_loss_weight: float = 0.8,
    min_roc_auc: float = 0.54,
    roc_auc_penalty: float = 0.00,
) -> float:
    """Compute Integral Calibration Error (ICE) from base ML metrics.

    ICE is a weighted sum of baseline losses minus rewards for sharp ranking
    (roc_auc, pr_auc). When ``roc_auc`` is weaker than ``min_roc_auc``, a
    penalty smoothly ramps up from 0 at the threshold to ``roc_auc_penalty``
    at the worst case ``roc_auc == 0.5`` (complete random). The ramp is
    linear in the deficit and symmetric about 0.5 (so an inverted ranker at
    e.g. 0.45 is penalised the same as one at 0.55-epsilon, matching the
    symmetric reward term ``-|roc_auc-0.5|*roc_auc_weight``).

    Keeping ``roc_auc_penalty`` as the "max penalty" knob preserves the
    prior semantics: old callers that set e.g. 3.0 still get a 3.0 bump at
    auc=0.5. What changed: the penalty now tapers smoothly to 0 as auc
    approaches ``min_roc_auc`` instead of dropping off a step cliff — this
    avoids jumpy early-stopping curves that could fixate just inside the
    penalty zone when the step was large.
    """
    # Guard against NaN roc_auc/pr_auc (single-class eval set, zero-variance
    # scores, etc. — fast_aucs_per_group_optimized returns NaN in those cases).
    # Without this guard the entire ICE becomes NaN, which silently breaks
    # early-stopping comparisons (NaN > best is always False, so the trainer
    # gets stuck on iteration-1 best instead of failing loud).
    base_loss = (
        brier_loss * brier_loss_weight
        + calibration_mae * mae_weight
        + calibration_std * std_weight
    )
    roc_term = 0.0 if np.isnan(roc_auc) else np.abs(roc_auc - 0.5) * roc_auc_weight
    pr_term = 0.0 if np.isnan(pr_auc) else pr_auc * pr_auc_weight
    res = base_loss - roc_term - pr_term
    threshold_width = min_roc_auc - 0.5
    if threshold_width > 0.0 and not np.isnan(roc_auc):
        deficit = threshold_width - np.abs(roc_auc - 0.5)
        if deficit > 0.0:
            res += (deficit / threshold_width) * roc_auc_penalty
    return res


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_brier_score_loss_seq(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return np.mean((y_true - y_prob) ** 2)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_brier_score_loss_par(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Parallel variant. ~7.7× faster than seq at N=10M (verified
    on 8-thread numba runtime). Loses to seq below N≈50k due to
    thread-spawn overhead -- the public ``fast_brier_score_loss``
    wrapper auto-dispatches based on N."""
    n = len(y_true)
    s = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_prob[i]
        s += d * d
    return s / n


# Crossover threshold for parallel kernels. Sum-reduction kernels
# (brier, log loss, prf1 counts) parallel-win from N≈50-100k upwards.
# Multilabel row-loop kernels (subset accuracy, jaccard) win from N≈10-50k.
# Conservative thresholds chosen to avoid the lose-band at low N.
_PARALLEL_REDUCTION_THRESHOLD: int = 100_000
_PARALLEL_MULTILABEL_THRESHOLD: int = 50_000


def fast_brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities), auto seq/par.

    Sequential numba kernel below ~100k rows (cold-start cost
    of the parallel runtime exceeds the per-element gain). Parallel
    kernel above the threshold -- 7.7× faster at N=10M on an 8-thread
    runtime. Tunable via ``_PARALLEL_REDUCTION_THRESHOLD``.
    """
    if len(y_true) >= _PARALLEL_REDUCTION_THRESHOLD:
        return _fast_brier_score_loss_par(y_true, y_prob)
    return _fast_brier_score_loss_seq(y_true, y_prob)


# Backward-compat alias — older code and tests import `brier_score_loss` from this module.
# Keep the name visible but route it to the renamed fast_brier_score_loss so the intent is clear.
brier_score_loss = fast_brier_score_loss


# ----------------------------------------------------------------------------------------------------------------------------
# Regression metrics: numba-accelerated drop-in replacements for sklearn
# ----------------------------------------------------------------------------------------------------------------------------
#
# sklearn input-validation overhead dominates these tiny reductions: at
# N=1M, sklearn's ``mean_absolute_error`` takes 12 ms while the numba
# kernel does the same work in ~1.5 ms (8x). Adding parallel=True drops
# it further to 0.7 ms (17x over sklearn).
# Speedups vs sklearn at N=1M:
#   MAE   17x  | MSE   15x  | RMSE  same as MSE | max_error 6x | R2 23x
#
# Full sklearn signature support:
#   - 1-D and 2-D ``y_true`` / ``y_pred`` (multioutput).
#   - ``sample_weight`` (1-D weights, broadcast across outputs).
#   - ``multioutput`` in ``'raw_values'``, ``'uniform_average'``, an
#     array of per-output weights, or (R² only) ``'variance_weighted'``.
#
# 1-D unweighted is the fastest path (single numba kernel call). 2-D
# multioutput loops per column inside Python (M is typically small).
# Weighted variants use a separate numba kernel.


# ---------- 1-D unweighted ----------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_mae_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in range(n):
        s += abs(y_true[i] - y_pred[i])
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_mae_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in numba.prange(n):
        s += abs(y_true[i] - y_pred[i])
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_mse_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in range(n):
        d = y_true[i] - y_pred[i]
        s += d * d
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_mse_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_pred[i]
        s += d * d
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_max_error_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    m = 0.0
    for i in range(n):
        d = abs(y_true[i] - y_pred[i])
        if d > m:
            m = d
    return m


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_r2_score_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Two-pass: mean of y_true, then SS_res and SS_tot."""
    n = len(y_true)
    ymean = 0.0
    for i in range(n):
        ymean += y_true[i]
    ymean /= n
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res
        ss_tot += d_tot * d_tot
    if ss_tot == 0.0:
        return 0.0  # sklearn convention for constant y_true
    return 1.0 - ss_res / ss_tot


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_r2_score_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    ymean = 0.0
    for i in numba.prange(n):
        ymean += y_true[i]
    ymean /= n
    ss_res = 0.0
    ss_tot = 0.0
    for i in numba.prange(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res
        ss_tot += d_tot * d_tot
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_r2_variance_seq(y_true: np.ndarray) -> float:
    """SS_tot for r2_variance_weighted multioutput aggregation."""
    n = len(y_true)
    ymean = 0.0
    for i in range(n):
        ymean += y_true[i]
    ymean /= n
    ss = 0.0
    for i in range(n):
        d = y_true[i] - ymean
        ss += d * d
    return ss


# ---------- 1-D weighted ----------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_mae_weighted_seq(y_true, y_pred, w):
    n = len(y_true)
    s = 0.0
    wsum = 0.0
    for i in range(n):
        s += abs(y_true[i] - y_pred[i]) * w[i]
        wsum += w[i]
    return s / wsum


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_mae_weighted_par(y_true, y_pred, w):
    n = len(y_true)
    s = 0.0
    wsum = 0.0
    for i in numba.prange(n):
        s += abs(y_true[i] - y_pred[i]) * w[i]
        wsum += w[i]
    return s / wsum


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_mse_weighted_seq(y_true, y_pred, w):
    n = len(y_true)
    s = 0.0
    wsum = 0.0
    for i in range(n):
        d = y_true[i] - y_pred[i]
        s += d * d * w[i]
        wsum += w[i]
    return s / wsum


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_mse_weighted_par(y_true, y_pred, w):
    n = len(y_true)
    s = 0.0
    wsum = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_pred[i]
        s += d * d * w[i]
        wsum += w[i]
    return s / wsum


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_r2_score_weighted_seq(y_true, y_pred, w):
    n = len(y_true)
    ymean = 0.0
    wsum = 0.0
    for i in range(n):
        ymean += y_true[i] * w[i]
        wsum += w[i]
    ymean /= wsum
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res * w[i]
        ss_tot += d_tot * d_tot * w[i]
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_r2_score_weighted_par(y_true, y_pred, w):
    n = len(y_true)
    ymean = 0.0
    wsum = 0.0
    for i in numba.prange(n):
        ymean += y_true[i] * w[i]
        wsum += w[i]
    ymean /= wsum
    ss_res = 0.0
    ss_tot = 0.0
    for i in numba.prange(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res * w[i]
        ss_tot += d_tot * d_tot * w[i]
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------- multioutput aggregation ----------


def _aggregate_multioutput(values: np.ndarray, multioutput) -> Union[float, np.ndarray]:
    """Apply sklearn's ``multioutput`` aggregation to a per-output values array.

    - ``'raw_values'`` -> ``values`` unchanged.
    - ``'uniform_average'`` -> ``np.mean(values)``.
    - array-like -> ``np.average(values, weights=multioutput)``.
    - other strings / inputs -> ValueError.
    """
    # Type check FIRST: ``multioutput == "raw_values"`` on an ndarray
    # broadcasts and raises 'ambiguous truth value'. Handle the
    # array-weights path before any string compare.
    if isinstance(multioutput, (np.ndarray, list, tuple)):
        return float(np.average(values, weights=np.asarray(multioutput, dtype=np.float64)))
    if multioutput == "raw_values":
        return values
    if multioutput == "uniform_average":
        return float(values.mean())
    raise ValueError(
        f"multioutput must be 'raw_values', 'uniform_average', or an array; got {multioutput!r}"
    )


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Promote ``(N,)`` to ``(N, 1)``; pass-through ``(N, M)``."""
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


# ---------- public wrappers ----------


def fast_mean_absolute_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput: Union[str, np.ndarray] = "uniform_average",
):
    """Drop-in for ``sklearn.metrics.mean_absolute_error``.

    Faster than sklearn (17× at N=1M) while supporting the full sklearn
    signature: ``sample_weight`` (1-D, broadcast across outputs) and
    ``multioutput`` in ``{'raw_values', 'uniform_average'}`` or an
    array of per-output weights.
    """
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if sample_weight is not None:
        w = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
    else:
        w = None

    if yt.ndim == 1:
        n = yt.shape[0]
        if w is None:
            return _fast_mae_par(yt, yp) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_mae_seq(yt, yp)
        return _fast_mae_weighted_par(yt, yp, w) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_mae_weighted_seq(yt, yp, w)

    yt2 = _to_2d(yt)
    yp2 = _to_2d(yp)
    n = yt2.shape[0]
    M = yt2.shape[1]
    per_out = np.empty(M, dtype=np.float64)
    use_par = n >= _PARALLEL_REDUCTION_THRESHOLD
    for j in range(M):
        col_yt = np.ascontiguousarray(yt2[:, j])
        col_yp = np.ascontiguousarray(yp2[:, j])
        if w is None:
            per_out[j] = _fast_mae_par(col_yt, col_yp) if use_par else _fast_mae_seq(col_yt, col_yp)
        else:
            per_out[j] = _fast_mae_weighted_par(col_yt, col_yp, w) if use_par else _fast_mae_weighted_seq(col_yt, col_yp, w)
    return _aggregate_multioutput(per_out, multioutput)


def fast_mean_squared_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput: Union[str, np.ndarray] = "uniform_average",
):
    """Drop-in for ``sklearn.metrics.mean_squared_error``. 15× faster at
    N=1M with full sample_weight + multioutput support."""
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if sample_weight is not None:
        w = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
    else:
        w = None

    if yt.ndim == 1:
        n = yt.shape[0]
        if w is None:
            return _fast_mse_par(yt, yp) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_mse_seq(yt, yp)
        return _fast_mse_weighted_par(yt, yp, w) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_mse_weighted_seq(yt, yp, w)

    yt2 = _to_2d(yt)
    yp2 = _to_2d(yp)
    n = yt2.shape[0]
    M = yt2.shape[1]
    per_out = np.empty(M, dtype=np.float64)
    use_par = n >= _PARALLEL_REDUCTION_THRESHOLD
    for j in range(M):
        col_yt = np.ascontiguousarray(yt2[:, j])
        col_yp = np.ascontiguousarray(yp2[:, j])
        if w is None:
            per_out[j] = _fast_mse_par(col_yt, col_yp) if use_par else _fast_mse_seq(col_yt, col_yp)
        else:
            per_out[j] = _fast_mse_weighted_par(col_yt, col_yp, w) if use_par else _fast_mse_weighted_seq(col_yt, col_yp, w)
    return _aggregate_multioutput(per_out, multioutput)


def fast_root_mean_squared_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput: Union[str, np.ndarray] = "uniform_average",
):
    """Drop-in for ``sklearn.metrics.root_mean_squared_error``.

    Computes per-output RMSE = sqrt(MSE) BEFORE aggregating
    (``raw_values`` returns per-output RMSEs; ``uniform_average``
    averages them after sqrt). Matches sklearn's behaviour exactly.
    """
    if isinstance(multioutput, str) and multioutput == "raw_values":
        per_out_mse = fast_mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values",
        )
        return np.sqrt(np.atleast_1d(per_out_mse))
    # For aggregated forms, sklearn computes per-output RMSE then averages.
    per_out_mse = np.atleast_1d(
        fast_mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values",
        )
    )
    per_out_rmse = np.sqrt(per_out_mse)
    return _aggregate_multioutput(per_out_rmse, multioutput)


def fast_max_error(
    y_true,
    y_pred,
    *,
    multioutput: Union[str, np.ndarray] = "raw_values",
):
    """Drop-in for ``sklearn.metrics.max_error``. ~6× faster at N=1M.

    sklearn's ``max_error`` raises on multioutput input; we extend with
    ``multioutput`` support (default ``'raw_values'`` returns per-output
    max). ``sample_weight`` does NOT apply to max-error (max is max).
    """
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if yt.ndim == 1:
        return _fast_max_error_seq(yt, yp)
    yt2 = _to_2d(yt)
    yp2 = _to_2d(yp)
    M = yt2.shape[1]
    per_out = np.empty(M, dtype=np.float64)
    for j in range(M):
        per_out[j] = _fast_max_error_seq(
            np.ascontiguousarray(yt2[:, j]),
            np.ascontiguousarray(yp2[:, j]),
        )
    return _aggregate_multioutput(per_out, multioutput)


def fast_r2_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput: Union[str, np.ndarray] = "uniform_average",
):
    """Drop-in for ``sklearn.metrics.r2_score``. 23× faster at N=1M with
    full sample_weight + multioutput support, including
    ``'variance_weighted'`` aggregation."""
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if sample_weight is not None:
        w = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
    else:
        w = None

    if yt.ndim == 1:
        n = yt.shape[0]
        if w is None:
            return _fast_r2_score_par(yt, yp) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_r2_score_seq(yt, yp)
        return _fast_r2_score_weighted_par(yt, yp, w) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_r2_score_weighted_seq(yt, yp, w)

    yt2 = _to_2d(yt)
    yp2 = _to_2d(yp)
    n = yt2.shape[0]
    M = yt2.shape[1]
    per_out = np.empty(M, dtype=np.float64)
    use_par = n >= _PARALLEL_REDUCTION_THRESHOLD
    for j in range(M):
        col_yt = np.ascontiguousarray(yt2[:, j])
        col_yp = np.ascontiguousarray(yp2[:, j])
        if w is None:
            per_out[j] = _fast_r2_score_par(col_yt, col_yp) if use_par else _fast_r2_score_seq(col_yt, col_yp)
        else:
            per_out[j] = _fast_r2_score_weighted_par(col_yt, col_yp, w) if use_par else _fast_r2_score_weighted_seq(col_yt, col_yp, w)

    if isinstance(multioutput, str) and multioutput == "variance_weighted":
        # Per-output variance of y_true (weighted if w supplied).
        ss_tots = np.empty(M, dtype=np.float64)
        for j in range(M):
            col_yt = np.ascontiguousarray(yt2[:, j])
            if w is None:
                ss_tots[j] = _fast_r2_variance_seq(col_yt)
            else:
                # Weighted variance = SS_tot_w / sum(w). The relative
                # weights are what matters for the average; using SS_tot
                # directly preserves the proportions sklearn uses.
                wsum = float(w.sum())
                wmean = float((col_yt * w).sum() / wsum)
                ss_tots[j] = float(((col_yt - wmean) ** 2 * w).sum())
        if ss_tots.sum() == 0.0:
            # Degenerate: all outputs constant. sklearn returns 0.0
            # when all SS_tot are zero (otherwise raises).
            return 0.0
        return float(np.average(per_out, weights=ss_tots))

    return _aggregate_multioutput(per_out, multioutput)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_log_loss_binary_seq(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Sequential numba binary log loss. See ``fast_log_loss_binary``
    public wrapper for auto seq/par dispatch."""
    n = len(y_true)
    if n == 0:
        return 0.0

    loss_sum = 0.0
    has_class_0 = False
    has_class_1 = False

    # Explicit out-of-range probability check: return NaN rather than silently clipping
    # whatever garbage the caller passed (previously a negative / >1 prob was just clipped
    # to eps / 1-eps and the result looked valid).
    for i in range(n):
        if y_pred[i] < 0.0 or y_pred[i] > 1.0:
            return np.nan

    for i in range(n):
        p = y_pred[i]
        # Clip to prevent log(0)
        p = max(eps, min(1 - eps, p))
        if y_true[i] == 1:
            loss_sum -= np.log(p)
            has_class_1 = True
        else:
            loss_sum -= np.log(1 - p)
            has_class_0 = True

    # Return nan if only one class present (mimics sklearn behavior)
    if not (has_class_0 and has_class_1):
        return np.nan

    return loss_sum / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_log_loss_binary_par(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Parallel binary log loss. ~4.8× faster than seq at N=10M.
    Auto-selected by ``fast_log_loss_binary`` above N=100k."""
    n = len(y_true)
    if n == 0:
        return 0.0

    # Out-of-range probability check via parallel sum-reduction (numba auto-detects ``+=``).
    bad = 0
    for i in numba.prange(n):
        if y_pred[i] < 0.0 or y_pred[i] > 1.0:
            bad += 1
    if bad > 0:
        return np.nan

    loss_sum = 0.0
    n_pos = 0
    for i in numba.prange(n):
        p = y_pred[i]
        if p < eps:
            p = eps
        elif p > 1 - eps:
            p = 1 - eps
        if y_true[i] == 1:
            loss_sum -= np.log(p)
            n_pos += 1
        else:
            loss_sum -= np.log(1 - p)

    # Need at least one of each class.
    if n_pos == 0 or n_pos == n:
        return np.nan
    return loss_sum / n


def fast_log_loss_binary(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Numba-accelerated binary log loss (cross-entropy), auto seq/par.

    Equivalent to sklearn.metrics.log_loss for binary classification.
    Faster than sklearn due to no input validation overhead. Above
    N=100k rows the parallel kernel is selected (~4.8× faster than
    seq at N=10M, 8-thread runtime).

    Returns np.nan if only one class is present in y_true.
    """
    if len(y_true) >= _PARALLEL_REDUCTION_THRESHOLD:
        return _fast_log_loss_binary_par(y_true, y_pred, eps)
    return _fast_log_loss_binary_seq(y_true, y_pred, eps)


def fast_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = None) -> float:
    """Fast log loss using numba. Drop-in replacement for sklearn.metrics.log_loss.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities for class 1.
        eps: Small value for clipping to prevent log(0). Default uses dtype's eps (sklearn-compatible).

    Returns:
        Binary cross-entropy loss.
    """
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, (pd.Series, pl.Series)):
        y_pred = y_pred.to_numpy()

    # Ensure float type for numba
    if y_true.dtype not in (np.float32, np.float64):
        y_true = y_true.astype(np.float64)
    if y_pred.dtype not in (np.float32, np.float64):
        y_pred = y_pred.astype(np.float64)

    # Use dtype's epsilon for sklearn compatibility (eps="auto" behavior)
    if eps is None:
        eps = np.finfo(y_pred.dtype).eps

    return fast_log_loss_binary(y_true, y_pred, eps)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _probability_separation_score_seq(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    """Sequential variant. Public wrapper auto-dispatches at N>=50k."""
    idx = y_true == class_label
    if idx.sum() == 0:
        return np.nan
    res = np.mean(y_prob[idx])
    if std_weight != 0.0:
        addend = np.std(y_prob[idx]) * std_weight
        if class_label == 1:
            res = res - addend
        else:
            res = res + addend
    return res


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _probability_separation_score_par(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    """Parallel variant. ~5× faster than seq at N=1M.

    Two prange passes: first computes count + sum (mean), second
    computes variance using the mean. Both passes use ``+=``
    reductions which numba auto-recognises."""
    n = len(y_true)
    if n == 0:
        return np.nan
    n_in = 0
    s = 0.0
    for i in numba.prange(n):
        if y_true[i] == class_label:
            n_in += 1
            s += y_prob[i]
    if n_in == 0:
        return np.nan
    mean = s / n_in
    if std_weight == 0.0:
        return mean

    sse = 0.0
    for i in numba.prange(n):
        if y_true[i] == class_label:
            d = y_prob[i] - mean
            sse += d * d
    std = np.sqrt(sse / n_in)
    addend = std * std_weight
    if class_label == 1:
        return mean - addend
    else:
        return mean + addend


def probability_separation_score(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    """Mean predicted probability for the in-class population, optionally
    discounted by std (separation = mean - std × weight). Auto seq/par
    dispatch above N=50k (~5× faster at N=1M)."""
    if len(y_true) >= _PARALLEL_MULTILABEL_THRESHOLD:
        return _probability_separation_score_par(y_true, y_prob, class_label, std_weight)
    return _probability_separation_score_seq(y_true, y_prob, class_label, std_weight)


def create_fairness_subgroups(
    df: pd.DataFrame,
    features: Sequence[Union[str, pd.Series]],
    cont_nbins: int = 3,
    min_pop_cat_thresh: Union[float, int] = 1000,
    merge_lowpop_cats: bool = True,
    exclude_terminal_lowpop_cats: bool = True,
    rare_group_name: str = "*RARE*",
) -> dict:
    """Create subgroups for fairness evaluation across demographic/categorical features.

    Fairness analysis evaluates model performance consistency across different
    demographic groups (e.g., age, gender, region) or categorical segments to
    ensure the model doesn't discriminate against specific subpopulations.

    Subgrouping splits observations into bins for which ML metrics are calculated separately.
    Use this when you need consistent & fair performance across subgroups - different geographical
    regions, client types, demographic segments, etc.

    For categorical variables, each category forms a natural bin.
    Low-populated categories (<min_pop_cat_thresh) are merged into a single 'rarevals' bin or excluded.
    Subgroups can have different weights (by default equal).

    How metrics are adjusted: From/to the original metric on entire dataset, weighted sum of stdevs over
    subgroups is deducted/added (depending on greater_is_better). An ideally fair model has zero stdevs.

    Final ML report includes: subgroup name, nbins, metric stdev, outliers, best/worst bins & performance."""

    if isinstance(min_pop_cat_thresh, float):
        assert min_pop_cat_thresh > 0 and min_pop_cat_thresh < 1.0
        min_pop_cat_thresh = int(len(df) * min_pop_cat_thresh)  # convert to abs value
    elif isinstance(min_pop_cat_thresh, int):
        assert min_pop_cat_thresh > 0 and min_pop_cat_thresh <= len(df) // 2

    subgroups = {}
    for feature_name in features:

        if feature_name in ("**ORDER**", "**RANDOM**"):
            subgroups[feature_name] = feature_name
            continue

        if isinstance(feature_name, pd.Series):
            feature_vals = feature_name
            feature_name = feature_vals.name
        else:
            feature_vals = df[feature_name]

        val_cnts = feature_vals.value_counts()

        if feature_vals.dtype.name not in ("category", "object", "date", "datetime"):
            if len(val_cnts) > cont_nbins:
                feature_vals = pd.qcut(feature_vals, q=cont_nbins, labels=None)  # use qcut for equipopulated binning
                val_cnts = feature_vals.value_counts()  # this needs recalculation now

        # use categories as natural bins. ensure that low-populated cats are merged if possible (merge_lowpop_cats)
        # or excluded (exclude_terminal_lowpop_cats).

        rarecats = val_cnts[val_cnts < min_pop_cat_thresh]
        if len(rarecats) > 0:
            cats = rarecats.index.values.tolist()
            if merge_lowpop_cats and rarecats.sum() >= min_pop_cat_thresh:
                # merging is possible
                feature_vals = feature_vals.copy().replace({cat: rare_group_name for cat in cats})
                val_cnts = feature_vals.value_counts()  # this needs recalculation now
                cats_to_use = val_cnts.index.values.tolist()
                logger.info(f"For feature {feature_name}, had to merge {len(cats):_} bins {','.join(map(str,cats))}, {rarecats.sum():_} records.")
            else:
                if exclude_terminal_lowpop_cats:
                    cats_to_use = val_cnts[val_cnts >= min_pop_cat_thresh].index.values.tolist()
                    logger.info(f"For feature {feature_name}, had to exclude {len(cats):_} bins {','.join(map(str,cats))}, {rarecats.sum():_} records.")
        else:
            cats_to_use = val_cnts.index.values.tolist()

        if len(cats_to_use) > 1:
            subgroups[feature_name] = dict(bins=feature_vals, bins_names=cats_to_use)
        else:
            logger.warning(f"Feature {feature_name} can't particiate in subgrouping: it has only one bin.")

    return subgroups


def create_fairness_subgroups_indices(
    subgroups: dict, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, group_weights: dict = None, cont_nbins: int = 3
) -> dict:
    """Create index mappings for fairness subgroups across train/val/test splits.

    Converts fairness subgroups (demographic/categorical bins) into index arrays
    for each data split, enabling per-subgroup metric computation.
    """
    if group_weights is None:
        group_weights = {}
    res = {}
    if len(val_idx) == len(test_idx):
        logger.warning(f"Validation and test sets have the same size. Fairness subgroups estimation will be incorrect.")
    for arr in (train_idx, test_idx, val_idx):
        npoints = len(arr)
        fairness_subgroups_indices = {}
        for group_name, group_params in subgroups.items():
            group_indices = {}
            if group_name in ("**ORDER**", "**RANDOM**"):
                bins, unique_bins = create_robustness_standard_bins(group_name=group_name, npoints=npoints, cont_nbins=cont_nbins)
            else:
                bins = group_params.get("bins")
                assert bins.index.is_unique
                bins = bins.loc[arr]
                unique_bins = None

            if unique_bins is None:
                if isinstance(bins, pd.Series):
                    unique_bins = bins.unique()
                else:
                    unique_bins = np.unique(bins)

            for bin_name in unique_bins:
                idx = bins == bin_name
                group_indices[bin_name] = np.where(idx)[0]

            fairness_subgroups_indices[group_name] = dict(bins=group_indices, weight=group_weights.get(group_name, 1.0))

        res[npoints] = fairness_subgroups_indices

    return res


def create_robustness_standard_bins(group_name: str, npoints: int, cont_nbins: int) -> tuple:

    step_size = npoints // cont_nbins
    bins = np.empty(shape=npoints, dtype=np.int16)
    start = 0
    unique_bins = range(cont_nbins)
    for i in unique_bins:
        bins[start : start + step_size] = i
        start += step_size
    if group_name == "**RANDOM**":
        np.random.shuffle(bins)

    return bins, unique_bins


def compute_fairness_metrics(
    metrics: dict,
    metrics_higher_is_better: dict,
    subgroups: dict,
    subset_index: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cont_nbins: int = 3,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compute fairness metrics across demographic/categorical subgroups.

    Evaluates model performance consistency across different subpopulations
    to identify potential bias or discrimination. * is added to the bin name
    if bin's metric is an outlier (computed using Tukey's fence & IQR)."""

    if subgroups:

        res = []
        quantile = 0.25
        quantiles_to_compute = [0.5 - quantile, 0.5, 0.5 + quantile]
        tukey_mult = get_tukey_fences_multiplier_for_quantile(quantile=quantile, sd_sigma=2.7)

        for group_name, group_params in subgroups.items():
            if group_name in ("**ORDER**", "**RANDOM**"):
                bins, unique_bins = create_robustness_standard_bins(group_name=group_name, npoints=len(y_true), cont_nbins=cont_nbins)
            else:
                bins = group_params.get("bins")
                if bins is not None:
                    assert subset_index is not None
                    bins = bins.loc[subset_index]
                bins_names = group_params.get("bins_names")
                unique_bins = None

            npoints = []
            perfs = defaultdict(dict)
            if unique_bins is None:
                if isinstance(bins, pd.Series):
                    unique_bins = bins.unique()
                else:
                    unique_bins = np.unique(bins)
            for bin_name in unique_bins:
                idx = bins == bin_name
                n_points = idx.sum()
                if n_points:
                    npoints.append(n_points)
                    for metric_name, metric_func in metrics.items():
                        if y_pred.ndim == 2:
                            metric_value = metric_func(y_true[idx], y_pred[idx, :])
                        else:
                            metric_value = metric_func(y_true[idx], y_pred[idx])
                        perfs[metric_name][f"{bin_name} [{n_points}]"] = metric_value

            for metric_name, metric_perf in perfs.items():

                metric_perf = sort_dict_by_value(metric_perf)
                npoints = np.array(npoints)
                line = dict(
                    factor=group_name,
                    metric=metric_name,
                    nbins=len(unique_bins),
                    npoints_from=npoints.min(),
                    npoints_median=int(np.median(npoints)),
                    npoints_to=npoints.max(),
                )

                # -----------------------------------------------------------------------------------------------------------------------------------------------------
                # Compute quantiles of the metric value.
                # -----------------------------------------------------------------------------------------------------------------------------------------------------

                performances = np.array(list(metric_perf.values()))
                quantiles = np.quantile(performances, q=quantiles_to_compute)
                iqr = quantiles[-1] - quantiles[0]
                min_boundary = quantiles[0] - tukey_mult * iqr
                max_boundary = quantiles[-1] + tukey_mult * iqr

                """
                for q, value in zip(quantiles_to_compute, quantiles):
                    line[f"q{q:.2f}"] = value
                """

                line[f"metric_mean"] = performances.mean()
                line[f"metric_std"] = performances.std()

                # -----------------------------------------------------------------------------------------------------------------------------------------------------
                # Show top-n best/worst groups. postfix * means metric value for the bin is an outlier.
                # -----------------------------------------------------------------------------------------------------------------------------------------------------

                l = len(metric_perf)
                real_top_n = min(l // 2, top_n)

                for i, (bin_name, metric_value) in enumerate(metric_perf.items()):
                    if metric_value < min_boundary or metric_value > max_boundary:
                        postfix = "*"
                    else:
                        postfix = ""
                    if i < real_top_n:
                        if metrics_higher_is_better[metric_name]:
                            line["bin-worst-" + str(i + 1)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                        else:
                            line["bin-best-" + str(i + 1)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                    elif i >= l - real_top_n:
                        if metrics_higher_is_better[metric_name]:
                            line["bin-best-" + str(l - i)] = f"{bin_name}: {metric_value:.3f}{postfix}"
                        else:
                            line["bin-worst-" + str(l - i)] = f"{bin_name}: {metric_value:.3f}{postfix}"

                res.append(line)
        if res:
            res = pd.DataFrame(res).set_index(["factor", "nbins", "npoints_from", "npoints_median", "npoints_to", "metric"])
            return res.reindex(sorted(res.columns), axis=1)


# Backward-compatible aliases for renamed fairness functions
create_robustness_subgroups = create_fairness_subgroups
create_robustness_subgroups_indices = create_fairness_subgroups_indices
compute_robustness_metrics = compute_fairness_metrics


def robust_mlperf_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Callable,
    higher_is_better: bool,
    subgroups: dict = None,
    whole_set_weight: float = 0.5,
    min_group_size: int = 100,
) -> float:
    """Bins idices need to be aware of arr sizes: boostings can call the metric on
    multiple sets of differnt lengths - train, val, etc. Arrays will be pure numpy, so no other means to
    distinguish except the arr size."""

    weights_sum = whole_set_weight
    total_metric_value = metric(y_true, y_score) * whole_set_weight

    l = len(y_true)
    if subgroups and l in subgroups:

        for group_name, group_params in subgroups[l].items():

            bins = group_params.get("bins")
            bin_weight = group_params.get("weight", 1.0)

            perfs = []
            for bin_name, bin_indices in bins.items():
                if len(bin_indices) < min_group_size:
                    continue
                if isinstance(y_score, Sequence):
                    if len(y_score) == 2:
                        metric_value = metric(y_true[bin_indices], [el[bin_indices] for el in y_score])
                    else:
                        metric_value = metric(y_true[bin_indices], y_score[1][bin_indices])
                else:
                    if y_score.ndim == 2:
                        metric_value = metric(y_true[bin_indices], y_score[bin_indices, :])
                    else:
                        metric_value = metric(y_true[bin_indices], y_score[bin_indices])
                perfs.append(metric_value)

            if perfs:
                perfs = np.array(perfs)
                bin_metric_value = perfs.mean()
                if higher_is_better:
                    bin_metric_value -= perfs.std()
                else:
                    bin_metric_value += perfs.std()

                weights_sum += bin_weight
                total_metric_value += bin_metric_value * bin_weight

    return total_metric_value / weights_sum


# ----------------------------------------------------------------------------------------------------------------------------
# Salvaged from OldEnsembling.py — combined Brier+precision scorer
# ----------------------------------------------------------------------------------------------------------------------------


def brier_and_precision_score(
    y_true,
    y_proba,
    precision_threshold: float = 0.5,
    brier_threshold: float = 0.25,
) -> float:
    """precision_score - fast_brier_score_loss when both thresholds pass, else 0.

    Brier must be <= brier_threshold and precision must be >= precision_threshold
    (at a 0.5 decision boundary) for a non-zero result. Useful as a conservative
    optimisation objective that rewards only models that are simultaneously
    calibrated AND precise at the top.
    """
    from sklearn.metrics import precision_score

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = fast_brier_score_loss(y_true=y_true, y_prob=y_proba)
    if brier > brier_threshold:
        return 0.0
    y_pred = (y_proba > 0.5).astype(int)
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
    except Exception:
        return 0.0
    if precision < precision_threshold:
        return 0.0
    return float(precision - brier)


def make_brier_precision_scorer(precision_threshold: float = 0.5, brier_threshold: float = 0.25):
    """Return an sklearn scorer wrapping brier_and_precision_score (needs_proba=True)."""
    from sklearn.metrics import make_scorer

    # New sklearn (>=1.4) replaces `needs_proba` with `response_method`; fall back
    # to the legacy kwarg for older versions.
    try:
        return make_scorer(
            brier_and_precision_score,
            response_method="predict_proba",
            greater_is_better=True,
            precision_threshold=precision_threshold,
            brier_threshold=brier_threshold,
        )
    except TypeError:
        return make_scorer(
            brier_and_precision_score,
            needs_proba=True,
            greater_is_better=True,
            precision_threshold=precision_threshold,
            brier_threshold=brier_threshold,
        )
