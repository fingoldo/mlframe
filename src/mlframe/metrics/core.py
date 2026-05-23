
from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

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
from mlframe.core.stats import get_tukey_fences_multiplier_for_quantile

# NUMBA_NJIT_PARAMS lives in ``._numba_params`` so split-out sibling modules
# (``_calibration_plot.py``, etc.) can import the same object without
# duplicating the dict. Re-exported here to preserve every historical
# ``from mlframe.metrics.core import NUMBA_NJIT_PARAMS`` import site.
from ._numba_params import (  # noqa: F401
    NUMBA_NJIT_PARAMS,
    _PARALLEL_REDUCTION_THRESHOLD,
    _PARALLEL_MULTILABEL_THRESHOLD,
)


def numba_warmup() -> None:
    """Force-compile the numba kernels invoked by the calibration metric
    fastpath so the first ``train_mlframe_models_suite`` call doesn't pay
    the 3-5s cold-JIT tax. Subsequent calls in the same process - and in
    fact subsequent fresh processes that find the on-disk ``cache=True``
    artefacts - then start near-instantly.

    Long-running services (web apps, scheduled batch jobs, Jupyter
    kernels) should call this once at process boundary; one-shot scripts
    can skip it (the cold-JIT cost is shifted, not eliminated).

    On a 1M-row regression suite, profiled cold-vs-warm: 4.7s -> 1.3s
    (the 3.4s delta is exactly the numba-compile work that this helper
    pulls forward into the warmup window).
    """
    import numpy as _np
    # Tiny inputs that exercise the same dtype signatures the real
    # eval-metric callbacks hit (float64 N x K, int8 indicator).
    _logits1 = _np.array([0.0, 0.5, 1.0], dtype=_np.float32)
    _logits2 = _np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], dtype=_np.float32)
    _y_pred_NK = _np.array([[0.1, 0.9], [0.4, 0.6]], dtype=_np.float64)
    _y_true_NK = _np.array([[0, 1], [1, 0]], dtype=_np.int8)
    try:
        _cb_logits_to_probs_binary_seq(_logits1)
        _cb_logits_to_probs_binary_par(_logits1)
        _cb_logits_to_probs_multiclass_seq(_logits2)
        _cb_logits_to_probs_multiclass_par(_logits2)
        _batch_per_class_ice_kernel(
            _y_true_NK, _y_pred_NK, 10, True,
            3.0, 2.0, 0.8, 1.5, 0.1, 0.54, 0.0,
        )
    except Exception as _exc:
        # Warmup is best-effort; if a kernel signature mismatches a future
        # numba version, the next-suite cold compile will still recover.
        logger.debug("numba_warmup: skipped kernel due to %s", _exc)


def _assert_numba_nogil_active() -> bool:
    """Verify JIT-compiled kernels actually released the GIL (nogil=True took effect).

    Numba silently retains the GIL when a kernel references Python objects it cannot prove safe - the compile succeeds (nopython mode) but parallelism under ThreadPoolExecutor is lost without any warning. Returns True iff every compiled signature on the canary kernel reports `release_gil`.
    """
    try:
        canary = globals().get("fast_roc_auc")
        if canary is None or not hasattr(canary, "signatures"):
            return True
        overloads = getattr(canary, "overloads", {})
        for sig, compile_result in overloads.items():
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

    Calls all @njit functions with small dummy data to trigger JIT compilation before timing-sensitive operations. Warms up both float32 and float64 paths.

    Re-entrancy guard: this function calls ``training.dummy_baselines._warmup_numba_kernels``
    (forward), and that function calls back into us (reverse). Without the
    ``_in_progress`` sentinel the pair mutually recurses past the stack limit
    before either try/except sees the failure (observed 2026-05-20 on S:
    full-suite run). Flag is set on the function itself so it's process-local
    and visible from both sides.
    """
    if getattr(prewarm_numba_cache, "_in_progress", False):
        return
    prewarm_numba_cache._in_progress = True
    try:
        _prewarm_numba_cache_body()
    finally:
        prewarm_numba_cache._in_progress = False


def _prewarm_numba_cache_body():
    # Kick the loky/wmic physical-core-count probe in a background thread before numba JIT. The probe is a Windows wmic subprocess (~1.5s wall) that loky caches per-process; running it in parallel with the JIT compile overlaps the wait so the suite never pays the 1.5s when it later asks for cpu_count via joblib.
    try:
        import threading

        def _kick_cpu_count():
            try:
                from joblib.parallel import cpu_count as _cc
                _cc()
            except Exception:
                # Wave 43 (2026-05-20): daemon thread is fire-and-forget; without
                # this debug log a failure of the perf prefetch would be completely
                # invisible. Keep the swallow (failure has no semantic effect, the
                # main path calls cpu_count again later) but at least surface it.
                logger.debug("_kick_cpu_count: prefetch failed", exc_info=True)

        threading.Thread(target=_kick_cpu_count, daemon=True).start()
    except Exception:
        pass

    # iter199 (2026-05-23): pre-warm polars group_by + agg path. c0042 binary
    # profile attributed 2.557s to a single group_by(...).agg(...) call in
    # _per_group_predict_polars on the first invocation per process. polars'
    # query optimizer / Rust hash-aggregate kernel has a ~2-3s cold-start cost
    # that warms in <2ms with ANY group_by call (verified via bench:
    # cold=1.9ms tiny + 2.5s production-size; after warm: 0.5ms tiny + 2.5ms
    # big). Trigger the warm here so the first dummy-baselines /
    # per_group_predict call doesn't pay it. Tiny 10-row toy enum frame
    # exercises the same code path at near-zero data cost. Same pattern as
    # numba kernel prewarm; ~3s saved per process on binary / multiclass
    # combos that compute per-group prior baselines.
    try:
        import polars as _pl
        _enum_t = _pl.Enum(["a", "b", "c"])
        _warm_df = _pl.DataFrame({
            "cat": _pl.Series("cat", ["a", "b", "a", "c", "b"], dtype=_enum_t),
            "__y__": _pl.Series("__y__", [1.0, 2.0, 1.5, 3.0, 2.0]),
        })
        _ = _warm_df.group_by("cat").agg(
            _pl.col("__y__").mean().alias("__mean__"),
            _pl.len().alias("__size__"),
        )
        # Also warm the join path (used in _per_group_predict_polars._predict):
        _ = _warm_df.select("cat").join(_warm_df, on="cat", how="left")
    except Exception:
        pass

    # Numba compiles for each dtype separately.
    for dtype in [np.float32, np.float64]:
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=dtype)
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5], dtype=dtype)

        _ = fast_roc_auc(y_true, y_pred)
        _ = fast_aucs(y_true, y_pred)

        _ = fast_calibration_binning(y_true, y_pred, nbins=10)
        _ = fast_calibration_metrics(y_true, y_pred, nbins=10)

        _ = brier_score_loss(y_true, y_pred)
        _ = fast_brier_score_loss(y_true, y_pred)
        _ = fast_log_loss(y_true, y_pred)
        _ = maximum_absolute_percentage_error(y_true, y_pred)
        _ = probability_separation_score(y_true, y_pred)

        freqs_p, freqs_t, hits = fast_calibration_binning(y_true, y_pred, nbins=10)
        _ = calibration_metrics_from_freqs(
            freqs_predicted=freqs_p, freqs_true=freqs_t, hits=hits,
            nbins=10, array_size=len(y_true), use_weights=True,
        )

    for dtype in [np.int32, np.int64]:
        y_true_int = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=dtype)
        y_pred_int = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=dtype)
        _ = fast_classification_report(y_true_int, y_pred_int, nclasses=2)
        _ = fast_precision(y_true_int, y_pred_int, nclasses=2)
        _ = compute_pr_recall_f1_metrics(y_true_int, y_pred_int)

    _ = integral_calibration_error_from_metrics(0.01, 0.01, 0.9, 0.25, 0.7, 0.7)

    # Prewarm calibration-report inner kernels using the dominant suite dtype combo. In the per-class loop of report_probabilistic_model_perf, y_true arrives as numpy bool (``targets == class_name``) and y_pred as float64; prewarming additional dtype-pairs costs ~5-10s each in JIT compile time.
    _yt_bool = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.bool_)
    _yp_f64 = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5], dtype=np.float64)
    try:
        _ = compute_ece_and_brier_decomposition(_yt_bool, _yp_f64, nbins=10)
        _ = fast_aucs_per_group_optimized(y_true=_yt_bool, y_score=_yp_f64, group_ids=None)
        _ = fast_log_loss(_yt_bool, _yp_f64)
        _ = fast_ice_only(_yt_bool, _yp_f64, nbins=10, use_weights=True)
    except Exception:
        pass

    # iter192 (2026-05-23): also prewarm fast_aucs_per_group_optimized with
    # group_ids supplied (different numba signature than group_ids=None) and
    # the (bool, float64) brier through the public wrapper to hit BOTH _seq
    # and _par branches. c0037 binary profile attributed 693ms compile to
    # fast_aucs_per_group_optimized (group_ids supplied) + 483ms to
    # fast_brier_score_loss _seq path (short per-group brier slices --
    # iter190 only covered the >=threshold _par path).
    try:
        _gi = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0], dtype=np.int64)
        _ = fast_aucs_per_group_optimized(y_true=_yt_bool, y_score=_yp_f64, group_ids=_gi)
        # Short array -> dispatches to _fast_brier_score_loss_seq, distinct
        # numba signature per dtype combo.
        _ = fast_brier_score_loss(_yt_bool, _yp_f64)
    except Exception:
        pass

    # iter198 (2026-05-23) bench-attempt-rejected: tried prewarming
    # ``fast_aucs(bool, float64)`` to eliminate the 667ms _compile_for_args
    # attributed to fast_aucs in c0133 multiclass. Verification re-run wall:
    # 7.97s -> 8.22s (within ~250ms run-to-run noise; (bool, f64) confirmed
    # warm via direct bench at 0.4ms / call). Suspected reason: the runtime
    # exercises ANOTHER dtype combo (likely int8/int64 + float32) that's
    # still cold, so the (bool, f64) prewarm doesn't move the needle.
    # Adding all 10 dtype combos would add ~3.4s to prewarm time -- worse
    # than the current state. Pinning a smaller subset is unbounded; skip.
    _yti = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    _ypi = np.array([0, 1, 0, 0, 1, 1], dtype=np.int64)
    try:
        _ = format_classification_report(_yti, _ypi, nclasses=2)
    except Exception:
        pass

    logits_binary = np.array([-1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5, -1.5, 0.25, -0.25], dtype=np.float64)
    _ = cb_logits_to_probs_binary(logits_binary)

    logits_multi = np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0], [0.0, 1.0, -1.0]], dtype=np.float64)
    _ = cb_logits_to_probs_multiclass(logits_multi)

    # Prewarm parallel-numba variants. Each `_par` variant is a separate numba compilation; the `parallel=True` IR adds ~1-3s per kernel on first call from a fresh process.
    try:
        _yt_f64 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
        _yp_f64 = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5], dtype=np.float64)
        _ = _fast_brier_score_loss_par(_yt_f64, _yp_f64)
        # iter190 (2026-05-23): also prewarm bool->float64 signature for the
        # _par reductions. c0023 profile attributed 4.156s of
        # _compile_for_args to fast_brier_score_loss across 2 fresh compiles
        # -- the (bool, float64) signature emitted by multilabel per-class
        # loops (``y_true = targets == class_name`` -> ndarray[bool]) was
        # NOT covered. Pre-warming once at import time pays the same 4s
        # upfront but moves it OUT of the first-fit hot path. Same fix
        # applied to fast_log_loss_binary_par for symmetry.
        _yt_bool = _yt_f64.astype(np.bool_)
        _ = _fast_brier_score_loss_par(_yt_bool, _yp_f64)
        _ = _fast_log_loss_binary_par(_yt_bool, _yp_f64, 1e-15)
        _ = _fast_log_loss_binary_par(_yt_f64, _yp_f64, 1e-15)
        _yt_i64 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        _yp_i64 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int64)
        _ = _compute_pr_recall_f1_metrics_par(_yt_i64, _yp_i64)
        _ml_yt = np.zeros((10, 3), dtype=np.uint8); _ml_yt[:5, 0] = 1
        _ml_yp = np.zeros((10, 3), dtype=np.uint8); _ml_yp[:5, 0] = 1
        _ = _fast_subset_accuracy_par(_ml_yt, _ml_yp)
        _ = _fast_jaccard_score_par(_ml_yt, _ml_yp)

        _logits_b = np.array([-1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5, -1.5, 0.25, -0.25], dtype=np.float64)
        _ = _cb_logits_to_probs_binary_par(_logits_b)
        _logits_mc = np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0], [0.0, 1.0, -1.0]], dtype=np.float64)
        _ = _cb_logits_to_probs_multiclass_par(_logits_mc)
        _ = _max_abs_pct_error_kernel_par(_yt_f64, _yp_f64)
        _yt_i64_psep = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        _ = _probability_separation_score_par(_yt_i64_psep, _yp_f64, 1, 0.5)

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
        # Non-fatal: a bad cache or numba-runtime hiccup; the seq path still works.
        pass

    yt_ml = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    yp_ml = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1]], dtype=np.uint8)
    _ = _fast_hamming_loss_seq(yt_ml, yp_ml)
    _ = _fast_hamming_loss_par(yt_ml, yp_ml)
    _ = _fast_subset_accuracy_seq(yt_ml, yp_ml)
    _ = _fast_jaccard_score_seq(yt_ml, yp_ml)
    # Bitmap variant takes packed uint64 + K; prewarm K<=64 path.
    yt_packed = np.array([0b011, 0b101, 0b110, 0b001], dtype=np.uint64)
    yp_packed = np.array([0b110, 0b101, 0b100, 0b011], dtype=np.uint64)
    _ = _fast_jaccard_bitmap_seq(yt_packed, yp_packed, 3)

    # Verify nogil=True actually stuck; silent fallback would make parallel val/test metric evaluation secretly sequential.
    _assert_numba_nogil_active()

    # Prewarm `_batch_per_class_ice_kernel`, the per-class parallel kernel inside `compute_probabilistic_multiclass_error`. Compiles separately from the sequential `fast_ice_only` variant prewarmed above; use the dtype combo the suite always sends (int8 indicator + float64 probs + K=3).
    try:
        from mlframe.metrics.core import _batch_per_class_ice_kernel  # noqa: F811
        _yt_nk4_pw = np.zeros((10, 3), dtype=np.int8)
        _yt_nk4_pw[0, 0] = 1; _yt_nk4_pw[1, 1] = 1; _yt_nk4_pw[2, 2] = 1
        _yp_nk4_pw = np.random.RandomState(0).rand(10, 3).astype(np.float64)
        _ = _batch_per_class_ice_kernel(
            _yt_nk4_pw, _yp_nk4_pw, 10, True,
            3.0, 2.0, 0.8, 1.5, 0.1, 0.54, 0.0,
        )
    except Exception:
        pass

    # Warm feature_selection numba kernels. Without this, the first MRMR.fit call pays ~60s of cumulative JIT compile. Lazy import keeps this module's import cost unchanged.
    try:
        from mlframe.feature_selection.filters._prewarm import (
            prewarm_fs_numba_cache,
        )
        prewarm_fs_numba_cache()
    except Exception:
        pass

    # Warm dummy_baselines kernels. The suite already calls `_warmup_numba_kernels` early in `train_mlframe_models_suite`, but that lands inside the suite wall-time; warming here shifts cost out of the user-visible timer.
    try:
        from mlframe.training.dummy_baselines import _warmup_numba_kernels
        _warmup_numba_kernels()
    except Exception:
        pass

    # Prewarm-import the heavy neural-net stack. `mlframe.lightninglib` / `mlframe.training.neural` pulls in PyTorch Lightning, which is a ~275s cold-import on Windows. The cost otherwise lands inside the suite call because the import is deferred until `mlp` is in the model list. Triggered ONLY when lightning is already discoverable; otherwise the import attempt itself would be a 5-10s ModuleNotFoundError walk through sys.path.
    try:
        import importlib.util as _ilu
        if _ilu.find_spec("lightning") is not None:
            try:
                import lightning.fabric  # noqa: F401
            except Exception:
                pass
            try:
                import mlframe.lightninglib  # noqa: F401
            except Exception:
                pass
        # `pytorch_lightning` is a separate package from `lightning` (legacy alias kept for back-compat); cold import is ~500s on Windows for the currently-pinned version.
        if _ilu.find_spec("pytorch_lightning") is not None:
            try:
                import pytorch_lightning  # noqa: F401
            except Exception:
                pass
        # `shap` cold import is ~228s on Windows (includes `shap.utils.transformers` walking the local transformers registry). The suite imports shap inside trainer.py when use_shap=True.
        if _ilu.find_spec("shap") is not None:
            try:
                import shap  # noqa: F401
                import shap.utils.transformers  # noqa: F401
                # Match the runtime monkeypatch so prewarm leaves shap in the state the suite expects.
                shap.utils.transformers.is_transformers_lm = lambda model: False
            except Exception:
                pass
        try:
            import mlframe.training.neural  # noqa: F401
        except Exception:
            pass
    except Exception:
        pass

    # Warm cupy GPU AUC kernels. `compute_batch_aucs` dispatches to `gpu_multiple_roc_auc_scores` / `gpu_multiple_pr_auc_scores` when N>=100k AND M>=5. cupy compiles CUDA kernels via NVRTC on first call (~128s per fresh process). No-op when cupy isn't installed.
    # Gate the WHOLE block on is_gpu_metrics_available() (which now probes
    # via an NVRTC compile, so it returns False on broken cupy / mismatched
    # CUDA installs). The try/except below only catches Python exceptions -
    # without the gate, a broken cupy install can HANG inside cp.argsort()
    # rather than raising, leaving the prewarm phase wedged (observed
    # 2026-05-20 on D: with cupy CUDA-Devices-Unavailable: prewarm timed
    # out at 180s before any test ran).
    if is_gpu_metrics_available():
        try:
            from mlframe.metrics.core import (
                gpu_multiple_roc_auc_scores, gpu_multiple_pr_auc_scores,
                gpu_multiple_rmse_scores,
            )
            _yt_gpu = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 16, dtype=np.int8)
            _yp_gpu = np.random.RandomState(0).rand(len(_yt_gpu), 3).astype(np.float64)
            _ = gpu_multiple_roc_auc_scores(_yt_gpu, _yp_gpu)
            _ = gpu_multiple_pr_auc_scores(_yt_gpu, _yp_gpu)
            # `gpu_multiple_rmse_scores` has separate cupy kernels for the 2-D fallback and the 1-D fastpath; each path's first call compiles a fresh NVRTC kernel.
            _yt_rmse = _yp_gpu[:, 0]
            _ = gpu_multiple_rmse_scores(_yt_rmse, _yp_gpu)
        except Exception:
            pass

    # Warm `ranking_metrics._summary_batched_kernel` (parallel njit). On LTR combos `compute_ranking_summary` is called once per dummy baseline; the first call eats the entire JIT-compile budget. Compile with the canonical dtype combo used by `compute_ranking_summary` itself.
    try:
        from mlframe.metrics.ranking import _summary_batched_kernel
        _yt_rank = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 0.0], dtype=np.float64)
        _ys_rank = np.array([0.3, 0.9, 0.5, 0.7, 0.2, 0.1], dtype=np.float64)
        _gs_rank = np.array([0, 3, 6], dtype=np.int64)
        _ks_rank = np.array([1, 5, 10], dtype=np.int64)
        _ = _summary_batched_kernel(_yt_rank, _ys_rank, _gs_rank, _ks_rank)
    except Exception:
        pass


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
    """Parallel sigmoid."""
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
    """Sequential variant. Public wrapper auto-dispatches at N>=100k.

    bench-attempt-rejected (2026-05-22, c0108 / iter165): transposing
    the (K, N) input to (N, K) first to make inner-loop reads
    contiguous is 11-21% SLOWER at every size (K=3, 8; N=50k..1M).
    Modern CPU prefetchers handle the stride-N reads well for small K;
    the upfront 24-100 MB transpose memcpy never earns itself back.
    Bit-equivalent output. Bench:
    profiling/bench_cb_logits_softmax_layout.py.
    """
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
    """Parallel softmax."""
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


def fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> float:
    """Compute ROC AUC efficiently using numba.

    Note: np.argsort needs to stay out of njitted func.
    """
    # **kwargs absorbs sklearn's unexpected params. Explicitly reject sample_weight rather than silently ignoring it.
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
    desc_score_indices = np.argsort(y_score, kind="stable")[::-1]  # Wave 57: stable sort for reproducibility on tied scores
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


# GPU dispatch + batch metrics live in ``_gpu_metrics.py``; re-exported here so
# historical ``from mlframe.metrics.core import compute_batch_aucs`` (and the other
# moved names) imports continue to resolve. See sibling for SSOT.
from ._gpu_metrics import (  # noqa: E402,F401
    _GPU_BATCH_THRESHOLD_N, _GPU_BATCH_THRESHOLD_M,
    _GPU_AVAILABLE, _NUMBA_CUDA_AVAILABLE,
    _CUPY_SSE_PER_COL, _NUMBA_RMSE_KERNEL,
    set_gpu_thresholds, is_gpu_metrics_available,
    _is_numba_cuda_available, _require_cupy,
    _get_cupy_sse_kernel, _get_numba_rmse_kernel,
    gpu_multiple_rmse_scores, gpu_multiple_roc_auc_scores,
    gpu_multiple_pr_auc_scores,
    _normalize_scores_2d, compute_batch_rmse, compute_batch_aucs,
    _resolve_backend,
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


# Multilabel metrics (hamming_loss, subset_accuracy, jaccard_score_multilabel +
# private numba kernels + bitmap-popcount fastpath) live in
# ``_multilabel_metrics.py``; re-exported below so historical
# ``from mlframe.metrics.core import hamming_loss`` imports keep resolving.
from ._multilabel_metrics import (  # noqa: E402,F401
    _fast_hamming_loss_seq, _fast_hamming_loss_par,
    _fast_subset_accuracy_seq, _fast_subset_accuracy_par,
    _fast_jaccard_score_seq, _fast_jaccard_score_par,
    _popcount64, _fast_jaccard_bitmap_seq,
    _can_use_bitmap_jaccard, _pack_for_bitmap,
    _coerce_multilabel_array, _validate_multilabel_pair,
    hamming_loss, subset_accuracy, jaccard_score_multilabel,
)


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


# calibration plot rendering (render_title_metric_token, fast_calibration_binning,
# _close_unless_interactive, show_calibration_plot, DEFAULT_TITLE_METRICS_TOKENS)
# moved to sibling _calibration_plot.py; re-exported below.
from ._calibration_plot import (  # noqa: F401, E402
    DEFAULT_TITLE_METRICS_TOKENS,
    render_title_metric_token,
    fast_calibration_binning,
    _close_unless_interactive,
    show_calibration_plot,
)


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
def _max_abs_pct_error_kernel_par(y_true: np.ndarray, y_pred: np.ndarray, nthr: int) -> Tuple[float, int]:
    """Parallel variant. ~2.3× faster than seq at N=1M.

    NOTE: ``if err > max_err: max_err = err`` inside ``prange`` is a
    race -- numba auto-detects ``+=`` as a reduction but NOT if-based
    max-update; concurrent threads can drop max-updates. Solution:
    per-thread max array + final reduction outside the prange.

    ``nthr`` is passed in (rather than called via numba.get_num_threads
    inside the kernel) so the @njit-cache can persist across runs.
    get_num_threads is a ctypes call that triggers the NumbaWarning
    "Cannot cache compiled function as it uses dynamic globals".
    """
    n = len(y_true)
    epsilon = np.finfo(np.float64).eps
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


# Module-level set: (n_zero, n_total) tuples for which the
# ``maximum_absolute_percentage_error: N of M y_true entries are zero``
# warning has already fired this process. Auto-cleared by interpreter
# shutdown. NOT thread-safe but the worst case is a duplicate warning in
# a rare race - the correctness signal is preserved.
_MAPE_ZERO_WARN_SEEN: set = set()


def maximum_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Auto seq/par dispatch. Parallel only wins at large N (race-free
    # max via per-thread accumulator + final reduction; lose-band runs
    # to ~200k due to setup cost).
    if len(y_true) >= 500_000:
        value, n_zero = _max_abs_pct_error_kernel_par(y_true, y_pred, numba.get_num_threads())
    else:
        value, n_zero = _max_abs_pct_error_kernel(y_true, y_pred)
    if n_zero > 0:
        # Rate-limit: emit the warning once per (n_zero, n_total) shape per
        # process. The metric is computed on train/val/test/OOF splits and
        # often by the per-feature ablation loop in BaselineDiagnostics, so
        # the same warning fires 4-15 times per training run with identical
        # content. Once is enough to alert the user that MAPE is mathematically
        # unreliable on their target; the rest is noise.
        _key = (int(n_zero), int(len(y_true)))
        if _key not in _MAPE_ZERO_WARN_SEEN:
            _MAPE_ZERO_WARN_SEEN.add(_key)
            logger.warning(
                "maximum_absolute_percentage_error: %d of %d y_true entries are zero; "
                "the epsilon fallback makes those ratios dominate the result. "
                "(further identical warnings suppressed this process)",
                n_zero, len(y_true),
            )
    return value


# Calibration metric kernels (CMAEW, ECE, Murphy Brier-decomp,
# fast_calibration_metrics) live in ``_calibration_metrics.py``; re-exported
# below so historical
# ``from mlframe.metrics.core import calibration_metrics_from_freqs`` (and the
# other moved names) imports continue to resolve.
from ._calibration_metrics import (  # noqa: E402,F401
    calibration_metrics_from_freqs,
    compute_ece_and_brier_decomposition,
    fast_calibration_metrics,
)

def fast_aucs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute both ROC AUC and PR AUC efficiently."""
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_score, (pd.Series, pl.Series)):
        y_score = y_score.to_numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, -1]
    desc_score_indices = np.argsort(y_score, kind="stable")[::-1]  # Wave 57: stable sort for reproducibility on tied scores
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


# Per-group AUC helpers live in ``_auc_per_group.py``; re-exported below so
# historical ``from mlframe.metrics.core import fast_aucs_per_group`` (and the
# other moved names) imports continue to resolve. See sibling for SSOT.
from ._auc_per_group import (  # noqa: E402,F401
    fast_aucs_per_group, fast_aucs_per_group_optimized,
    compute_grouped_group_aucs, fast_numba_aucs_simple,
    compute_mean_aucs_per_group,
)


# Classification + calibration report block moved to _classification_report.py.
from ._classification_report import (  # noqa: F401, E402
    format_classification_report,
    _compute_pr_recall_f1_metrics_seq,
    _compute_pr_recall_f1_metrics_par,
    compute_pr_recall_f1_metrics,
    fast_calibration_report,
    _batch_per_class_ice_kernel,
    fast_ice_only,
    predictions_time_instability,
)

# ICE metric + ``compute_probabilistic_multiclass_error`` live in
# ``_ice_metric.py``; re-exported below so historical
# ``from mlframe.metrics.core import ICE`` / ``compute_probabilistic_multiclass_error``
# imports continue to resolve. See sibling for SSOT.
from ._ice_metric import (  # noqa: E402,F401
    compute_probabilistic_multiclass_error,
    ICE,
    _install_catboost_sklearn_clone_patch,
)


# ICE-from-base-metrics aggregator lives in ``_calibration_metrics.py`` (it
# composes the calibration outputs into the single ICE score). Re-exported
# below so historical ``from mlframe.metrics.core import integral_calibration_error_from_metrics``
# imports keep resolving.
from ._calibration_metrics import integral_calibration_error_from_metrics  # noqa: E402,F401

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


# Crossover threshold for parallel kernels. See ``_numba_params.py`` (SSOT).
# Sum-reduction kernels (brier, log loss, prf1 counts) parallel-win from
# N≈50-100k upwards. Multilabel row-loop kernels (subset accuracy, jaccard)
# win from N≈10-50k. Conservative thresholds chosen to avoid the lose-band
# at low N.


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


# Regression metrics live in ``_regression_metrics.py``; re-exported below to keep
# ``from mlframe.metrics.core import fast_*`` imports stable. See sibling for SSOT.
from ._regression_metrics import (  # noqa: E402,F401
    _fast_mae_seq, _fast_mae_par, _fast_mse_seq, _fast_mse_par,
    _fast_max_error_seq, _fast_r2_score_seq, _fast_r2_score_par,
    _fast_r2_variance_seq,
    _fast_mae_weighted_seq, _fast_mae_weighted_par,
    _fast_mse_weighted_seq, _fast_mse_weighted_par,
    _fast_r2_score_weighted_seq, _fast_r2_score_weighted_par,
    _aggregate_multioutput, _to_2d,
    fast_mean_absolute_error, fast_mean_squared_error,
    fast_root_mean_squared_error, fast_max_error, fast_r2_score,
    _fused_regression_pass1_seq, _fused_regression_pass1_par,
    _fused_regression_pass2_seq, _fused_regression_pass2_par,
    fast_regression_metrics_block,
)


# Binary log-loss + probability-separation kernels live in
# ``_log_loss_and_separation.py``; re-exported below so historical
# ``from mlframe.metrics.core import fast_log_loss`` imports keep resolving.
from ._log_loss_and_separation import (  # noqa: E402,F401
    _fast_log_loss_binary_seq, _fast_log_loss_binary_par,
    fast_log_loss_binary, fast_log_loss,
    _probability_separation_score_seq, _probability_separation_score_par,
    probability_separation_score,
)


# Fairness / robustness subgrouping + metrics live in ``_fairness_metrics.py``;
# re-exported below so historical
# ``from mlframe.metrics.core import create_fairness_subgroups`` (and the other
# moved names) imports continue to resolve. See sibling for SSOT.
from ._fairness_metrics import (  # noqa: E402,F401
    create_fairness_subgroups, create_fairness_subgroups_indices,
    create_robustness_standard_bins, compute_fairness_metrics,
    create_robustness_subgroups, create_robustness_subgroups_indices,
    compute_robustness_metrics, robust_mlperf_metric,
)


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
