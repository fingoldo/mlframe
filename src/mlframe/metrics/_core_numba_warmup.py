"""Numba warmup helpers for ``mlframe.metrics.core``.

Carved from ``core.py`` to keep the parent facade below the LOC budget.
Public symbols are re-exported from the parent so historical
``from mlframe.metrics.core import numba_warmup`` imports continue to
resolve.
"""

from __future__ import annotations

import logging
import os as _os

import numba
import numpy as np

logger = logging.getLogger(__name__)


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
    from .core import (
        _cb_logits_to_probs_binary_seq, _cb_logits_to_probs_binary_par,
        _cb_logits_to_probs_multiclass_seq, _cb_logits_to_probs_multiclass_par,
        _batch_per_class_ice_kernel,
    )
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
        from .core import fast_roc_auc as canary
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

    Re-entrancy guard: this function calls ``training.baselines.dummy._warmup_numba_kernels``
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
    from .core import (
        fast_roc_auc, fast_aucs, fast_calibration_binning, fast_calibration_metrics,
        brier_score_loss, fast_brier_score_loss, fast_log_loss,
        maximum_absolute_percentage_error, probability_separation_score,
        calibration_metrics_from_freqs, fast_classification_report, fast_precision,
        compute_pr_recall_f1_metrics, integral_calibration_error_from_metrics,
        compute_ece_and_brier_decomposition, compute_ece_debiased, compute_brier_decomposition_debiased,
        compute_ece_brier_full_and_debiased,
        fast_aucs_per_group_optimized,
        fast_ice_only, format_classification_report,
        cb_logits_to_probs_binary, cb_logits_to_probs_multiclass,
        _fast_brier_score_loss_par, _fast_log_loss_binary_par,
        _compute_pr_recall_f1_metrics_par, _fast_subset_accuracy_par,
        _fast_jaccard_score_par, _cb_logits_to_probs_binary_par,
        _cb_logits_to_probs_multiclass_par, _max_abs_pct_error_kernel_par,
        _probability_separation_score_par,
        _fast_mae_seq, _fast_mae_par, _fast_mae_weighted_seq, _fast_mae_weighted_par,
        _fast_mse_seq, _fast_mse_par, _fast_mse_weighted_seq, _fast_mse_weighted_par,
        _fast_max_error_seq, _fast_r2_score_seq, _fast_r2_score_par,
        _fast_r2_score_weighted_seq, _fast_r2_score_weighted_par, _fast_r2_variance_seq,
        _fast_hamming_loss_seq, _fast_hamming_loss_par,
        _fast_subset_accuracy_seq, _fast_jaccard_score_seq,
        _fast_jaccard_bitmap_seq,
        is_gpu_metrics_available,
    )

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
        from mlframe.metrics.calibration import _fast_calibration_binning_prange
        _ = _fast_calibration_binning_prange(y_true, y_pred, nbins=10)
        _ = fast_calibration_metrics(y_true, y_pred, nbins=10)

        _ = brier_score_loss(y_true, y_pred)
        _ = fast_brier_score_loss(y_true, y_pred)
        _ = fast_log_loss(y_true, y_pred)
        # MAPE warmup needs a NON-ZERO y_true vector: the classifier-style {0,1}
        # array used above would trigger the rate-limited "N of M y_true entries
        # are zero" warning at import time, scaring users with a 5-of-10-zero
        # message that has nothing to do with their actual training data. The
        # numba kernel compiles on dtype, not on values, so any non-zero vector
        # works.
        _y_mape = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0], dtype=dtype)
        _p_mape = np.array([1.1, 0.9, 2.2, 1.8, 3.3, 2.7, 4.4, 3.6, 5.5, 4.5], dtype=dtype)
        _ = maximum_absolute_percentage_error(_y_mape, _p_mape)
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
        _ = compute_ece_debiased(_yt_bool, _yp_f64, nbins=10)
        _ = compute_brier_decomposition_debiased(_yt_bool, _yp_f64, nbins=10)
        _ = compute_ece_brier_full_and_debiased(_yt_bool, _yp_f64, nbins=10)
        _ = fast_aucs_per_group_optimized(y_true=_yt_bool, y_score=_yp_f64, group_ids=None)
        # iter86: the report now takes the fused ROC/PR/KS walk (return_ks=True -> fast_numba_aucs_with_ks), a distinct numba signature.
        _ = fast_aucs_per_group_optimized(y_true=_yt_bool, y_score=_yp_f64, group_ids=None, return_order=True, return_ks=True)
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
        _ = _max_abs_pct_error_kernel_par(_yt_f64, _yp_f64, numba.get_num_threads())
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

    # Wrapped in try/except for the same defensive reason as the regression block above: a bad numba
    # cache or runtime hiccup on an exotic build should degrade to seq, not abort the whole prewarm.
    # The parallel multilabel kernels (``_fast_hamming_loss_par`` / ``_fast_jaccard_score_par``) were
    # rewritten to a per-row MAP + serial final reduction so they compile cleanly on numba 0.63.x,
    # which previously aborted them with ``AssertionError: unexpected cycle in lookup()`` from
    # ``numba/parfors/parfor.py`` (the public dispatcher would then crash on the large-N path).
    try:
        yt_ml = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=np.uint8)
        yp_ml = np.array([[1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1]], dtype=np.uint8)
        _ = _fast_hamming_loss_seq(yt_ml, yp_ml)
        _ = _fast_hamming_loss_par(yt_ml, yp_ml)
        _ = _fast_subset_accuracy_seq(yt_ml, yp_ml)
        _ = _fast_jaccard_score_seq(yt_ml, yp_ml)
        _ = _fast_jaccard_score_par(yt_ml, yp_ml)
        # Bitmap variant takes packed uint64 + K; prewarm K<=64 path.
        yt_packed = np.array([0b011, 0b101, 0b110, 0b001], dtype=np.uint64)
        yp_packed = np.array([0b110, 0b101, 0b100, 0b011], dtype=np.uint64)
        _ = _fast_jaccard_bitmap_seq(yt_packed, yp_packed, 3)
    except Exception:
        # Catches the numba-internal ``AssertionError`` raised from
        # ``parfor.py:3886`` lookup() as well as any compile / runtime fault
        # in the sequential helpers. AssertionError inherits from Exception
        # so the bare ``except Exception`` is sufficient.
        pass

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
        from mlframe.feature_selection.filters import prewarm_fs_numba_cache
        prewarm_fs_numba_cache()
    except Exception:
        pass

    # Warm dummy_baselines kernels. The suite already calls `_warmup_numba_kernels` early in `train_mlframe_models_suite`, but that lands inside the suite wall-time; warming here shifts cost out of the user-visible timer.
    try:
        from mlframe.training.baselines import _warmup_numba_kernels
        _warmup_numba_kernels()
    except Exception:
        pass

    # Prewarm-import the heavy neural-net stack. `mlframe.lightninglib` / `mlframe.training.neural` pulls in PyTorch Lightning, which is a ~275s cold-import on Windows. The cost otherwise lands inside the suite call because the import is deferred until `mlp` is in the model list. Triggered ONLY when lightning is already discoverable; otherwise the import attempt itself would be a 5-10s ModuleNotFoundError walk through sys.path.
    #
    # iter604 (perf gate): the heavy-lib prewarm here ALWAYS fires when
    # dummy_baselines is enabled (the suite default), even when the
    # caller's model list has no MLP/recurrent and the user never sets
    # ``use_shap=True``. Profile measurements at 100k:
    #   c0073 (regression hgb+xgb)      9.07s in this block (46% of 19.8s wall)
    #   c0002 (LTR lgb)                 7.98s (72% of 11.1s wall)
    #   c0006 (binary cb+xgb+ens)       9.65s (20% of 49.3s wall)
    #   c0143 (multiclass xgb)          6.74s (33% of 20.3s wall)
    # For non-neural / non-shap workloads the cost is pure waste -- the
    # imports never get used. The MLFRAME_PREWARM_HEAVY_LIBS env var
    # lets a caller opt out:
    #   "0" / "false" / "no" / "skip"  -> skip all heavy-lib prewarms
    #   anything else (incl. unset)    -> current behavior (warm if
    #                                     discoverable)
    # Default unchanged so production / interactive sessions keep the
    # "shift cost out of user-visible timer" property; short-lived
    # CLI tools / fuzz harnesses / CI jobs that know they don't use
    # neural can set the env var once and save 6-10s per process.
    #
    # iter618 end-to-end validation on c0089_777f5fe3 (binary lgb-only
    # @100k) measured 32.17s suite default vs 13.86s with gate=0
    # (-18.3s, -56% suite wall). The full saving is 2.3x larger than
    # the initial subprocess-prewarm bench predicted because the
    # heavy-lib IMPORT CASCADE (lightning -> torch -> cuda probe;
    # shap -> transformers registry walk; pytorch_lightning callback
    # registry) rippling through other module loads is not captured by
    # pstats cumtime on the prewarm body itself (pstats shows only
    # +0.31s on the body call), but DOES land in the user-visible
    # wall-clock. The env var is therefore the highest-impact perf
    # toggle in mlframe for short-lived non-neural runs.
    _prewarm_heavy = _os.environ.get("MLFRAME_PREWARM_HEAVY_LIBS", "").strip().lower()
    _skip_heavy = _prewarm_heavy in {"0", "false", "no", "skip"}
    if not _skip_heavy:
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
