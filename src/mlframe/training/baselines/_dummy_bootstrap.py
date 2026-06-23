"""Bootstrap CIs + paired-bootstrap robustness for ``dummy_baselines``.

Split out of ``dummy_baselines.py`` to keep the parent below the 1k-line
monolith threshold. Behaviour preserved bit-for-bit; the parent re-exports
the public-looking entries so the orchestrator in
``compute_dummy_baselines`` continues to call them via the same names.

What lives here:
  - ``_paired_bootstrap_vs_runner_up`` (D2 paired-bootstrap robustness check)
  - ``_vectorized_bootstrap_logloss_samples`` (numba-backed log-loss
    bootstrap sample generator)
  - ``_bootstrap_ci_for_strongest`` (per-metric bootstrap CI for the
    strongest baseline)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error

# Numba kernels live in the ``_dummy_numba_kernels.py`` leaf so this sibling
# can depend on them without re-entering the parent module.
from ._dummy_numba_kernels import _NUMBA_AVAILABLE
if _NUMBA_AVAILABLE:
    from ._dummy_numba_kernels import (
        _numba_bootstrap_logloss_binary_samples,
        _numba_bootstrap_mae_samples,
        _numba_bootstrap_rmse_samples,
        _numba_paired_bootstrap_logloss_binary,
        _numba_paired_bootstrap_mae,
        _numba_paired_bootstrap_rmse,
    )
# ``_pick_strongest`` lives in ``dummy_baselines.py`` itself; imported
# lazily inside the function bodies that need it to keep the
# ``dummy_baselines -> _dummy_bootstrap -> dummy_baselines`` import cycle
# broken.

logger = logging.getLogger(__name__)


def _paired_bootstrap_vs_runner_up(
    target_type: str,
    strongest: str,
    primary_metric: str,
    table: pd.DataFrame,
    val_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """D2 paired-bootstrap robustness check.

    Picks the runner-up baseline by primary metric on val (test
    fallback), runs a paired bootstrap (1000 resamples by default) on
    the same resample-indices for both predictors, and returns:

      ``{"runner_up": name,
         "delta": strongest_val - runner_up_val (or model_val - dummy_val),
         "delta_ci": (lo, hi),
         "p_strongest_beats": fraction of resamples where strongest wins}``

    Returns ``None`` when no runner-up exists or metric not computable.
    """
    if strongest not in table.index:
        return None

    # Pick runner-up = second-best by primary_metric on the reference split.
    if primary_metric in table.columns:
        ref_col = primary_metric
    else:
        ref_col = primary_metric.replace("val_", "test_") if primary_metric.startswith("val_") else None
        if ref_col is None or ref_col not in table.columns:
            return None
    series = table[ref_col].dropna()
    if strongest not in series.index or len(series) < 2:
        return None
    # Wave 20 fix: registry dispatcher (same shape as _pick_strongest above).
    from ..metrics_registry import metric_name_higher_is_better as _mhb
    _direction = _mhb(primary_metric)
    minimize = True if _direction is None else (not _direction)
    series_excl_strongest = series.drop(index=strongest)
    if series_excl_strongest.empty:
        return None
    runner_up = series_excl_strongest.idxmin() if minimize else series_excl_strongest.idxmax()

    # Need predictions for both on the same split.
    sp_val = val_preds.get(strongest)
    sp_test = test_preds.get(strongest)
    rp_val = val_preds.get(runner_up)
    rp_test = test_preds.get(runner_up)

    # Pick split where both have predictions + target is present.
    use_val = (
        val_y is not None and sp_val is not None and rp_val is not None
        and len(sp_val) == len(val_y) and len(rp_val) == len(val_y)
    )
    use_test = (
        test_y is not None and sp_test is not None and rp_test is not None
        and len(sp_test) == len(test_y) and len(rp_test) == len(test_y)
    )
    if use_val:
        y_ref, p1, p2 = val_y, sp_val, rp_val
    elif use_test:
        y_ref, p1, p2 = test_y, sp_test, rp_test
    else:
        return None

    # Metric callable. Limited to RMSE / MAE / log_loss for robustness;
    # NDCG / AUC paired-bootstrap requires per-query / per-class plumbing
    # that is out of scope here (returns None to skip TIE check).
    n = len(y_ref)
    if n < 10:
        return None

    # Numba-accelerated paths for RMSE / MAE / binary log-loss -- ~30-340x
    # faster than the Python loop with sklearn metric inside (measured:
    # 1100ms -> 3.4ms on n=1500, 1000 resamples for RMSE). Falls back to
    # sklearn loop for log_loss with non-binary preds, multilabel macro
    # log-loss (no numba kernel -- cost > value at the n<2000 gate), and
    # when numba unavailable.
    deltas = None
    if _NUMBA_AVAILABLE:
        try:
            if "RMSE" in primary_metric:
                y_arr = np.ascontiguousarray(y_ref, dtype=np.float64)
                p1_arr = np.ascontiguousarray(p1, dtype=np.float64)
                p2_arr = np.ascontiguousarray(p2, dtype=np.float64)
                deltas = _numba_paired_bootstrap_rmse(
                    y_arr, p1_arr, p2_arr, int(n_resamples), int(seed),
                )
                if not minimize:
                    deltas = -deltas
            elif "MAE" in primary_metric:
                y_arr = np.ascontiguousarray(y_ref, dtype=np.float64)
                p1_arr = np.ascontiguousarray(p1, dtype=np.float64)
                p2_arr = np.ascontiguousarray(p2, dtype=np.float64)
                deltas = _numba_paired_bootstrap_mae(
                    y_arr, p1_arr, p2_arr, int(n_resamples), int(seed),
                )
                if not minimize:
                    deltas = -deltas
            elif "log_loss" in primary_metric and "macro" not in primary_metric:
                # Binary-only log-loss kernel: requires 1D y in {0,1} and
                # 1D probs in [0,1]. For 2D-prob multiclass the predictions
                # are (N, K) softmax, not directly compatible with the
                # binary kernel -- fall through to sklearn for those cases.
                y_arr_1d = np.ascontiguousarray(y_ref).ravel()
                p1_arr = np.asarray(p1)
                p2_arr = np.asarray(p2)
                # Detect binary 1D case: targets in {0, 1} and probs are 1D
                if (
                    p1_arr.ndim == 1 and p2_arr.ndim == 1
                    and y_arr_1d.dtype.kind in "iu"
                    and len(np.unique(y_arr_1d)) <= 2
                ):
                    y_int = np.ascontiguousarray(y_arr_1d, dtype=np.int64)
                    p1_f = np.ascontiguousarray(p1_arr, dtype=np.float64)
                    p2_f = np.ascontiguousarray(p2_arr, dtype=np.float64)
                    deltas = _numba_paired_bootstrap_logloss_binary(
                        y_int, p1_f, p2_f, int(n_resamples), int(seed),
                    )
                    if not minimize:
                        deltas = -deltas
        except Exception as _numba_err:
            # Numba fast-path can fail on dtype edges / contiguity / shape mismatches.
            # The sklearn loop fallback below is ~60x slower; an operator who doesn't
            # see this WARN re-benches "bootstrap CI got slow" without realising the
            # numba path silently fell back. Emit the type+message so they can grep.
            logger.warning(
                "dummy_baselines: numba paired-bootstrap-logloss fast-path failed "
                "(%s: %s); falling back to sklearn loop (~60x slower). n_resamples=%d, "
                "y_ref.shape=%s, p1.shape=%s.",
                type(_numba_err).__name__, _numba_err, n_resamples,
                getattr(y_ref, "shape", None),
                getattr(p1, "shape", None),
            )
            deltas = None  # fall through to sklearn loop

    if deltas is None:
        # Vectorised numpy path for paired log_loss bootstrap. Calls
        # _vectorized_bootstrap_logloss_samples twice with the SAME seed so
        # the index matrices match and deltas line up element-wise. Same
        # rng-seeded path matches the legacy sklearn-per-call loop's
        # statistical contract (same idx every iter for both p1 and p2) but
        # ~60x faster on log_loss metrics at n=600 / 1000 resamples. Skips
        # to the legacy sklearn loop for RMSE / MAE (already covered by the
        # numba kernel above) and for "log_loss_macro" (returns None per
        # the legacy gate -- multi-output paired CI considered too cheap-
        # value to compute).
        if "log_loss" in primary_metric and "macro" not in primary_metric:
            s1 = _vectorized_bootstrap_logloss_samples(y_ref, p1, int(n_resamples), int(seed))
            s2 = _vectorized_bootstrap_logloss_samples(y_ref, p2, int(n_resamples), int(seed))
            if s1 is not None and s2 is not None and s1.shape == s2.shape:
                finite_mask = np.isfinite(s1) & np.isfinite(s2)
                if finite_mask.sum() >= max(1, n_resamples // 4):
                    raw = s1[finite_mask] - s2[finite_mask]
                    deltas = raw if minimize else -raw

    if deltas is None:
        # Final fallback: sklearn metric loop. Used for log_loss_macro (skipped
        # via the None return below) and as a safety net if the numba / vectorised
        # paths raised.
        if "RMSE" in primary_metric:
            def fn(y, p):
                return float(np.sqrt(mean_squared_error(y, p)))
        elif "MAE" in primary_metric:
            def fn(y, p):
                return float(mean_absolute_error(y, p))
        elif "log_loss_macro" in primary_metric:
            return None  # multi-output; cost > value at this gate
        elif "log_loss" in primary_metric:
            from sklearn.metrics import log_loss as _ll
            def fn(y, p):
                return float(_ll(y, p))
        else:
            return None

        rng = np.random.default_rng(seed)
        deltas = np.empty(n_resamples, dtype=np.float64)
        valid = 0
        for _i in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            try:
                v1 = fn(y_ref[idx], p1[idx])
                v2 = fn(y_ref[idx], p2[idx])
            except Exception:
                continue
            if not (np.isfinite(v1) and np.isfinite(v2)):
                continue
            deltas[valid] = (v1 - v2) if minimize else (v2 - v1)
            valid += 1
        if valid < n_resamples // 4:
            return None
        deltas = deltas[:valid]

    # For minimize metrics: strongest wins iff strongest_val < runner_up_val
    # -> delta = (strongest - runner_up) < 0. P(strongest beats) = mean(delta < 0).
    # For maximize metrics: strongest wins iff strongest > runner_up
    # -> delta = (runner_up - strongest) < 0. Same condition.
    p_strongest_beats = float(np.mean(deltas < 0))
    point = float(np.mean(deltas))
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))

    return {
        "runner_up": str(runner_up),
        "delta": point,
        "delta_ci": (lo, hi),
        "p_strongest_beats": p_strongest_beats,
        "split_used": "val" if use_val else "test",
    }


def _vectorized_bootstrap_logloss_samples(
    y: np.ndarray,
    p: np.ndarray,
    n_resamples: int,
    seed: int,
    eps: float = 1e-12,
) -> np.ndarray | None:
    """Vectorised bootstrap log-loss; handles 1D binary and 2D multilabel-macro.

    Returns a length-``n_resamples`` ndarray of per-resample log-loss values, or
    None when the input shapes aren't supported. Avoids the per-resample sklearn
    metric call (~10 ms each, dominated by input validation) by generating all
    bootstrap indices in one shot and computing log-loss via numpy broadcasting
    -- ~40x speedup at n=600, n_resamples=1000 vs the sklearn-per-call loop.

    Binary path: y shape (n,), p shape (n,). p is the predicted probability of
    class 1. Per-resample value is ``mean(-y*log(p_clip) - (1-y)*log(1-p_clip))``.

    Multilabel-macro path: y shape (n, K), p shape (n, K) with K binary labels;
    per-resample value is the unweighted mean of per-label binary log-loss.

    Memory: one (n_resamples, n) index matrix + the gathered y / p arrays.
    For typical n=600, n_resamples=1000, K=4 the gather is ~20 MB -- fine for
    the dummy-baseline phase budget. None is returned when shapes don't match
    the binary or multilabel layouts; caller falls back to the sklearn loop.
    """
    if n_resamples <= 0:
        return None
    n = len(y)
    if n < 10:
        return None
    rng = np.random.default_rng(seed)
    # Three supported shapes:
    #   (A) Binary 1-D     : y (n,) {0/1}, p (n,)             -- per-row BCE.
    #   (B) Multilabel 2-D : y (n, K) {0/1}, p (n, K)         -- macro BCE.
    #   (C) Multiclass 2-D : y (n,) int labels, p (n, K) softmax probs -- CE.
    #
    # In every case we pre-compute the per-row loss ONCE on the input-shape
    # arrays, THEN bootstrap via a single ``elem_n[idx]`` gather. Avoids
    # running np.log / np.where over the (n_resamples, n) gathered tensor
    # (the original sklearn-per-resample loop's 1000-call cost).
    if y.shape == p.shape:
        # (A) / (B): same-shape elementwise BCE.
        p_clip = np.clip(p, eps, 1.0 - eps)
        log_p = np.log(p_clip)
        # log1p(-p) preserves precision near p->1 where 1.0 - p_clip cancels
        # catastrophically; with eps=1e-12 in float64, 1-(1-eps) keeps ~4 digits.
        log_1mp = np.log1p(-p_clip)
        is_pos = y > 0.5
        elem_n = -np.where(is_pos, log_p, log_1mp)
        idx = rng.integers(0, n, size=(n_resamples, n))
        elem_r = elem_n[idx]
        if y.ndim == 1:
            return elem_r.mean(axis=1)
        if y.ndim == 2:
            # Macro mean: avg across rows + labels equally.
            return elem_r.mean(axis=(1, 2))
        return None
    # (C) Multiclass: y (n,) integer class labels, p (n, K) class probs.
    # Per-row CE = -log(p[i, y[i]]). Vectorise the true-class lookup via
    # fancy indexing, then bootstrap-mean as before.
    if (
        y.ndim == 1
        and p.ndim == 2
        and y.shape[0] == p.shape[0]
        and y.dtype.kind in ("i", "u")
    ):
        y_int = y.astype(np.intp, copy=False)
        # Out-of-range labels would index past K-1 silently; bail to the
        # sklearn fallback so it produces the correct error path.
        if y_int.min() < 0 or y_int.max() >= p.shape[1]:
            return None
        p_clip = np.clip(p, eps, 1.0 - eps)
        true_class_p = p_clip[np.arange(n, dtype=np.intp), y_int]
        elem_n = -np.log(true_class_p)
        idx = rng.integers(0, n, size=(n_resamples, n))
        return elem_n[idx].mean(axis=1)
    return None


def _bootstrap_ci_for_strongest(
    target_type: str,
    strongest: str,
    primary_metric: str,
    val_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    val_y: np.ndarray | None,
    test_y: np.ndarray | None,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Bootstrap CI on val + test for the strongest baseline only.

    Returns ``{"val": (lo, point, hi), "test": (lo, point, hi)}`` or
    ``None`` when not computable. 1000 resamples by default; cost ~1s on
    n=10^4. Seed is per-target for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Pick the metric callable matching primary_metric. Minimize
    # convention follows _pick_strongest naming.
    def _resample_metric(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float] | None:
        n = len(y)
        if n < 10:
            return None

        # Numba-accelerated path for RMSE / MAE / binary log-loss
        # (~30-340x faster than the sklearn-per-call loop on n=1500
        # x 1000 resamples).
        if _NUMBA_AVAILABLE and y.ndim == 1 and p.ndim == 1:
            try:
                y_arr = np.ascontiguousarray(y, dtype=np.float64)
                p_arr = np.ascontiguousarray(p, dtype=np.float64)
                if "RMSE" in primary_metric:
                    samples = _numba_bootstrap_rmse_samples(
                        y_arr, p_arr, int(n_resamples), int(seed),
                    )
                    point = float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))
                    if not np.isfinite(point):
                        return None
                    lo = float(np.percentile(samples, 2.5))
                    hi = float(np.percentile(samples, 97.5))
                    return (lo, point, hi)
                if "MAE" in primary_metric:
                    samples = _numba_bootstrap_mae_samples(
                        y_arr, p_arr, int(n_resamples), int(seed),
                    )
                    point = float(np.mean(np.abs(y_arr - p_arr)))
                    if not np.isfinite(point):
                        return None
                    lo = float(np.percentile(samples, 2.5))
                    hi = float(np.percentile(samples, 97.5))
                    return (lo, point, hi)
                if (
                    "log_loss" in primary_metric
                    and "macro" not in primary_metric
                    and y.dtype.kind in "iu"
                    and len(np.unique(y)) <= 2
                ):
                    # Binary 1D-prob case: matches the binary log-loss
                    # numba kernel signature.
                    y_int = np.ascontiguousarray(y, dtype=np.int64)
                    samples = _numba_bootstrap_logloss_binary_samples(
                        y_int, p_arr, int(n_resamples), int(seed),
                    )
                    # Point estimate via the same eps-clipped formula
                    # the kernel uses (matches sklearn's eps=1e-15).
                    eps = 1e-15
                    p_clip = np.clip(p_arr, eps, 1.0 - eps)
                    point = float(np.mean(
                        -np.where(y_int == 1, np.log(p_clip), np.log1p(-p_clip))
                    ))
                    if not np.isfinite(point):
                        return None
                    lo = float(np.percentile(samples, 2.5))
                    hi = float(np.percentile(samples, 97.5))
                    return (lo, point, hi)
            except Exception as _numba_err:
                # Numba single-bootstrap fast-path failed; same operator-blindness as
                # the paired-bootstrap twin above. Log so the perf regression is
                # traceable to the fallback rather than mystery slowness.
                logger.warning(
                    "dummy_baselines: numba single-bootstrap fast-path failed (%s: %s); "
                    "falling back to vectorised numpy / sklearn loop. primary_metric=%s n_resamples=%d.",
                    type(_numba_err).__name__, _numba_err, primary_metric, n_resamples,
                )

        # Vectorised numpy path for log_loss / log_loss_macro that don't
        # match the numba binary-int guard above (e.g. float-binary 1D y,
        # multilabel 2D y / p). Avoids 1000 sklearn metric calls and runs
        # ~40x faster than the sklearn-per-call loop at n=600.
        if "log_loss" in primary_metric:
            try:
                samples = _vectorized_bootstrap_logloss_samples(
                    y, p, int(n_resamples), int(seed),
                )
            except Exception:
                samples = None
            if samples is not None and len(samples) >= max(1, n_resamples // 4):
                eps = 1e-15
                p_clip = np.clip(p, eps, 1.0 - eps)
                is_pos = y > 0.5
                elem = -np.where(is_pos, np.log(p_clip), np.log1p(-p_clip))
                if y.ndim == 1:
                    point = float(np.mean(elem))
                elif y.ndim == 2:
                    point = float(np.mean(elem))
                else:
                    point = float("nan")
                if np.isfinite(point):
                    lo = float(np.percentile(samples, 2.5))
                    hi = float(np.percentile(samples, 97.5))
                    return (lo, point, hi)

        # Fallback path: sklearn metric loop. Used for log_loss
        # variants and as a safety net if the numba kernel raises.
        try:
            if "RMSE" in primary_metric:
                def fn(yi, pi):
                    return float(np.sqrt(mean_squared_error(yi, pi)))
            elif "MAE" in primary_metric:
                def fn(yi, pi):
                    return float(mean_absolute_error(yi, pi))
            elif "log_loss_macro" in primary_metric:
                # Multilabel macro: average over labels; here we use per-row
                # log_loss approx (cheap CI is best-effort for multilabel).
                from sklearn.metrics import log_loss as _ll
                K = y.shape[1] if y.ndim == 2 else 1
                def fn(yi, pi):
                    if yi.ndim == 1:
                        return float(_ll(yi, pi, labels=[0, 1]))
                    losses = []
                    failed: list[tuple[int, str]] = []
                    for k in range(K):
                        try:
                            losses.append(float(_ll(yi[:, k], pi[:, k], labels=[0, 1])))
                        except Exception as _e:
                            # Pre-fix `continue` silently dropped failing classes
                            # from the mean -- the multilabel log-loss reported
                            # back was a biased average over surviving classes only.
                            failed.append((k, str(_e)))
                            losses.append(float("nan"))
                    if failed:
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "multilabel log-loss: %d/%d class component(s) failed "
                            "and are reported as NaN (not silently dropped); "
                            "use np.nanmean downstream. failures: %s",
                            len(failed), K, failed[:5],
                        )
                    return float(np.nanmean(losses)) if losses else float("nan")
            elif "log_loss" in primary_metric:
                from sklearn.metrics import log_loss as _ll
                # 1D label, 1D or 2D pred
                def fn(yi, pi):
                    return float(_ll(yi, pi))
            else:
                # Maximize metrics (NDCG / AUC) -- bootstrap point estimate
                # works the same; the CI is naturally returned in metric
                # units regardless of direction.
                return None
        except Exception:
            return None

        # Point estimate
        try:
            point = fn(y, p)
        except Exception:
            return None
        if not np.isfinite(point):
            return None
        # Bootstrap resamples
        samples: list[float] = []
        failures = 0
        first_err: Optional[str] = None
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            try:
                v = fn(y[idx], p[idx])
                if np.isfinite(v):
                    samples.append(float(v))
            except Exception as _e_boot:
                # Pre-fix `continue` was silent. Track failure count so we
                # can WARN-log if more than a small fraction failed -- the
                # `< n_resamples // 4` guard below only catches extreme
                # under-sampling, not the partial-bias case where (say) 40%
                # of resamples raised and the CI is computed over the
                # surviving 60% (likely the most well-behaved tail).
                failures += 1
                if first_err is None:
                    first_err = str(_e_boot)
                continue
        if failures > max(1, n_resamples // 10):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "dummy_baselines: bootstrap CI: %d/%d resamples failed "
                "(first error: %s); CI computed over %d surviving samples "
                "may be biased.", failures, n_resamples, first_err, len(samples),
            )
        if len(samples) < n_resamples // 4:
            return None
        lo = float(np.percentile(samples, 2.5))
        hi = float(np.percentile(samples, 97.5))
        return (lo, point, hi)

    out: dict[str, Any] = {}
    val_p = val_preds.get(strongest)
    test_p = test_preds.get(strongest)
    if val_y is not None and val_p is not None and len(val_y) == len(val_p):
        v = _resample_metric(val_y, val_p)
        if v is not None:
            out["val"] = v
    if test_y is not None and test_p is not None and len(test_y) == len(test_p):
        v = _resample_metric(test_y, test_p)
        if v is not None:
            out["test"] = v
    return out if out else None

