"""Pseudo-BMA (pseudo Bayesian model averaging) weights for composite-component ensembles -- overfit-resistant, uncertainty-aware blend weights.

The ensemble layer (``composite/ensemble/``) combines several discovered composite components (+ raw + lag) into one
prediction. Equal / stacked weights either ignore predictive quality or overfit a tiny OOF set. Pseudo-BMA (Yao, Vehtari,
Simpson & Gelman 2018, "Using Stacking to Average Bayesian Predictive Distributions") weights each component by its
expected log predictive density (elpd) with proper uncertainty:

    w_k proportional to exp(elpd_k),   elpd_k = sum_i lpd_{i,k}

where ``lpd_{i,k}`` is the per-row log predictive density of component ``k`` at row ``i``. For a point regressor we turn
the OOF prediction into a Gaussian predictive density using a per-component residual-scale estimate ``sigma_k`` (weighted
RMSE of that component's OOF residuals):

    lpd_{i,k} = -0.5*log(2*pi) - log(sigma_k) - 0.5*(r_{i,k}/sigma_k)**2,   r_{i,k} = y_i - pred_{i,k}

For quantile components the log-score is replaced by the (negated) pinball loss, a proper scoring rule for quantiles.

Point pseudo-BMA (``bb_draws=0``) softmaxes the summed elpd -- on a small / noisy OOF set a single component can win by
sampling noise. BB-pseudo-BMA (``bb_draws>0``) is the Bayesian-bootstrap-stabilised variant: draw ``Dirichlet(1,...,1)``
weights over the ``n`` rows, recompute a reweighted elpd per draw, softmax each draw, and AVERAGE the resulting weight
vectors. Averaging over row-reweightings shrinks weights toward each other and away from a noise-driven winner, so weight
variance across seeds drops sharply (see ``tests/training/composite/test_biz_val_pseudo_bma.py``).

Wiring point for the parent (do NOT edit here -- this module is a pure function family): the ensemble builder in
``composite/ensemble/_cross_target.py`` (``CompositeCrossTargetEnsemble.from_train_metrics`` / the OOF-weighted path around
line 521, which today does ``weights = gains / gains.sum()`` on RMSE-gains) can call ``pseudo_bma_weights(oof_preds, y,
sample_weight=..., bb_draws=...)`` on the SAME leakage-free OOF matrix produced by ``compute_oof_holdout_predictions`` and
pass the returned vector as the ``weights=`` of the ``oof_weighted`` strategy. ``blend`` mirrors that strategy's convex
combination for callers that hold the OOF matrix directly.

Performance (cProfile, ``_benchmarks/bench_pseudo_bma.py``, n=2000 / K=5 / bb_draws=1000). The point path is a single
vectorised residual + Gaussian-lpd pass (~0.1 ms). The BB path's cost is the ``(bb_draws, n)`` Dirichlet draw + the
``(bb_draws, n) @ (n, K)`` matmul; both are BLAS / RNG C loops with no actionable Python-level speedup at this shape (the
per-draw softmax is a fused ``numba.njit`` reduction). A gamma-based Dirichlet in ``numpy`` dominated by RNG is not worth an
njit rewrite -- the draw is already in C. Documented: no actionable speedup beyond the njit softmax already applied.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import numba

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    numba = None  # type: ignore
    _HAS_NUMBA = False


_LOG_2PI = float(np.log(2.0 * np.pi))
_TINY_SIGMA = 1e-12


def _softmax_rows(elpd: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max-subtraction so summed elpd (can be huge / very negative) does not overflow exp."""
    m = elpd.max(axis=1, keepdims=True)
    e = np.exp(elpd - m)
    return np.asarray(e / e.sum(axis=1, keepdims=True))


if _HAS_NUMBA:

    @numba.njit(cache=True, fastmath=True)
    def _softmax_rows_njit(elpd: np.ndarray) -> np.ndarray:  # pragma: no cover - exercised via dispatch
        B, K = elpd.shape
        out = np.empty((B, K), dtype=np.float64)
        for b in range(B):
            m = elpd[b, 0]
            for k in range(1, K):
                if elpd[b, k] > m:
                    m = elpd[b, k]
            s = 0.0
            for k in range(K):
                v = np.exp(elpd[b, k] - m)
                out[b, k] = v
                s += v
            for k in range(K):
                out[b, k] /= s
        return out


def _row_softmax(elpd: np.ndarray) -> np.ndarray:
    if _HAS_NUMBA:
        return np.asarray(_softmax_rows_njit(np.ascontiguousarray(elpd)))
    return _softmax_rows(elpd)


def _validate_inputs(oof_preds: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(oof_preds, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"pseudo_bma_weights: oof_preds must be 2-D (n, K); got shape {P.shape}.")
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    n, K = P.shape
    if yv.shape[0] != n:
        raise ValueError(f"pseudo_bma_weights: y length {yv.shape[0]} != oof_preds rows {n}.")
    if K == 0:
        raise ValueError("pseudo_bma_weights: oof_preds has zero components (K=0).")
    if not np.isfinite(P).all() or not np.isfinite(yv).all():
        raise ValueError("pseudo_bma_weights: oof_preds / y must be finite (no NaN / inf).")
    return P, yv


def _pointwise_lpd(
    oof_preds: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    quantile: Optional[float],
) -> np.ndarray:
    """Per-row per-component log predictive density matrix ``(n, K)``.

    Gaussian log-score for point regressors (default), or the negated pinball loss when ``quantile`` is set (a proper
    scoring rule for a single quantile level). ``sample_weight`` only rescales the per-component ``sigma_k`` estimate here;
    the row-level elpd reduction applies weights in :func:`pseudo_bma_weights` / the BB draws.
    """
    P, yv = oof_preds, y
    n, K = P.shape
    resid = yv[:, None] - P  # (n, K)

    if quantile is not None:
        if not (0.0 < quantile < 1.0):
            raise ValueError(f"pseudo_bma_weights: quantile must be in (0, 1); got {quantile}.")
        # Pinball loss -> higher (less negative) log-score for sharper, better-calibrated quantile predictions.
        under = resid >= 0.0
        pinball = np.where(under, quantile * resid, (quantile - 1.0) * resid)
        return -pinball

    if sample_weight is None:
        sigma = np.sqrt((resid * resid).mean(axis=0))
    else:
        w = sample_weight
        wsum = w.sum()
        sigma = np.sqrt((w[:, None] * resid * resid).sum(axis=0) / wsum)
    # Degenerate (perfect fit / n<K collinear) component: floor sigma so its lpd stays finite and it does not swamp others.
    sigma = np.maximum(sigma, _TINY_SIGMA)
    return np.asarray(-0.5 * _LOG_2PI - np.log(sigma)[None, :] - 0.5 * (resid / sigma[None, :]) ** 2)


def pseudo_bma_weights(
    oof_preds,
    y,
    *,
    sample_weight=None,
    bb_draws: int = 0,
    quantile: Optional[float] = None,
    random_state: Optional[int] = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """Pseudo-BMA blend weights over per-component OOF predictions.

    Parameters
    ----------
    oof_preds : array (n, K)
        Leakage-free out-of-fold predictions, one column per ensemble component. The caller MUST supply OOF (not train-fit)
        predictions -- this function cannot verify out-of-foldness, exactly like the NNLS / ridge stackers it complements.
    y : array (n,)
        Targets aligned with ``oof_preds`` rows.
    sample_weight : array (n,), optional
        Non-negative per-row weights; used for the ``sigma_k`` estimate and the elpd row reduction.
    bb_draws : int, default 0
        ``0`` -> point pseudo-BMA (softmax of the summed elpd). ``>0`` -> BB-pseudo-BMA: average softmax weights over that
        many ``Dirichlet(1,...,1)`` row reweightings for stability on small / noisy OOF sets.
    quantile : float in (0, 1), optional
        When set, components are quantile predictions and the log-score is the negated pinball loss at this level.
    random_state : int, optional
        Seed for the Bayesian-bootstrap draws (ignored when ``bb_draws == 0``).
    temperature : float, default 1.0
        Softmax temperature on the elpd. ``>1`` flattens weights (more conservative), ``<1`` sharpens toward the best
        component. ``1.0`` is the standard pseudo-BMA.

    Returns
    -------
    weights : ndarray (K,)
        Non-negative weights summing to 1.0.
    """
    P, yv = _validate_inputs(oof_preds, y)
    n, K = P.shape
    if temperature <= 0.0:
        raise ValueError(f"pseudo_bma_weights: temperature must be > 0; got {temperature}.")

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"pseudo_bma_weights: sample_weight length {w.shape[0]} != n {n}.")
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("pseudo_bma_weights: sample_weight must be finite and non-negative.")
        if w.sum() <= 0:
            raise ValueError("pseudo_bma_weights: sample_weight sums to zero.")
    else:
        w = None

    lpd = _pointwise_lpd(P, yv, w, quantile)  # (n, K)

    if K == 1:
        return np.ones(1, dtype=np.float64)

    row_w = w if w is not None else np.ones(n, dtype=np.float64)

    if bb_draws <= 0:
        elpd = (row_w[:, None] * lpd).sum(axis=0, keepdims=True) / temperature  # (1, K)
        return np.asarray(_row_softmax(elpd)[0])

    # BB-pseudo-BMA: Dirichlet(1,...,1) over rows == normalized Exponential(1) draws, scaled by the base row weights so
    # sample_weight is honored. Each draw reweights the elpd; averaging the per-draw softmax weights stabilises the blend.
    rng = np.random.default_rng(random_state)
    g = rng.standard_gamma(1.0, size=(bb_draws, n))
    dir_w = g * row_w[None, :]
    dir_w /= dir_w.sum(axis=1, keepdims=True)
    # elpd^(b)_k = n * sum_i dir_w[b,i] * lpd[i,k] keeps the summed-elpd scale so temperature acts as in the point path.
    elpd = (n * (dir_w @ lpd)) / temperature  # (bb_draws, K)
    return np.asarray(_row_softmax(elpd).mean(axis=0))


def blend(preds, weights) -> np.ndarray:
    """Convex combination of per-component predictions: ``sum_k weights[k] * preds[:, k]``.

    Mirrors the ``oof_weighted`` strategy's blend in ``CompositeCrossTargetEnsemble.predict`` so a caller holding the OOF /
    inference prediction matrix can apply pseudo-BMA weights directly without going through the ensemble object.
    """
    P = np.asarray(preds, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"blend: preds must be 2-D (n, K); got shape {P.shape}.")
    wv = np.asarray(weights, dtype=np.float64).reshape(-1)
    if wv.shape[0] != P.shape[1]:
        raise ValueError(f"blend: weights length {wv.shape[0]} != preds columns {P.shape[1]}.")
    return P @ wv
