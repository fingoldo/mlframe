"""Log-odds (logit-space) combination of conditionally-independent probability estimates.

Naive-Bayes-style ensembles — one sub-model per (near-)independent feature or feature group — should combine
member probabilities MULTIPLICATIVELY in odds space, not arithmetically or by raw probability product:
``P(y=1|X) / P(y=0|X) ~ prod_i P(y=1|x_i) / P(y=0|x_i)`` under conditional independence of the ``x_i`` given
``y``. Averaging or multiplying raw probabilities directly does not correspond to any correct combination
rule; summing log-odds (logits) does, and is numerically the same operation with none of the probability-
product's under/overflow risk on many members.

Backend: numpy baseline, fused ``numba.njit`` kernels (single-thread and ``prange``-parallel), and a cupy
GPU path — size- AND hardware-dependent, so the choice routes through
``kernel_tuning_cache`` (see ``_ktc_dispatch.py``) rather than a hardcoded threshold. Measured on the dev
host (2026-07-03): njit_single wins at n=1,000 (0.02ms), cupy wins at n=100,000 (5.23ms vs njit_par 8.38ms),
njit_parallel wins at n=1,000,000 (25.9ms vs cupy 28.2ms) — no backend dominates uniformly across shapes or
(implicitly) hardware, which is exactly the scenario the tuning cache exists for.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlframe.calibration._independence_check import member_residual_correlation

try:
    import numba
    from numba import prange

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a core mlframe dependency; exercised only if absent
    _NUMBA_AVAILABLE = False

_PARALLEL_THRESHOLD = 20_000  # measurement-backed fallback, used when the KTC has not tuned this shape yet


def _odds_combine_numpy(p: np.ndarray, clip: float) -> np.ndarray:
    """Combine member probabilities via unweighted log-odds summation, numpy baseline."""
    p = np.clip(p, clip, 1.0 - clip)
    logits = np.log(p / (1.0 - p))
    combined_logit = logits.sum(axis=1)
    return np.asarray(1.0 / (1.0 + np.exp(-combined_logit)))


def _odds_combine_numpy_weighted(p: np.ndarray, weights: np.ndarray, clip: float) -> np.ndarray:
    """Combine member probabilities via per-member weighted log-odds summation, numpy baseline."""
    p = np.clip(p, clip, 1.0 - clip)
    logits = np.log(p / (1.0 - p))
    combined_logit = logits @ weights
    return np.asarray(1.0 / (1.0 + np.exp(-combined_logit)))


def _odds_combine_cupy(p: np.ndarray, clip: float) -> np.ndarray:
    """Combine member probabilities via unweighted log-odds summation, cupy GPU backend."""
    import cupy as cp

    p_gpu = cp.asarray(p)
    p_c = cp.clip(p_gpu, clip, 1.0 - clip)
    logits = cp.log(p_c / (1.0 - p_c))
    combined_logit = logits.sum(axis=1)
    result = 1.0 / (1.0 + cp.exp(-combined_logit))
    return np.asarray(cp.asnumpy(result))


def _odds_combine_cupy_weighted(p: np.ndarray, weights: np.ndarray, clip: float) -> np.ndarray:
    """Combine member probabilities via per-member weighted log-odds summation, cupy GPU backend."""
    import cupy as cp

    p_gpu = cp.asarray(p)
    w_gpu = cp.asarray(weights)
    p_c = cp.clip(p_gpu, clip, 1.0 - clip)
    logits = cp.log(p_c / (1.0 - p_c))
    combined_logit = logits @ w_gpu
    result = 1.0 / (1.0 + cp.exp(-combined_logit))
    return np.asarray(cp.asnumpy(result))


if _NUMBA_AVAILABLE:

    @numba.njit(fastmath=True, cache=True)
    def _odds_combine_njit(p: np.ndarray, clip: float) -> np.ndarray:
        """Combine member probabilities via unweighted log-odds summation, single-thread njit backend."""
        n, k = p.shape
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            acc = 0.0
            for j in range(k):
                v = p[i, j]
                if v < clip:
                    v = clip
                elif v > 1.0 - clip:
                    v = 1.0 - clip
                acc += np.log(v / (1.0 - v))
            out[i] = 1.0 / (1.0 + np.exp(-acc))
        return out

    @numba.njit(fastmath=True, cache=True, parallel=True)
    def _odds_combine_njit_parallel(p: np.ndarray, clip: float) -> np.ndarray:
        """Combine member probabilities via unweighted log-odds summation, ``prange``-parallel njit backend."""
        n, k = p.shape
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            acc = 0.0
            for j in range(k):
                v = p[i, j]
                if v < clip:
                    v = clip
                elif v > 1.0 - clip:
                    v = 1.0 - clip
                acc += np.log(v / (1.0 - v))
            out[i] = 1.0 / (1.0 + np.exp(-acc))
        return out

    @numba.njit(fastmath=True, cache=True)
    def _odds_combine_njit_weighted(p: np.ndarray, weights: np.ndarray, clip: float) -> np.ndarray:
        """Combine member probabilities via per-member weighted log-odds summation, single-thread njit backend."""
        n, k = p.shape
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            acc = 0.0
            for j in range(k):
                v = p[i, j]
                if v < clip:
                    v = clip
                elif v > 1.0 - clip:
                    v = 1.0 - clip
                acc += weights[j] * np.log(v / (1.0 - v))
            out[i] = 1.0 / (1.0 + np.exp(-acc))
        return out

    @numba.njit(fastmath=True, cache=True, parallel=True)
    def _odds_combine_njit_parallel_weighted(p: np.ndarray, weights: np.ndarray, clip: float) -> np.ndarray:
        """Combine member probabilities via per-member weighted log-odds summation, ``prange``-parallel njit backend."""
        n, k = p.shape
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            acc = 0.0
            for j in range(k):
                v = p[i, j]
                if v < clip:
                    v = clip
                elif v > 1.0 - clip:
                    v = 1.0 - clip
                acc += weights[j] * np.log(v / (1.0 - v))
            out[i] = 1.0 / (1.0 + np.exp(-acc))
        return out


def _dispatch(p: np.ndarray, clip: float) -> np.ndarray:
    """Route unweighted log-odds combination to the KTC-chosen backend, falling back to CPU on GPU failure."""
    n, k = p.shape
    from mlframe.calibration._ktc_dispatch import choose_odds_combine_backend

    fallback = "njit_parallel" if (n >= _PARALLEL_THRESHOLD and _NUMBA_AVAILABLE) else "njit_single"
    backend = choose_odds_combine_backend(n, k, fallback=fallback if _NUMBA_AVAILABLE else "numpy")
    if backend == "cupy":
        try:
            return _odds_combine_cupy(p, clip)
        except Exception:  # GPU path failed at runtime (OOM, driver hiccup) -> CPU fallback, never raise
            backend = "njit_parallel" if _NUMBA_AVAILABLE else "numpy"
    if not _NUMBA_AVAILABLE:
        return _odds_combine_numpy(p, clip)
    if backend == "njit_parallel":
        return np.asarray(_odds_combine_njit_parallel(p, clip))
    return np.asarray(_odds_combine_njit(p, clip))


def _dispatch_weighted(p: np.ndarray, w: np.ndarray, clip: float) -> np.ndarray:
    """Route per-member weighted log-odds combination to the KTC-chosen backend, falling back to CPU on GPU failure."""
    n, k = p.shape
    from mlframe.calibration._ktc_dispatch import choose_odds_combine_backend

    fallback = "njit_parallel" if (n >= _PARALLEL_THRESHOLD and _NUMBA_AVAILABLE) else "njit_single"
    backend = choose_odds_combine_backend(n, k, fallback=fallback if _NUMBA_AVAILABLE else "numpy")
    if backend == "cupy":
        try:
            return _odds_combine_cupy_weighted(p, w, clip)
        except Exception:
            backend = "njit_parallel" if _NUMBA_AVAILABLE else "numpy"
    if not _NUMBA_AVAILABLE:
        return _odds_combine_numpy_weighted(p, w, clip)
    if backend == "njit_parallel":
        return np.asarray(_odds_combine_njit_parallel_weighted(p, w, clip))
    return np.asarray(_odds_combine_njit_weighted(p, w, clip))


def odds_ratio_combine(
    member_probs: np.ndarray,
    weights: np.ndarray | None = None,
    clip: float = 1e-7,
    check_independence: bool = False,
    correlation_threshold: float = 0.85,
    on_correlation_violation: str = "warn",
) -> np.ndarray:
    """Combine conditionally-independent member probabilities via log-odds (logit) summation.

    Parameters
    ----------
    member_probs
        ``(n_samples, n_members)`` array of P(y=1) estimates, one column per conditionally-independent
        sub-model (e.g. one per feature or per feature group, as in a naive-Bayes-style ensemble).
    weights
        Optional ``(n_members,)`` per-member weight applied to each member's logit before summing (e.g. to
        down-weight a noisier sub-model). Defaults to equal weight (1.0) for every member.
    clip
        Probabilities are clipped to ``[clip, 1 - clip]`` before the logit transform to avoid +/-inf logits
        from exactly-0/1 member predictions.
    check_independence
        Opt-in. When ``True``, empirically estimates each member's correlation against the leave-one-out
        consensus (mean logit) of the other members (``mlframe.calibration._independence_check.
        member_residual_correlation``) before combining, and acts on it per ``on_correlation_violation`` if
        the conditional-independence assumption underlying log-odds summation is meaningfully violated
        (correlated/redundant members otherwise cause systematically over-confident combined probabilities,
        since shared evidence gets double-counted). Costs an extra O(n*k^2) pass; default ``False`` keeps
        the hot path exactly as before.
    correlation_threshold
        Mean absolute member-vs-consensus correlation above which members are considered meaningfully
        redundant/correlated (0 = a member's logit is unrelated to what the others already say, ~1 =
        the member is essentially a redundant re-measurement of the same signal). Only used when
        ``check_independence=True``. Default 0.85: genuinely conditionally-independent members still show
        moderate (~0.5-0.7) consensus correlation purely through their shared dependence on y, so the
        threshold sits above that band and only fires on members carrying near-duplicate signal.
    on_correlation_violation
        ``"warn"`` (default when triggered): emit a ``UserWarning`` and still return the ordinary log-odds
        sum. ``"fallback"``: emit a ``UserWarning`` AND return an equal-weight log opinion pool instead
        (``sigmoid(mean_i logit(p_i))``, i.e. the geometric mean of the member odds) — bounded amplification
        that does not compound double-counted shared evidence the way an unbounded sum does. Ignored when
        ``check_independence=False``.

    Returns
    -------
    np.ndarray
        ``(n_samples,)`` combined probability, computed as ``sigmoid(sum_i w_i * logit(p_i))`` (or the
        correlation-triggered geometric-mean fallback above). This is the conditional-independence-correct
        combination rule; it is NOT the same as ``mean(p_i)`` or ``prod(p_i)`` (both of which are ad-hoc and
        do not correspond to any probabilistic combination law).
    """
    p = np.asarray(member_probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"odds_ratio_combine: member_probs must be 2D (n_samples, n_members); got shape {p.shape}")

    if check_independence and p.shape[1] >= 2:
        diag = member_residual_correlation(p, clip=clip)
        mean_abs_corr = diag["mean_abs_residual_correlation"]
        if mean_abs_corr > correlation_threshold:
            warnings.warn(
                f"odds_ratio_combine: mean absolute residual correlation between members ({mean_abs_corr:.3f}) "
                f"exceeds correlation_threshold ({correlation_threshold:.3f}) -> conditional-independence "
                "assumption looks violated; log-odds summation will over-state confidence. "
                + ("Falling back to equal-weight log opinion pool." if on_correlation_violation == "fallback" else "Proceeding with log-odds summation anyway (on_correlation_violation='warn')."),
                stacklevel=2,
            )
            if on_correlation_violation == "fallback":
                equal_weights = np.full(p.shape[1], 1.0 / p.shape[1], dtype=np.float64)
                return _dispatch_weighted(p, equal_weights, clip)

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (p.shape[1],):
            raise ValueError(f"odds_ratio_combine: weights shape {w.shape} must equal (n_members,) = ({p.shape[1]},)")
        return _dispatch_weighted(p, w, clip)

    return _dispatch(p, clip)


__all__ = ["odds_ratio_combine"]
