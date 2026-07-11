"""``hull_moving_average``: lag-reduced moving average, ``2*SMA(n/2) - SMA(n)`` smoothed by ``sqrt(n)``.

Source: 9th_g-research-crypto-forecasting.md -- explicit Hull MA implementation used as "most precious
feature": ``last_close - HullMA``. A plain SMA/EMA lags behind sharp trend changes by roughly half its
window length; the Hull MA construction (weighted-difference of a fast and slow SMA, then re-smoothed at a
shorter window) cancels most of that lag while still suppressing high-frequency noise -- a genuinely
different lag/smoothness tradeoff than mlframe's existing EWMA/rolling-window transforms, not a
re-parameterization of them.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _cumsum_with_prefix(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Skip the leading-NaN run of ``x`` and cumsum the valid suffix (prefixed with a 0.0 anchor).

    Factored out of ``_sma`` so ``hull_moving_average_multi`` can compute this ONCE per input series and
    reuse it across every requested window's fast/slow SMA pass, instead of recomputing an identical cumsum
    once per window (the dominant per-call cost, per cProfile).
    """
    n = x.shape[0]
    first_valid = 0
    while first_valid < n and np.isnan(x[first_valid]):
        first_valid += 1
    valid = x[first_valid:]
    n_valid = valid.shape[0]
    cumsum = np.concatenate([[0.0], np.cumsum(valid)])
    return cumsum, first_valid, n_valid


def _sma_from_cumsum(cumsum: np.ndarray, first_valid: int, n_valid: int, n: int, window: int) -> np.ndarray:
    """Windowed-mean reduction over a precomputed ``_cumsum_with_prefix`` result -- see ``_sma`` for the
    single-call entry point and the NaN-prefix rationale."""
    if window > n_valid:
        return np.full(n, np.nan)
    cumsum_valid = cumsum[window:] - cumsum[:-window]
    sma_valid = cumsum_valid / window
    return np.concatenate([np.full(first_valid + window - 1, np.nan), sma_valid])


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average via cumulative sum -- O(n), no pandas rolling-engine overhead.

    ``hull_moving_average`` calls this 3x per invocation (fast SMA, slow SMA, final re-smoothing pass);
    pandas' generic ``Series.rolling().mean()`` pays real per-call setup cost (window-bounds computation,
    Series wrapping) that a direct cumsum reduction skips entirely -- measured as the dominant cProfile cost.

    A leading-NaN prefix in ``x`` (as produced by the FIRST two SMA calls, feeding into the third) would
    otherwise contaminate a naive cumsum: NaN propagates through every subsequent cumulative sum, not just
    the windows that actually contain it. Skip the leading-NaN run and apply the cumsum reduction only to
    the valid suffix, then re-pad.
    """
    n = x.shape[0]
    cumsum, first_valid, n_valid = _cumsum_with_prefix(x)
    return _sma_from_cumsum(cumsum, first_valid, n_valid, n, window)


def hull_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Hull Moving Average: ``WMA_style(2*SMA(n/2) - SMA(n), sqrt(n))`` via nested SMA (the source's own
    simplified SMA-based construction, not the canonical WMA-based Hull formula -- kept faithful to the
    winning solution's actual code).

    Parameters
    ----------
    values
        ``(n,)`` time-ordered series (e.g. close prices).
    window
        Hull MA period; internally uses ``SMA(window // 2)``, ``SMA(window)``, and a final
        ``SMA(round(sqrt(window)))`` re-smoothing pass.

    Returns
    -------
    np.ndarray
        ``(n,)`` Hull MA values; the first ``~window`` entries are NaN (insufficient history), matching
        standard rolling-window edge behavior.
    """
    x = np.asarray(values, dtype=np.float64)
    half_window = max(1, window // 2)
    sqrt_window = max(1, round(np.sqrt(window)))

    sma_half = _sma(x, half_window)
    sma_full = _sma(x, window)
    raw_hma_input = 2.0 * sma_half - sma_full
    hma = _sma(raw_hma_input, sqrt_window)
    return hma


def hull_ma_deviation(values: np.ndarray, window: int) -> np.ndarray:
    """``value - hull_moving_average(value, window)`` -- the reduced-lag trend-deviation feature the source
    used directly ("most precious feature"), a Composite Target ``diff``-style base variant."""
    x = np.asarray(values, dtype=np.float64)
    return np.asarray(x - hull_moving_average(x, window))


def hull_moving_average_multi(values: np.ndarray, windows: Sequence[int]) -> dict[int, np.ndarray]:
    """``hull_moving_average`` for several ``windows`` at once -- a common real usage (regime detection
    typically compares a fast and a slow Hull MA rather than trusting a single window).

    Bit-identical to calling ``hull_moving_average(values, w)`` once per ``w`` in ``windows``: same formula,
    same NaN-prefix handling, just restructured so the two first-stage SMA passes (fast/slow, both windowed
    reductions of the SAME input series) share one ``_cumsum_with_prefix`` call instead of each window
    recomputing an identical cumsum of ``values`` from scratch. Only the final re-smoothing pass (its input
    differs per window) still needs a per-window cumsum.

    Parameters
    ----------
    values
        ``(n,)`` time-ordered series (e.g. close prices).
    windows
        Hull MA periods to compute, e.g. ``[10, 20, 50]`` for a fast/medium/slow regime-detection stack.

    Returns
    -------
    dict[int, np.ndarray]
        ``{window: hma}``, one ``(n,)`` array per requested window.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    cumsum, first_valid, n_valid = _cumsum_with_prefix(x)

    results: dict[int, np.ndarray] = {}
    for window in windows:
        half_window = max(1, window // 2)
        sqrt_window = max(1, round(np.sqrt(window)))

        sma_half = _sma_from_cumsum(cumsum, first_valid, n_valid, n, half_window)
        sma_full = _sma_from_cumsum(cumsum, first_valid, n_valid, n, window)
        raw_hma_input = 2.0 * sma_half - sma_full
        hma = _sma(raw_hma_input, sqrt_window)
        results[window] = hma
    return results


__all__ = ["hull_moving_average", "hull_ma_deviation", "hull_moving_average_multi"]
