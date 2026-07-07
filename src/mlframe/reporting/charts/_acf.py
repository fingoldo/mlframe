"""FFT-based autocorrelation / partial-autocorrelation kernels for diagnostic panels.

Both the regression residual-ACF panel and the temporal target ACF/PACF panels reduce to the same
two computations: the sample ACF (via the Wiener-Khinchin FFT autocovariance, O(n log n) instead of
the O(n * nlags) naive double loop) and the sample PACF (Durbin-Levinson recursion over the ACF, which
is O(nlags^2) on the tiny ``nlags``-length ACF vector, never on n).

The series length fed to the FFT is tail-capped (mirrors the ACF-period detector in the timeseries
baselines): only the most recent ``MAX_ACF_SERIES`` points carry the autocorrelation structure a
diagnostic cares about, and the FFT cost is then bounded regardless of a 1e6-row input. The number of
lags is capped at ``MAX_ACF_LAGS`` so the returned vectors stay plot-sized.

White-noise significance bounds are +-z/sqrt(n) (Bartlett, the standard ACF band); callers draw them as
horizontal reference lines. ``n`` here is the (post-cap) length actually used, so the band matches the
data the bars were computed from.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Tail-cap the series the FFT runs on: 200k recent points reproduce the low-lag autocorrelation structure
# within plotting noise while bounding the FFT cost at any n (a 1e6-row residual series caps to this).
MAX_ACF_SERIES: int = 200_000
# Plot-sized lag cap: low-order lags carry the diagnostic signal (AR(1) spike, weekly/seasonal echo); a
# 50-lag bar chart is already dense. Bounds the Durbin-Levinson PACF recursion (O(nlags^2)) as well.
MAX_ACF_LAGS: int = 50
# 95% two-sided normal quantile for the Bartlett white-noise band.
_Z_95: float = 1.959963984540054


def _prep_series(x: np.ndarray) -> np.ndarray:
    """Finite, mean-centred, tail-capped float64 view of ``x`` for the FFT autocovariance."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size > MAX_ACF_SERIES:
        arr = arr[-MAX_ACF_SERIES:]
    if arr.size:
        arr = arr - arr.mean()
    return np.asarray(arr)


def acf_fft(x: np.ndarray, nlags: int = MAX_ACF_LAGS) -> Tuple[np.ndarray, int]:
    """Sample autocorrelation at lags ``1..min(nlags, n-1)`` via the FFT autocovariance.

    Returns ``(acf_lags, n_used)`` where ``acf_lags[k]`` is the lag-(k+1) autocorrelation (lag 0 is the
    trivial 1.0 and is omitted) and ``n_used`` is the post-cap series length the band should use. The
    autocovariance is the inverse FFT of the periodogram (Wiener-Khinchin) zero-padded to avoid circular
    wrap-around, then normalised by the lag-0 variance (biased 1/n estimator, the statsmodels default).
    """
    arr = _prep_series(x)
    n = arr.size
    if n < 2:
        return np.zeros(0, dtype=np.float64), n
    k = min(int(nlags), n - 1)
    if k < 1:
        return np.zeros(0, dtype=np.float64), n
    nfft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(arr, n=nfft)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft)[: k + 1]
    acov /= n
    if acov[0] <= 0.0:
        # Zero variance (constant series after centring): autocorrelation is undefined -> empty so the
        # panel emits an honest annotation rather than a row of spurious zero bars.
        return np.zeros(0, dtype=np.float64), n
    return (acov[1:] / acov[0]), n


def pacf_levinson(x: np.ndarray, nlags: int = MAX_ACF_LAGS) -> Tuple[np.ndarray, int]:
    """Sample partial autocorrelation at lags ``1..k`` via the Durbin-Levinson recursion.

    PACF[k] is the last reflection coefficient of the order-k Yule-Walker fit; the recursion runs over the
    small ``nlags``-length ACF vector (O(nlags^2)), never over n. Returns ``(pacf_lags, n_used)`` with the
    same lag-1.. convention as ``acf_fft`` (lag 0 omitted).
    """
    r_lags, n = acf_fft(x, nlags=nlags)
    k = r_lags.size
    if k < 1:
        return np.zeros(0, dtype=np.float64), n
    # r[0]=1 (lag 0), r[1..k] = autocorrelations. phi holds the current-order AR coefficients.
    r = np.empty(k + 1, dtype=np.float64)
    r[0] = 1.0
    r[1:] = r_lags
    pacf = np.zeros(k, dtype=np.float64)
    phi = np.zeros(k + 1, dtype=np.float64)
    phi_prev = np.zeros(k + 1, dtype=np.float64)
    v = 1.0
    for m in range(1, k + 1):
        acc = r[m]
        for j in range(1, m):
            acc -= phi_prev[j] * r[m - j]
        refl = acc / v if v > 1e-300 else 0.0
        refl = float(np.clip(refl, -1.0, 1.0))
        phi[m] = refl
        for j in range(1, m):
            phi[j] = phi_prev[j] - refl * phi_prev[m - j]
        v *= 1.0 - refl * refl
        pacf[m - 1] = refl
        phi_prev[: m + 1] = phi[: m + 1]
    return pacf, n


def significance_band(n_used: int, z: float = _Z_95) -> float:
    """Bartlett white-noise +-band half-width ``z/sqrt(n)``; 0.0 for an empty series."""
    return (z / np.sqrt(n_used)) if n_used > 0 else 0.0


__all__ = [
    "MAX_ACF_SERIES",
    "MAX_ACF_LAGS",
    "acf_fft",
    "pacf_levinson",
    "significance_band",
]
