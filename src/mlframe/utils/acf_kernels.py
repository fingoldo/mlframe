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
horizontal reference lines. ``n`` here is the number of OBSERVATIONS actually used (post-cap, non-finite rows
excluded), so the band matches the data the bars were computed from rather than the grid they sit on.

Non-finite rows are holes, not rows to delete: deleting them closes the gap, so lag k silently stops meaning k
steps of the caller's grid. The autocovariance is normalised by the number of pairs observed at each lag instead,
which is the pairwise-complete estimator. Measured on an AR(1) with phi=0.7 and one row in seven punched out, the
gap-closing form returned 0.669 / 0.446 / 0.294 at lags 1/3/5 against a truth of 0.700 / 0.343 / 0.168 -- biased
toward zero, and increasingly so with the lag; the pairwise form returns 0.700 / 0.344 / 0.160. A gap-free series
takes the original 1/n normalisation and is bit-identical to before.
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


def lag_tick_labels(nlags: int, max_labels: int = 12) -> tuple:
    """Category labels for a per-lag bar axis: every lag keeps its bar, only every k-th keeps a visible label."""
    step = max(1, int(np.ceil(nlags / max(max_labels, 1))))
    return tuple(str(i) if (i == 1 or i % step == 0) else " " * i for i in range(1, nlags + 1))


def gap_fraction(x: np.ndarray) -> float:
    """Share of rows dropped as non-finite before the autocovariance -- i.e. how holed the series is.

    The gaps are no longer closed -- ``acf_fft`` keeps them and normalises by the observed PAIR count per lag -- so
    this number is no longer a caveat about a biased estimate. It is still worth showing: an ACF resting on a series
    that is a third holes is estimated from far fewer pairs than its length suggests, and the reader should know.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(1.0 - np.count_nonzero(np.isfinite(arr)) / arr.size)


def _prep_series(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-centred, tail-capped float64 series plus its presence mask, for the FFT autocovariance.

    Non-finite rows are ZEROED AFTER centring rather than removed. Removing them closes the gap they leave, so lag
    k stops meaning k steps of the original grid and starts meaning "k surviving observations" -- which biases the
    autocorrelation toward zero, increasingly so at higher lags. Keeping the hole and carrying the mask lets the
    autocovariance be normalised by the number of PAIRS actually observed at each lag (see ``acf_fft``), which is
    the pairwise-complete estimator and needs no interpolation.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size > MAX_ACF_SERIES:
        arr = arr[-MAX_ACF_SERIES:]
    mask = np.isfinite(arr).astype(np.float64)
    n_obs = mask.sum()
    if n_obs:
        arr = np.where(mask > 0, arr - (arr[mask > 0].mean()), 0.0)
    else:
        arr = np.zeros_like(arr)
    return np.asarray(arr), mask


def acf_fft(x: np.ndarray, nlags: int = MAX_ACF_LAGS) -> Tuple[np.ndarray, int]:
    """Sample autocorrelation at lags ``1..min(nlags, n-1)`` via the FFT autocovariance.

    Returns ``(acf_lags, n_used)`` where ``acf_lags[k]`` is the lag-(k+1) autocorrelation (lag 0 is the
    trivial 1.0 and is omitted) and ``n_used`` is the post-cap series length the band should use. The
    autocovariance is the inverse FFT of the periodogram (Wiener-Khinchin) zero-padded to avoid circular
    wrap-around, then normalised by the lag-0 variance (biased 1/n estimator, the statsmodels default).
    """
    arr, mask = _prep_series(x)
    n_grid = arr.size
    n = int(mask.sum())  # observations actually present; the Bartlett band must use this, not the grid length
    if n < 2 or n_grid < 2:
        return np.zeros(0, dtype=np.float64), n
    k = min(int(nlags), n_grid - 1)
    if k < 1:
        return np.zeros(0, dtype=np.float64), n
    nfft = 1 << int(np.ceil(np.log2(2 * n_grid - 1)))
    f = np.fft.rfft(arr, n=nfft)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft)[: k + 1]
    # Pair counts per lag by the same Wiener-Khinchin route on the presence mask: pairs[j] is how many (t, t+j)
    # pairs had BOTH ends observed. Dividing by it instead of by n is what makes a holed series unbiased; on a
    # gap-free series every pairs[j] is exactly n-j, so this reduces to the usual estimator.
    if n == n_grid:
        # No holes: keep the biased 1/n normalisation exactly as before. It is the statsmodels default, it keeps the
        # autocovariance sequence positive-semidefinite (which the Durbin-Levinson PACF recursion relies on), and
        # leaving it alone means a gap-free series is bit-identical to the pre-fix output.
        pairs = np.full(k + 1, float(n))
        acov /= n
    else:
        fm = np.fft.rfft(mask, n=nfft)
        pairs = np.rint(np.fft.irfft(fm * np.conjugate(fm), n=nfft)[: k + 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            acov = np.where(pairs > 0, acov / np.maximum(pairs, 1.0), np.nan)
    if not np.isfinite(acov[0]) or acov[0] <= 0.0:
        # Zero variance (constant series after centring): autocorrelation is undefined -> empty so the
        # panel emits an honest annotation rather than a row of spurious zero bars.
        return np.zeros(0, dtype=np.float64), n
    out = acov[1:] / acov[0]
    # A lag with too few observed pairs has no usable estimate; NaN so the panel skips the bar instead of drawing a
    # number built on a handful of rows.
    return np.where(pairs[1:] >= 2, out, np.nan), n


def pacf_levinson(x: np.ndarray, nlags: int = MAX_ACF_LAGS) -> Tuple[np.ndarray, int]:
    """Sample partial autocorrelation at lags ``1..k`` via the Durbin-Levinson recursion.

    PACF[k] is the last reflection coefficient of the order-k Yule-Walker fit; the recursion runs over the
    small ``nlags``-length ACF vector (O(nlags^2)), never over n. Returns ``(pacf_lags, n_used)`` with the
    same lag-1.. convention as ``acf_fft`` (lag 0 omitted).
    """
    r_lags, n = acf_fft(x, nlags=nlags)
    # A holed series can leave a lag with too few observed pairs to estimate (NaN). The recursion is sequential in
    # the lag, so it can only run up to the first gap; truncating is honest, propagating a NaN through every
    # higher order is not.
    finite_run = int(np.argmin(np.isfinite(r_lags))) if r_lags.size and not np.all(np.isfinite(r_lags)) else r_lags.size
    r_lags = r_lags[:finite_run]
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
    "gap_fraction",
    "MAX_ACF_SERIES",
    "MAX_ACF_LAGS",
    "acf_fft",
    "pacf_levinson",
    "significance_band",
]
