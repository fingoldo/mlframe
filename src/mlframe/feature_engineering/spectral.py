"""FFT-based rolling spectral features.

Per-row spectral band energies, spectral entropy, dominant frequency,
and HF/LF ratio computed over a fixed-K trailing window inside each
group. Vectorised via ``np.fft.rfft`` on the sliding-window stack.

Distinct from ``wavelet_dwt.py``: this is straight FFT (linear basis),
not wavelets. Use this when the signal-of-interest is frequency-banded
(seismic, audio, telemetry, sensor noise) and use wavelets when the
signal has localised transients (spikes, edges).

Underpinned by ``grouped.per_group_sliding_window`` so per-group
boundary handling is consistent with other rolling-feature modules.
"""

from __future__ import annotations

__all__ = [
    "rolling_spectral_band_energies",
    "rolling_spectral_entropy",
    "rolling_dominant_freq_idx",
    "rolling_hf_lf_ratio",
    "rolling_spectral_centroid",
    "rolling_spectral_bandwidth",
    "rolling_spectral_rolloff",
    "rolling_spectral_flatness",
    "rolling_spectral_flux",
    "rolling_periodicity_score",
]

from typing import Tuple

import numpy as np

from .grouped import per_group_sliding_window


def _bands_for(K: int, n_bands: int = 3) -> Tuple[Tuple[int, int], ...]:
    """Default lo/mid/hi band edges for an rfft of length K.

    Skips DC bin (bin 0). The default 3-band split lo/mid/hi divides
    the non-DC spectrum into thirds by index. Configurable via the
    ``bands`` kwarg on the public functions when a finer split is
    needed.
    """
    n_freq = K // 2 + 1
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    if n_bands == 3:
        return (
            (1, max(2, n_freq // 6)),
            (n_freq // 6, n_freq // 2),
            (n_freq // 2, n_freq),
        )
    # Generic equal-width split skipping DC
    width = max(1, (n_freq - 1) // n_bands)
    edges = [(1 + i * width, min(n_freq, 1 + (i + 1) * width)) for i in range(n_bands)]
    return tuple(edges)


def _detrend_zero_mean(wins: np.ndarray) -> np.ndarray:
    """Subtract per-window mean. Avoids DC dominating the spectrum."""
    return np.asarray(wins - wins.mean(axis=1, keepdims=True))


def rolling_spectral_band_energies(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    bands: Tuple[Tuple[int, int], ...] | None = None,
    detrend: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Energy in each frequency band of an rfft of the K-window.

    Output shape: ``(len(values), n_bands)`` where ``n_bands = len(bands)``
    (default 3 = lo/mid/hi after dropping DC).

    Energy = ``|rfft(window)|^2`` summed inside the band's bin range.
    Bands default to a 1/6, 1/2, n-split of the non-DC spectrum, which
    matches the convention in `mlframe.training` composite-target FE.
    Pass explicit ``bands=((1, 8), (8, 32), ...)`` for custom splits.

    NaN-safe: non-finite input values are replaced with the per-window
    mean before FFT so a single missing observation in the window
    does not propagate to all bands.
    """
    if bands is None:
        bands = _bands_for(window_K)
    n_bands = len(bands)
    out = np.full((values.size, n_bands), fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        finite_mask = np.isfinite(wins)
        if not finite_mask.all():
            # per-window mean impute
            mu = np.where(
                finite_mask.any(axis=1, keepdims=True),
                np.nanmean(wins, axis=1, keepdims=True),
                0.0,
            )
            wins = np.where(finite_mask, wins, mu)
        if detrend:
            wins = _detrend_zero_mean(wins)
        spec = np.abs(np.fft.rfft(wins, axis=1)) ** 2
        for b, (lo, hi) in enumerate(bands):
            out[write_idx, b] = spec[:, lo:hi].sum(axis=1)
    return out


def rolling_spectral_entropy(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    detrend: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Shannon entropy of the normalised power spectrum per K-window.

    Entropy is in nats (natural log). Higher = energy spread evenly
    across frequencies (noise-like); lower = energy concentrated at
    a few peaks (oscillatory / structured).

    ``H = -sum(p_k * log(p_k))`` where ``p_k = |F_k|^2 /
    sum(|F_k|^2)``. DC bin included by default to match
    ``scipy.signal.welch`` + ``antropy.spectral_entropy`` conventions.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        finite_mask = np.isfinite(wins)
        if not finite_mask.all():
            mu = np.where(
                finite_mask.any(axis=1, keepdims=True),
                np.nanmean(wins, axis=1, keepdims=True),
                0.0,
            )
            wins = np.where(finite_mask, wins, mu)
        if detrend:
            wins = _detrend_zero_mean(wins)
        spec = np.abs(np.fft.rfft(wins, axis=1)) ** 2
        spec_norm = spec / np.maximum(spec.sum(axis=1, keepdims=True), 1e-9)
        # -sum p log p; clip avoids log(0) (NaN under div-by-zero).
        ent = -np.sum(spec_norm * np.log(spec_norm + 1e-12), axis=1)
        out[write_idx] = ent
    return out


def rolling_dominant_freq_idx(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    detrend: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Index of the dominant frequency bin (excluding DC) per K-window.

    Returns the argmax of ``|rfft(window)|`` over bins 1..n_freq-1.
    Distance from 1 = how high-frequency the signal is locally.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _sort_idx_seg, wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        finite_mask = np.isfinite(wins)
        if not finite_mask.all():
            mu = np.where(
                finite_mask.any(axis=1, keepdims=True),
                np.nanmean(wins, axis=1, keepdims=True),
                0.0,
            )
            wins = np.where(finite_mask, wins, mu)
        if detrend:
            wins = _detrend_zero_mean(wins)
        spec = np.abs(np.fft.rfft(wins, axis=1))
        # Skip DC (column 0)
        dom = np.argmax(spec[:, 1:], axis=1) + 1
        out[write_idx] = dom.astype(np.float64)
    return out


def rolling_hf_lf_ratio(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    detrend: bool = True,
    fill_value: float = 1.0,
    clip_range: Tuple[float, float] = (0.0, 100.0),
) -> np.ndarray:
    """Ratio of high-frequency band energy to low-frequency band energy.

    Convenience wrapper around ``rolling_spectral_band_energies`` with
    the default 3-band split: returns ``e_hi / (e_lo + 1e-6)`` clipped
    to ``clip_range``. Values >>1 = signal energy concentrated in HF;
    <<1 = LF-dominated; ~1 = balanced.

    Useful as a single-scalar "smoothness" feature.
    """
    bands = rolling_spectral_band_energies(
        values, group_ids, window_K=window_K, detrend=detrend,
        fill_value=np.nan,
    )
    # bands[:, 0] = lo, bands[:, -1] = hi
    e_lo = bands[:, 0]
    e_hi = bands[:, -1]
    ratio = e_hi / (e_lo + 1e-6)
    ratio = np.where(np.isfinite(ratio), ratio, fill_value)
    return np.clip(ratio, clip_range[0], clip_range[1])


def _spec_pow(seg_f: np.ndarray, K: int, detrend: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Window into K-stride, optional zero-mean detrend, return |rfft|^2.

    Helper shared by centroid/bandwidth/rolloff/flatness/flux/periodicity.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    wins = sliding_window_view(seg_f, K)
    if detrend:
        wins = wins - wins.mean(axis=1, keepdims=True)
    return np.abs(np.fft.rfft(wins, axis=1)) ** 2, wins


def rolling_spectral_centroid(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    detrend: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Power-weighted mean bin index per window: ``Sum(k * P_k) / Sum(P_k)``.

    "Where the mass of the spectrum sits": low = LF-dominated, high =
    HF-dominated. Used in audio brightness / timbre, bearing-wear
    monitoring (centroid drifts up as raceway pits develop), road-
    roughness telemetry, EEG alpha-vs-beta shift detection.
    Almost always shipped together with ``rolling_spectral_bandwidth``.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for sort_idx_seg, _wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        seg = values[sort_idx_seg].astype(np.float64)
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0  # all-NaN window: nanmean is NaN and `NaN or 0` keeps NaN (NaN is truthy), so guard explicitly
        seg_f = np.where(np.isfinite(seg), seg, seg_mean)
        spec, _ = _spec_pow(seg_f, window_K, detrend)
        n_freq = spec.shape[1]
        k = np.arange(n_freq, dtype=np.float64)
        denom = spec.sum(axis=1) + 1e-12
        # spec @ k == (spec * k[None, :]).sum(axis=1) but via BLAS gemv (no (rows, n_freq) temporary): 5-30x faster on the reduction, ~1e-16 reduction-order delta.
        out[write_idx] = (spec @ k) / denom
    return out


def rolling_spectral_bandwidth(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    detrend: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Power-weighted std of bin index around the centroid.

    "How spread the spectral mass is". Pair with ``rolling_spectral_
    centroid`` for a (location, scale) descriptor of the spectrum as
    a 1D distribution.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for sort_idx_seg, _wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        seg = values[sort_idx_seg].astype(np.float64)
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0  # all-NaN window: nanmean is NaN and `NaN or 0` keeps NaN (NaN is truthy), so guard explicitly
        seg_f = np.where(np.isfinite(seg), seg, seg_mean)
        spec, _ = _spec_pow(seg_f, window_K, detrend)
        n_freq = spec.shape[1]
        k = np.arange(n_freq, dtype=np.float64)
        denom = spec.sum(axis=1) + 1e-12
        centroid = (spec @ k) / denom  # BLAS gemv; see rolling_spectral_centroid for the equivalence note.
        # variance about centroid
        var = (spec * (k[None, :] - centroid[:, None]) ** 2).sum(axis=1) / denom
        out[write_idx] = np.sqrt(np.clip(var, 0.0, None))
    return out


def rolling_spectral_rolloff(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    percentile: float = 0.85,
    detrend: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Smallest bin k* such that the cumulative power up to k* >= q * total.

    Robust tail descriptor: unlike ``dominant_freq_idx`` it doesn't
    collapse to a single peak; unlike ``centroid`` it isn't pulled by
    distant outliers. Cheap CDF lookup per window.

    Use cases: voiced/unvoiced speech, audio fingerprinting, network
    burstiness, predictive maintenance (rolloff jump = new HF noise
    source even when total energy unchanged).
    """
    if not (0.0 < percentile <= 1.0):
        raise ValueError(f"percentile must be in (0, 1], got {percentile}")
    out = np.full(values.size, fill_value, dtype=np.float64)
    for sort_idx_seg, _wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        seg = values[sort_idx_seg].astype(np.float64)
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0  # all-NaN window: nanmean is NaN and `NaN or 0` keeps NaN (NaN is truthy), so guard explicitly
        seg_f = np.where(np.isfinite(seg), seg, seg_mean)
        spec, _ = _spec_pow(seg_f, window_K, detrend)
        cum = np.cumsum(spec, axis=1)
        total = cum[:, -1:] + 1e-12
        targets = percentile * total
        # First column index where cum >= target per row.
        idx = (cum >= targets).argmax(axis=1)
        out[write_idx] = idx.astype(np.float64)
    return out


def rolling_spectral_flatness(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    detrend: bool = True,
    eps: float = 1e-12,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Wiener entropy: ``geomean(P) / mean(P)``, bounded [0, 1].

    ~0 = pure tone / strong periodicity, ~1 = white-noise-like.
    Distinct from ``rolling_spectral_entropy`` (unbounded in nats);
    flatness is a normalised ratio downstream models prefer.

    Use cases: tonal vs noisy regime classifier (engine knock, ECG vs
    noise, voice-activity), volatility regime compression in finance,
    audio codec quality.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for sort_idx_seg, _wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        seg = values[sort_idx_seg].astype(np.float64)
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0  # all-NaN window: nanmean is NaN and `NaN or 0` keeps NaN (NaN is truthy), so guard explicitly
        seg_f = np.where(np.isfinite(seg), seg, seg_mean)
        spec, _ = _spec_pow(seg_f, window_K, detrend)
        # Drop DC bin (always 0 after detrend; would give log(0) noise).
        s = spec[:, 1:] + eps
        log_g = np.log(s).mean(axis=1)
        a = s.mean(axis=1)
        out[write_idx] = np.exp(log_g) / (a + eps)
    return out


def rolling_spectral_flux(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    normalize: bool = True,
    detrend: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """L2 distance between consecutive magnitude spectra: ``Sum_k (|F_k|_t - |F_k|_{t-1})^2``.

    Captures *temporal change of the spectrum* — the derivative side
    that all 4 base spectral features (snapshot statistics) miss.
    Canonical "is something happening NOW?" feature in audio/sensor
    pipelines.

    Use cases: audio onset detection (MIR), change-point in vibration
    monitoring (tool-breakage moment), EEG event detection, regime
    change in market microstructure.

    ``normalize=True`` divides by the sum of both spectra (cosine-like
    bound) so flux is comparable across signal amplitude levels.
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for sort_idx_seg, _wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        seg = values[sort_idx_seg].astype(np.float64)
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0  # all-NaN window: nanmean is NaN and `NaN or 0` keeps NaN (NaN is truthy), so guard explicitly
        seg_f = np.where(np.isfinite(seg), seg, seg_mean)
        spec, _ = _spec_pow(seg_f, window_K, detrend)
        mag = np.sqrt(spec)
        # Flux at row r = sum over k of (mag[r, k] - mag[r-1, k])^2.
        # First row of segment has no previous -> NaN.
        diff = mag[1:] - mag[:-1]
        flux = (diff**2).sum(axis=1)
        if normalize:
            tot = mag[1:].sum(axis=1) + mag[:-1].sum(axis=1) + 1e-12
            flux = flux / tot
        # write_idx points to the LAST-position anchor for each window.
        # First sliding-window row is at write_idx[0]; flux is undefined
        # there (no previous window). Map flux to write_idx[1:].
        out[write_idx[1:]] = flux
    return out


def rolling_periodicity_score(
    values: np.ndarray,
    group_ids: np.ndarray,
    window_K: int = 100,
    *,
    exclude_lag_zero: bool = True,
    detrend: bool = True,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Periodicity strength via autocorrelation (Wiener-Khinchin).

    ACF = irfft(|rfft(window)|^2). Score =
    ``max(ACF[lag>0]) / mean(|ACF[lag>0]|)``. High = strong repeating
    pattern at SOME lag (no need to know which); low = aperiodic.

    Use cases: rhythmic pattern detection without committing to a
    fundamental (HRV breathing-rate coupling, motor cycle counting
    from accelerometer, recurring fault signatures, periodic-vs-
    aperiodic user behaviour).
    """
    out = np.full(values.size, fill_value, dtype=np.float64)
    for sort_idx_seg, _wins, write_idx in per_group_sliding_window(
        values, group_ids, window_K=window_K,
    ):
        seg = values[sort_idx_seg].astype(np.float64)
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0  # all-NaN window: nanmean is NaN and `NaN or 0` keeps NaN (NaN is truthy), so guard explicitly
        seg_f = np.where(np.isfinite(seg), seg, seg_mean)
        spec, _ = _spec_pow(seg_f, window_K, detrend)
        acf = np.fft.irfft(spec, n=window_K, axis=1)
        # Use first half (lags 0 .. K//2) where ACF is meaningful.
        half = window_K // 2
        start = 1 if exclude_lag_zero else 0
        acf_lags = acf[:, start : half + 1]
        peak = np.abs(acf_lags).max(axis=1)
        mean_abs = np.abs(acf_lags).mean(axis=1) + 1e-12
        out[write_idx] = peak / mean_abs
    return out
