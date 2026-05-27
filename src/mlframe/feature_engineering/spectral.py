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
    return wins - wins.mean(axis=1, keepdims=True)


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
        spec_norm = spec / (spec.sum(axis=1, keepdims=True) + 1e-12)
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
