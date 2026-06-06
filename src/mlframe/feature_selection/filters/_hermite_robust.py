"""Outlier-robust axis normalisation for the orthogonal-polynomial preprocessors.

A cheap per-column spike-contamination gate (``_detect_heavy_tail``) plus the
MAD-anchored robust bounds (``_robust_lo_hi`` / ``_robust_scale``) the basis
preprocessors use when the raw per-column scale is corrupted by injected
spikes. GATED on ``_robust_axis_enabled`` (default ON; ``MLFRAME_ROBUST_AXIS=0``
replays the legacy raw-scale path) and byte-identical to legacy on clean columns.
"""
from __future__ import annotations

import numpy as np

# Robust bounds = median +/- _ROBUST_AXIS_K * (1.4826*MAD). MAD is contamination-proof up to ~50% of the column, so the
# derived span ~ 6*sigma stays anchored to the CLEAN core regardless of how many 1000x spikes are injected -- unlike an
# inner-quantile trim, which only excludes the tail when the trim fraction exceeds the contamination fraction. k=3 covers
# ~99.7% of a Gaussian core, matching the legacy intent that the working axis span the bulk of the data.
_ROBUST_AXIS_K = 3.0
# Spike-contamination detector parameters. The gate must trip on INJECTED SPIKE contamination (a thin fraction of points
# orders of magnitude beyond a dense bulk) WITHOUT tripping on a genuinely heavy-tailed-but-clean column (lognormal,
# Student-t, exponential) -- robustifying those would change engineered byte values on legitimate data. A single
# half-range/scale ratio cannot separate the two (a heavy lognormal and a 1%-spike column have similar max/MAD ratios), so
# we test for the SEPARATION SIGNATURE instead: contamination leaves a clear multiplicative GAP between the bulk and the
# spikes, while a smooth heavy tail is continuous. ``_ROBUST_AXIS_OUTER_K`` is the robust-scale multiple defining the
# candidate-outlier band; ``_ROBUST_AXIS_GAP`` is the bulk->outer jump that marks a true gap; ``_ROBUST_AXIS_MAX_FRAC``
# caps the outlier fraction so a column that is >20% beyond 10 sigma is treated as genuinely heavy, not spike-contaminated.
_ROBUST_AXIS_OUTER_K = 10.0
_ROBUST_AXIS_GAP = 3.0
_ROBUST_AXIS_MAX_FRAC = 0.20


def _robust_axis_enabled() -> bool:
    """Default ON. ``MLFRAME_ROBUST_AXIS=0`` forces the legacy raw-scale path for replay / A-B compare."""
    import os as _os
    flag = _os.environ.get("MLFRAME_ROBUST_AXIS", "").strip().lower()
    return flag not in ("0", "false", "off", "no")


def _detect_heavy_tail(x: np.ndarray) -> bool:
    """Cheap per-column SPIKE-contamination gate. True iff a thin fraction of points sit beyond ``_ROBUST_AXIS_OUTER_K``
    robust scales AND are separated from the bulk by a multiplicative GAP of at least ``_ROBUST_AXIS_GAP``.

    The gap test is what distinguishes injected spike contamination (a dense bulk + a handful of order-of-magnitude
    outliers with empty space between them) from a genuinely heavy-tailed-but-clean column (lognormal / Student-t /
    exponential), whose tail is CONTINUOUS with the bulk (gap ~ 1.0-1.3, measured). Only spike contamination corrupts the
    raw scale; a smooth heavy tail is the legitimate home of the skewed bases and must stay on the byte-identical legacy
    path. Degenerate columns (<8 finite values, near-constant, all-non-finite) never trip -- there is no scale to corrupt."""
    xf = x[np.isfinite(x)]
    if xf.size < 8:
        return False
    med = float(np.median(xf))
    robust_scale = _robust_scale(xf, med)
    if robust_scale <= 1e-12:
        return False
    dev = np.abs(xf - med)
    thr = _ROBUST_AXIS_OUTER_K * robust_scale
    outer_mask = dev > thr
    n_outer = int(np.count_nonzero(outer_mask))
    if n_outer == 0 or n_outer > _ROBUST_AXIS_MAX_FRAC * xf.size:
        # No extreme points, or so many that the tail is genuinely heavy rather than a thin contaminating spike.
        return False
    # Gap test without a full sort: the bulk edge is the largest deviation still inside the bulk, the outer edge the
    # smallest deviation in the candidate-outlier band -- two masked reductions (O(n)) instead of an O(n log n) sort.
    bulk_edge = float(dev[~outer_mask].max())
    outer_min = float(dev[outer_mask].min())
    return (outer_min / max(bulk_edge, 1e-12)) >= _ROBUST_AXIS_GAP


def _robust_scale(xf: np.ndarray, med: float) -> float:
    """Contamination-proof scale: 1.4826*MAD, with an IQR fallback when MAD collapses on a discrete / tied core. Returns
    0.0 only on a genuinely degenerate (near-constant) column, which the caller treats as 'no robust path'."""
    mad = float(np.median(np.abs(xf - med)))
    scale = 1.4826 * mad
    if scale > 1e-12:
        return scale
    q25, q75 = np.quantile(xf, [0.25, 0.75])
    iqr_scale = float(q75 - q25) / 1.349  # IQR/1.349 ~ sigma for a Gaussian; recovers a scale when MAD ties to 0.
    return iqr_scale if iqr_scale > 1e-12 else 0.0


def _robust_lo_hi(x: np.ndarray) -> tuple[float, float]:
    """MAD-anchored [lo, hi] bounds = median +/- k*(1.4826*MAD) on the finite subset. The span tracks the CLEAN core even
    under heavy contamination (MAD ignores up to ~50% outliers), unlike a fixed inner-quantile trim which only excludes
    the tail once the trim fraction exceeds the contamination fraction."""
    xf = x[np.isfinite(x)]
    med = float(np.median(xf))
    scale = _robust_scale(xf, med)
    if scale <= 1e-12:
        # Degenerate core: fall back to the actual finite min/max so the caller still gets a usable (non-zero) span.
        return float(np.min(xf)), float(np.max(xf))
    return med - _ROBUST_AXIS_K * scale, med + _ROBUST_AXIS_K * scale
