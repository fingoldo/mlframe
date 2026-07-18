"""biz_value: outlier-robust basis-axis normalisation (DEFAULT ON).

The univariate-basis FE preprocessors fit their normalisation scale from RAW per-column statistics: the Fourier axis uses
``lo/span = min/max-min``; the Hermite z-score uses raw ``std``; the Legendre/Chebyshev min-max uses raw ``min/max``. On a
heavy-tailed / spike-contaminated column (e.g. 5% of values at +/-1000) the raw span / std blows up ~1000x, collapsing 99%
of the data into a sliver of the axis. The engineered transform then (a) carries an OUTLIER-INFLATED plug-in MI that can
hijack selection, and (b) is SHIFT-FRAGILE -- a single new extreme value in production shifts the axis and changes every
row's engineered value.

The robust-axis fix estimates the scale from a contamination-proof MAD range and clamps the mapped axis, GATED on a cheap
per-column spike-contamination detector so it is byte-identical to legacy on clean columns and engages only where the raw
scale is provably corrupted.

Each assertion pins a measured quantitative win with a wide margin (set 5-15% below the measured value) so mild
measurement noise does not trip it but a real regression (gate stuck off, clamp removed, robust scale broken) fails it.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
    _evaluate_basis_column,
    _fit_fourier_for_col,
)

_N = 4000
_OUTLIER_FRAC = 0.05
_EXTREME = 1000.0


def _contaminate(base: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Inject ``_OUTLIER_FRAC`` of +/-_EXTREME spikes into a copy of ``base``. Returns ``(contaminated, inlier_mask)``."""
    cont = base.copy()
    n = base.size
    idx = rng.choice(n, int(n * _OUTLIER_FRAC), replace=False)
    cont[idx] = rng.choice([-1.0, 1.0], idx.size) * _EXTREME
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    return cont, mask


def _fourier_axis(x: np.ndarray, lo: float, span: float, freq: float = 1.0) -> np.ndarray:
    """Fourier axis."""
    z = (np.asarray(x, dtype=np.float64) - lo) / max(span, 1e-12)
    return np.sin(2.0 * np.pi * freq * z)


# ---------------------------------------------------------------------------
# Win 1: the bin-collapse mechanism is fixed -- the bulk spans the axis again.
# ---------------------------------------------------------------------------


def test_biz_val_robust_axis_fourier_inlier_spread_not_collapsed():
    """On a 5%-spike column the legacy Fourier axis collapses the 95% inlier mass into a ~0.0005-wide sliver of the [0, 1]
    axis (one bin); the robust axis spreads it across ~0.156. Measured spread ratio ~312x; floor 50x. This is the ROOT
    mechanism behind both the MI inflation and the shift-fragility -- if it regresses, the bulk re-collapses."""
    spread_ratios = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(_N)
        cont, mask = _contaminate(base, rng)

        os.environ["MLFRAME_ROBUST_AXIS"] = "0"
        lo_l, sp_l = _fit_fourier_for_col(cont)
        z_l = (cont[mask] - lo_l) / max(sp_l, 1e-12)
        legacy_spread = float(np.std(z_l))

        os.environ["MLFRAME_ROBUST_AXIS"] = "1"
        lo_r, sp_r = _fit_fourier_for_col(cont)
        z_r = (cont[mask] - lo_r) / max(sp_r, 1e-12)
        robust_spread = float(np.std(z_r))

        spread_ratios.append(robust_spread / max(legacy_spread, 1e-12))
    os.environ.pop("MLFRAME_ROBUST_AXIS", None)
    median_ratio = float(np.median(spread_ratios))
    assert median_ratio >= 50.0, (
        f"robust Fourier axis must spread the inlier mass >=50x wider than the collapsed legacy axis (measured ~312x); "
        f"got median {median_ratio:.1f}x -- the gate is off or the robust span is broken."
    )


# ---------------------------------------------------------------------------
# Win 2: shift-fragility eliminated -- a new outlier no longer moves the axis.
# ---------------------------------------------------------------------------


def test_biz_val_robust_axis_fourier_shift_stable_under_new_outlier():
    """The SHIFT-FRAGILE defect: with the legacy raw-span axis, ONE new extreme value at fit time shifts the engineered
    value of every clean row by up to ~0.88 (out of the sin's [-1, 1] range). The robust MAD-anchored axis is essentially
    unmoved (~0.0016). Measured legacy >=0.87, robust <=0.007; assert legacy >=0.5 AND robust <=0.05 (>10x margin)."""
    legacy_drifts, robust_drifts = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(_N)
        cont, _ = _contaminate(base, rng)
        probe = base[:1000]
        for ev, store in (("0", legacy_drifts), ("1", robust_drifts)):
            os.environ["MLFRAME_ROBUST_AXIS"] = ev
            lo1, sp1 = _fit_fourier_for_col(cont)
            v1 = _fourier_axis(probe, lo1, sp1)
            cont2 = np.concatenate([cont, [5.0 * _EXTREME]])
            lo2, sp2 = _fit_fourier_for_col(cont2)
            v2 = _fourier_axis(probe, lo2, sp2)
            store.append(float(np.max(np.abs(v1 - v2))))
    os.environ.pop("MLFRAME_ROBUST_AXIS", None)
    legacy_drift = float(np.median(legacy_drifts))
    robust_drift = float(np.median(robust_drifts))
    assert legacy_drift >= 0.5, (
        f"legacy axis is supposed to be shift-fragile (drift ~0.88); got {legacy_drift:.4f} -- the contrast test is no longer measuring the bug."
    )
    assert robust_drift <= 0.05, (
        f"robust axis must be shift-stable under a new outlier (measured ~0.0016); got {robust_drift:.4f} -- the robust "
        f"path is not engaging or the MAD anchor is leaking the new spike into the span."
    )
    assert (legacy_drift / max(robust_drift, 1e-12)) >= 10.0, (
        f"robust axis must be >=10x more shift-stable than legacy; got {legacy_drift / max(robust_drift, 1e-12):.1f}x."
    )


# ---------------------------------------------------------------------------
# Win 3: engineered column does not collapse to a degenerate near-constant.
# ---------------------------------------------------------------------------


def test_biz_val_robust_axis_engineered_column_not_degenerate_under_outliers():
    """The plug-in-MI hijack risk comes from the engineered column COLLAPSING under the legacy axis: a 5%-spike column
    maps 95% of rows to a ~9e-05-wide sliver of He_2 values (effectively a constant), and a degenerate near-constant
    column produces an erratic / inflated quantile-bin MI that can out-rank genuine signal. The robust axis keeps the
    engineered inlier values spread (IQR ~1.07 vs legacy ~9e-05). Measured ratio ~11600x; floor: legacy IQR <=1e-3 AND
    robust IQR >=0.3 (so the engineered column carries an honest, non-degenerate distribution). This is the deterministic
    root-cause assertion -- the MI inflation on a pure-noise column is finite-sample binning noise and is NOT a stable
    sensor, but the engineered-value collapse that CAUSES it is."""
    legacy_iqrs, robust_iqrs = [], []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        base = rng.standard_normal(_N)
        cont, mask = _contaminate(base, rng)
        for ev, store in (("0", legacy_iqrs), ("1", robust_iqrs)):
            os.environ["MLFRAME_ROBUST_AXIS"] = ev
            v = np.asarray(_evaluate_basis_column(cont, "hermite", 2), dtype=np.float64)[mask]
            q25, q75 = np.quantile(v, [0.25, 0.75])
            store.append(float(q75 - q25))
    os.environ.pop("MLFRAME_ROBUST_AXIS", None)
    legacy_iqr = float(np.median(legacy_iqrs))
    robust_iqr = float(np.median(robust_iqrs))
    assert legacy_iqr <= 1e-3, (
        f"legacy engineered column is supposed to COLLAPSE under outliers (IQR ~9e-05); got {legacy_iqr:.2e} -- the "
        f"contrast test is no longer measuring the bug."
    )
    assert robust_iqr >= 0.3, (
        f"robust engineered column must stay non-degenerate (measured inlier IQR ~1.07); got {robust_iqr:.4f} -- the "
        f"robust axis is not engaging or the clamp is over-compressing the bulk."
    )


# ---------------------------------------------------------------------------
# Win 4: clean-column byte-identity (non-negotiable byte-stability guarantee).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis,degree", [("hermite", 2), ("hermite", 3), ("legendre", 2), ("chebyshev", 3), ("laguerre", 2)])
def test_biz_val_robust_axis_clean_column_byte_identical(basis, degree):
    """On a CLEAN column the robust path must be byte-identical to legacy (the gate stays off). Asserts np.array_equal
    (0 tolerance) between the engineered values with MLFRAME_ROBUST_AXIS on vs off, for every basis. This is the
    non-negotiable byte-stability guarantee that keeps the wide FE suite green."""
    rng = np.random.default_rng(7)
    if basis == "laguerre":
        x = np.abs(rng.standard_normal(_N)) + 0.01  # positive-domain column for the shift preprocessor.
    else:
        x = rng.standard_normal(_N)

    os.environ["MLFRAME_ROBUST_AXIS"] = "1"
    v_on = np.asarray(_evaluate_basis_column(x, basis, degree), dtype=np.float64)
    os.environ["MLFRAME_ROBUST_AXIS"] = "0"
    v_off = np.asarray(_evaluate_basis_column(x, basis, degree), dtype=np.float64)
    os.environ.pop("MLFRAME_ROBUST_AXIS", None)
    assert np.array_equal(v_on, v_off), (
        f"{basis} He{degree} on a CLEAN column is NOT byte-identical with the robust axis on vs off -- the gate is "
        f"leaking into the clean path. max|diff|={float(np.max(np.abs(v_on - v_off))):.3e}"
    )


def test_biz_val_robust_axis_clean_fourier_byte_identical():
    """Same byte-identity guarantee for the Fourier axis ``(lo, span)`` on a clean column."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(_N)
    os.environ["MLFRAME_ROBUST_AXIS"] = "1"
    lo_on, sp_on = _fit_fourier_for_col(x)
    os.environ["MLFRAME_ROBUST_AXIS"] = "0"
    lo_off, sp_off = _fit_fourier_for_col(x)
    os.environ.pop("MLFRAME_ROBUST_AXIS", None)
    assert lo_on == lo_off and sp_on == sp_off, f"clean Fourier (lo, span) not byte-identical robust-on vs off: ({lo_on}, {sp_on}) != ({lo_off}, {sp_off})."
