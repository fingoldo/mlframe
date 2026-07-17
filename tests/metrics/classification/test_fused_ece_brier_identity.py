"""Identity regression: fused single-pass ECE/Brier kernel == three separate binning kernels.

``compute_ece_brier_full_and_debiased`` bins ONCE and emits the plug-in decomposition + both debiased
estimators. It must be BIT-IDENTICAL to calling ``compute_ece_and_brier_decomposition`` +
``compute_ece_debiased`` + ``compute_brier_decomposition_debiased`` separately (same histogram, same
per-bin formulas). The default ``fast_calibration_report`` headline path dispatches to the fused kernel,
so a divergence here would silently shift every reported ECE / Brier REL / RES.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_metrics import (
    compute_ece_and_brier_decomposition,
    compute_ece_debiased,
    compute_brier_decomposition_debiased,
    compute_ece_brier_full_and_debiased,
)


def _separate(y, p, nbins):
    ece_pl, rel_pl, res_pl, unc, br_pl = compute_ece_and_brier_decomposition(y, p, nbins)
    ece_db = compute_ece_debiased(y, p, nbins)
    rel_db, res_db, _unc2, br_db = compute_brier_decomposition_debiased(y, p, nbins)
    return ece_pl, rel_pl, res_pl, unc, br_pl, ece_db, rel_db, res_db, br_db


@pytest.mark.parametrize("n", [1, 2, 5, 100, 5_000])
@pytest.mark.parametrize("nbins", [1, 10, 15, 100])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_fused_matches_separate_kernels_bit_identical(n, nbins, seed):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(0.0, 1.0, n), 1e-9, 1 - 1e-9).astype(np.float64)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(np.float64)

    sep = _separate(y, p, nbins)
    fused = compute_ece_brier_full_and_debiased(y, p, nbins)
    # fused order: ece_pl, rel_pl, res_pl, unc, br_pl, ece_db, rel_db, res_db, br_db
    for a, b in zip(sep, fused):
        assert a == b, f"divergence n={n} nbins={nbins} seed={seed}: {a} != {b}"


def test_fused_empty_input_sentinels():
    e = np.empty(0, dtype=np.float64)
    sep = _separate(e, e, 10)
    fused = compute_ece_brier_full_and_debiased(e, e, 10)
    for a, b in zip(sep, fused):
        assert a == b


def test_fused_extremes_and_boundary_probs():
    # p exactly at bin boundaries and 0/1, both classes.
    p = np.array([0.0, 1.0, 0.5, 0.25, 0.75, 0.999999], dtype=np.float64)
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
    for nbins in (1, 4, 10, 50):
        sep = _separate(y, p, nbins)
        fused = compute_ece_brier_full_and_debiased(y, p, nbins)
        for a, b in zip(sep, fused):
            assert a == b


def test_fused_bool_y_true_matches():
    rng = np.random.default_rng(3)
    p = np.clip(rng.uniform(0, 1, 500), 1e-9, 1 - 1e-9).astype(np.float64)
    yb = rng.uniform(0, 1, 500) < p
    sep = _separate(yb.astype(np.float64), p, 15)
    fused = compute_ece_brier_full_and_debiased(yb.astype(np.float64), p, 15)
    for a, b in zip(sep, fused):
        assert a == b
