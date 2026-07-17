"""Identity pin for the njit-fused Family-D conditional z-score
(``_extra_fe_families_dispersion._zscore_from_bins`` -> ``_zscore_from_bins_njit``).

The njit single-pass kernel must be BIT-IDENTICAL to the prior numpy multi-pass
body (clip codes + gather bin_mean/bin_std + floor-std + isfinite-masked divide).
Pins the edge cases that matter: NaN x_i rows -> 0.0, a floored (degenerate) bin
std -> divide by 1.0, and out-of-range codes clipped to the last bin.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _extra_fe_families_dispersion as M

_FLOOR = M._DISPERSION_SIGMA_FLOOR


def _old_zscore(xi, codes_j, bin_mean, bin_std):
    """Verbatim pre-njit numpy body (the A/B baseline)."""
    codes_j = np.clip(codes_j, 0, bin_mean.size - 1)
    per_row_mean = bin_mean[codes_j]
    per_row_std = bin_std[codes_j]
    per_row_std = np.where(per_row_std >= _FLOOR, per_row_std, 1.0)
    finite_i = np.isfinite(xi)
    z = np.zeros_like(xi, dtype=np.float64)
    z[finite_i] = (xi[finite_i] - per_row_mean[finite_i]) / per_row_std[finite_i]
    return z


@pytest.mark.parametrize("n", [37, 1000, 50_000])
@pytest.mark.parametrize("nb", [1, 3, 10])
def test_zscore_njit_bit_identical_to_numpy(n, nb):
    rng = np.random.default_rng(n * 100 + nb)
    xi = rng.standard_normal(n)
    # inject NaNs (the masked-to-0 path)
    xi[rng.random(n) < 0.07] = np.nan
    codes = rng.integers(0, nb, size=n).astype(np.int64)
    bin_mean = rng.standard_normal(nb)
    bin_std = np.abs(rng.standard_normal(nb)) + 0.1
    if nb >= 2:
        bin_std[nb // 2] = 0.0  # force the std-floor branch
    old = _old_zscore(xi, codes, bin_mean, bin_std)
    new = M._zscore_from_bins(xi, codes, bin_mean, bin_std)
    assert new.dtype == np.float64
    np.testing.assert_array_equal(new, old)  # EXACT bit-identity


def test_zscore_out_of_range_codes_clipped():
    """A code >= nb (defensive at replay) clips to the last bin, == numpy clip."""
    nb = 4
    xi = np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float64)
    codes = np.array([0, nb + 5, -3, 1], dtype=np.int64)
    bin_mean = np.array([0.0, 1.0, 2.0, 3.0])
    bin_std = np.array([1.0, 2.0, 3.0, 4.0])
    old = _old_zscore(xi, codes, bin_mean, bin_std)
    new = M._zscore_from_bins(xi, codes, bin_mean, bin_std)
    np.testing.assert_array_equal(new, old)


def test_generate_full_output_bit_identical(monkeypatch):
    """End-to-end: full generate_conditional_dispersion_features output is
    unchanged vs the numpy baseline."""
    rng = np.random.default_rng(7)
    n = 4000
    base = rng.standard_normal(n)
    X = pd.DataFrame({f"c{i}": rng.standard_normal(n) * (1 + np.abs(base)) for i in range(4)})
    enc_new, _ = M.generate_conditional_dispersion_features(X, list(X.columns), n_bins=10)
    monkeypatch.setattr(M, "_zscore_from_bins", _old_zscore)
    enc_old, _ = M.generate_conditional_dispersion_features(X, list(X.columns), n_bins=10)
    assert list(enc_new.columns) == list(enc_old.columns)
    np.testing.assert_array_equal(enc_new.to_numpy(), enc_old.to_numpy())
