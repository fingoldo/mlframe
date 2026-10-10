"""The two-kernel dyadic-Haar leg builder equals the per-leg cupy build entry for entry."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _wavelet_basis_fe_batched as wb
from mlframe.feature_selection.filters._wavelet_legs_fused_gpu import leg_code_matrices


def _reference(z, max_scale, min_half):
    """The per-leg cupy build with the same i % 3 split."""
    n = int(z.shape[0])
    idx = np.arange(n)
    va = (idx % 3) == 0
    return wb._per_leg_matrices(cp, z, max_scale, ~va, va, min_half)


def _host_codes(z_h, max_scale, min_half):
    """Straightforward numpy build of the eligible legs and their ``leg + 1`` code matrices, independent of both GPU implementations."""
    n = z_h.shape[0]
    is_val = (np.arange(n) % 3) == 0
    metas, cols = [], []
    for j in range(max_scale + 1):
        for k in range(2**j):
            left, right = k / 2.0**j, (k + 1) / 2.0**j
            mid = (left + right) / 2.0
            leg = np.where((z_h >= left) & (z_h < mid), 1, np.where((z_h >= mid) & (z_h < right), -1, 0))
            if (leg > 0).sum() >= min_half and (leg < 0).sum() >= min_half:
                metas.append((j, k))
                cols.append(leg + 1)
    if not metas:
        return metas, None, None
    codes = np.stack(cols, axis=1).astype(np.int64)
    return metas, codes[~is_val], codes[is_val]


def _same(z, max_scale, min_half):
    """Assert metas and both code matrices equal the per-leg cupy build and the numpy build; return the metas."""
    want_metas, want_tr, want_va = _reference(z, max_scale, min_half)
    host_metas, host_tr, host_va = _host_codes(cp.asnumpy(z), max_scale, min_half)
    got = leg_code_matrices(cp, z, max_scale, min_half)
    assert got is not None
    metas, tr, va = got
    assert metas == want_metas == host_metas
    if metas:
        assert tr.dtype == want_tr.dtype and tr.flags.c_contiguous and va.flags.c_contiguous
        np.testing.assert_array_equal(cp.asnumpy(tr), cp.asnumpy(want_tr))
        np.testing.assert_array_equal(cp.asnumpy(va), cp.asnumpy(want_va))
        np.testing.assert_array_equal(cp.asnumpy(tr), host_tr)
        np.testing.assert_array_equal(cp.asnumpy(va), host_va)
    return metas


@pytest.mark.parametrize("max_scale", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n", [2_000, 9_001, 30_000])
def test_uniform_columns(max_scale, n):
    """Uniform z over [0, 1]: all legs eligible, every code identical."""
    z = cp.asarray(np.random.default_rng(n + max_scale).uniform(size=n))
    assert len(_same(z, max_scale, 5)) == (1 << (max_scale + 1)) - 1


def test_exact_cell_boundaries_and_the_closed_right_edge():
    """Values exactly on k/2^j, (k+0.5)/2^j and 1.0 take the same side in both builds."""
    edges = np.concatenate([np.arange(0, 17) / 16.0, (np.arange(0, 16) + 0.5) / 16.0, np.arange(0, 9) / 8.0, [1.0, 0.0, 1.0 - 1e-16]])
    z = cp.asarray(np.tile(edges, 40))
    metas = _same(z, 4, 1)
    assert (0, 0) in metas and (4, 15) in metas
    got_metas, tr, _ = leg_code_matrices(cp, z, 4, 1)
    tr_h = cp.asnumpy(tr)
    z_tr = np.delete(cp.asnumpy(z), np.arange(0, z.shape[0], 3))
    col = got_metas.index((1, 1))
    # (j=1, k=1) is the cell [0.5, 1): +1 on [0.5, 0.75) -> code 2, -1 on [0.75, 1) -> code 0, 0 elsewhere -> code 1; the closed edge 1.0 is outside every cell
    want = np.where((z_tr >= 0.5) & (z_tr < 0.75), 2, np.where((z_tr >= 0.75) & (z_tr < 1.0), 0, 1))
    np.testing.assert_array_equal(tr_h[:, col], want)
    assert set(np.unique(tr_h[z_tr == 1.0, col])) == {1}


def test_clipped_and_nan_values_encode_as_outside():
    """z clipped to the ends and NaN rows are 0 in every leg, as the comparisons give."""
    rng = np.random.default_rng(0)
    z_h = rng.uniform(-0.3, 1.3, size=6_000)
    z_h[::17] = np.nan
    z = cp.clip(cp.asarray(z_h), 0.0, 1.0)
    metas = _same(z, 3, 5)
    assert len(metas) == 15
    _, tr, va = leg_code_matrices(cp, z, 3, 5)
    nan_rows = np.isnan(np.delete(z_h, np.arange(0, 6_000, 3)))
    assert nan_rows.any()
    assert (cp.asnumpy(tr)[nan_rows] == 1).all()
    assert (cp.asnumpy(va)[np.isnan(z_h[::3])] == 1).all()


def test_thin_support_removes_legs_like_the_loop():
    """With z concentrated in one cell only the legs covering it stay eligible, with the same order."""
    z = cp.asarray(np.random.default_rng(1).uniform(0.0, 0.2, size=5_000))
    metas = _same(z, 4, 100)
    assert 0 < len(metas) < 31


def test_no_eligible_leg_gives_an_empty_result():
    """A constant column supports no leg."""
    z = cp.asarray(np.full(3_000, 0.3))
    got = leg_code_matrices(cp, z, 3, 50)
    assert got[0] == [] and got[1] is None and got[2] is None
    assert _reference(z, 3, 50)[0] == []


def test_the_selection_is_unchanged_end_to_end(monkeypatch):
    """The admitted legs equal those of the per-leg path for a column with real dyadic structure."""
    rng = np.random.default_rng(5)
    n = 30_000
    x = rng.uniform(size=n)
    y = np.where((x > 0.25) & (x < 0.375), 1, 0) + 0.2 * rng.normal(size=n)
    monkeypatch.setenv("MLFRAME_FE_WAVELET_FUSED", "0")
    want = wb._select_wavelet_legs_batched_device(x, y, 0.0, 1.0, max_scale=3, max_legs=4, scale_sigma=3.0)
    monkeypatch.setenv("MLFRAME_FE_WAVELET_FUSED", "1")
    got = wb._select_wavelet_legs_batched_device(x, y, 0.0, 1.0, max_scale=3, max_legs=4, scale_sigma=3.0)
    assert got == want
