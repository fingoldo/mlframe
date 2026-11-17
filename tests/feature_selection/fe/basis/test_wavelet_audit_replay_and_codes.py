"""Wavelet FE: replay in float64, non-dense integer target codes, fused GPU leg builder on float64 z."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests._cuda_require import require_cuda_device

from mlframe.feature_selection.filters import _wavelet_basis_fe as W
from mlframe.feature_selection.filters._wavelet_basis_fe_recipes import _apply_orth_wavelet, build_orth_wavelet_recipe


def test_replay_cells_are_cut_on_float64_z_like_the_fit():
    """A float32 column value that lands on the other side of a cell edge in float32 arithmetic is replayed as the fit saw it."""
    lo, span = -1.3, 2.9
    x = np.array([0.15000002] * 5 + [0.5] * 5, dtype=np.float32)
    rec = build_orth_wavelet_recipe(name="a__haar_j0k0", src_name="a", j=0, k=0, lo=lo, span=span)
    got = _apply_orth_wavelet(rec, pd.DataFrame({"a": x}))
    z64 = np.clip((x.astype(np.float64) - lo) / span, 0.0, 1.0)
    np.testing.assert_array_equal(got, W._dyadic_haar_leg(z64, 0, 0))


def test_integer_target_with_negative_labels_selects_the_same_legs_as_zero_based_labels():
    """Labels {-1, 1} must not collide in the leg-MI histogram: the legs equal those of the {0, 1} coding."""
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.random(n)
    y01 = (((x > 0.55) & (x < 0.7)) ^ (rng.random(n) < 0.05)).astype(np.int64)
    lo, span = float(x.min()), float(x.max() - x.min())
    base = W._select_wavelet_legs(x, y01, lo, span, use_device=False)
    assert base
    assert W._select_wavelet_legs(x, 2 * y01 - 1, lo, span, use_device=False) == base
    codes = W._bin_y_codes(2 * y01 - 1)
    assert int(codes.min()) == 0 and int(codes.max()) == 1


def test_fused_leg_builder_cuts_cells_on_float64_z(monkeypatch):
    """z values 1e-9 below a cell edge stay below it in the fused GPU builder, as in the per-leg float64 reference."""
    cp = pytest.importorskip("cupy")
    require_cuda_device()
    from mlframe.feature_selection.filters._wavelet_basis_fe_batched import _per_leg_matrices
    from mlframe.feature_selection.filters._wavelet_legs_fused_gpu import leg_code_matrices

    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_vram.fe_gpu_has_vram_cushion", lambda nbytes: True)
    rng = np.random.default_rng(0)
    z = rng.random(600)
    z[::7] = 0.5 - 1e-9
    z_g = cp.asarray(z)
    idx = np.arange(z.size)
    va = idx % 3 == 0
    fused = leg_code_matrices(cp, z_g, 1, 1)
    if fused is None:
        pytest.skip("fused builder declined (no free VRAM)")
    metas_f, tr_f, va_f = fused
    metas_r, tr_r, va_r = _per_leg_matrices(cp, z_g, 1, ~va, va, 1)
    assert metas_f == metas_r
    np.testing.assert_array_equal(cp.asnumpy(tr_f), cp.asnumpy(tr_r))
    np.testing.assert_array_equal(cp.asnumpy(va_f), cp.asnumpy(va_r))
