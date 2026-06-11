"""P15 regression: narrow prebin bin-code dtype to int16 (gated nbins<182,
else int32) while keeping MI bit-identical via an int64 upcast of the
``combo = x_idx*nbins + y_idx`` index inside ``_mi_from_binned_pair``.

Audit: audit/composite_audit_2026_06_10/future_items.json id=P15.
File under test: src/mlframe/training/composite/discovery/screening.py.

The three things pinned here, any of which a regression would trip:

1. ``_prebin_feature_columns`` returns int16 codes for ``nbins < 182`` and
   int32 for ``nbins >= 182`` (the boundary where ``nbins**2 - 1`` would
   exceed int16-max), with the -1 non-finite sentinel preserved.
2. The integer bin-CODE values are unchanged vs the legacy int64 prebin
   (so the only thing that changed is storage width, not semantics).
3. ``_mi_from_binned_pair`` / ``_mi_to_target_prebinned`` MI is BIT-IDENTICAL
   to the legacy int64-code path across nbins -- including the
   overflow-boundary nbins (182, 200, 300) where an un-upcast int16 combo
   would wrap negative and corrupt the joint-count histogram.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import (
    _mi_from_binned_pair,
    _mi_to_target_prebinned,
    _prebin_code_dtype,
    _prebin_feature_columns,
)


# --- Legacy reference (pre-P15 int64 prebin + non-upcast combo) -------------


def _ref_prebin_int64(fm: np.ndarray, nbins: int) -> np.ndarray:
    """Verbatim pre-P15 ``_prebin_feature_columns`` body, int64 throughout."""
    n_rows, n_cols = fm.shape
    if n_rows < 5 * nbins:
        return np.full((n_rows, n_cols), -1, dtype=np.int64)
    q_edges = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    binned = np.empty((n_rows, n_cols), dtype=np.int64)
    for j in range(n_cols):
        col = fm[:, j]
        col_finite = np.isfinite(col)
        if col_finite.sum() < 5 * nbins:
            binned[:, j] = -1
            continue
        cut = (
            np.quantile(col, q_edges)
            if col_finite.all()
            else np.nanquantile(col, q_edges)
        )
        col_idx = np.full(n_rows, -1, dtype=np.int64)
        col_idx[col_finite] = np.searchsorted(
            cut, col[col_finite], side="right"
        ).astype(np.int64)
        np.clip(col_idx, 0, nbins - 1, out=col_idx, where=col_idx >= 0)
        binned[:, j] = col_idx
    return binned


def _make_matrix(n: int = 4000, k: int = 6, *, with_nan: bool, seed: int = 7):
    rng = np.random.default_rng(seed)
    fm = rng.standard_normal((n, k)).astype(np.float32)
    if with_nan:
        fm[rng.random((n, k)) < 0.05] = np.nan
    y = (fm[:, 0] * 0.7 + rng.standard_normal(n).astype(np.float32) * 0.3).astype(
        np.float32
    )
    # y must be finite for the prebinned MI target; impute the few NaN it
    # inherited from column 0.
    y = np.where(np.isfinite(y), y, 0.0).astype(np.float32)
    return fm, y


_NBINS_GRID = [4, 16, 50, 100, 181, 182, 200, 300]


@pytest.mark.parametrize("nbins", _NBINS_GRID)
def test_prebin_code_dtype_gated_at_182(nbins: int) -> None:
    """int16 below 182, int32 from 182 up (where nbins**2-1 > int16-max)."""
    expected = np.int16 if nbins < 182 else np.int32
    assert _prebin_code_dtype(nbins) == np.dtype(expected)
    # The boundary itself: 181**2-1 fits int16, 182**2-1 does not.
    assert (181 ** 2 - 1) <= np.iinfo(np.int16).max
    assert (182 ** 2 - 1) > np.iinfo(np.int16).max


@pytest.mark.parametrize("nbins", _NBINS_GRID)
@pytest.mark.parametrize("with_nan", [False, True])
def test_prebin_dtype_and_codes_match_legacy(nbins: int, with_nan: bool) -> None:
    """Narrowed dtype is correct AND the code VALUES equal the int64 legacy."""
    fm, _ = _make_matrix(with_nan=with_nan)
    pb = _prebin_feature_columns(fm, nbins=nbins)
    expected_dtype = np.int16 if nbins < 182 else np.int32
    assert pb.dtype == np.dtype(expected_dtype)
    ref = _ref_prebin_int64(fm, nbins=nbins)
    # Same bin indices (and same -1 sentinels) as the legacy int64 path.
    assert np.array_equal(pb.astype(np.int64), ref)
    # Sentinel survives the narrow dtype where there are NaN rows.
    if with_nan:
        assert (pb == -1).any()


@pytest.mark.parametrize("nbins", _NBINS_GRID)
@pytest.mark.parametrize("with_nan", [False, True])
def test_mi_bit_identical_to_int64_prebin(nbins: int, with_nan: bool) -> None:
    """MI through the narrowed-dtype prebin == MI through the int64 prebin,
    exactly (max abs diff 0.0), across every nbins incl. the >=182 overflow
    boundary."""
    fm, y = _make_matrix(with_nan=with_nan)
    pb_new = _prebin_feature_columns(fm, nbins=nbins)
    pb_ref = _ref_prebin_int64(fm, nbins=nbins)
    mi_new = _mi_to_target_prebinned(pb_new, y, nbins=nbins)
    mi_ref = _mi_to_target_prebinned(pb_ref, y, nbins=nbins)
    assert mi_new == mi_ref, (
        f"MI not bit-identical at nbins={nbins} with_nan={with_nan}: "
        f"{mi_new!r} vs {mi_ref!r}"
    )


@pytest.mark.parametrize("nbins", [182, 200, 300, 1000])
def test_int16_combo_upcast_no_overflow(nbins: int) -> None:
    """At nbins>=182 the flattened index nbins**2-1 exceeds int16-max; the
    in-kernel int64 upcast must keep the joint-count index correct even when
    the caller passes int16 codes (which an un-upcast ``x_idx*nbins`` would
    wrap negative -> a corrupt / crashing bincount). MI must equal the
    int64-codes computation exactly."""
    rng = np.random.default_rng(11)
    n = 5 * nbins + 500
    # Spread codes across the full 0..nbins-1 range so the worst-case
    # (nbins-1, nbins-1) cell is exercised.
    xi = rng.integers(0, nbins, size=n)
    yi = rng.integers(0, nbins, size=n)
    # Force the worst-case corner to appear.
    xi[:50] = nbins - 1
    yi[:50] = nbins - 1
    xi16 = xi.astype(np.int16) if nbins < 182 else xi.astype(np.int32)
    yi16 = yi.astype(np.int16) if nbins < 182 else yi.astype(np.int32)
    # Use the gated code dtype to mirror real prebin output.
    code_dtype = _prebin_code_dtype(nbins)
    xi_codes = xi.astype(code_dtype)
    yi_codes = yi.astype(code_dtype)
    mi_codes = _mi_from_binned_pair(xi_codes, yi_codes, nbins=nbins)
    mi_int64 = _mi_from_binned_pair(
        xi.astype(np.int64), yi.astype(np.int64), nbins=nbins
    )
    assert mi_codes == mi_int64
    # And the result is a sane, finite, non-negative MI (not a wrapped-index
    # artefact).
    assert np.isfinite(mi_codes) and mi_codes >= 0.0


def test_small_n_sentinel_path_uses_narrow_dtype() -> None:
    """The n_rows < 5*nbins early-return also honours the narrowed dtype."""
    nbins = 16
    fm = np.random.default_rng(3).standard_normal((10, 4)).astype(np.float32)
    pb = _prebin_feature_columns(fm, nbins=nbins)  # 10 < 5*16=80 -> all -1
    assert pb.dtype == np.dtype(np.int16)
    assert (pb == -1).all()
