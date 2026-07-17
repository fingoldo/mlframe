"""Unit tests for ``compute_rff_features``, ``compute_positional_encoding``, and ``positions_within_group``.

Coverage:
- Shape, dtype, deterministic-from-seed contract for all three.
- Validation paths (NaN, non-numeric, sparse-like, bad dims, bad params, missing seed).
- ``positions_within_group`` ordinal correctness with and without ``sort_col``.

Marked @pytest.mark.fast for the fast subset - none of these need anything beyond numpy + polars + numba (and the warm-jit fixture).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlframe.feature_engineering.transformer import (
    compute_positional_encoding,
    compute_rff_features,
    positions_within_group,
)


pytestmark = pytest.mark.fast


# ---------- compute_rff_features ----------


def test_rff_shape_and_dtype():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 8)).astype(np.float32)
    df = compute_rff_features(X, seed=42, n_features=64, use_gpu=False)
    assert df.shape == (100, 64)
    assert all(df.dtypes[i] == pl.Float32 for i in range(64))
    # Half cos / half sin column naming.
    assert any("rff_cos_0" == c for c in df.columns)
    assert any("rff_sin_0" == c for c in df.columns)
    assert any(f"rff_sin_{32 - 1}" == c for c in df.columns)


def test_rff_deterministic_same_seed():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8)).astype(np.float32)
    df1 = compute_rff_features(X, seed=7, n_features=32, use_gpu=False)
    df2 = compute_rff_features(X, seed=7, n_features=32, use_gpu=False)
    np.testing.assert_array_equal(df1.to_numpy(), df2.to_numpy())


def test_rff_output_buffer_is_fortran_order_no_per_column_copy(monkeypatch):
    """The CPU/cupy kernels write into an F-contiguous ``out`` so each per-column slice handed to the polars DataFrame builder is already contiguous.

    Pre-fix the buffer was C-order, so polars called ``np.ascontiguousarray`` on every strided column slice -- a full per-column copy (8.4s of a 14s N=10M,
    d=8, n_features=64 e2e). This sensor counts ACTUAL copies (a returned array that does not share memory with its input) of large 1-D column slices and
    requires zero. On the pre-fix C-order buffer this count equals ``n_features``; here it must be 0.
    """
    orig = np.ascontiguousarray
    copies = {"n": 0}

    def _spy(a, *args, **kwargs):
        r = orig(a, *args, **kwargs)
        if isinstance(a, np.ndarray) and a.ndim == 1 and a.size >= 100_000:
            if r is not a and not np.shares_memory(r, a):
                copies["n"] += 1
        return r

    monkeypatch.setattr(np, "ascontiguousarray", _spy)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200_000, 8)).astype(np.float32)
    df = compute_rff_features(X, seed=1, n_features=64, use_gpu=False)
    assert df.shape == (200_000, 64)
    assert copies["n"] == 0, f"expected zero per-column copies on F-order buffer; got {copies['n']} (C-order regression)"


def test_rff_different_seed_changes_output():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8)).astype(np.float32)
    df1 = compute_rff_features(X, seed=7, n_features=32, use_gpu=False)
    df2 = compute_rff_features(X, seed=8, n_features=32, use_gpu=False)
    assert not np.allclose(df1.to_numpy(), df2.to_numpy())


def test_rff_polars_input_accepted():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5)).astype(np.float32)
    df_in = pl.DataFrame({f"f{i}": X[:, i] for i in range(5)})
    out = compute_rff_features(df_in, seed=3, n_features=16, use_gpu=False)
    assert out.shape == (30, 16)


def test_rff_raises_on_nan():
    X = np.array([[1.0, 2.0], [np.nan, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        compute_rff_features(X, seed=0, n_features=8, use_gpu=False)


def test_rff_raises_on_1d_input():
    X = np.zeros(50, dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        compute_rff_features(X, seed=0, n_features=8, use_gpu=False)


def test_rff_raises_on_non_numeric():
    X = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(TypeError, match="numeric"):
        compute_rff_features(X, seed=0, n_features=8, use_gpu=False)


def test_rff_raises_on_odd_n_features():
    X = np.zeros((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="even"):
        compute_rff_features(X, seed=0, n_features=15, use_gpu=False)


def test_rff_raises_on_missing_seed():
    X = np.zeros((10, 4), dtype=np.float32)
    # Python's "missing 1 required keyword-only argument: 'seed'" fires before our require_seed; both messages contain "seed".
    with pytest.raises(TypeError, match="seed"):
        compute_rff_features(X, n_features=8, use_gpu=False)  # type: ignore[call-arg]


def test_rff_explicit_sigma_float():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    df_med = compute_rff_features(X, seed=0, n_features=16, sigma="median", use_gpu=False)
    df_explicit = compute_rff_features(X, seed=0, n_features=16, sigma=0.5, use_gpu=False)
    # Different sigma -> different output.
    assert not np.allclose(df_med.to_numpy(), df_explicit.to_numpy())


def test_rff_no_standardize_path():
    rng = np.random.default_rng(0)
    X = (rng.standard_normal((50, 4)) * 100.0).astype(np.float32)  # large-scale input
    df1 = compute_rff_features(X, seed=0, n_features=16, standardize=False, use_gpu=False)
    df2 = compute_rff_features(X, seed=0, n_features=16, standardize=True, use_gpu=False)
    assert df1.shape == df2.shape == (50, 16)
    # With vs without standardisation should give different output unless input is already centred / scaled.
    assert not np.allclose(df1.to_numpy(), df2.to_numpy())


def test_rff_outputs_unit_kernel_features():
    # The RFF approximation: |phi(x_i) . phi(x_j)|^2 estimates the RBF kernel between x_i and x_j with variance dropping in 1/n_features.
    # Sanity check: for two close points, output rows should have higher dot product than for two distant points.
    X = np.array(
        [
            [0.0, 0.0],
            [0.01, 0.01],  # close to row 0
            [5.0, 5.0],  # far
        ],
        dtype=np.float32,
    )
    df = compute_rff_features(X, seed=0, n_features=2048, sigma=1.0, standardize=False, use_gpu=False)
    arr = df.to_numpy()
    close_dot = float(arr[0] @ arr[1])
    far_dot = float(arr[0] @ arr[2])
    assert close_dot > far_dot, f"Close points dot ({close_dot:.4f}) should exceed far points dot ({far_dot:.4f})"


# ---------- compute_positional_encoding ----------


def test_pe_shape_and_dtype():
    pos = np.arange(50)
    df = compute_positional_encoding(pos, d_model=16)
    assert df.shape == (50, 16)
    assert all(df.dtypes[i] == pl.Float32 for i in range(16))
    assert "pe_0" in df.columns
    assert "pe_15" in df.columns


def test_pe_deterministic():
    pos = np.arange(20)
    df1 = compute_positional_encoding(pos, d_model=8)
    df2 = compute_positional_encoding(pos, d_model=8)
    np.testing.assert_array_equal(df1.to_numpy(), df2.to_numpy())


def test_pe_first_pair_is_sin_cos():
    # PE(pos, 0) = sin(pos / base^0) = sin(pos); PE(pos, 1) = cos(pos / base^0) = cos(pos).
    pos = np.array([0, 1, 2, 3], dtype=np.int64)
    df = compute_positional_encoding(pos, d_model=4, base=10_000.0)
    arr = df.to_numpy()
    np.testing.assert_allclose(arr[:, 0], np.sin(pos), atol=1e-6)
    np.testing.assert_allclose(arr[:, 1], np.cos(pos), atol=1e-6)


def test_pe_fused_kernel_matches_numpy_reference():
    # Fused positional_encoding_njit must match the original three-array numpy path (angles temporary + separate sin/cos into strided views) within a single
    # float32 ULP, and preserve the interleaved [sin_j @ col 2j, cos_j @ col 2j+1] layout, across several d_model. A regression that bypasses the kernel or
    # transposes the sin/cos columns trips this.
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 500_000, size=5000).astype(np.int64)
    base = 10_000.0
    for d_model in (4, 16, 32):
        got = compute_positional_encoding(pos, d_model=d_model, base=base).to_numpy()
        # Reference: exact pre-fusion numpy computation in float32.
        pos_f = np.fmod(pos.astype(np.float32), np.float32(1_000_000.0))
        half = d_model // 2
        div = (1.0 / np.power(base, 2.0 * np.arange(half, dtype=np.float64) / d_model)).astype(np.float32)
        ang = pos_f[:, None] * div[None, :]
        ref = np.empty((pos_f.size, d_model), dtype=np.float32)
        ref[:, 0::2] = np.sin(ang)
        ref[:, 1::2] = np.cos(ang)
        np.testing.assert_allclose(got, ref, atol=1e-6, rtol=0, err_msg=f"d_model={d_model}")


def test_pe_raises_on_odd_d_model():
    with pytest.raises(ValueError, match="even"):
        compute_positional_encoding(np.arange(10), d_model=7)


def test_pe_raises_on_bad_base():
    with pytest.raises(ValueError, match="base"):
        compute_positional_encoding(np.arange(10), d_model=4, base=1.0)


def test_pe_accepts_polars_series():
    s = pl.Series("pos", list(range(20)))
    df = compute_positional_encoding(s, d_model=8)
    assert df.shape == (20, 8)


def test_pe_clamps_huge_positions():
    # 2e6 % 1e6 = 0; PE at clamped 0 is all-zero sins, all-one cosines.
    pos = np.array([2_000_000, 3_000_000], dtype=np.int64)
    df = compute_positional_encoding(pos, d_model=4, base=10_000.0)
    arr = df.to_numpy()
    # Both rows clamp to 0; sin(0) = 0, cos(0) = 1.
    np.testing.assert_allclose(arr[:, 0::2], np.zeros((2, 2)), atol=1e-6)
    np.testing.assert_allclose(arr[:, 1::2], np.ones((2, 2)), atol=1e-6)


# ---------- positions_within_group ----------


def test_positions_within_group_simple_unsorted():
    df = pl.DataFrame({"g": ["a", "a", "b", "a", "b"], "ts": [3, 1, 2, 2, 1]})
    pos = positions_within_group(df, group_col="g")
    # Without sort_col: ordinal in arrival order within each group.
    # Group 'a' rows at indices 0, 1, 3 -> ordinals 0, 1, 2; group 'b' at 2, 4 -> ordinals 0, 1.
    np.testing.assert_array_equal(pos.to_numpy(), np.array([0, 1, 0, 2, 1]))


def test_positions_within_group_with_sort_col():
    df = pl.DataFrame({"g": ["a", "a", "b", "a", "b"], "ts": [3, 1, 2, 2, 1]})
    pos = positions_within_group(df, group_col="g", sort_col="ts")
    # Sorted within g by ts: g='a' ts=[1,2,3] at original indices [1,3,0] -> ordinals [2,0,1] respectively at original rows;
    # g='b' ts=[1,2] at original indices [4,2] -> ordinals [1,0].
    np.testing.assert_array_equal(pos.to_numpy(), np.array([2, 0, 1, 1, 0]))


def test_positions_within_group_raises_on_missing_group_col():
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError, match="group_col"):
        positions_within_group(df, group_col="missing")


def test_positions_within_group_raises_on_missing_sort_col():
    df = pl.DataFrame({"g": ["a", "b"], "x": [1, 2]})
    with pytest.raises(KeyError, match="sort_col"):
        positions_within_group(df, group_col="g", sort_col="missing")
