"""Integer columns with values above 2^24 keep their distinct values through the float cast of ``categorize_dataset`` (five epoch timestamps one second apart used to become one code)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization import categorize_dataset
from mlframe.feature_selection.filters.discretization import _discretization_dataset as D

pl = pytest.importorskip("polars")


def _epoch_frame(n=4000, seed=0):
    """Five timestamps one second apart around 1.6e9 that determine the target, plus a noise column."""
    rng = np.random.default_rng(seed)
    ts = 1_600_000_000 + rng.integers(0, 5, n)
    return pd.DataFrame({"ts": ts.astype(np.int64), "z": rng.standard_normal(n)}), (ts % 2 == 0).astype(int)


@pytest.mark.parametrize("make", [lambda d: d, pl.from_pandas], ids=["pandas", "polars"])
def test_supervised_binning_keeps_every_timestamp(make):
    """MDLP sees the five values and keeps five codes."""
    X, y = _epoch_frame()
    data, cols, _nb = categorize_dataset(make(X), nbins_strategy="mdlp", y_for_strategy=y, dtype=np.int32)
    assert len(np.unique(data[:, cols.index("ts")])) == 5


def test_codes_equal_those_of_the_shifted_column():
    """Shifting by the minimum is exact: the codes equal those of the same column written as 0..4."""
    X, y = _epoch_frame()
    small = X.assign(ts=X["ts"] - X["ts"].min())
    a, ca, _ = categorize_dataset(X, nbins_strategy="mdlp", y_for_strategy=y, dtype=np.int32)
    b, cb, _ = categorize_dataset(small, nbins_strategy="mdlp", y_for_strategy=y, dtype=np.int32)
    np.testing.assert_array_equal(a[:, ca.index("ts")], b[:, cb.index("ts")])


def test_range_wider_than_float32_switches_to_float64():
    """A column whose range exceeds 2^24 cannot be shifted into float32: the block is binned in float64."""
    W = pd.DataFrame({"w": np.array([0, 2**40, 2**41, 3, 5] * 200, dtype=np.int64)})
    _df, dt = D._exact_integer_columns(W, ["w"], np.float32, False)
    assert np.dtype(dt) == np.float64


def test_frames_without_large_integers_are_untouched():
    """Small integers and floats come back as the same frame and dtype."""
    X = pd.DataFrame({"s": np.arange(100), "f": np.linspace(0, 1, 100)})
    df, dt = D._exact_integer_columns(X, ["s", "f"], np.float32, False)
    assert df is X and dt == np.float32


def test_integral_float64_column_keeps_its_values():
    """A float64 column holding integers above 2^24 (a densified nullable Int64) is shifted like an integer column; MDLP keeps five codes."""
    X, y = _epoch_frame()
    data, cols, _nb = categorize_dataset(X.assign(ts=X["ts"].astype(np.float64)), nbins_strategy="mdlp", y_for_strategy=y, dtype=np.int32)
    assert len(np.unique(data[:, cols.index("ts")])) == 5


def test_nullable_int64_through_mrmr_densify_keeps_its_values():
    """The fit densifies a nullable Int64 above 2^24 to float64, not float32."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._inputs import _densify_dtype

    X, _y = _epoch_frame()
    col = pd.Series(pd.array(X["ts"].to_numpy(), dtype="Int64"))
    col[0] = pd.NA
    assert np.dtype(_densify_dtype(col)) == np.float64
    assert col.astype(_densify_dtype(col)).dropna().nunique() == 5


def test_fractional_large_floats_are_left_to_the_float32_relaxation():
    """Non-integral large floats are not shifted (their float32 rounding is the accepted relaxation)."""
    X = pd.DataFrame({"f": 1.6e9 + np.linspace(0, 1, 50) + 0.5})
    df, dt = D._exact_integer_columns(X, ["f"], np.float32, False)
    assert df is X and dt == np.float32
