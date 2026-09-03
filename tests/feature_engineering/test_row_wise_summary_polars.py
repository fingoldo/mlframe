"""The polars-native per-row summary must agree with the numpy reference, value for value.

A production run spent 11.3s here after a polars->pandas conversion it did not otherwise need. A naive polars
rewrite is much SLOWER than the numpy path (measured: 0.09x with ``concat_list`` + ``list.eval``), so the point
of this module is the specific construct -- horizontal reductions where possible, one sorted fixed-width array
where an order statistic is unavoidable -- and its exactness is what makes it substitutable at all.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from mlframe.feature_engineering.row_wise_summary import row_wise_summary_stats
from mlframe.feature_engineering.row_wise_summary_polars import row_wise_summary_stats_polars

STATS = ["mean", "std", "min", "max", "median", "q10", "q90"]


def _frame(n_rows: int = 3000, n_cols: int = 12, nan_frac: float = 0.08, seed: int = 0) -> pl.DataFrame:
    """A float frame with NaN density high enough that the nan-handling paths matter."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_rows, n_cols))
    data[rng.random(data.shape) < nan_frac] = np.nan
    return pl.DataFrame({f"f{i}": data[:, i] for i in range(n_cols)})


class TestItMatchesTheNumpyReference:
    """Substitutability is the whole contract; a faster answer that differs is not an answer."""

    @pytest.mark.parametrize("stat", STATS)
    def test_each_stat_matches(self, stat):
        """Per stat, so a failure names which one drifted."""
        df = _frame()
        expected = row_wise_summary_stats(df.to_pandas(), stats=[stat])[f"row_summary_{stat}"].to_numpy()
        got = row_wise_summary_stats_polars(df, stats=[stat])[f"row_summary_{stat}"].to_numpy()
        assert np.array_equal(np.isnan(expected), np.isnan(got)), f"{stat}: NaN placement differs"
        assert np.nanmax(np.abs(expected - got)) < 1e-12

    def test_std_uses_the_reference_ddof(self):
        """polars defaults to ddof=1 and numpy's nanstd to ddof=0 -- a silent third-decimal drift if unfixed."""
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 6.0], "c": [5.0, 10.0]})
        got = row_wise_summary_stats_polars(df, stats=["std"])["row_summary_std"].to_numpy()
        assert np.allclose(got, [np.nanstd([1, 3, 5]), np.nanstd([2, 6, 10])])

    def test_quantiles_interpolate_like_numpy(self):
        """Nearest-rank would be a different definition; the reference interpolates and so must this."""
        df = pl.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]})
        got = row_wise_summary_stats_polars(df, stats=["q10", "q90"])
        assert np.isclose(got["row_summary_q10"][0], np.nanquantile([1, 2, 3, 4], 0.10))
        assert np.isclose(got["row_summary_q90"][0], np.nanquantile([1, 2, 3, 4], 0.90))

    def test_per_row_null_counts_are_respected(self):
        """Each row's quantile position is over ITS OWN non-null count, not the column width."""
        df = pl.DataFrame({"a": [1.0, 1.0], "b": [2.0, None], "c": [3.0, None], "d": [4.0, 9.0]})
        got = row_wise_summary_stats_polars(df, stats=["median"])["row_summary_median"].to_numpy()
        assert np.isclose(got[0], np.nanmedian([1, 2, 3, 4]))
        assert np.isclose(got[1], np.nanmedian([1, 9]))


class TestSemanticsAndShape:
    """The edges where polars and numpy disagree unless told not to."""

    def test_nan_is_treated_as_missing_not_as_a_value(self):
        """polars treats NaN as an ordinary float; the reference skips it. Without the conversion they diverge."""
        df = pl.DataFrame({"a": [1.0, np.nan], "b": [3.0, 5.0]})
        got = row_wise_summary_stats_polars(df, stats=["mean", "max"])
        assert np.isclose(got["row_summary_mean"][1], 5.0)
        assert np.isclose(got["row_summary_max"][1], 5.0)

    def test_all_null_row_yields_null(self):
        """Nothing to summarise is not zero."""
        df = pl.DataFrame({"a": [None, 1.0], "b": [None, 3.0]}, schema={"a": pl.Float64, "b": pl.Float64})
        got = row_wise_summary_stats_polars(df, stats=["mean"])["row_summary_mean"].to_list()
        assert got[0] is None and np.isclose(got[1], 2.0)

    def test_row_count_and_order_are_preserved(self):
        """Output is joined back positionally, so a reorder would silently mislabel every row."""
        df = _frame(n_rows=500)
        out = row_wise_summary_stats_polars(df, stats=["mean"])
        assert out.height == df.height
        assert np.allclose(
            out["row_summary_mean"].to_numpy()[:5],
            row_wise_summary_stats(df.to_pandas(), stats=["mean"])["row_summary_mean"].to_numpy()[:5],
            equal_nan=True,
        )

    def test_grouped_mode_names_and_splits_correctly(self):
        """Per-family stats must not blur families together, and must be named per group."""
        df = _frame(n_cols=6)
        out = row_wise_summary_stats_polars(df, stats=["mean"], groups={"lo": ["f0", "f1"], "hi": ["f4", "f5"]})
        assert set(out.columns) == {"row_summary_lo_mean", "row_summary_hi_mean"}
        expected = row_wise_summary_stats(df.to_pandas(), stats=["mean"], groups={"lo": ["f0", "f1"]})
        assert np.allclose(out["row_summary_lo_mean"].to_numpy(), expected["row_summary_lo_mean"].to_numpy(), equal_nan=True)

    def test_unknown_stat_raises(self):
        """A typo must fail loudly rather than silently emitting fewer columns than asked for."""
        with pytest.raises(ValueError, match="unrecognized stat"):
            row_wise_summary_stats_polars(_frame(n_rows=10), stats=["nonsense"])

    def test_numeric_columns_are_selected_by_default(self):
        """A string column must not be dragged into a numeric reduction."""
        df = _frame(n_rows=50, n_cols=3).with_columns(pl.lit("x").alias("label"))
        out = row_wise_summary_stats_polars(df, stats=["mean"])
        assert out.height == 50
