"""When the row-wise summary is the only extension stage, compute it in polars and skip the pandas bridge.

A production run converted a 2.18M x 113 polars frame to pandas and spent 11.3s reducing it with numpy, for a
stage that needs no sklearn estimator at all. Natively in polars the same call is 5.6x faster at 400k x 85 and
6.0x at 2M -- with every order statistic bit-identical -- because the streaming engine reduces morsels instead
of materialising one float64 matrix of the whole frame.

The fastpath must decline for every configuration that genuinely needs pandas, or it would silently skip a
transform the caller asked for.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from mlframe.training.pipeline._pipeline_extensions import (
    _only_row_wise_summary_requested,
    _row_wise_summary_polars_fastpath,
)


def _config(**overrides):
    """An extensions config with every stage off except the row-wise summary."""
    base = dict(
        row_wise_summary_stats_enabled=True,
        row_wise_summary_stats_list=None,
        row_wise_extreme_columns_enabled=False,
        pysr_enabled=False,
        tfidf_columns=None,
        scaler=None,
        binarization_threshold=None,
        kbins=None,
        polynomial_degree=None,
        nonlinear_features=None,
        dim_reducer=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _frames(n: int = 400, p: int = 6):
    """Three polars splits sharing a schema."""
    rng = np.random.default_rng(0)

    def one(rows):
        """One split with a little NaN density."""
        data = rng.normal(size=(rows, p))
        data[rng.random(data.shape) < 0.05] = np.nan
        return pl.DataFrame({f"f{i}": data[:, i] for i in range(p)})

    return one(n), one(n // 4), one(n // 4)


class TestWhenItTakesOver:
    """It may only take over when nothing else in the block needs pandas."""

    def test_only_summary_requested(self):
        """The configuration from the production run."""
        assert _only_row_wise_summary_requested(_config()) is True

    @pytest.mark.parametrize(
        "override",
        [
            {"row_wise_extreme_columns_enabled": True},
            {"pysr_enabled": True},
            {"tfidf_columns": ["txt"]},
            {"scaler": "standard"},
            {"kbins": 5},
            {"polynomial_degree": 2},
            {"dim_reducer": "pca"},
            {"nonlinear_features": ["x"]},
            {"binarization_threshold": 0.5},
        ],
    )
    def test_any_other_stage_declines(self, override):
        """Each of these needs the pandas bridge, so the fastpath must not swallow the call."""
        assert _only_row_wise_summary_requested(_config(**override)) is False

    def test_summary_disabled_declines(self):
        """Nothing to compute."""
        assert _only_row_wise_summary_requested(_config(row_wise_summary_stats_enabled=False)) is False


class TestTheFastpath:
    """What it returns, and that it declines rather than guessing."""

    def test_columns_are_added_to_every_split(self):
        """A partial application would schema-drift the fitted pipeline against whichever split missed out."""
        train, val, test = _frames()
        t, v, s, handled = _row_wise_summary_polars_fastpath(train, val, test, _config(), verbose=0)
        assert handled is True
        added = set(t.columns) - set(train.columns)
        assert added, "no summary columns were added"
        assert added == set(v.columns) - set(val.columns) == set(s.columns) - set(test.columns)

    def test_row_counts_and_frame_type_are_preserved(self):
        """The frames stay polars -- taking the fastpath and then converting anyway would defeat it."""
        train, val, test = _frames()
        t, v, s, _ = _row_wise_summary_polars_fastpath(train, val, test, _config(), verbose=0)
        assert isinstance(t, pl.DataFrame) and t.height == train.height
        assert v.height == val.height and s.height == test.height

    def test_values_match_the_numpy_reference(self):
        """Substitutability is the contract; a faster answer that differs is not an answer."""
        from mlframe.feature_engineering.row_wise_summary import row_wise_summary_stats

        train, val, test = _frames()
        t, _v, _s, _ = _row_wise_summary_polars_fastpath(train, val, test, _config(), verbose=0)
        expected = row_wise_summary_stats(train.to_pandas())
        for col in expected.columns:
            assert np.allclose(t[col].to_numpy(), expected[col].to_numpy(), equal_nan=True, atol=1e-12)

    def test_pandas_input_declines(self):
        """The fastpath is polars-only; a pandas caller must take the original route."""
        train, val, test = _frames()
        out = _row_wise_summary_polars_fastpath(train.to_pandas(), val, test, _config(), verbose=0)
        assert out[3] is False

    def test_no_numeric_columns_declines(self):
        """Nothing to summarise means nothing to hand back as handled."""
        empty = pl.DataFrame({"txt": ["a", "b"]})
        assert _row_wise_summary_polars_fastpath(empty, empty, empty, _config(), verbose=0)[3] is False

    def test_a_missing_split_is_tolerated(self):
        """val/test are optional at this point in the suite."""
        train, _val, _test = _frames()
        t, v, s, handled = _row_wise_summary_polars_fastpath(train, None, None, _config(), verbose=0)
        assert handled is True and v is None and s is None and t.width > train.width

    def test_configured_stat_list_is_honoured(self):
        """A caller narrowing the stat list must not silently get the default set."""
        train, val, test = _frames()
        t, _v, _s, _ = _row_wise_summary_polars_fastpath(train, val, test, _config(row_wise_summary_stats_list=["mean"]), verbose=0)
        added = [c for c in t.columns if c.startswith("row_summary_")]
        assert added == ["row_summary_mean"]
