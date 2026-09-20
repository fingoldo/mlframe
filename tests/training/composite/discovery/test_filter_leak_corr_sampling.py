"""The leak-corr test must pick its rows before the feature columns are gathered, not after.

``_filter_features`` held every numeric column over every train row, stacked a second full copy, and only then asked
whether the allocation was too large: about 14 GB on a 3.2M x 500 frame, for a test that compares ``|corr(x, y)``| to
0.99999. The rows are now chosen up front, so only the sampled block is ever held; the constancy and finite-row checks
still read the whole column, and the drop list is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import _filter as filter_mod
from mlframe.training.composite.discovery._filter import _filter_features, _leak_corr_sample_rows
from mlframe.training.configs import CompositeTargetDiscoveryConfig


class _Disc:
    """The attribute surface ``_filter_features`` reads off the discovery instance."""

    def __init__(self):
        self.config = CompositeTargetDiscoveryConfig(enabled=True, random_state=0)
        self._target_col = "y"
        self._patterns_compiled = []
        self._filter_drops = []


def _frame(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """A frame carrying one exact copy of y, one near-copy, a constant column and ordinary features."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    df = pd.DataFrame({
        "y": y,
        "y_copy": y.copy(),
        "y_near": y + rng.normal(0.0, 1e-9, n),
        "constant": np.full(n, 3.0),
        "f1": rng.normal(size=n),
        "f2": y * 0.3 + rng.normal(size=n),
    })
    return df, y


def test_below_the_minimum_every_row_is_read():
    """Small frames keep the exact full-row correlation: there is nothing to save."""
    assert _leak_corr_sample_rows(1000) is None
    assert _leak_corr_sample_rows(500_000) is None


def test_above_the_minimum_the_rows_are_strided_and_bounded():
    """A large frame reads a stride, and the stride keeps at least the minimum sample."""
    rows = _leak_corr_sample_rows(3_200_000)
    assert rows is not None
    assert rows.size >= 500_000, f"the sample must stay above the precision floor; got {rows.size}"
    assert rows.size < 3_200_000
    assert np.all(np.diff(rows) == rows[1] - rows[0]), "the sample must be a plain stride, so it is reproducible"


def test_the_drop_list_is_unchanged_when_sampling_kicks_in(monkeypatch):
    """Sampling is a memory change: the exact copy and the y-derived column are still dropped, f1 still kept."""
    df, y = _frame()
    idx = np.arange(len(df))
    full = _filter_features(_Disc(), df, list(df.columns), y, idx)

    monkeypatch.setattr(filter_mod, "_LEAK_CORR_MIN_SAMPLE_ROWS", 1000)
    disc = _Disc()
    sampled = _filter_features(disc, df, list(df.columns), y, idx)
    assert sampled == full, f"sampling changed the survivor list: {sampled} vs {full}"
    assert "y_copy" not in sampled and "y_near" not in sampled, "a y copy must still be caught as leakage"
    assert "f1" in sampled and "f2" in sampled


def test_the_constancy_check_still_reads_every_row(monkeypatch):
    """A column constant everywhere except a few rows must still be recognised as varying, whatever the stride."""
    n = 4000
    rng = np.random.default_rng(1)
    col = np.full(n, 2.0)
    col[1] = 50.0  # a row a coarse stride would skip
    df = pd.DataFrame({"y": rng.normal(size=n), "sparse_variation": col, "f1": rng.normal(size=n)})
    monkeypatch.setattr(filter_mod, "_LEAK_CORR_MIN_SAMPLE_ROWS", 100)
    disc = _Disc()
    kept = _filter_features(disc, df, list(df.columns), df["y"].to_numpy(), np.arange(n))
    assert "sparse_variation" in kept, "the constancy check must read the whole column, not the leak-corr sample"


def test_only_the_sampled_rows_are_held_per_column(monkeypatch):
    """The gathered block is the sample, which is what bounds the peak allocation."""
    df, y = _frame(n=4000)
    monkeypatch.setattr(filter_mod, "_LEAK_CORR_MIN_SAMPLE_ROWS", 500)
    held: list[int] = []
    original = filter_mod._maybe_sample_for_leak_corr

    def spy(candidates, candidate_arrays, y_train):
        """Record the row count of the gathered columns, then defer."""
        held.append(candidate_arrays[0].shape[0] if candidate_arrays else 0)
        return original(candidates, candidate_arrays, y_train)

    monkeypatch.setattr(filter_mod, "_maybe_sample_for_leak_corr", spy)
    _filter_features(_Disc(), df, list(df.columns), y, np.arange(len(df)))
    assert held and held[0] <= 1000, f"the per-column block must be the sample, not all 4000 rows; got {held}"
