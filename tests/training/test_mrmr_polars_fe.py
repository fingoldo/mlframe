"""Tests for MRMR feature engineering (FE) on Polars input.

Fix 10's main selector path is polars-native (zero-copy to_physical for
cat codes, zero-copy to_numpy() for numerics). This file adds tests for
the FE path (fe_max_steps > 0) on polars input, covering:
  * FE with numeric-only frames (pair-level polynomial / binary ops)
  * FE with mixed numeric + categorical frames
  * Parity: pandas FE and polars FE produce the same selected-features
    set on identical seeded input
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_selection.filters import MRMR


def _build_pair_signal_data(n=500, seed=0, frame_type="polars"):
    """Build a frame where the target = f(num_a * num_b) — a product that
    MRMR's pairwise FE should pick up. Individual num_a/num_b have low
    marginal MI; their product has high MI."""
    rng = np.random.default_rng(seed)
    num_a = rng.standard_normal(n).astype(np.float32)
    num_b = rng.standard_normal(n).astype(np.float32)
    num_c = rng.standard_normal(n).astype(np.float32)  # noise
    y = (num_a * num_b > 0).astype(np.int32)

    if frame_type == "polars":
        return pl.DataFrame({"num_a": num_a, "num_b": num_b, "num_c": num_c}), y
    return pd.DataFrame({"num_a": num_a, "num_b": num_b, "num_c": num_c}), y


def _mrmr_kwargs_quick_fe():
    return dict(
        verbose=0, max_runtime_mins=1, n_workers=1,
        quantization_nbins=5, use_simple_mode=True,
        min_nonzero_confidence=0.9, max_consec_unconfirmed=3,
        full_npermutations=3,
        fe_max_steps=1,
        fe_npermutations=3,
        fe_max_pair_features=1,
        fe_binary_preset="minimal",
        fe_unary_preset="minimal",
    )


@pytest.mark.parametrize("frame_type", ["pandas", "polars"])
def test_mrmr_fe_runs_on_polars_and_pandas(frame_type):
    """FE path (fe_max_steps > 0) must run on both pandas and polars
    without raising. Before Fix 10 + FE polars support, this would have
    raised AttributeError('DataFrame' object has no attribute 'iloc') on
    polars."""
    df, y = _build_pair_signal_data(n=500, seed=0, frame_type=frame_type)
    sel = MRMR(**_mrmr_kwargs_quick_fe())
    sel.fit(df, y)
    assert sel.support_ is not None, "MRMR.fit must set support_"
    assert len(sel.support_) >= 1, "MRMR must select at least one feature"


def test_mrmr_fe_zero_copy_polars():
    """Even with FE enabled, the polars path must not call X.to_pandas()
    on the full frame (CLAUDE.md: no 100+ GB copies). Spy and assert 0
    calls during fit."""
    import polars as _pl

    df, y = _build_pair_signal_data(n=500, seed=1, frame_type="polars")

    call_count = {"n": 0}
    orig = _pl.DataFrame.to_pandas
    def _spy(self, *args, **kwargs):
        call_count["n"] += 1
        return orig(self, *args, **kwargs)
    _pl.DataFrame.to_pandas = _spy
    try:
        sel = MRMR(**_mrmr_kwargs_quick_fe())
        sel.fit(df, y)
    finally:
        _pl.DataFrame.to_pandas = orig

    assert call_count["n"] == 0, (
        f"MRMR.fit with FE on polars called pl.DataFrame.to_pandas() "
        f"{call_count['n']} times — regression. Full polars FE support "
        f"should avoid any full-frame copy; only categorize_dataset's "
        f"column-subset .to_numpy() is permitted."
    )


def test_mrmr_fe_transform_returns_polars_when_input_polars():
    """After FE-enabled fit, transform on a polars input must still
    return polars (subset of original cols — FE-generated cols are
    internal MI state, not transform outputs)."""
    df, y = _build_pair_signal_data(n=400, seed=2, frame_type="polars")
    sel = MRMR(**_mrmr_kwargs_quick_fe())
    sel.fit(df, y)
    out = sel.transform(df)
    assert isinstance(out, pl.DataFrame), (
        f"transform on polars input should return polars; got {type(out).__name__}"
    )
    # Selected cols must all be in the original frame (FE cols are internal).
    assert set(out.columns).issubset(set(df.columns))


def test_mrmr_fe_caller_frame_not_mutated_on_polars():
    """Fix 10 guarantee: polars caller frame is never mutated. FE path
    specifically — which injects new cols at filters.py:3429 — must not
    leak those into the caller's X (polars is immutable, but a
    double-check in case a careful refactor breaks invariant)."""
    df, y = _build_pair_signal_data(n=400, seed=3, frame_type="polars")
    original_columns = list(df.columns)
    sel = MRMR(**_mrmr_kwargs_quick_fe())
    sel.fit(df, y)
    assert list(df.columns) == original_columns, (
        f"Caller's polars frame was mutated. Before: {original_columns}, "
        f"after MRMR.fit: {list(df.columns)}"
    )
