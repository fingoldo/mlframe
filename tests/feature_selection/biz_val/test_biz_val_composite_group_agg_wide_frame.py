"""Business value of the composite group aggregates on a wide frame: with auto-detected sources the planted per-cell effect is found among many noise columns.

The family's own tests name one numeric column, so the conditioning support was a single column. In a real frame the sources are auto-detected: with every numeric column in the
support the joint class puts each row in a cell of its own and no candidate could score above 0 (the defect this test pins)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _composite_group_agg_fe as cga


def _wide_frame(n: int = 6000, n_noise: int = 24, seed: int = 11) -> "tuple[pd.DataFrame, np.ndarray]":
    """Two group columns, one noisy source whose per-cell mean depends on the (g0, g1) cell, ``n_noise`` independent noise columns, and a target driven by that cell level."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"g0": rng.integers(0, 5, n), "g1": rng.integers(0, 4, n)})
    offset = 1.5 * rng.standard_normal((5, 4))
    cell = offset[X["g0"].to_numpy(), X["g1"].to_numpy()]
    X["x_sig"] = (cell + 4.0 * rng.standard_normal(n)).astype(np.float32)  # a noisy source: only its per-cell mean carries the target
    for j in range(n_noise):
        X[f"noise{j}"] = rng.standard_normal(n).astype(np.float32)
    y = (cell + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def test_wide_frame_finds_the_planted_cell_effect():
    """Auto-detected sources on 25 numeric columns: the aggregates of the signal source are selected, none of a noise column."""
    X, y = _wide_frame()
    _x_aug, appended, recipes, scores = cga.hybrid_composite_group_agg_fe(X, y, group_col_sets=[("g0", "g1")])
    assert appended, "no composite aggregate selected on a frame with a planted per-cell effect"
    assert all("x_sig" in name for name in appended), appended
    assert len(recipes) == len(appended) and float(scores["cmi"].max()) > 1e-3


def test_named_sources_are_trusted_as_given():
    """A caller that names its columns gets those columns scored (the relevance filter is for auto-detection only)."""
    X, y = _wide_frame(n_noise=3)
    _x_aug, _appended, _recipes, scores = cga.hybrid_composite_group_agg_fe(X, y, group_col_sets=[("g0", "g1")], num_cols=["noise0", "noise1"])
    assert set(scores["source_col"]) == {"noise0", "noise1"}
