"""Surviving pair features are returned as a list whose column index points at the dense survivor buffer."""

from __future__ import annotations

import types

import numpy as np

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit import _emit_pair_features_step3_pair_features_part1


def _run(configs):
    """Run the survivor packing for ``configs`` (each ``(pair, bin_name, i)``) over a 3-column candidate buffer."""
    rng = np.random.default_rng(0)
    n = 60
    buffer = rng.normal(size=(n, 3))
    st = types.SimpleNamespace()
    out = _emit_pair_features_step3_pair_features_part1(
        {(c, 0) for c in configs},  # multi-emit extras are recorded with j=0
        lambda cfg: f"feat_{cfg[2]}",
        1, False, None, None, None, {}, {}, {}, {}, None, None, buffer, False, None, {}, st, None, None, False, [], n, 8,
    )
    return out, st, buffer


def test_survivors_are_a_list_with_dense_column_indices_in_name_order():
    """Three forms from one pair: the returned list has j = 0, 1, 2 in name order, matching new_cols and the stacked buffer columns."""
    configs = [(("a", "b"), "mul", 2), (("a", "b"), "div", 0), (("a", "b"), "add", 1)]
    out, st, buffer = _run(configs)
    assert isinstance(out, list) and [j for _c, j in out] == [0, 1, 2]
    assert st.new_cols == ["feat_0", "feat_1", "feat_2"]
    for cfg, j in out:
        np.testing.assert_allclose(st.transformed_vals[:, j], buffer[:, cfg[2]])
