"""Regression guards for the cluster-hierarchy fixes (audit 2026-06-03):

- hierarchy-stability-3: the ndarray quantization branch must filter non-finite
  values before np.quantile (mirroring the DataFrame branch). A NaN-bearing
  column otherwise yields NaN bin edges -> a single bin -> SU=0 -> the anchor
  silently never merges.
- hierarchy-stability-12: the super-anchor (cluster label) is the medoid
  (most-central anchor by mean within-component SU), not the lexicographically
  smallest name.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._cluster_hierarchy import (
    _component_medoid,
    _quantize_for_su,
)


class TestQuantizeNanHandling:
    def test_ndarray_nan_column_still_binned(self):
        rng = np.random.default_rng(0)
        good = rng.standard_normal(300)
        nan_col = good.copy()
        nan_col[:5] = np.nan
        X = np.column_stack([good, nan_col])
        fd, fn, names = _quantize_for_su(X)
        assert fd is not None
        # The NaN-bearing column must still receive >1 bin (finite values get
        # quantized), matching the DataFrame path. Pre-fix: NaN edges -> 1 bin.
        assert int(fn[1]) > 1, f"NaN column collapsed to {int(fn[1])} bin(s)"

    def test_ndarray_matches_dataframe_for_clean_data(self):
        import pandas as pd

        rng = np.random.default_rng(1)
        arr = rng.standard_normal((300, 3))
        fd_a, fn_a, _ = _quantize_for_su(arr)
        fd_d, fn_d, _ = _quantize_for_su(pd.DataFrame(arr, columns=["a", "b", "c"]))
        assert list(fn_a) == list(fn_d)
        assert np.array_equal(fd_a, fd_d)


class TestComponentMedoid:
    def test_picks_central_anchor(self):
        # b is central: high SU to both a and c; a-c link is weak.
        comp = ["a", "b", "c"]
        pair_sus = {("a", "b"): 0.9, ("b", "c"): 0.9, ("a", "c"): 0.1}
        assert _component_medoid(comp, pair_sus) == "b"

    def test_handles_reversed_key_order(self):
        # pair_sus keyed in original anchor order; medoid must look up both ways.
        comp = ["c", "b", "a"]
        pair_sus = {("b", "c"): 0.9, ("a", "b"): 0.9, ("a", "c"): 0.1}
        assert _component_medoid(comp, pair_sus) == "b"

    def test_two_member_alphabetical(self):
        assert _component_medoid(["y", "x"], {}) == "x"
