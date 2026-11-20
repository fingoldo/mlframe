"""The oof_cell2d recipe keeps its fitted edges in float64 so replay bins every value exactly as the fit did."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._oof_cell2d_fe import _cells, apply_oof_cell2d_recipe, build_oof_cell2d_recipe, fit_oof_cell2d


def test_oof_cell2d_replay_bins_values_against_the_float64_fit_edges():
    """A float32 rounding of an edge that lies half an ulp above a data value used to move that value into the next bin at replay."""
    n = 1000
    one = np.float32(1.0)
    a = np.where(np.arange(n) < n // 2, one, np.nextafter(one, np.float32(2.0))).astype(np.float32)
    rng = np.random.default_rng(3)
    b = rng.random(n).astype(np.float32)
    y = rng.random(n)
    fit = fit_oof_cell2d(a, b, y)
    rec = build_oof_cell2d_recipe(name="oofcell(a,b)", src=("a", "b"), ea=fit["ea"], eb=fit["eb"], tab=fit["tab"], fill=fit["fill"], lo=-1.0, hi=2.0)
    got = apply_oof_cell2d_recipe(rec, pd.DataFrame({"a": a, "b": b}))
    cell, _ = _cells(a, b, fit["ea"], fit["eb"])
    np.testing.assert_allclose(got, fit["tab"].reshape(-1)[cell], atol=1e-6)
