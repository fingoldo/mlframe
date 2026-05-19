"""Tests for ``mlframe.training.quantile_postproc.fix_quantile_crossing``."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.quantile_postproc import fix_quantile_crossing


class TestFixCrossing:
    def test_sort_fixes_crossings(self):
        P = np.array([[0.5, 0.3, 0.7], [0.1, 0.2, 0.15]])
        out = fix_quantile_crossing(P, [0.1, 0.5, 0.9], mode="sort")
        expected = np.array([[0.3, 0.5, 0.7], [0.1, 0.15, 0.2]])
        assert np.allclose(out, expected)

    def test_sort_idempotent(self):
        P = np.array([[0.1, 0.5, 0.9]])
        once = fix_quantile_crossing(P, [0.1, 0.5, 0.9], mode="sort")
        twice = fix_quantile_crossing(once, [0.1, 0.5, 0.9], mode="sort")
        assert np.allclose(once, twice)

    def test_sort_already_monotone_unchanged(self):
        P = np.array([[1.0, 2.0, 3.0]])
        out = fix_quantile_crossing(P, [0.1, 0.5, 0.9], mode="sort")
        assert np.allclose(out, P)

    def test_isotonic_strictly_monotone(self):
        # Crossings on every row.
        rng = np.random.default_rng(0)
        P = rng.standard_normal((50, 3))
        out = fix_quantile_crossing(P, [0.1, 0.5, 0.9], mode="isotonic")
        assert np.all(np.diff(out, axis=1) >= -1e-9)

    def test_isotonic_idempotent(self):
        P = np.array([[0.5, 0.3, 0.7]])
        once = fix_quantile_crossing(P, [0.1, 0.5, 0.9], mode="isotonic")
        twice = fix_quantile_crossing(once, [0.1, 0.5, 0.9], mode="isotonic")
        assert np.allclose(once, twice)

    def test_none_no_op(self):
        P = np.array([[0.5, 0.3, 0.7]])
        out = fix_quantile_crossing(P, [0.1, 0.5, 0.9], mode="none")
        assert np.allclose(out, P)

    def test_invalid_mode_raises(self):
        P = np.array([[0.5, 0.3, 0.7]])
        with pytest.raises(ValueError, match="mode must be"):
            fix_quantile_crossing(P, [0.1, 0.5, 0.9], mode="quantile-snap")

    def test_shape_mismatch_raises(self):
        P = np.array([[0.5, 0.3]])
        with pytest.raises(ValueError, match="preds_NK.shape\\[1\\]"):
            fix_quantile_crossing(P, [0.1, 0.5, 0.9], mode="sort")

    def test_1d_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            fix_quantile_crossing(np.array([0.1, 0.5, 0.9]), [0.1, 0.5, 0.9])


# E-P0.5: polars DataFrame input parametrize completion.

pl = pytest.importorskip("polars")


@pytest.mark.parametrize("frame_kind", ["numpy", "polars_df", "polars_lazy"])
@pytest.mark.parametrize("mode", ["sort", "isotonic"])
def test_fix_quantile_crossing_accepts_frame_kinds(frame_kind: str, mode: str):
    P = np.array([[0.5, 0.3, 0.7], [0.1, 0.2, 0.15]])
    if frame_kind == "polars_df":
        arr = pl.DataFrame(P, schema=["q1", "q5", "q9"]).to_numpy()
    elif frame_kind == "polars_lazy":
        arr = pl.DataFrame(P, schema=["q1", "q5", "q9"]).lazy().collect().to_numpy()
    else:
        arr = P
    out = fix_quantile_crossing(arr, [0.1, 0.5, 0.9], mode=mode)
    assert out.shape == arr.shape
    assert np.all(np.diff(out, axis=1) >= -1e-9)
