"""The two-sort device presort gives the order and tie flags of the host stable sort for float64 and float32 columns (structural launch reduction, 2026-11-20)."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _honest_null_presort_gpu as pg
from mlframe.feature_selection.filters._honest_null_accept import _presort_columns, _tie_flags


def _cols(dtype, n=4000, k=9, seed=0):
    """Columns with heavy ties, a constant column, both zeros and negative values."""
    rng = np.random.default_rng(seed)
    out = np.round(rng.normal(size=(n, k)), 1).astype(dtype)
    out[:, 3] = 2.5
    out[::7, 4] = 0.0
    out[1::7, 4] = -0.0
    return np.ascontiguousarray(out)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_two_sort_presort_equals_the_host_stable_sort(dtype):
    """Order and tie flags of the device two-sort route equal ``_presort_columns`` / ``_tie_flags`` column for column."""
    raw = _cols(dtype)
    got = pg._presort_by_two_sorts(cp, raw)
    assert got is not None
    order, tie = got
    ref_order = np.asarray(_presort_columns(raw))
    ref_tie = np.asarray(_tie_flags(raw, ref_order))
    assert np.array_equal(order, ref_order)
    assert np.array_equal(tie, ref_tie)


def test_float64_columns_take_the_two_sort_route_not_a_sort_per_column(monkeypatch):
    """``presort_and_ties_gpu`` on float64 columns launches no per-column device sort (the composite-key route declines them, the two-sort route answers)."""
    raw = _cols(np.float64)
    calls = []
    real = pg.device_argsort

    def counting(a, *args, **kw):
        """Record the shape of every device argsort."""
        calls.append(tuple(a.shape))
        return real(a, *args, **kw)

    monkeypatch.setattr(pg, "device_argsort", counting)
    order, _tie = pg.presort_and_ties_gpu(raw)
    assert all(len(shape) == 1 for shape in calls) and len(calls) == 2
    np.testing.assert_array_equal(order, np.asarray(_presort_columns(raw)))
