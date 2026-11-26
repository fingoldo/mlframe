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


def _mixed_cols(n=4000, k=10, seed=1):
    """A double-precision block: the even columns are float32 in value (widened, heavy ties, a constant, both zeros), the odd ones are not."""
    rng = np.random.default_rng(seed)
    out = np.round(rng.normal(size=(n, k)), 1)
    out[:, 1::2] = np.round(rng.normal(size=(n, k // 2)), 1) + 1e-9 * rng.normal(size=(n, k // 2))  # not representable in float32
    out[:, 0::2] = out[:, 0::2].astype(np.float32).astype(np.float64)
    out[:, 2] = 2.5
    out[::7, 4] = 0.0
    out[1::7, 4] = -0.0
    return np.ascontiguousarray(out)


def test_a_double_block_with_float32_columns_equals_the_host_stable_sort():
    """Columns that survive float32 go through the composite-key route and the others through two sorts: order and ties equal the host's, column for column and equal the two-sort route's."""
    raw = _mixed_cols()
    got = pg._presort_by_exact_groups(cp, raw)
    assert got is not None
    ref_order = np.asarray(_presort_columns(raw))
    assert np.array_equal(got[0], ref_order) and np.array_equal(got[1], np.asarray(_tie_flags(raw, ref_order)))
    two = pg._presort_by_two_sorts(cp, raw)
    assert two is not None and np.array_equal(got[0], two[0]) and np.array_equal(got[1], two[1])
    via_entry = pg.presort_and_ties_gpu(raw)
    assert via_entry is not None and np.array_equal(via_entry[0], ref_order)


def test_the_routes_follow_the_exactness_of_the_columns(monkeypatch):
    """A block of float32 values in double runs no argsort (one composite-key sort), a block of inexact values runs the two sorts only, a mixed block runs both; a float32 block is not this route's."""
    seen = {"argsort": 0, "sort": 0}
    real_argsort, real_sort = pg.device_argsort, pg.device_sort
    monkeypatch.setattr(pg, "device_argsort", lambda *a, **k: (seen.__setitem__("argsort", seen["argsort"] + 1), real_argsort(*a, **k))[1])
    monkeypatch.setattr(pg, "device_sort", lambda *a, **k: (seen.__setitem__("sort", seen["sort"] + 1), real_sort(*a, **k))[1])
    rng = np.random.default_rng(4)
    exact = np.ascontiguousarray(np.round(rng.normal(size=(3000, 6)), 1).astype(np.float32).astype(np.float64))
    inexact = np.ascontiguousarray(np.round(rng.normal(size=(3000, 6)), 1))
    pg._presort_by_exact_groups(cp, exact)
    assert seen == {"argsort": 0, "sort": 1}
    seen.update(argsort=0, sort=0)
    pg._presort_by_exact_groups(cp, inexact)
    assert seen == {"argsort": 2, "sort": 0}
    seen.update(argsort=0, sort=0)
    pg._presort_by_exact_groups(cp, np.ascontiguousarray(np.hstack([exact, inexact])))
    assert seen == {"argsort": 2, "sort": 1}
    assert pg._presort_by_exact_groups(cp, exact.astype(np.float32)) is None
