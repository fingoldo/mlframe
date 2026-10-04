"""Polars gather in the ranking prep helpers must not materialise the sort index as Python ints."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from mlframe.training.ranking.ranking import prepare_cb_inputs, prepare_lgb_inputs


def _frame(n: int):
    """Unsorted-group polars frame, labels and group ids of ``n`` rows."""
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    return X, rng.random(n), rng.integers(0, 1000, n)


@pytest.mark.parametrize("fn", [prepare_cb_inputs, prepare_lgb_inputs])
def test_polars_sort_gather_matches_numpy_order_and_avoids_python_int_list(fn):
    """Result rows equal the stable-argsort gather, and the traced Python-heap peak stays far below a per-row Python int list."""
    n = 400_000
    X, y, g = _frame(n)
    fn(*(_frame(1000)))  # warm imports
    tracemalloc.start()
    out = fn(X, y, g)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    idx = np.argsort(g, kind="stable")
    assert np.array_equal(out[0]["a"].to_numpy(), X["a"].to_numpy()[idx])
    assert np.array_equal(out[3], idx)
    assert peak < 40 * n
