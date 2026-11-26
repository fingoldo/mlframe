"""``heldout_ridge_r2_of_columns``: the held-out standardise-then-ridge probe read in row chunks equals the dense ``ScaledRidge`` probe, does not depend on the chunk size, and does not form the design."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from mlframe.feature_selection.filters import _streaming_ridge_probe as probe
from mlframe.feature_selection.filters._fe_raw_redundancy_helpers import _heldout_ridge_r2
from mlframe.feature_selection.filters._linear_probes import ScaledRidge
from mlframe.metrics.core import fast_r2_score


def _dense_reference(X: np.ndarray, y: np.ndarray, frac: float = 0.7) -> float:
    """The probe as the dense path computed it: round to float32, zero the non-finite values, ``ScaledRidge`` on the first rows, R^2 on the rest."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    t = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sp = int(X.shape[0] * frac)
    model = ScaledRidge(alpha=1.0).fit(X[:sp], t[:sp])
    return float(fast_r2_score(t[sp:], model.predict(X[sp:])))


def _problem(n: int, p: int, seed: int = 0):
    """A frame with a large offset, a constant column, a NaN column and a target that depends on a few columns."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    X[:, 3] = X[:, 3] * 1e3 + 5e4
    X[:, 5] = 7.0
    X[::97, 7] = np.nan
    X[::211, 8] = np.inf
    y = (X[:, 0] - 0.5 * X[:, 1] + 0.01 * X[:, 3] + 0.3 * rng.standard_normal(n)).astype(np.float32)
    return X, y


@pytest.mark.parametrize("n,p", [(3000, 12), (60000, 40), (20001, 9)])
def test_the_streamed_probe_equals_the_dense_probe(n, p):
    """The same value to rounding: the probe rounds, zeroes and standardises as the dense path did."""
    X, y = _problem(n, p)
    got = probe.heldout_ridge_r2_of_columns([X[:, j] for j in range(p)], y)
    assert got is not None
    assert got == pytest.approx(_dense_reference(X, y), abs=1e-9)


@pytest.mark.parametrize("chunk_bytes", [1, 40_000, 5_000_000])
def test_the_chunk_size_does_not_change_the_value(monkeypatch, chunk_bytes):
    """A chunk of the minimum 1024 rows, of a few thousand and of all the rows give one value."""
    X, y = _problem(9000, 14, seed=1)
    cols = [X[:, j] for j in range(14)]
    monkeypatch.setattr(probe, "CHUNK_BYTES", 128 * 1024 * 1024)
    whole = probe.heldout_ridge_r2_of_columns(cols, y)
    monkeypatch.setattr(probe, "CHUNK_BYTES", chunk_bytes)
    chunked = probe.heldout_ridge_r2_of_columns(cols, y)
    assert whole is not None and chunked is not None
    assert chunked == pytest.approx(whole, abs=1e-9)


def test_the_helper_of_a_column_list_is_the_streamed_probe():
    """``_heldout_ridge_r2`` of a list of columns gives the value of the dense reference, and still separates a faithful set from a linearly lossy one."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal(4000)
    b = rng.standard_normal(4000)
    y = 2.0 * b + 0.3 * a
    faithful = _heldout_ridge_r2([a, b], y)
    lossy = _heldout_ridge_r2([a, b**3], y)
    assert faithful == pytest.approx(_dense_reference(np.column_stack([a, b]), y), abs=1e-6)
    assert faithful is not None and lossy is not None and lossy < faithful - 0.05


@pytest.mark.parametrize("rows", [10, 60])
def test_too_few_rows_give_none(rows):
    """Fewer than 50 training or held-out rows: no verdict."""
    X, y = _problem(rows, 9)
    assert probe.heldout_ridge_r2_of_columns([X[:, j] for j in range(3)], y) is None


def test_a_constant_target_and_no_columns_give_none():
    """A target without spread and an empty column list leave the caller's decision unchanged."""
    X, _ = _problem(500, 9)
    assert probe.heldout_ridge_r2_of_columns([X[:, j] for j in range(3)], np.zeros(500)) is None
    assert probe.heldout_ridge_r2_of_columns([], np.arange(500.0)) is None


def test_a_column_of_another_length_gives_none():
    """Columns that do not match the target length cannot be read: no verdict."""
    X, y = _problem(500, 9)
    assert probe.heldout_ridge_r2_of_columns([X[:400, 0], X[:, 1]], y) is None


def test_integer_columns_are_read_as_they_are():
    """Binned codes (int8) are a valid column: the probe widens them chunk by chunk."""
    rng = np.random.default_rng(5)
    codes = rng.integers(0, 6, size=(5000, 4)).astype(np.int8)
    y = codes[:, 0] * 1.5 - codes[:, 1] + rng.standard_normal(5000) * 0.1
    got = probe.heldout_ridge_r2_of_columns([codes[:, j] for j in range(4)], y)
    assert got is not None and got > 0.95


def test_the_design_is_never_formed(monkeypatch):
    """Peak traced allocation stays near one chunk: far below the dense float64 design, which is 8 bytes a cell."""
    monkeypatch.setattr(probe, "CHUNK_BYTES", 8 * 1024 * 1024)
    n, p = 200_000, 60
    rng = np.random.default_rng(7)
    X = rng.standard_normal((p, n)).astype(np.float32)  # columns are the contiguous rows of this array: no copy is made by the probe
    y = (X[0] + rng.standard_normal(n)).astype(np.float32)
    columns = [X[j] for j in range(p)]
    dense_bytes = n * p * 8
    tracemalloc.start()
    try:
        value = probe.heldout_ridge_r2_of_columns(columns, y)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert value is not None
    # the chunk, its standardised copy and the product temporaries: a few chunks, whatever the row count
    assert peak < 4 * probe.CHUNK_BYTES and peak < dense_bytes / 2, (peak, dense_bytes)
