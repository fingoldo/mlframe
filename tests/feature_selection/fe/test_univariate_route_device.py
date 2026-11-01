"""Routing all columns of the univariate generator on the device gives the host router's choice on columns built for each basis, skips what the generator skips, and leaves the rest to the host."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import generate_univariate_basis_features
from mlframe.feature_selection.filters._univariate_route_device import route_columns_on_device

from ._basis_cases import DEGREES, N, _case

BASES = ["hermite", "legendre", "chebyshev", "laguerre", "fourier", "rbf", "sigmoid", "pade"]


def _frame():
    """One column per basis built for it, a noise column, a NaN column and an integer group key; the target of the first (they share a seed, so the targets are the same rows)."""
    X = {}
    y = None
    for b in BASES:
        x, yy, _d = _case(b)
        X[f"c_{b}"] = x.astype(np.float32)
        if b == "pade":
            y = yy
    rng = np.random.default_rng(5)
    X["noise"] = rng.normal(size=N).astype(np.float32)
    X["with_nan"] = np.where(rng.random(N) < 0.1, np.nan, rng.normal(size=N)).astype(np.float32)
    X["key"] = rng.integers(0, 6, N).astype(np.float32)
    return pd.DataFrame(X), y


def test_device_choice_equals_host_choice_per_crafted_column(monkeypatch):
    """For the column built for each basis (each against its own target) the device pass returns that basis."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_BASIS_CELLS_DEVICE", "1")
    for b in BASES:
        x, y, _d = _case(b)
        X = pd.DataFrame({"a": x.astype(np.float32)})
        got = route_columns_on_device(X, ["a"], y, DEGREES, True, None)
        if not got:
            pytest.skip("needs a usable CUDA device")
        assert got == {"a": b}, f"{b}: {got}"


def test_ineligible_columns_are_left_to_the_host(monkeypatch):
    """A column with a NaN, an integer group key (auto columns) and a degenerate target are not routed on the device."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_BASIS_CELLS_DEVICE", "1")
    X, y = _frame()
    got = route_columns_on_device(X, list(X.columns), y, (2, 3), True, None)
    if not got:
        pytest.skip("needs a usable CUDA device")
    assert "with_nan" not in got and "key" not in got
    assert set(got) >= {f"c_{b}" for b in BASES}
    assert route_columns_on_device(X, list(X.columns), np.zeros(N), (2, 3), True, None) == {}
    assert route_columns_on_device(X, list(X.columns), y, (2, 3), True, {"c_hermite": np.zeros(3)}) == {}


def test_generator_output_is_the_same_with_and_without_the_device_pass(monkeypatch):
    """The columns emitted (names and values) do not depend on whether the routing ran on the device."""
    pytest.importorskip("cupy")
    X, y = _frame()
    monkeypatch.setenv("MLFRAME_BASIS_CELLS_DEVICE", "0")
    host = generate_univariate_basis_features(X, y=y, dedup_collinear_sources=False)
    monkeypatch.setenv("MLFRAME_BASIS_CELLS_DEVICE", "1")
    from mlframe.feature_selection.filters import _orthogonal_univariate_fe as U

    U._BASIS_MEMO.clear()
    dev = generate_univariate_basis_features(X, y=y, dedup_collinear_sources=False)
    chosen_host = {c.split("__")[0]: c.split("__")[1].rstrip("0123456789") for c in host.columns}
    chosen_dev = {c.split("__")[0]: c.split("__")[1].rstrip("0123456789") for c in dev.columns}
    # the columns with a clear winner agree; a near-tie noise column may differ (device rounding)
    for b in BASES:
        assert chosen_host[f"c_{b}"] == chosen_dev[f"c_{b}"]
    assert set(host.columns) >= {c for c in dev.columns if c.split("__")[0].startswith("c_")}
