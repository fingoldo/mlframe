"""The device modular candidate codes equal the host pool's family codes, few-level candidates included."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._family_maxt_gate import family_codes_of_rows
from mlframe.feature_selection.filters._modular_codes_gpu import modular_family_codes_gpu
from mlframe.feature_selection.filters._periodic_fe import modular_candidate_block


def _frame(n: int = 30000) -> pd.DataFrame:
    """Continuous, integer (few residue levels for small periods), binary, heavy-tailed and partly non-finite columns."""
    rng = np.random.default_rng(11)
    a = rng.standard_normal(n) * 40
    a[::53] = np.nan
    return pd.DataFrame(
        {"a": a, "b": rng.integers(0, 500, n).astype(float), "c": rng.integers(0, 2, n).astype(float), "d": rng.standard_cauchy(n), "e": rng.exponential(100, n)}
    )


@pytest.mark.parametrize("n_bins", [4, 10])
def test_device_codes_equal_the_host_pool(monkeypatch, n_bins):
    """Same names, codes and code counts as coding the host-built pool."""
    monkeypatch.setenv("MLFRAME_MODULAR_CODES_DEVICE", "1")
    X = _frame()
    periods = (7, 12.5, 30, 365)
    names, values = modular_candidate_block(X, list(X.columns), periods)
    want_codes, want_nx = family_codes_of_rows(values, n_bins)
    got = modular_family_codes_gpu(X, list(X.columns), periods, n_bins)
    assert got is not None
    assert got[0] == names
    assert np.array_equal(got[1], want_codes)
    assert np.array_equal(got[2], want_nx)


def test_forced_off_returns_none(monkeypatch):
    """``MLFRAME_MODULAR_CODES_DEVICE=0`` leaves the pool to the host path."""
    monkeypatch.setenv("MLFRAME_MODULAR_CODES_DEVICE", "0")
    assert modular_family_codes_gpu(_frame(2000), ["a", "b"], (7,), 10) is None


def test_family_size_decides_for_every_chunk(monkeypatch):
    """A chunk below the device bound still takes the device when the whole family is above it (all chunks of one gate take one path)."""
    import mlframe.feature_selection.filters._quantile_codes_gpu as Q

    monkeypatch.delenv("MLFRAME_MODULAR_CODES_DEVICE", raising=False)
    monkeypatch.setattr(Q, "_device_min_elements", lambda: 10_000_000)
    X = _frame(20000)
    small = modular_family_codes_gpu(X, ["a", "b"], (7,), 10)  # 2 x 3 x 20000 = 120000 elements alone
    assert small is None
    assert modular_family_codes_gpu(X, ["a", "b"], (7,), 10, family_elements=20_000_000) is not None
