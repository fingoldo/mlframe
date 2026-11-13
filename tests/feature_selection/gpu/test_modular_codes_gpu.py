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
    """Same names as the host-built pool; the residue rows (``mod``) code exactly as on the host. The sine / cosine rows are taken at the stored type on the device: for the
    continuous and binary columns (and the heavy-tailed ones) at most a few in a million cells cross a bin edge; the integer column with 500 levels has many exactly tied
    sine / cosine values (sin(x) = sin(pi - x)) whose order the host's double rounding noise and the device's float rounding break differently, so a tied group straddling an edge
    splits differently (under 15 percent of its cells), which the quantile coder's tie handling makes arbitrary on either side."""
    monkeypatch.setenv("MLFRAME_MODULAR_CODES_DEVICE", "1")
    X = _frame()
    periods = (7, 12.5, 30, 365)
    names, values = modular_candidate_block(X, list(X.columns), periods)
    want_codes, want_nx = family_codes_of_rows(values, n_bins)
    got = modular_family_codes_gpu(X, list(X.columns), periods, n_bins)
    assert got is not None
    assert got[0] == names
    assert np.array_equal(got[1][0::3], want_codes[0::3])
    assert np.array_equal(got[2][0::3], want_nx[0::3])
    per_col = 3 * len(periods)
    for i, col in enumerate(X.columns):
        moved = float((got[1][i * per_col : (i + 1) * per_col] != want_codes[i * per_col : (i + 1) * per_col]).mean())
        assert moved < (0.15 if col == "b" else 1e-4), (col, moved)


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


def test_a_tuner_entry_exists_and_the_fallback_follows_the_measured_crossover():
    """Below the measured bound the host pool, from it the device (the sweep on the dev card put the crossover between 300000 and 1500000 elements)."""
    from mlframe.feature_selection.filters import _modular_codes_gpu as M

    assert M._SPEC is not None
    assert M._fallback_choice(300_000) == "host"
    assert M._fallback_choice(6_000_000) == "device"


def test_the_pool_choice_asks_the_tuner_outside_the_strict_path_and_keeps_the_strict_bound(monkeypatch):
    """Non-strict: the tuner's verdict decides. Strict-resident: the residency bound (20M elements) stays, whatever the tuner says."""
    from mlframe.feature_selection.filters import _modular_codes_gpu as M
    from mlframe.feature_selection.filters import _quantile_codes_gpu as Q

    for name in ("MLFRAME_MODULAR_CODES_DEVICE", "MLFRAME_MODULAR_DROP_COPIES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: False)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_usability_signal._crit_np_dtype", lambda: np.float32)
    asked = []
    monkeypatch.setattr(M, "modular_device_chosen", lambda n: asked.append(n) or True)
    monkeypatch.setattr(Q, "_device_min_elements", lambda: Q.DEVICE_MIN_ELEMENTS)
    assert M._use_device(2_000_000) is True and asked == [2_000_000]
    monkeypatch.setattr(Q, "_device_min_elements", lambda: Q.STRICT_DEVICE_MIN_ELEMENTS)
    asked.clear()
    assert M._use_device(2_000_000) is False and not asked
    assert M._use_device(Q.STRICT_DEVICE_MIN_ELEMENTS) is True


def test_forced_host_and_forced_device_pools_have_the_same_bin_shares_and_nearly_all_codes():
    """The sweep's callables agree on the share of cells per bin to 1e-6; cell by cell, only the values the stored-type sine / cosine moves across a bin edge differ (a few in a million)."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _modular_codes_gpu as M

    (X,) = M._sweep_inputs({"n_samples": 6_000_000})
    assert np.allclose(M._sweep_host(X), M._sweep_device(X), rtol=1e-6, atol=1e-6)
    _names, values = modular_candidate_block(X, list(X.columns), list(M._SWEEP_PERIODS))
    host_codes, _ = family_codes_of_rows(values, M._SWEEP_BINS)
    _n, dev_codes, _nx = modular_family_codes_gpu(X, list(X.columns), list(M._SWEEP_PERIODS), M._SWEEP_BINS, family_elements=1 << 40)
    assert float((np.asarray(host_codes) != np.asarray(dev_codes)).mean()) < 1e-4
