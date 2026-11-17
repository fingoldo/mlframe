"""The extra-basis generator detects the adaptive Fourier frequencies of all its columns in one device batch and emits the same columns as the per-column detector."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_batch as fb
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe_generate as gen


def _frame(n=6000, p=7, seed=0):
    """Seven columns, two of them carrying a periodic signal the linear and fixed-grid bases miss, and a binary-valued target."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.uniform(0, 1, (n, p)).astype(np.float32), columns=[f"x{i}" for i in range(p)])
    y = (np.sin(2 * np.pi * 7.3 * X["x0"]) + 0.6 * np.sin(2 * np.pi * 11.1 * X["x3"]) + 0.3 * r.standard_normal(n) > 0).astype(np.float64)
    return X, y.to_numpy()


def test_prefetch_gives_the_per_column_columns(monkeypatch):
    """Same output columns, values and metadata whether the frequencies come from the batch or from the per-column calls; the batch call really happens."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    X, y = _frame()
    kw = dict(extra_bases=("fourier",), y=y, fourier_adaptive=True, fourier_freqs=(1.0, 2.0), fourier_powers=(1,), dedup_collinear_sources=False)
    calls = []
    real = fb.detect_fourier_freqs_batch
    monkeypatch.setattr(gen, "device_batch_enabled", None, raising=False)
    monkeypatch.setattr(fb, "device_batch_enabled", lambda: True)
    monkeypatch.setattr(fb, "detect_fourier_freqs_batch", lambda *a, **k: (calls.append(len(a[0])), real(*a, **k))[1])
    got, meta_got = gen.generate_extra_basis_features(X, **kw)
    assert calls and calls[0] >= 2, "the batch detector must run for the gated columns"
    monkeypatch.setattr(fb, "device_batch_enabled", lambda: False)
    want, meta_want = gen.generate_extra_basis_features(X, **kw)
    assert list(got.columns) == list(want.columns)
    assert set(meta_got) == set(meta_want)
    for name, m in meta_want.items():  # the per-column detector carries float32 rounding of the grid frequency (7.287499904632568 for 7.2875); the batch holds the grid value itself
        assert {k: v for k, v in meta_got[name].items() if not isinstance(v, float)} == {k: v for k, v in m.items() if not isinstance(v, float)}
        for k, v in m.items():
            if isinstance(v, float):
                assert meta_got[name][k] == pytest.approx(v, rel=1e-6, abs=1e-6), (name, k)
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=0, atol=1e-4)


def _chirp_frame(n=6000, p=6, seed=1):
    """Six columns, two of them carrying a growing-frequency (chirp) signal, and a binary-valued target."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame(r.standard_normal((n, p)).astype(np.float32), columns=[f"x{i}" for i in range(p)])
    z0, z2 = X["x0"].to_numpy(), X["x2"].to_numpy()
    y = (np.sin(2 * np.pi * 0.12 * np.sign(z0) * z0**2) + 0.7 * np.sin(2 * np.pi * 0.17 * np.sign(z2) * z2**2) + 0.3 * r.standard_normal(n) > 0).astype(np.float64)
    return X, y


def test_chirp_prefetch_gives_the_per_column_columns(monkeypatch):
    """The chirp-axis legs come out the same whether their frequencies are detected in one device batch or column by column, and the batch really runs for the chirp axis."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    X, y = _chirp_frame()
    kw = dict(extra_bases=("fourier",), y=y, fourier_adaptive=False, fourier_chirp=True, fourier_freqs=(1.0,), fourier_powers=(1,), dedup_collinear_sources=False)
    calls = []
    real = fb.detect_fourier_freqs_batch
    monkeypatch.setattr(fb, "device_batch_enabled", lambda: True)
    monkeypatch.setattr(fb, "detect_fourier_freqs_batch", lambda *a, **k: (calls.append((len(a[0]), k.get("min_rows"))), real(*a, **k))[1])
    got, meta_got = gen.generate_extra_basis_features(X, **kw)
    assert any(c[0] >= 2 and c[1] == 800 for c in calls), "the chirp batch detector must run for the gated columns"
    monkeypatch.setattr(fb, "device_batch_enabled", lambda: False)
    want, meta_want = gen.generate_extra_basis_features(X, **kw)
    assert list(got.columns) == list(want.columns) and any("__qsin" in c for c in want.columns)
    assert set(meta_got) == set(meta_want)
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=0, atol=1e-4)
