"""The strict-resident MI keeps a float32 candidate matrix float32 (no widening to double): same ranking and values within the order-statistic tolerance of the double path, and the switch restores the double path."""

import numpy as np
import pytest

pytest.importorskip("cupy")
from tests._cuda_require import require_cuda_device

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import _strict_mi_keeps_float32


def _matrix():
    """Continuous, tied, binary, heavy-tailed and offset float32 columns and a binary target driven by the first."""
    rng = np.random.default_rng(3)
    n = 20000
    cols = [rng.standard_normal(n), np.round(rng.standard_normal(n) * 2) / 2, (rng.random(n) > 0.5), np.exp(2 * rng.standard_normal(n)), 1e5 + rng.standard_normal(n) * 1e-1, rng.standard_t(1.5, n)]
    cols += [rng.standard_normal(n) for _ in range(70)]
    X = np.column_stack(cols).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 3] / np.abs(X[:, 3]).max() + rng.standard_normal(n) * 0.5) > 0).astype(np.int64)
    return X, y


def test_float32_operand_matches_the_double_path(monkeypatch):
    """Values agree within 1e-3 (edges differ only at ties) and the ranking of the informative columns is the same."""
    require_cuda_device()
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    X, y = _matrix()
    monkeypatch.setenv("MLFRAME_STRICT_MI_F32", "0")
    ref = np.asarray(_mi_classif_batch(X, y, nbins=10), dtype=np.float64)
    monkeypatch.setenv("MLFRAME_STRICT_MI_F32", "1")
    got = np.asarray(_mi_classif_batch(X, y, nbins=10), dtype=np.float64)
    assert np.abs(got - ref).max() < 1e-3
    assert set(np.argsort(-ref)[:5]) == set(np.argsort(-got)[:5])


def test_switch_and_dtype_gate(monkeypatch):
    """Only a float32 matrix keeps float32; the env switch turns it off."""
    X, _y = _matrix()
    monkeypatch.setenv("MLFRAME_STRICT_MI_F32", "1")
    assert _strict_mi_keeps_float32(X)
    assert not _strict_mi_keeps_float32(X.astype(np.float64))
    monkeypatch.setenv("MLFRAME_STRICT_MI_F32", "0")
    assert not _strict_mi_keeps_float32(X)
