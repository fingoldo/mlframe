"""The pair-FE analytic noise gate engages from 10k rows and agrees with the permutation gate up to Monte-Carlo noise."""
import numpy as np

from mlframe.feature_selection.filters import _analytic_mi_null as A
from mlframe.feature_selection.filters.info_theory._batch_kernels_v3 import batch_mi_with_noise_gate_v3 as perm_gate


def test_floor_only_moves_for_the_pair_gate(monkeypatch):
    """Global floor stays 25k, the pair gate floor is 10k, and an explicit env override applies to both."""
    monkeypatch.delenv("MLFRAME_MI_ANALYTIC_NULL_MIN_N", raising=False)
    assert A.analytic_null_min_n() == 25_000
    assert A.pair_gate_min_n() == 10_000
    assert A.analytic_null_applicable(12_000, 10, 2, min_n=A.pair_gate_min_n())
    assert not A.analytic_null_applicable(12_000, 10, 2)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL_MIN_N", "50000")
    assert A.pair_gate_min_n() == 50_000


def test_analytic_gate_agrees_with_the_permutation_gate_at_12k_rows():
    """40 candidates from pure noise to weak signal: at most 4 of 40 decisions differ (the 99-shuffle gate is itself noisy at the boundary)."""
    n, nb = 12_000, 10
    rng = np.random.default_rng(0)
    y = (rng.random(n) > 0.5).astype(np.int16)
    cols = []
    for k in range(40):
        z = rng.normal(size=n) + [0, 0, 0, 0.01, 0.02, 0.03, 0.05, 0.08][k % 8] * (2 * y - 1) * 3
        edges = np.quantile(z, np.linspace(0, 1, nb + 1)[1:-1])
        cols.append(np.searchsorted(edges, z).astype(np.int32))
    codes = np.ascontiguousarray(np.stack(cols, axis=1))
    nbins = np.full(codes.shape[1], nb, np.int64)
    fy = np.bincount(y, minlength=2) / n
    args = (codes, nbins, y, y, fy)
    perm = perm_gate(*args, 99, np.uint64(1), 0.99, False, np.int32, np.int16)
    obs = perm_gate(*args, 0, np.uint64(0), 0.99, False, np.int32, np.int16)
    got = A.analytic_batch_noise_gate(codes, obs, y, n, 0.99, by=2, min_n=A.pair_gate_min_n())
    assert int(((np.asarray(perm) > 0) != (np.asarray(got) > 0)).sum()) <= 4
