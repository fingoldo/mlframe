"""The analytic triage of a family never contradicts the exact permutation test, and spares it when candidates are far from the null."""

import numpy as np
import pytest

import mlframe.feature_selection.filters._family_maxt_gate as fg


def _family(seed, n=4000, n_noise=40, signal=(0.0, 0.5, 1.0), bins=8):
    """Columns of ``bins`` codes: noise, plus columns that follow the binary target with the given strength."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    cols = [rng.integers(0, bins, n) for _ in range(n_noise)]
    for s in signal:
        base = rng.integers(0, bins, n)
        follow = np.where(rng.random(n) < s, y * (bins - 1), base)
        cols.append(follow)
    return y, [c.astype(np.int8).reshape(-1, 1) for c in cols]


def _exact(batches, y, monkeypatch):
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    try:
        return fg.family_pass_batches(batches, y, alpha=0.01, n_perm=999)
    finally:
        monkeypatch.delenv("MLFRAME_MI_ANALYTIC_NULL")


@pytest.mark.parametrize("seed", range(4))
def test_triage_agrees_with_the_exact_test(seed, monkeypatch):
    y, batches = _family(seed)
    exact = _exact(batches, y, monkeypatch)
    got = fg.family_pass_batches(batches, y, alpha=0.01, n_perm=999)
    assert got == exact
    assert got[-1] and not any(got[:40])


def test_strong_family_runs_no_permutation(monkeypatch):
    y, batches = _family(1, signal=(1.0, 0.9, 0.8), n_noise=5)
    calls = []
    real = fg.family_null_max
    monkeypatch.setattr(fg, "family_null_max", lambda *a, **k: calls.append(1) or real(*a, **k))
    got = fg.family_pass_batches(batches, y, alpha=0.01, n_perm=999)
    assert got[-3:] == [True, True, True]
    assert not calls, "candidates far above the null must be decided without drawing permutations"


def test_sparse_table_uses_the_permutation_test_alone(monkeypatch):
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    batches = [rng.integers(0, 60, n).astype(np.int16).reshape(-1, 1) for _ in range(5)]
    cols, widths = fg._stack_codes_transposed(batches, [1] * 5), [1] * 5
    n_x = cols.max(axis=1).astype(np.int64) + 1
    obs = fg.mi_all_columns_njit(cols, n_x, np.ascontiguousarray(y.astype(np.int64)), 2)
    assert fg._analytic_triage(cols, n_x, y.astype(np.int64), 2, obs, np.arange(6), 0.01) is None
