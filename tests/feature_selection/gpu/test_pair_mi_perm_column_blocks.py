"""The pair-MI permutation kernel streams a pool that does not fit the card in column blocks instead of declining to the CPU."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _pair_mi_perm_cuda as P

pytestmark = pytest.mark.gpu


def _case(n=3000, n_features=40, n_pairs=120, k=5, seed=0):
    """Byte codes of ``n_features`` columns, random pairs, ``k`` permuted binary targets and their frequencies."""
    rng = np.random.default_rng(seed)
    nb = rng.integers(2, 6, n_features).astype(np.int64)
    x = np.stack([rng.integers(0, b, n) for b in nb], axis=1).astype(np.uint8)
    a = rng.integers(0, n_features, n_pairs).astype(np.int64)
    b = rng.integers(0, n_features, n_pairs).astype(np.int64)
    y = rng.integers(0, 2, n).astype(np.uint8)
    perms = np.stack([rng.permutation(y) for _ in range(k)])
    freqs = np.bincount(y, minlength=2).astype(np.float64) / n
    return x, a, b, nb, perms, freqs


def test_blocked_matrix_equals_the_whole_pool_matrix(monkeypatch):
    """With free memory reported as just above the whole-pool need only a few columns fit at a time; the MI matrix equals the unblocked one."""
    x, a, b, nb, perms, freqs = _case()
    whole = P.pair_mi_perm_matrix_cuda(x, a, b, nb, perms, freqs, 10_000)
    assert whole is not None
    n, k, n_features, n_pairs = x.shape[0], perms.shape[0], x.shape[1], a.shape[0]
    need = n_features * n + k * n + k * n_pairs * 8 + n_pairs * 16 + n_features * 4 + freqs.shape[0] * 8  # the module's whole-pool estimate
    tight = 2 * need - 1  # one byte short of the whole-pool admission: the pool streams, about 18 columns at a time
    calls = []
    real = P._launch_pairs
    monkeypatch.setattr(P, "_launch_pairs", lambda *args: calls.append(args[1].shape[1]) or real(*args))
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (tight, 4 << 30))
    blocked = P.pair_mi_perm_matrix_cuda(x, a, b, nb, perms, freqs, 10_000)
    assert blocked is not None and len(calls) > 1
    assert max(calls) < x.shape[1]
    assert np.array_equal(blocked, whole)


def test_too_little_memory_for_two_columns_declines(monkeypatch):
    """When not even two columns plus the targets fit, the answer is still ``None`` (the CPU kernel)."""
    x, a, b, nb, perms, freqs = _case()
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (1000, 4 << 30))
    assert P.pair_mi_perm_matrix_cuda(x, a, b, nb, perms, freqs, 10_000) is None


def test_targets_per_pass_follow_the_shared_budget():
    """The pass takes 8 targets when their histograms fit, fewer as the joint table grows, never more than the number of targets, and 1 at the budget's edge."""
    assert P._targets_per_pass(100, 10, 25) == 8
    assert P._targets_per_pass(100, 10, 3) == 2
    assert P._targets_per_pass(100, 10, 1) == 1
    assert P._targets_per_pass(1300, 8, 25) == 1  # (1300 * 8 + 1300) * 4 = 45.6 KB is over the 44 KB budget even for one target... falls to 1
    assert P._targets_per_pass(500, 8, 25) == 2


def test_every_targets_per_pass_gives_the_same_matrix():
    """Forcing each of 1, 2, 4, 8 targets per pass gives the same MI matrix (the counts are exact integers and the sums add in a fixed order)."""
    x, a, b, nb, perms, freqs = _case(k=9)
    n = x.shape[0]
    results = []
    for kb in (1, 2, 4, 8):
        orig = P._targets_per_pass
        P._targets_per_pass = lambda *args, _kb=kb: _kb
        try:
            results.append(P.pair_mi_perm_matrix_cuda(x, a, b, nb, perms, freqs, 10_000))
        finally:
            P._targets_per_pass = orig
    assert len(results) == 4 and all(r is not None for r in results) and n
    for r in results[1:]:
        assert np.allclose(r, results[0], rtol=0, atol=1e-15)
