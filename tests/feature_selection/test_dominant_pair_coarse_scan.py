"""The two-stage dominant-pair scan returns the pair of the exact scan, and only rescans a few pairs when the pool has a clear leader."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairs_prevalence as sp

P = 210  # 21945 pairs: past the pool size from which the coarse pass is used
N = 30000


def _case(kind, seed):
    """Random operands and a continuous target built from the product of operands 3 and 7: clean, carried by 1 % outlier rows, or pure noise."""
    rng = np.random.default_rng(seed)
    ops = rng.standard_normal((P, N)).astype(np.float32)
    noise = rng.standard_normal(N).astype(np.float32)
    if kind == "clean":
        y = ops[3] * ops[7] + 0.5 * noise
    elif kind == "tail":
        y = noise.copy()
        rows = rng.choice(N, N // 100, replace=False)
        y[rows] = 60.0 * ops[3, rows] * ops[7, rows] + noise[rows]
    else:
        y = noise
    pairs = [(a, b) for a in range(P) for b in range(a + 1, P)]
    return pairs, y.astype(np.float64), ops


@pytest.mark.parametrize("kind", ["clean", "tail", "noise"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coarse_scan_returns_the_exact_winner(kind, seed):
    """Same pair as the exact scan over all pairs."""
    pytest.importorskip("numba")
    pairs, y, ops = _case(kind, seed)
    cached = lambda i: ops[i]
    want = sp._dominant_pair_by_linear_corr_exact(pairs, y, cached)
    got = sp._dominant_pair_by_linear_corr(pairs, y, cached)
    assert got == want


def test_a_clear_leader_is_rescanned_on_few_pairs(monkeypatch):
    """With a planted product pair the exact pass covers a small share of the pairs; without one it covers all."""
    seen = []
    real = sp._scan_best_corrs

    def spy(scan_pks, ya, cached, extra_stride=1):
        """Record the size of every pass."""
        seen.append((len(scan_pks), extra_stride))
        return real(scan_pks, ya, cached, extra_stride)

    monkeypatch.setattr(sp, "_scan_best_corrs", spy)
    pairs, y, ops = _case("clean", 0)
    sp._dominant_pair_by_linear_corr(pairs, y, lambda i: ops[i])
    assert len(seen) == 2 and seen[0][0] == len(pairs) and seen[0][1] > 1 and seen[1][0] < 0.05 * len(pairs)
    seen.clear()
    pairs, y, ops = _case("noise", 0)
    sp._dominant_pair_by_linear_corr(pairs, y, lambda i: ops[i])
    assert seen[-1][0] == len(pairs)  # no leader: the exact scan on all pairs
