"""``_propose_fourier`` is the single-direction predecessor of ``_propose_fourier_both_warps``; the escalation path calls only the latter.

Kept as the reference: for one direction the batched proposer must find the same frequencies and warp kinds.
"""

import numpy as np

from mlframe.feature_selection.filters._fe_auto_escalation import _propose_fourier_both_warps
from mlframe.feature_selection.filters._fe_auto_escalation_proposers import _propose_fourier


def _signature(proposals):
    """Kind and rounded frequencies of each proposal, sorted so two proposers compare regardless of order."""
    return sorted((p["kind"], tuple(np.round(np.asarray(p["freqs"], dtype=float), 9))) for p in proposals)


def test_propose_fourier_matches_the_forward_direction_of_both_warps():
    """The single-direction proposer finds the same frequencies as the forward direction of the batched one."""
    rng = np.random.default_rng(7)
    n = 3000
    x_a = rng.uniform(-1.0, 1.0, n)
    x_b = rng.uniform(0.5, 1.5, n)
    y = np.sin(2.0 * np.pi * 3.0 * x_a) * x_b + 0.05 * rng.standard_normal(n)

    single = _propose_fourier(x_a, x_b, y, min_val_corr=0.15, max_freqs=3)
    pair_cands: list = []
    _propose_fourier_both_warps(x_a, x_b, "a", "b", y, 0.15, 3, pair_cands)
    forward = [c for c in pair_cands if c["src_a"] == "a"]

    assert single, "the planted 3-cycle warp must be detected by the single-direction proposer"
    assert len(forward) == len(single)
    assert _signature(single) == _signature(forward)


def test_propose_fourier_finds_nothing_on_pure_noise():
    """Pure noise proposes no Fourier warp."""
    rng = np.random.default_rng(11)
    n = 3000
    x_a = rng.uniform(-1.0, 1.0, n)
    x_b = rng.uniform(0.5, 1.5, n)
    y = rng.standard_normal(n)

    assert _propose_fourier(x_a, x_b, y, min_val_corr=0.15, max_freqs=3) == []
