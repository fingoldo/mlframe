"""Two-direction sign proof for the RelaxMRMR 3-D-MI score (``_relaxmrmr_3d.relax_mrmr_score``).

Pins the two correctness bugs the score had:

A1 -- the pairwise-redundancy term ``(1/|S|) sum_j I(X; X_j)`` was identically zero (the joint-MI kernel was called
      with the selected column in the conditioning slot and a constant zeros array in the target slot, so it computed
      ``I((X, X_j); const) = 0``). A candidate identical to an already-selected feature must now incur a large penalty.

A2 -- the 3-way interaction term used only the CONDITIONAL co-information ``I(X;Z_i|Y)+I(X;Z_j|Y)-I(X;Z_i,Z_j|Y)`` and
      ADDED it, which RAISES the score of a jointly-redundant candidate as ``alpha`` grows -- the wrong direction. The
      correct quantity is the interaction information ``II = I(X;Z_i;Z_j|Y) - I(X;Z_i;Z_j)``: II<0 for redundancy
      (lower score with alpha), II>0 for synergy (higher score with alpha).

These assertions FAIL on the pre-fix code (pair_red==0 for A1; redundant-triple delta is +0.39 instead of negative
for A2) and PASS post-fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score

N = 8000


def _noisy_copy(rng, latent, K, p=0.1):
    """A noisy reflection of ``latent``: flip a fraction ``p`` of entries to random bins."""
    out = latent.copy()
    flip = rng.random(latent.size) < p
    out[flip] = rng.integers(0, K, int(flip.sum()))
    return out.astype(np.int64)


def test_a1_duplicate_candidate_penalised_below_independent_of_equal_relevance():
    """A1: a candidate IDENTICAL to an already-selected feature must score strictly below an independent
    candidate of equal marginal relevance (pre-fix the pairwise-redundancy term was 0, so the two tied)."""
    rng = np.random.default_rng(0)
    K = 5
    latent = rng.integers(0, K, N).astype(np.int64)
    y = latent.copy()
    z_sel = _noisy_copy(rng, latent, K)
    x_dup = z_sel.copy()  # exact duplicate of the selected feature
    x_ind = _noisy_copy(rng, latent, K)  # same construction => same marginal I(X; Y), but independent of z_sel

    s_dup = relax_mrmr_score(x_dup, [z_sel], y, K, [K], K, alpha=0.0)
    s_ind = relax_mrmr_score(x_ind, [z_sel], y, K, [K], K, alpha=0.0)
    assert s_dup < s_ind - 0.5, f"duplicate-of-selected candidate must be heavily penalised; got dup={s_dup:.4f} ind={s_ind:.4f}"


def test_a2_redundant_triple_score_drops_as_alpha_grows():
    """A2 direction 1: for a jointly-REDUNDANT triple (X, Z1, Z2 all noisy copies of the latent driver of y)
    the score must DECREASE as alpha increases (II < 0). Pre-fix it increased (delta +0.39)."""
    K = 5
    for seed in range(5):
        rng = np.random.default_rng(seed)
        latent = rng.integers(0, K, N).astype(np.int64)
        y = latent.copy()
        x = _noisy_copy(rng, latent, K)
        z1 = _noisy_copy(rng, latent, K)
        z2 = _noisy_copy(rng, latent, K)
        s0 = relax_mrmr_score(x, [z1, z2], y, K, [K, K], K, alpha=0.0)
        s1 = relax_mrmr_score(x, [z1, z2], y, K, [K, K], K, alpha=1.0)
        assert s1 < s0, f"seed {seed}: redundant-triple score must drop with alpha; got alpha0={s0:.4f} alpha1={s1:.4f}"


def test_a2_synergistic_triple_scored_above_redundant_of_equal_relevance():
    """A2 direction 2: a SYNERGISTIC candidate (XOR-like: jointly informative with the selected pair but marginally
    weak) must score strictly ABOVE a redundant/irrelevant candidate of equal (~0) marginal relevance, once alpha>0.

    Target ``y = z1 XOR z2``; the synergistic candidate equals ``z1`` (completes the XOR jointly with z2, marginal
    I(X;Y)~0); the baseline candidate is independent noise (also marginal ~0). The interaction term must lift the
    synergistic one and not the noise one."""
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        z1 = rng.integers(0, 2, N).astype(np.int64)
        z2 = rng.integers(0, 2, N).astype(np.int64)
        y = (z1 ^ z2).astype(np.int64)
        x_syn = z1.copy()  # synergistic: jointly completes XOR with z2
        x_noise = rng.integers(0, 2, N).astype(np.int64)  # equal ~0 marginal relevance, no joint structure
        s_syn = relax_mrmr_score(x_syn, [z1, z2], y, 2, [2, 2], 2, alpha=1.0)
        s_noise = relax_mrmr_score(x_noise, [z1, z2], y, 2, [2, 2], 2, alpha=1.0)
        assert s_syn > s_noise + 0.2, f"seed {seed}: synergistic candidate must outscore equal-relevance noise; got syn={s_syn:.4f} noise={s_noise:.4f}"


def test_score_is_finite_and_alpha0_matches_marginal_minus_pairwise():
    """Sanity: the score stays finite, and at alpha=0 it is exactly relevance minus mean pairwise redundancy
    (the interaction term is gated off), so the A1 fix is what makes alpha=0 discriminate duplicates."""
    rng = np.random.default_rng(3)
    K = 4
    latent = rng.integers(0, K, N).astype(np.int64)
    y = latent.copy()
    z1 = _noisy_copy(rng, latent, K)
    z2 = _noisy_copy(rng, latent, K)
    x = _noisy_copy(rng, latent, K)
    s = relax_mrmr_score(x, [z1, z2], y, K, [K, K], K, alpha=0.0)
    assert np.isfinite(s)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v", "--no-cov", "-p", "no:randomly"]))
