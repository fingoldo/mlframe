"""RelaxMRMR's interaction term must not reward independent candidates through plug-in MI bias (mrmr_audit_2026-09-14 RO-2).

``inter = co_cond - co_uncond`` subtracts ``I(X; Z_i, Z_j)``, estimated on a (K_x, K_i*K_j) table, from marginal MIs on (K_x, K_i) and
(K_x, K_j) tables. Plug-in MI is biased upward by roughly (occupied cells - 1)/2n per table, so the composite term carries the largest
bias; with no correction the difference leaves a systematic positive "synergy" for every candidate, growing with the selected pair's joint
cardinality and shrinking with n. Under full independence every term of the score except relevance should be ~0.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score


def _independent(n, k_sel, seed=0, k_x=10, k_y=3):
    """Mutually independent integer codes for the candidate, two selected columns and the target."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, k_x, size=n)
    z1 = rng.integers(0, k_sel, size=n)
    z2 = rng.integers(0, k_sel, size=n)
    y = rng.integers(0, k_y, size=n)
    return x, [z1, z2], y, k_x, [k_sel, k_sel], k_y


def _score_parts(x, sel, y, k_x, k_sel, k_y):
    """The full score, and the score with the interaction term switched off (alpha=0); their difference is the interaction term."""
    full = relax_mrmr_score(x, sel, y, k_x, k_sel, k_y, alpha=1.0)
    no_inter = relax_mrmr_score(x, sel, y, k_x, k_sel, k_y, alpha=0.0)
    return full, full - no_inter


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_interaction_term_is_near_zero_under_independence(seed):
    """With X, Z_i, Z_j and Y all independent there is no synergy to reward."""
    _, inter = _score_parts(*_independent(n=2000, k_sel=10, seed=seed))
    assert abs(inter) < 0.01, f"interaction term {inter:.4f} under full independence (seed={seed})"


def test_interaction_term_does_not_scale_with_the_selected_pair_cardinality():
    """The same independent data scored with a 4-level vs a 12-level selected pair must get about the same interaction term."""
    gaps = []
    for k_sel in (4, 12):
        vals = [_score_parts(*_independent(n=2000, k_sel=k_sel, seed=s))[1] for s in range(3)]
        gaps.append(float(np.mean(vals)))
    assert abs(gaps[1] - gaps[0]) < 0.01, f"mean interaction term moved from {gaps[0]:.4f} (K=4) to {gaps[1]:.4f} (K=12)"


def test_genuine_synergy_is_still_rewarded():
    """Control: when y is the XOR of the two selected columns and x copies z1, the interaction term must remain clearly non-zero."""
    rng = np.random.default_rng(0)
    n = 4000
    z1 = rng.integers(0, 2, size=n)
    z2 = rng.integers(0, 2, size=n)
    y = z1 ^ z2
    x = z1.copy()
    _, inter = _score_parts(x, [z1, z2], y, 2, [2, 2], 2)
    assert abs(inter) > 0.1, f"a genuine XOR interaction was corrected away: {inter:.4f}"
