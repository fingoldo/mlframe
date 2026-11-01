"""On pure noise the family gate admits some candidate at no more than about its nominal level (tracker 2026-10-15, S-F08).

The existing gate tests check that two code paths agree, never that the gate keeps its error rate. Here the labels are independent of every
candidate, so the share of trials in which ANY candidate passes is the family-wise error rate; an uncorrected per-candidate test over 20 columns
would pass in about 64% of trials at alpha 0.05.
"""

import numpy as np

from mlframe.feature_selection.filters._family_maxt_gate import family_maxt_pass

_ALPHA = 0.05
_TRIALS = 100
_COLUMNS = 20


def _noise_trial(seed: int) -> np.ndarray:
    """One trial: 400 rows, 20 candidate columns and a binary label, all independent."""
    rng = np.random.default_rng(seed)
    return family_maxt_pass(rng.normal(size=(400, _COLUMNS)), rng.integers(0, 2, 400), n_bins=8, alpha=_ALPHA, n_perm=199, seed=seed)


def test_pure_noise_family_wise_pass_rate_stays_near_alpha():
    """Over 100 independent noise trials the family-wise pass rate is at most alpha plus Monte Carlo slack (0.12), far under the uncorrected 0.64."""
    any_pass = sum(bool(_noise_trial(s).any()) for s in range(_TRIALS))
    assert any_pass / _TRIALS <= 0.12, any_pass


def test_a_real_signal_still_passes_among_noise():
    """The same gate admits a column that carries the label, so a gate that rejected everything would not satisfy the rate test by accident."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 400)
    X = rng.normal(size=(400, _COLUMNS))
    X[:, 3] = y * 2.0 + rng.normal(scale=0.5, size=400)
    flags = family_maxt_pass(X, y, n_bins=8, alpha=_ALPHA, n_perm=199, seed=0)
    assert bool(flags[3])
    assert int(flags.sum()) <= 2
