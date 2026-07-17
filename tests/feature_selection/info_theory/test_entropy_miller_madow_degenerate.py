"""Wave 9.1 loop-iter-20 regression: ``entropy_miller_madow`` must
satisfy H >= 0 on degenerate distributions.

Pre-fix at ``info_theory.py:113``: the function unconditionally added
``(k - 1) / (2 * n_samples)`` where ``k`` is the number of non-empty
bins (count AFTER the ``freqs[freqs > 0]`` filter). On degenerate
inputs (empty array, all-zero freqs, single-bin) ``k`` was 0 or 1,
giving correction terms ``-1/(2n)`` or ``0``. Negative entropy VIOLATES
the H >= 0 invariant and propagates through ``conditional_mi``::

    I(X; Y | Z) = H(XZ) + H(YZ) - H(Z) - H(XYZ)

When any of the four entropies hits a degenerate-Z slice (no support
after MDLP filtering, all-zero joint cell, etc.) and Miller-Madow is
active, ``conditional_mi`` goes NEGATIVE - which then propagates into
``expected_gains`` ranking, falsely boosting features near degenerate
conditioning.

Severity: medium-high (opt-in path, but documented Miller-Madow correctness
guarantee broken when active). Wave 8 PID composition consumes these
values without clamping.

Fix at info_theory.py:113: return ``h_plugin`` directly when k <= 1.
Plug-in entropy is exact (= 0) for deterministic distributions, so the
Miller-Madow bias correction is undefined / unnecessary in that regime.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_miller_madow_empty_freqs_returns_zero():
    """Empty freqs (no non-empty bins) must give H = 0, not -1/(2n)."""
    from mlframe.feature_selection.filters.info_theory import entropy_miller_madow

    h = entropy_miller_madow(np.array([], dtype=np.float64), n_samples=100)
    assert h >= 0.0, f"empty freqs gave H={h}, must be >= 0"
    assert abs(h) < 1e-12


def test_miller_madow_all_zero_freqs_returns_zero():
    """All-zero freqs (every count filtered out) must give H = 0."""
    from mlframe.feature_selection.filters.info_theory import entropy_miller_madow

    h = entropy_miller_madow(np.array([0.0, 0.0, 0.0]), n_samples=100)
    assert h >= 0.0


def test_miller_madow_single_bin_returns_zero():
    """Single populated bin (deterministic distribution): H = 0 exactly.
    The Miller-Madow correction adds (k-1)/(2n) = 0 here naturally,
    but the explicit guard keeps it safely 0 even if k=1 logic
    diverged in numba.
    """
    from mlframe.feature_selection.filters.info_theory import entropy_miller_madow

    h = entropy_miller_madow(np.array([1.0]), n_samples=100)
    assert h >= 0.0
    assert abs(h) < 1e-12


def test_miller_madow_valid_distribution_unchanged():
    """Negative control: a valid 2-bin distribution should give the
    plug-in entropy PLUS Miller-Madow correction (k=2 -> +1/(2n)).
    The fix must not alter behaviour on the multi-bin path.
    """
    from mlframe.feature_selection.filters.info_theory import entropy_miller_madow

    n = 100
    # H_plugin for [0.5, 0.5] -> 0.5 ln(2) + 0.5 ln(2) = ln(2) = 0.693
    # MM correction = (2-1) / (2*100) = 0.005
    # NOTE: freqs are RAW COUNTS not probabilities in this estimator's
    # internal expectations -- the plug-in branch is -sum(log(p) * p).
    # For probabilities summing to 1 it's the entropy in nats.
    h = entropy_miller_madow(np.array([0.5, 0.5]), n_samples=n)
    # Plug-in = ln(2) = 0.6931; MM = 0.6931 + 1/(200) = 0.6981
    assert h > 0.69
    assert h < 0.70


def test_miller_madow_three_bin_uniform():
    """Three-bin uniform: H_plugin = ln(3); MM adds (3-1)/(2*100) = 0.01."""
    from mlframe.feature_selection.filters.info_theory import entropy_miller_madow

    n = 100
    third = 1.0 / 3.0
    h = entropy_miller_madow(np.array([third, third, third]), n_samples=n)
    # ln(3) ~= 1.0986
    assert h > 1.10
    assert h < 1.12
