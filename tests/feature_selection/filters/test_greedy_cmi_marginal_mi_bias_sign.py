"""The marginal binned MI of independent columns is corrected toward zero, not inflated (tracker 2026-11-02 wave 11, Miller-Madow sign, second family)."""

import numpy as np

from mlframe.feature_selection.filters._fe_synergy_screen import _combo_mm_mi_njit
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned


def test_marginal_mi_of_independent_columns_is_near_zero():
    """20x20 bins at n=2000: the plug-in bias is ~0.09 nats; the old sign added it again (mean ~0.18)."""
    rng = np.random.default_rng(0)
    vals = [_cmi_from_binned(rng.integers(0, 20, 2000), rng.integers(0, 20, 2000), None) for _ in range(30)]
    assert np.mean(vals) < 0.03


def test_combo_mm_mi_of_independent_codes_is_near_zero():
    """One 3-level combined code against an independent 2-class target stays below a small bound."""
    rng = np.random.default_rng(1)
    n = 3000
    vals = []
    for _ in range(20):
        combo = rng.integers(0, 60, (n, 1)).astype(np.int64)
        t = rng.integers(0, 2, n).astype(np.int64)
        vals.append(_combo_mm_mi_njit(combo, np.array([60], dtype=np.int64), t, 2, 60))
    assert np.mean(vals) < 0.02
