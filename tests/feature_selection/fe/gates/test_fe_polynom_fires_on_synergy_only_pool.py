"""Regression guard for the smart-polynom search firing on an interaction-only
target (2026-06-03 wave-9 follow-up; pins the fix for default_filtering.py:165).

On a pure-interaction target (each operand marginally ~0 MI, the PAIR strong)
marginal screening keeps 0-1 features, so every prospective FE pair carries a
synergy-bootstrap operand. The smart-polynom optimiser excludes speculative
synergy pairs (so Optuna can't fit a high-MI cell to a noise-operand pair) --
but when the selected pool is too small to form ANY non-synergy pair, that
exclusion withheld EVERY pair and the polynom search never fired (0 hermite
features), even though the interaction signal is genuine. The fix applies the
exclusion only when it leaves a non-empty pool. This test asserts the polynom
search produces a hermite_pair recipe in that regime (pre-fix: 0).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _interaction_only_frame(n=3000, seed=42):
    """Interaction only frame."""
    rng = np.random.default_rng(seed)
    cols = {k: rng.normal(size=n) for k in ["x_a", "x_b", "x_c", "x_d", "x_e", "x_f", "noise1", "noise2"]}
    yc = 1.0 * cols["x_a"] * cols["x_b"] + 0.5 * cols["x_c"] * cols["x_d"] + 0.3 * cols["x_e"] * cols["x_f"] + 0.2 * rng.normal(size=n)
    y = pd.Series((yc > np.median(yc)).astype(np.int64))
    return pd.DataFrame(cols), y


def test_smart_polynom_fires_when_pool_is_all_synergy():
    """Smart polynom fires when pool is all synergy."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = _interaction_only_frame()
    m = MRMR(
        verbose=0,
        random_seed=42,
        fe_max_steps=1,
        fe_smart_polynom_iters=2,
        fe_smart_polynom_optimization_steps=30,
    ).fit(df, y)

    recipes = getattr(m, "_engineered_recipes_", []) or []
    hermite_pairs = [r for r in recipes if getattr(r, "kind", None) == "hermite_pair"]
    hermite_feats = getattr(m, "_hermite_features_", []) or []
    evaluated = max(len(hermite_pairs), len(hermite_feats))
    assert evaluated >= 1, (
        "smart-polynom search produced 0 hermite features on a pure-interaction "
        "target. The synergy-exclusion withheld every pair because the selected "
        "pool was too small to form a non-synergy pair (default_filtering.py:165 "
        f"regression). recipes kinds={sorted(set(getattr(r, 'kind', '?') for r in recipes))}"
    )
