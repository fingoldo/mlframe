"""Tests for the permuted-y NOISE-FLOOR feature-count cut (wrappers/_noise_floor.py).

The plateau rule is PURE (operates on curve arrays), so its logic is tested deterministically with synthetic curves
-- no model fits. One small integration test exercises select_features_noise_floor end-to-end on a cheap dataset.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.wrappers import select_features_noise_floor, noise_floor_plateau


# --------------------------------------------------------------------- pure plateau-rule logic (no model fits)
def test_plateau_stops_at_signal_onset_then_flat():
    # real curve climbs to N=3 (idx 2) then is GENUINELY FLAT (no further real gain); permuted curve is ~flat noise
    # -> N* should be 3 (the onset of the flat = the plateau). Any post-N=3 wiggle that exceeds the noise envelope
    # would correctly NOT be a plateau, so the flat region must be truly flat to isolate the onset.
    n_grid = [1, 2, 3, 5, 8, 12]
    real = np.array([0.60, 0.75, 0.90, 0.90, 0.90, 0.90])
    perm = np.zeros((5, len(n_grid))) + np.linspace(0.50, 0.505, len(n_grid))  # flat ~0.5 noise
    n_star, idx, _, _ = noise_floor_plateau(n_grid, real, perm, pct=95.0)
    assert n_star == 3, f"plateau should stop where the real curve flattens (N=3), got {n_star}"


def test_plateau_keeps_climbing_curve():
    # a curve that keeps genuinely climbing past the noise envelope -> N* at (near) the largest grid point.
    n_grid = [1, 2, 5, 10, 20]
    real = np.array([0.55, 0.62, 0.70, 0.78, 0.86])           # monotone real gains
    perm = np.tile(np.linspace(0.50, 0.51, len(n_grid)), (5, 1))  # tiny noise gains
    n_star, _, _, _ = noise_floor_plateau(n_grid, real, perm, pct=95.0)
    assert n_star == 20, f"a genuinely-climbing curve should not be cut early, got {n_star}"


def test_plateau_cuts_noise_only_curve_early():
    # real curve is itself within the noise envelope from the start -> stop at the very first grid point.
    n_grid = [1, 5, 10, 50, 100]
    real = np.array([0.70, 0.705, 0.706, 0.707, 0.707])
    perm = np.tile([0.0, 0.01, 0.012, 0.013, 0.013], (6, 1)) + np.array([0.70, 0.71, 0.712, 0.713, 0.713])
    n_star, _, _, _ = noise_floor_plateau(n_grid, real, perm, pct=95.0)
    assert n_star == 1, f"a noise-tolerance plateau should be cut to the first N, got {n_star}"


# --------------------------------------------------------------------- end-to-end cut (small, cheap)
def test_select_features_noise_floor_cuts_overselected_ranking():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    n = 600
    # 3 informative features + 40 noise; ranking puts the 3 informative first then noise (an over-selected ranking).
    z = rng.standard_normal((n, 3))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(z @ np.array([1.8, -1.5, 1.2]))))).astype(int)
    import pandas as pd
    cols = {f"inf_{i}": z[:, i] for i in range(3)}
    cols.update({f"noise_{k}": rng.standard_normal(n) for k in range(40)})
    X = pd.DataFrame(cols)
    ranking = [f"inf_{i}" for i in range(3)] + [f"noise_{k}" for k in range(40)]  # informative-first

    out = select_features_noise_floor(
        lambda: LogisticRegression(max_iter=500), X, pd.Series(y), ranking,
        n_grid=[1, 2, 3, 5, 10, 20, 43], cv=3, n_perm=3, random_state=0,
    )
    assert 1 <= out["n_star"] <= 10, f"noise-floor should cut the 43-feature over-selection to a small N, got {out['n_star']}"
    assert set(out["selected"]) <= set(ranking) and out["selected"] == ranking[: out["n_star"]]
    # the informative block should survive the cut (it leads the ranking and carries the signal)
    assert {"inf_0", "inf_1"} <= set(out["selected"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--no-cov", "-p", "no:randomly", "-p", "no:cacheprovider"]))
