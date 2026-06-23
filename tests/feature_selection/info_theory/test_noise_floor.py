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


# --------------------------------------------------------------------- SA3: noise-envelope estimated from too few permutations
def _envelope_max_excess(n_perm: int, seed: int, pct: float = 95.0, G: int = 6) -> float:
    """The per-grid-point noise envelope is ``percentile(perm[:,j]-perm[:,i], pct)``. Reproduce its seed-variance directly:
    with a tiny n_perm the ``pct`` percentile is an extreme order statistic (n_perm=3 -> MAX of 3 draws) and swings wildly.
    """
    rng = np.random.default_rng(seed)
    perm = 0.5 + 0.02 * rng.standard_normal((n_perm, G))
    return max(float(np.percentile(perm[:, j] - perm[:, 0], pct)) for j in range(1, G))


def test_noise_floor_envelope_stable_at_default_unstable_at_three():
    """SA3: at the OLD default n_perm=3 the 95th-percentile envelope is a high-variance sample maximum; at the raised
    default it is a low-variance interior order statistic. Pin that the envelope std shrinks markedly with more perms."""
    env3 = np.array([_envelope_max_excess(3, s) for s in range(24)])
    env50 = np.array([_envelope_max_excess(50, s) for s in range(24)])
    assert env50.std() < 0.5 * env3.std(), (
        f"noise envelope must be more stable at n_perm=50 than n_perm=3: std3={env3.std():.4f} std50={env50.std():.4f}"
    )


def test_select_features_noise_floor_default_nperm_is_defensible():
    """The default n_perm must be large enough that the 95th percentile is not a sample maximum (>=20 for pct=95)."""
    import inspect
    sig = inspect.signature(select_features_noise_floor)
    default_n_perm = sig.parameters["n_perm"].default
    assert default_n_perm >= 20, f"default n_perm={default_n_perm} is too small for a 95th-percentile envelope"


def test_select_features_noise_floor_warns_below_floor(caplog):
    """A too-small n_perm must emit the order-statistic warning so callers aren't silently handed an unstable floor."""
    import logging
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_normal((n, 2))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(z @ np.array([1.8, -1.5]))))).astype(int)
    X = pd.DataFrame({"a": z[:, 0], "b": z[:, 1], "c": rng.standard_normal(n)})
    with caplog.at_level(logging.WARNING):
        select_features_noise_floor(
            lambda: LogisticRegression(max_iter=300), X, pd.Series(y), ["a", "b", "c"],
            n_grid=[1, 2, 3], cv=3, n_perm=3, random_state=0,
        )
    assert any("noise floor" in r.message.lower() or "percentile" in r.message.lower() for r in caplog.records), (
        "expected a warning that n_perm=3 is too small for the 95th-percentile noise floor"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--no-cov", "-p", "no:randomly", "-p", "no:cacheprovider"]))
