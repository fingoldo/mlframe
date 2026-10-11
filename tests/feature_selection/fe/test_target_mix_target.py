"""The mix-offset target is built from the training split only and equals ``y - beta * u`` on every row."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mlframe.feature_selection.filters._target_mix_target import build_mix_offset_target
from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(seed, n=2000):
    """A frame whose target carries a strong linear term in ``u`` plus a product of two other columns."""
    rng = np.random.default_rng(seed)
    u, x2, x3, nz = (rng.standard_normal(n) for _ in range(4))
    y = 3.0 * u + x2 * x3 + 0.1 * rng.standard_normal(n)
    return pd.DataFrame({"u": u, "x2": x2, "x3": x3, "nz": nz}), y


def test_accepted_base_gives_the_offset_target_on_every_row():
    """When a base is accepted, the values equal ``y - beta * u`` on all rows, with the slope close to the true 3."""
    X, y = _frame(0)
    train_idx = np.arange(1500)
    out = build_mix_offset_target(X, y, train_idx, selector=MRMR(verbose=0, n_jobs=1, random_seed=0, fe_target_mix_enable=True, fe_target_mix_max_bases=1, fe_max_steps=1), gate_regressor=HistGradientBoostingRegressor(max_iter=100, random_state=0))
    assert out is not None
    assert abs(out["beta"] - 3.0) < 0.3
    assert out["values"].shape == y.shape
    from mlframe.feature_selection.filters._target_offset_regressor import _replay_base

    assert np.allclose(out["values"], y - out["beta"] * _replay_base(out["base"], X))


def test_pure_noise_gives_no_target():
    """No base clears the gate on pure noise, so the builder returns None and the suite keeps the original target."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.standard_normal(1500), "b": rng.standard_normal(1500)})
    y = rng.standard_normal(1500)
    out = build_mix_offset_target(X, y, np.arange(1000), selector=MRMR(verbose=0, n_jobs=1, random_seed=1, fe_target_mix_enable=True, fe_target_mix_max_bases=1, fe_max_steps=1), gate_regressor=HistGradientBoostingRegressor(max_iter=50, random_state=0))
    assert out is None
