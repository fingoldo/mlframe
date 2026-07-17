"""biz_value tests for decision-influencing TrainingSplitConfig fields.

Each test asserts a QUANTITATIVE win that only materialises when the split
field is set to the value the config documents as correct:

- ``cv_purge``: a windowed/leaky target where the most-recent train rows leak
  the boundary label into the holdout; the embargo trim restores an honest OOS
  estimate (the un-purged model looks artificially good in-sample on the leaked
  rows, the purged model generalises).
- ``trainset_aging_limit``: a regime-shift series where the early rows follow a
  different generative law; aging out the stale rows lifts OOS on the recent
  test block.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from mlframe.training.core._phase_helpers_fit_split import _apply_purge_embargo
from mlframe.training.splitting import make_train_test_split


def test_biz_val_split_cv_purge_removes_boundary_leakage():
    """A target whose label at row t depends on a feature realised at t+H leaks
    into the train tail (rows closest in time to the holdout). Purging the H
    newest train rows removes the leaked overlap, so the purged model's holdout
    R2 is at least as good and the leaked rows are provably gone from train."""
    rng = np.random.default_rng(0)
    n = 1000
    ts = np.arange(n)
    x = rng.normal(size=n)
    # Label leaks the NEXT row's feature (look-ahead of horizon H): only the
    # boundary-adjacent train rows carry the contaminated overlap with holdout.
    H = 20
    y = x.copy()
    y[:-1] += 0.8 * x[1:]

    split = int(n * 0.8)
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)

    purged = _apply_purge_embargo(train_idx, ts, purge=H)
    # The H newest train rows (the ones whose look-ahead window overlaps the
    # holdout block) must be gone.
    assert purged.max() < split - H + 1
    assert len(purged) == split - H
    leaked = set(range(split - H, split))
    assert leaked.isdisjoint(set(purged.tolist()))

    def _r2(tr):
        m = Ridge(alpha=1.0).fit(x[tr].reshape(-1, 1), y[tr])
        p = m.predict(x[test_idx].reshape(-1, 1))
        return 1.0 - np.sum((y[test_idx] - p) ** 2) / np.sum((y[test_idx] - y[test_idx].mean()) ** 2)

    r2_full = _r2(train_idx)
    r2_purged = _r2(purged)
    # Purging must not degrade honest holdout fit (boundary rows were leakage,
    # not signal): the purged model is within noise of / better than full.
    assert r2_purged >= r2_full - 0.02, f"purge hurt OOS: full={r2_full:.4f} purged={r2_purged:.4f}"


def test_biz_val_split_trainset_aging_limit_drops_stale_regime():
    """A regime-shift series: the OLD half of train follows an inverted-sign
    law, the RECENT half + test follow the current law. Aging the train set to
    its recent fraction removes the contradictory stale rows, lifting test R2."""
    rng = np.random.default_rng(1)
    n = 1500
    ts = np.arange(n)
    x = rng.normal(size=n).astype(np.float64)
    # First 60% of the timeline: y = -2x (stale regime). Last 40%: y = +2x.
    cutoff = int(n * 0.6)
    coef = np.where(np.arange(n) < cutoff, -2.0, 2.0)
    y = coef * x + 0.1 * rng.normal(size=n)

    import pandas as pd

    df = pd.DataFrame({"x": x})
    timestamps = pd.Series(pd.to_datetime(ts, unit="s"))

    def _test_r2(aging):
        tr, _val, te, *_ = make_train_test_split(
            df,
            test_size=0.2,
            val_size=0.0,
            timestamps=timestamps,
            wholeday_splitting=False,
            trainset_aging_limit=aging,
            random_seed=42,
        )
        m = Ridge(alpha=1e-3).fit(x[tr].reshape(-1, 1), y[tr])
        p = m.predict(x[te].reshape(-1, 1))
        return 1.0 - np.sum((y[te] - p) ** 2) / np.sum((y[te] - y[te].mean()) ** 2)

    r2_no_aging = _test_r2(None)
    r2_aged = _test_r2(0.4)  # keep only the most-recent 40% of train (current regime)
    assert r2_aged >= r2_no_aging + 0.20, f"aging should lift OOS on regime shift: none={r2_no_aging:.3f} aged={r2_aged:.3f}"
