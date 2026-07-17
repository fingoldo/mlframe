"""biz_value: the cross-target ensemble OOF weight-estimation subsample preserves the blend weights.

The honest-OOF stacking refit (K folds x ~dozen components on millions of rows) was the prod 4.5h hog.
The NNLS / dummy-floor weights it estimates saturate far below millions of rows, so a group-aware
subsample (whole groups kept) must give an ensemble RMSE within noise of the full-data weights while
cutting the refit wall. Pins: (a) the subsample keeps WHOLE groups, (b) ensemble RMSE from subsample
weights matches full within ~1%, (c) the subsample is strictly smaller (the speed lever).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Ridge

from mlframe.training.composite import compute_oof_holdout_predictions
from mlframe.training.core._phase_composite_post_xt_ensemble import (
    _oof_subsample_positions,
    _slice_frame_rows,
)


def _grouped(n=30_000, n_groups=20, seed=0):
    rng = np.random.default_rng(seed)
    levels = rng.uniform(-3.0, 3.0, n_groups)
    groups = rng.integers(0, n_groups, size=n).astype(np.int64)
    f0 = levels[groups] + rng.normal(0, 0.5, n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    X = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    y = (1.5 * f0 + 0.7 * f1 - 0.4 * f2 + rng.normal(0, 0.3, n)).astype(np.float64)
    return X, y, groups


def _oof(X, y, groups, kfold=5):
    models = [Ridge(alpha=a).fit(X, y) for a in (0.1, 1.0, 10.0, 100.0)]
    names = [f"c{i}" for i in range(len(models))]
    t0 = time.perf_counter()
    oof, _h, _s = compute_oof_holdout_predictions(
        component_models=models,
        component_names=names,
        component_specs=[None] * len(models),
        train_X=X,
        y_train_full=y,
        base_train_full_per_spec={},
        holdout_frac=0.2,
        random_state=42,
        kfold=kfold,
        group_ids=groups,
    )
    return oof, time.perf_counter() - t0


def _nnls_w(oof, y):
    fin = np.isfinite(oof).all(axis=1) & np.isfinite(y)
    w, _ = nnls(oof[fin], y[fin])
    s = w.sum()
    return (w / s) if s > 0 else w


def test_biz_val_oof_subsample_keeps_whole_groups():
    _X, _y, groups = _grouped()
    pos = _oof_subsample_positions(groups.size, groups, cap=6_000, seed=42)
    assert pos is not None and 0 < pos.size < groups.size
    kept_groups = set(groups[pos].tolist())
    # WHOLE groups: every row of a kept group is present (no partial group).
    for g in kept_groups:
        assert int(np.count_nonzero(groups[pos] == g)) == int(np.count_nonzero(groups == g)), f"group {g} only partially kept -- group-aware subsample broken"


def test_biz_val_oof_subsample_weights_match_full():
    X, y, groups = _grouped()
    oof_full, _wall_full = _oof(X, y, groups)
    w_full = _nnls_w(oof_full, y)

    pos = _oof_subsample_positions(groups.size, groups, cap=6_000, seed=42)
    Xs = _slice_frame_rows(X, pos)
    oof_sub, _wall_sub = _oof(Xs, y[pos], groups[pos])
    w_sub = _nnls_w(oof_sub, y[pos])

    fin = np.isfinite(oof_full).all(axis=1) & np.isfinite(y)
    rmse_full = float(np.sqrt(np.mean((oof_full[fin] @ w_full - y[fin]) ** 2)))
    rmse_sub = float(np.sqrt(np.mean((oof_full[fin] @ w_sub - y[fin]) ** 2)))
    # The subsample weights, applied to the FULL OOF surface, must be within ~1% of the full weights'
    # RMSE -- the components are near-interchangeable so the weight VECTORS may differ, but the blended
    # surface they produce must not (RMSE-equivalence is the contract, not weight-identity).
    assert rmse_sub <= rmse_full * 1.01, f"subsample weights regress RMSE: {rmse_sub} vs {rmse_full}"
    assert pos.size < groups.size  # the speed lever: fewer refit rows


def test_biz_val_oof_subsample_disabled_returns_none():
    _X, _y, groups = _grouped(n=5_000)
    assert _oof_subsample_positions(groups.size, groups, cap=0, seed=42) is None  # disabled
    assert _oof_subsample_positions(groups.size, groups, cap=10_000, seed=42) is None  # n <= cap
