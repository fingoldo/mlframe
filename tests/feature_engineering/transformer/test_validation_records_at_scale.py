"""Priority 2 of the multi-seed honesty pass: do the 2 SURVIVING records
(iter68 kin8nm RFF, iter69 abalone disagreement+cdist) still hold on a
LARGER regression dataset?

Test target: California Housing (~20640 rows x 8 numeric features), which is
~2.5x the size of kin8nm (8192) and ~5x the size of abalone (4177).

The 4000-row cap is gone (already in test_validation_records.py). Each seed
gets its own KFold + train/test split. We run 3 seeds {0, 17, 42} to keep
runtime under ~30min while still letting fold-noise become visible.

Verdict rule (same as test_validation_records.py): SURVIVES iff
``median > 0 AND min > -0.3 * median``; FOLD-NOISE? otherwise.

Caveat already documented in RESULTS.md: California Housing's +rff result on
its OWN test_biz_val matrix is negative (-3.6% LGB R2), so iter68 generalising
here would be informative. iter69's cdist component is currently seeded
internally with seed=42 (in test_biz_val_real_datasets._features_cdist), so
the seed variation here exercises baseline_disagreement + train/test split,
but not the cdist neighbours themselves.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("catboost")
pytest.importorskip("xgboost")
pytest.importorskip("sklearn")

from tests.feature_engineering.transformer.test_biz_val_real_datasets import _load_california
from tests.feature_engineering.transformer.test_validation_records import (
    _build_iter68,
    _build_iter69,
    _measure_lift_seeded,
)


pytestmark = pytest.mark.fast


_SCALE_SEEDS = (0, 17, 42)


def _validate_scale(loader_fn, builder, target_model, target_metric, claimed_lift, label):
    X_full, y_full, task = loader_fn()
    print(f"\n  Dataset shape: {X_full.shape}  task: {task}")
    lifts = []
    for seed in _SCALE_SEEDS:
        try:
            lift = _measure_lift_seeded(X_full, y_full, task, builder, seed, target_model, target_metric)
        except Exception as exc:
            print(f"  [seed={seed}] ERROR: {type(exc).__name__}: {exc}")
            lift = float("nan")
        print(f"  [seed={seed}] {label} {target_model} {target_metric} lift: {lift:+.4f}")
        lifts.append(lift)
    arr = np.array([l for l in lifts if not np.isnan(l)])
    if arr.size == 0:
        print(f">>> {label}: ALL ERROR")
        return
    median = float(np.median(arr))
    iqr = float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)) if arr.size > 1 else 0.0
    lo = float(arr.min())
    hi = float(arr.max())
    survives = "SURVIVES" if median > 0 and lo > -abs(median) * 0.3 else "FOLD-NOISE?"
    print(f">>> {label} {target_model} {target_metric}: median={median:+.4f} IQR={iqr:.4f} min={lo:+.4f} max={hi:+.4f} | small-N claimed={claimed_lift:+.4f} | {survives}")


def test_scale_iter68_california_lgb_r2():
    """iter68 (multi_baseline_hard_row + RFF) was +11.42% median on kin8nm 8k. Does it hold on California 20k?"""
    _validate_scale(_load_california, _build_iter68, "lgb", "R2", 0.1142, "iter68_mbhrattn+rff_CA20k")


def test_scale_iter69_california_cb_r2():
    """iter69 (baseline_disagreement + cdist) was +2.26% median on abalone 4k. Does it hold on California 20k?"""
    _validate_scale(_load_california, _build_iter69, "cb", "R2", 0.0226, "iter69_blagreement+cdist_CA20k")
