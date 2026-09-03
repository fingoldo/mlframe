"""TRAINING_LOOSE_C-1 (2026-08-05 audit): _compute_oof_preds constructed GroupKFold(n_splits=1) OUTSIDE
its try/except when a train fold has only 1 distinct group. sklearn raises ValueError at GroupKFold
CONSTRUCTION time (requires n_splits >= 2), which the surrounding try (only wrapping cross_val_predict)
never caught -- an uncaught ValueError crashed the whole per-model training call after the model had
already trained. Fixed by checking the distinct-group count before constructing the splitter and skipping
OOF gracefully (matching this function's other "not computable" branches) when fewer than 2 groups exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.trainer import _compute_oof_preds


def _make_dataset(n=120, seed=0):
    """Small synthetic regression dataset for OOF computation tests."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 0.5 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


def test_single_distinct_group_skips_oof_instead_of_crashing():
    """A train fold with only 1 distinct group must NOT crash GroupKFold at construction -- it must skip
    OOF gracefully, returning (None, None), same as the other 'not computable' branches."""
    from sklearn.tree import DecisionTreeRegressor

    X, y = _make_dataset(n=120, seed=0)
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X, y)

    # Every row belongs to the SAME group -- exactly the degenerate case that used to crash.
    group_ids = np.zeros(len(y), dtype=int)

    oof_preds, oof_probs = _compute_oof_preds(
        model=model,
        train_df=X,
        train_target=y.to_numpy(),
        is_classifier_model=False,
        n_splits=5,
        random_seed=0,
        group_ids=group_ids,
    )
    assert oof_preds is None
    assert oof_probs is None


def test_two_distinct_groups_still_computes_oof():
    """Sanity: with >= 2 distinct groups (the normal case), OOF must still compute successfully -- the
    fix must not accidentally disable group-aware OOF for the genuinely valid case."""
    from sklearn.tree import DecisionTreeRegressor

    X, y = _make_dataset(n=120, seed=1)
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X, y)

    group_ids = np.array([0] * 60 + [1] * 60)

    oof_preds, _oof_probs = _compute_oof_preds(
        model=model,
        train_df=X,
        train_target=y.to_numpy(),
        is_classifier_model=False,
        n_splits=5,
        random_seed=0,
        group_ids=group_ids,
    )
    assert oof_preds is not None
    assert oof_preds.shape == (len(y),)
