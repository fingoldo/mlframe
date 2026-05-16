"""Tests for Fix 1: K-fold OOF predictions replace in-sample train_preds for level-1 stacking.

In-sample ``train_preds`` are produced by predicting on rows the model already saw during fit, which leaks the
training-set residual structure into any meta-learner that consumes them. The replacement is OOF predictions from
``sklearn.model_selection.cross_val_predict``: for every row, the held-out fold predicts that row, so the meta-learner
sees only out-of-fold signal. Two assertions cover the contract:

1. ``_compute_oof_preds`` returns an array materially different from in-sample preds on a model that overfits its
   training data (tree on small N).
2. ``score_ensemble(max_ensembling_level=2, ...)`` raises ValueError when any member lacks ``oof_preds`` / ``oof_probs``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _make_tree_overfitting_dataset(n: int = 120, seed: int = 0):
    """Synthetic regression dataset where a deep tree memorises noise; OOF and in-sample diverge sharply."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = X[:, 0] * 0.5 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


def test_oof_preds_differ_from_in_sample_train_preds():
    """A DecisionTreeRegressor with no depth limit memorises train; OOF preds must materially diverge from in-sample."""
    from sklearn.tree import DecisionTreeRegressor
    from mlframe.training.trainer import _compute_oof_preds

    X, y = _make_tree_overfitting_dataset(n=200, seed=42)
    # Unconstrained tree -> in-sample preds equal targets (zero residuals); OOF preds carry generalisation noise.
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X, y)
    in_sample_preds = model.predict(X)

    oof_preds, oof_probs = _compute_oof_preds(
        model=model,
        train_df=X,
        train_target=y.to_numpy(),
        is_classifier_model=False,
        n_splits=5,
        random_seed=42,
    )
    assert oof_probs is None
    assert oof_preds is not None
    assert oof_preds.shape == (len(y),)

    # In-sample tree fits perfectly -> RMSE(in_sample, y) ~ 0; OOF must diverge from in_sample by > epsilon.
    rmse_oof_vs_insample = float(np.sqrt(np.mean((oof_preds - in_sample_preds) ** 2)))
    assert rmse_oof_vs_insample > 0.1, (
        f"OOF preds collapsed to in-sample preds (rmse={rmse_oof_vs_insample:.6f}); "
        f"cross_val_predict is not actually folding."
    )


def test_score_ensemble_level2_raises_without_oof():
    """max_ensembling_level=2 requires OOF on every member; missing OOF -> immediate ValueError, no silent leak."""
    from mlframe.models.ensembling import score_ensemble

    # Two regression members WITHOUT oof_preds: only train_preds present. score_ensemble at level=2 must reject.
    rng = np.random.default_rng(0)
    n = 100
    member_a = SimpleNamespace(
        model=None,
        val_preds=rng.normal(size=n),
        val_probs=None,
        test_preds=rng.normal(size=n),
        test_probs=None,
        train_preds=rng.normal(size=n),
        train_probs=None,
        # No oof_preds / oof_probs attributes -> getattr default None -> guard fires.
    )
    member_b = SimpleNamespace(
        model=None,
        val_preds=rng.normal(size=n),
        val_probs=None,
        test_preds=rng.normal(size=n),
        test_probs=None,
        train_preds=rng.normal(size=n),
        train_probs=None,
    )

    with pytest.raises(ValueError, match="oof_preds"):
        score_ensemble(
            models_and_predictions=[member_a, member_b],
            ensemble_name="test",
            train_target=rng.normal(size=n),
            val_target=rng.normal(size=n),
            test_target=rng.normal(size=n),
            max_ensembling_level=2,
            verbose=False,
        )
