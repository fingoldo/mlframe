"""biz_value test for ``models.additive_interaction_diagnostic``.

The win: on a purely additive dataset (no true feature interactions), the ``num_leaves=2`` additive-only
model should nearly match the full model's CV score (high ``additive_signal_ratio``, no interaction
engineering recommended). On a purely interaction-driven dataset (target = product of two features, zero
marginal/additive signal), the additive model should score far worse than the full model (low/negative
ratio, interaction engineering correctly recommended) -- the diagnostic must actually distinguish the two
regimes, not just report a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.models.additive_interaction_diagnostic import additive_interaction_diagnostic


def test_biz_val_additive_interaction_diagnostic_distinguishes_additive_from_interaction_signal():
    """Biz val additive interaction diagnostic distinguishes additive from interaction signal."""
    rng = np.random.default_rng(0)
    n = 3000

    X_additive = rng.normal(0, 1, (n, 4))
    y_additive = X_additive[:, 0] * 2 + np.sin(X_additive[:, 1] * 2) + X_additive[:, 2] ** 2 + rng.normal(0, 0.2, n)

    X_interaction = rng.normal(0, 1, (n, 4))
    y_interaction = X_interaction[:, 0] * X_interaction[:, 1] * 3 + rng.normal(0, 0.2, n)

    splits = list(KFold(5, shuffle=True, random_state=0).split(X_additive))

    result_additive = additive_interaction_diagnostic(X_additive, y_additive, splits, metric_fn=r2_score, objective="regression")
    result_interaction = additive_interaction_diagnostic(X_interaction, y_interaction, splits, metric_fn=r2_score, objective="regression")

    assert result_additive["additive_signal_ratio"] > 0.9, result_additive
    assert result_additive["recommend_interaction_engineering"] is False

    assert result_interaction["additive_signal_ratio"] < 0.5, result_interaction
    assert result_interaction["recommend_interaction_engineering"] is True

    assert result_additive["additive_signal_ratio"] > result_interaction["additive_signal_ratio"]


def test_additive_interaction_diagnostic_pandas_series_y_with_nondefault_index_matches_ndarray():
    """MODELS-9 (2026-08-05 audit): _cv_score indexed y[train_idx]/y[test_idx] directly; for a pandas
    Series y with a non-default index (the common real-world case -- e.g. a Series carried over from an
    upstream .sample()/.sort_values() call), this performs LABEL-based indexing instead of positional,
    silently misaligning train/test rows against cv_splits' positional index arrays. Pins that passing y
    as a pandas Series with a shuffled (non-default) index gives the SAME result as the equivalent plain
    ndarray."""
    rng = np.random.default_rng(1)
    n = 800
    X = rng.normal(0, 1, (n, 3))
    y_arr = X[:, 0] * 2 + rng.normal(0, 0.2, n)
    splits = list(KFold(4, shuffle=True, random_state=0).split(X))

    shuffled_index = rng.permutation(n)
    y_series = pd.Series(y_arr, index=shuffled_index)

    result_ndarray = additive_interaction_diagnostic(X, y_arr, splits, metric_fn=r2_score, objective="regression")
    result_series = additive_interaction_diagnostic(X, y_series, splits, metric_fn=r2_score, objective="regression")

    assert result_series["full_model_cv_score"] == pytest.approx(result_ndarray["full_model_cv_score"], abs=1e-9)
    assert result_series["additive_model_cv_score"] == pytest.approx(result_ndarray["additive_model_cv_score"], abs=1e-9)
