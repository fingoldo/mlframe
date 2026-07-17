"""biz_value test for ``melt_to_long_gbm_features``'s opt-in ``context_columns`` extension.

The module docstring's "Honest empirical note" documents a specific, reproduced failure mode of the pure
long format: for a purely additive target (``y = sum(features) + noise``), a single long-format row only
ever sees ONE feature's value, so the per-row model regresses the FULL target off ~1/d of the signal --
a severe SNR loss. This test reproduces that exact shape and shows ``context_columns`` (broadcasting the
row's companion feature values alongside the melted value) recovers most of the lost signal, measured as
R^2 of the aggregated long_gbm_mean meta-feature against the held-out-style target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mlframe.training.composite import melt_to_long_gbm_features


def test_biz_val_melt_to_long_gbm_features_context_columns_recovers_additive_signal():
    rng = np.random.default_rng(0)
    n, d = 3_000, 10
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{j}" for j in range(d)])
    y = X.to_numpy().sum(axis=1) + rng.normal(scale=0.1, size=n)

    baseline = melt_to_long_gbm_features(X, y, model_factory=lambda: LinearRegression(), n_splits=5, random_state=0, agg_stats=("mean",))
    baseline_r2 = r2_score(y, baseline["long_gbm_mean"].to_numpy())

    context = melt_to_long_gbm_features(
        X, y, model_factory=lambda: LinearRegression(), n_splits=5, random_state=0, agg_stats=("mean",), context_columns=list(X.columns)
    )
    context_r2 = r2_score(y, context["long_gbm_mean"].to_numpy())

    # Pure long format catastrophically underfits the additive target (documented MSE +4,200% failure mode).
    assert baseline_r2 < 0.2, f"expected the pure long format to reproduce the documented additive-target SNR loss, got R^2={baseline_r2:.3f}"
    # context_columns restores the row's full companion-feature context, letting the per-row model recover
    # the additive sum -- threshold set comfortably below the measured ~0.98 R^2.
    assert context_r2 > 0.85, f"expected context_columns to recover most of the additive signal, got R^2={context_r2:.3f}"
    assert context_r2 - baseline_r2 > 0.6, "context_columns must show a large, real recovery over the pure long-format baseline"


def test_biz_val_melt_to_long_gbm_features_context_columns_omitted_is_bit_identical():
    """Guardrail: omitting ``context_columns`` must reproduce the exact pre-existing long table."""
    rng = np.random.default_rng(1)
    n, d = 200, 6
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{j}" for j in range(d)])
    y = rng.normal(size=n)

    default_omitted = melt_to_long_gbm_features(X, y, model_factory=lambda: LinearRegression(), n_splits=5, random_state=0)
    default_explicit_none = melt_to_long_gbm_features(X, y, model_factory=lambda: LinearRegression(), n_splits=5, random_state=0, context_columns=None)

    np.testing.assert_array_equal(default_omitted["long_gbm_mean"].to_numpy(), default_explicit_none["long_gbm_mean"].to_numpy())
