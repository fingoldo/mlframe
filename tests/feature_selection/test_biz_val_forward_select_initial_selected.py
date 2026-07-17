"""biz_value test for ``forward_select``'s ``initial_selected`` parameter.

The win (3rd_home-credit-default-risk.md): building a 2nd-level stacker -- start with only base-model OOF
predictions, then greedily forward-select raw features one at a time by CV-delta, on top of that fixed core
(never removing the base predictions). This is distinct from plain forward selection over the raw features
alone: the meta-model needs the base predictions ALWAYS present (they carry most of the signal), with raw
features added only where they genuinely improve on top of what the base predictions already capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.forward_select import forward_select


def _make_stacking_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    true_signal = rng.normal(size=n)
    # base-model OOF predictions: a good but imperfect estimate of the true signal.
    base_pred = true_signal + rng.normal(scale=0.5, size=n)
    # a handful of raw features: one genuinely captures the RESIDUAL the base model misses, the rest are
    # pure noise.
    residual_capturing_feature = (true_signal - base_pred) + rng.normal(scale=0.05, size=n)
    noise_features = {f"noise{i}": rng.normal(size=n) for i in range(5)}

    X = pd.DataFrame({"base_pred": base_pred, "useful_raw": residual_capturing_feature, **noise_features})
    y = true_signal
    return X, y


def test_biz_val_forward_select_with_fixed_stacking_core_finds_useful_raw_feature():
    X, y = _make_stacking_dataset(n=1500, seed=0)
    raw_candidates = [c for c in X.columns if c != "base_pred"]

    selected = forward_select(
        X,
        y,
        lambda: Ridge(alpha=1.0),
        scoring="neg_mean_squared_error",
        cv=5,
        candidate_features=raw_candidates,
        initial_selected=["base_pred"],
        min_improvement=0.001,
    )

    assert selected[0] == "base_pred", f"expected the fixed core to always be first in the selected list, got {selected}"
    assert "useful_raw" in selected, f"expected the residual-capturing raw feature to be greedily selected, got {selected}"
    assert not any(c.startswith("noise") for c in selected), f"expected pure-noise raw features to be rejected, got {selected}"


def test_forward_select_initial_selected_beats_base_alone_and_none_kept():
    X, y = _make_stacking_dataset(n=1500, seed=1)
    raw_candidates = [c for c in X.columns if c != "base_pred"]

    base_alone_mse = -cross_val_score(Ridge(alpha=1.0), X[["base_pred"]], y, cv=5, scoring="neg_mean_squared_error").mean()

    selected = forward_select(
        X,
        y,
        lambda: Ridge(alpha=1.0),
        scoring="neg_mean_squared_error",
        cv=5,
        candidate_features=raw_candidates,
        initial_selected=["base_pred"],
        min_improvement=0.001,
    )
    augmented_mse = -cross_val_score(Ridge(alpha=1.0), X[selected], y, cv=5, scoring="neg_mean_squared_error").mean()

    assert augmented_mse < base_alone_mse, (
        f"expected the raw-augmented stacker to beat base predictions alone, got augmented={augmented_mse:.4f} base_alone={base_alone_mse:.4f}"
    )


def test_forward_select_initial_selected_none_preserves_original_behavior():
    X, y = _make_stacking_dataset(n=500, seed=2)
    result_default = forward_select(X, y, lambda: Ridge(alpha=1.0), scoring="neg_mean_squared_error", cv=5)
    result_explicit_none = forward_select(X, y, lambda: Ridge(alpha=1.0), scoring="neg_mean_squared_error", cv=5, initial_selected=None)
    assert result_default == result_explicit_none
