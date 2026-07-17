"""biz_value test for ``forward_select``'s opt-in ``patience`` early-stop mechanism.

The win: on a candidate pool where only the first K columns carry real signal and the rest are pure noise,
exhaustive forward selection (the default) keeps paying for a CV fit per remaining noise candidate every
round until the pool is exhausted, even though nothing past K can ever improve the score. ``patience`` stops
the loop once the best remaining candidate's improvement is statistically indistinguishable from noise for
``patience`` consecutive rounds, avoiding that wasted compute -- without changing the default (patience=None)
behavior at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.feature_selection.forward_select import forward_select


def _make_dataset(n: int, n_signal: int, n_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    signal_cols = {f"signal{i}": y * (1.0 / (i + 1)) + rng.normal(scale=0.3, size=n) for i in range(n_signal)}
    noise_cols = {f"noise{i}": rng.normal(size=n) for i in range(n_noise)}
    X = pd.DataFrame({**signal_cols, **noise_cols})
    return X, y


def test_biz_val_forward_select_patience_stops_near_signal_boundary_avoiding_noise_evaluation():
    n_signal, n_noise = 4, 40
    X, y = _make_dataset(n=600, n_signal=n_signal, n_noise=n_noise, seed=0)

    selected_early, report = forward_select(
        X,
        y,
        lambda: Ridge(alpha=1.0),
        scoring="neg_mean_squared_error",
        cv=5,
        patience=3,
        significance_level=0.05,
        return_report=True,
    )
    rounds_early = len(report.steps)
    pool_fraction_evaluated_early = rounds_early / (n_signal + n_noise)

    selected_exhaustive = forward_select(
        X,
        y,
        lambda: Ridge(alpha=1.0),
        scoring="neg_mean_squared_error",
        cv=5,
    )

    assert report.stopped_early, "expected the patience mechanism to trigger on the noise tail"
    assert not any(c.startswith("noise") for c in selected_early), f"expected no noise column selected before early-stop, got {selected_early}"
    # the exhaustive run must walk the full noise tail (all 44 candidates get at least one round of
    # evaluation before the pool is exhausted), while the early-stop run gives up well before that.
    assert len(selected_exhaustive) >= n_signal
    assert pool_fraction_evaluated_early <= 0.5, (
        f"expected early-stop to evaluate at most half the candidate pool, evaluated fraction={pool_fraction_evaluated_early:.2f}"
    )
    assert rounds_early < (n_signal + n_noise), f"expected early-stop rounds ({rounds_early}) to be fewer than the full pool ({n_signal + n_noise})"


def test_forward_select_patience_none_preserves_original_behavior():
    X, y = _make_dataset(n=400, n_signal=2, n_noise=6, seed=1)
    result_default = forward_select(X, y, lambda: Ridge(alpha=1.0), scoring="neg_mean_squared_error", cv=5)
    result_explicit_none = forward_select(X, y, lambda: Ridge(alpha=1.0), scoring="neg_mean_squared_error", cv=5, patience=None, return_report=False)
    assert result_default == result_explicit_none
