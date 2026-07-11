"""biz_value test for ``feature_selection.varying_size_top_k_subsets.varying_size_top_k_subsets``.

The win (7th_elo-merchant-category-recommendation.md): committing to a single "best" feature-count cutoff is
a model-selection decision with its own uncertainty (the true optimal cutoff isn't known in advance, and
depends on noise in the importance ranking itself). Training several models at DIFFERENT feature-count
cutoffs and averaging their predictions should be more robust than betting everything on one arbitrarily-
chosen cutoff, since each model's specific cutoff-choice error averages out across the ensemble.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.varying_size_top_k_subsets import varying_size_top_k_subsets


def _make_ranked_feature_dataset(n: int, n_signal: int, n_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n_signal) / np.sqrt(n_signal)
    X_signal = rng.normal(size=(n, n_signal))
    X_noise = rng.normal(size=(n, n_noise))
    y = X_signal @ w + rng.normal(scale=0.5, size=n)
    X = np.concatenate([X_signal, X_noise], axis=1)
    # "ranked_features" simulates a noisy importance ranking: signal features rank first on average, but
    # with some genuine ranking noise (a few noise features occasionally outrank weak signal features).
    true_importance = np.concatenate([np.abs(w) + rng.normal(scale=0.02, size=n_signal), rng.normal(scale=0.01, size=n_noise)])
    ranking = np.argsort(-true_importance)
    feature_names = [f"f{i}" for i in range(n_signal + n_noise)]
    ranked_features = [feature_names[i] for i in ranking]
    return X, y, feature_names, ranked_features


def test_biz_val_varying_size_subsets_ensemble_beats_single_arbitrary_cutoff():
    X, y, feature_names, ranked_features = _make_ranked_feature_dataset(n=600, n_signal=8, n_noise=40, seed=0)
    import pandas as pd

    df = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

    subsets = varying_size_top_k_subsets(ranked_features, sizes=[3, 8, 15, 30])
    ensemble_preds = []
    for subset in subsets:
        model = Ridge(alpha=1.0).fit(X_train[subset], y_train)
        ensemble_preds.append(model.predict(X_test[subset]))
    ensemble_pred = np.mean(ensemble_preds, axis=0)
    mse_ensemble = mean_squared_error(y_test, ensemble_pred)

    # single arbitrary cutoff: an unlucky guess of the "right" feature count, picked independently of the
    # true signal size (a realistic model-selection scenario -- the true cutoff (8) isn't known in advance).
    arbitrary_cutoff_mse = []
    for size in [3, 15, 30]:
        subset = varying_size_top_k_subsets(ranked_features, sizes=[size])[0]
        model = Ridge(alpha=1.0).fit(X_train[subset], y_train)
        arbitrary_cutoff_mse.append(mean_squared_error(y_test, model.predict(X_test[subset])))
    mean_single_cutoff_mse = float(np.mean(arbitrary_cutoff_mse))

    assert mse_ensemble < mean_single_cutoff_mse, f"expected the varying-size-subset ensemble to beat the average single-cutoff model (robust to not knowing the true optimal cutoff), got ensemble={mse_ensemble:.4f} avg_single={mean_single_cutoff_mse:.4f}"


def test_varying_size_top_k_subsets_exact_prefixes():
    ranked = ["a", "b", "c", "d", "e"]
    subsets = varying_size_top_k_subsets(ranked, sizes=[1, 3, 10])
    assert subsets == [["a"], ["a", "b", "c"], ["a", "b", "c", "d", "e"]]
