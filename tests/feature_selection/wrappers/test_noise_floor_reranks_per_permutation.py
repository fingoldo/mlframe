"""The permuted-y null curve must re-rank the features on the shuffled labels, as the real ranking was chosen on the real labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from mlframe.feature_selection.wrappers._noise_floor import select_features_noise_floor


def _noise_bed():
    """Pure-noise features ranked by their correlation with a random binary label (selection on the labels)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 30)), columns=[f"f{i}" for i in range(30)])
    y = pd.Series((rng.random(160) < 0.5).astype(int))
    corr = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]))
    return X, y, list(corr.sort_values(ascending=False).index)


def _run(rerank):
    """Noise-floor cut of the correlation ranking with the given rerank mode."""
    X, y, ranking = _noise_bed()
    return select_features_noise_floor(
        lambda: RandomForestClassifier(n_estimators=15, max_depth=3, random_state=0, n_jobs=1),
        X, y, ranking, n_grid=[1, 3, 6, 12], cv=3, n_perm=20, rerank=rerank,
    )


def test_rerank_raises_the_null_curve_by_the_selection_optimism():
    """Opt-in re-ranked shuffles score above chance because their features were picked on the shuffled labels; the fixed ranking stays near 0.5."""
    fixed = _run("none")
    reranked = _run("permutation")
    assert float(np.mean(fixed["perm_mean"])) < 0.53
    assert float(np.mean(reranked["perm_mean"])) > float(np.mean(fixed["perm_mean"])) + 0.02


def test_unknown_rerank_mode_is_rejected():
    """A misspelled mode raises instead of silently running one of the two."""
    X, y, ranking = _noise_bed()
    with pytest.raises(ValueError, match="rerank"):
        select_features_noise_floor(lambda: RandomForestClassifier(n_estimators=5), X, y, ranking, rerank="sometimes")
