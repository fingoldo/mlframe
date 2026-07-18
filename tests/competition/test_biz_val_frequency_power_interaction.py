"""biz_value test for ``mlframe.competition.frequency_power_interaction``.

Builds a Santander-style synthetic dataset where a feature's per-value occurrence
count carries target signal beyond what the raw feature value alone (or the raw
count alone) exposes to a linear model: the true generating process depends on
``scaled_X ** clip(count, 1, 3)``, a genuinely multiplicative interaction that a
linear-in-[X, count] model cannot represent but a linear-in-[X, count, interaction]
model can. Proves the power-interaction feature adds real downstream predictive
signal (OOF AUC) over raw feature + plain count feature alone.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.competition.frequency_power_interaction import frequency_power_interaction


def _make_dataset(n_unique: int, n_pairs: int, seed: int):
    """Santander-style pattern: some feature values are unique (count=1), others are
    exact duplicates occurring exactly twice (count=2). The generating process routes
    through ``scaled_X ** clip(count, 1, 3)``: count=1 rows get a (roughly) linear-in-X
    effect, count=2 rows get a sign-symmetric quadratic-in-X effect. A model linear in
    [X, count] alone cannot represent this count-dependent nonlinearity (it has no X^2
    term); only the interaction feature exposes it directly.
    """
    rng = np.random.default_rng(seed)
    x_unique = rng.uniform(-3.0, 3.0, size=n_unique)
    x_pair_base = rng.uniform(-3.0, 3.0, size=n_pairs)
    x = np.concatenate([x_unique, x_pair_base, x_pair_base])
    rng.shuffle(x)

    result = frequency_power_interaction(x, feature_range=(-4.0, 4.0), count_clip_range=(1.0, 3.0))

    z = 1.0 * result.interaction_feature + rng.normal(scale=0.3, size=x.size)
    p = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, p)
    return x, result.counts.astype(np.float64), result.interaction_feature, y


def _oof_auc(features: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Return the 5-fold stratified out-of-fold AUC of a standardized logistic regression on features."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    oof_pred = cross_val_predict(clf, features, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, oof_pred))


def test_biz_val_frequency_power_interaction_beats_raw_plus_count():
    """Power-interaction feature adds >=0.06 absolute OOF AUC over raw feature + plain count alone."""
    x, counts, interaction, y = _make_dataset(n_unique=3000, n_pairs=1500, seed=0)

    baseline_features = np.column_stack([x, counts])
    interaction_features = np.column_stack([x, counts, interaction])

    baseline_auc = _oof_auc(baseline_features, y, seed=0)
    interaction_auc = _oof_auc(interaction_features, y, seed=0)

    assert (
        interaction_auc > baseline_auc
    ), f"expected power-interaction feature to improve OOF AUC over raw+count alone, baseline_auc={baseline_auc:.4f}, interaction_auc={interaction_auc:.4f}"
    improvement = interaction_auc - baseline_auc
    # measured ~0.10 absolute AUC improvement on this fixture; threshold set well below that
    assert improvement >= 0.06, f"expected >=0.06 absolute AUC improvement, got {improvement:.4f}"


def test_biz_val_frequency_power_interaction_clip_bounds_exponent():
    """Clipped counts stay within count_clip_range, scaled feature within feature_range, and interaction is scaled**clipped_count."""
    x, _counts, _interaction, _ = _make_dataset(n_unique=1000, n_pairs=500, seed=1)
    result = frequency_power_interaction(x, feature_range=(-4.0, 4.0), count_clip_range=(1.0, 3.0))
    assert np.all(result.clipped_counts >= 1.0)
    assert np.all(result.clipped_counts <= 3.0)
    # scaled feature must land within the requested MinMax range
    assert result.scaled_feature.min() >= -4.0 - 1e-9
    assert result.scaled_feature.max() <= 4.0 + 1e-9
    # interaction feature is exactly scaled ** clipped_count, not count ** scaled
    expected = np.power(result.scaled_feature, result.clipped_counts)
    assert np.allclose(result.interaction_feature, expected)
