"""biz_value test for ``feature_selection.unanimous_permutation_prune.unanimous_permutation_prune``.

The win (2nd_ieee-cis-fraud-detection.md, and this idea's own critique -- "more conservative than
mean-importance thresholding"): a regime-dependent feature can have a NEGATIVE average permutation importance
across walk-forward CV folds (it's actively costly in most regimes) while still being genuinely, strongly
useful in at least one regime/fold. A naive mean-importance threshold (drop if mean < 0) WRONGLY discards
such a feature. The unanimous criterion only prunes when permuting IMPROVED the score in EVERY fold -- a
feature that helps in even one fold survives -- so it correctly retains this feature where the naive
mean-threshold baseline would not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_selection.unanimous_permutation_prune import unanimous_permutation_prune


def _scoring(y_true, y_pred):
    return -float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _make_regime_reversing_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    stable_feature = rng.normal(size=n)
    regime_feature = rng.normal(size=n)
    y = 2.0 * stable_feature + 0.3 * rng.standard_normal(n)

    # regime_feature is strongly useful in the EARLY minority regime but its effect REVERSES sign for the
    # majority later regime -- a single linear fit can't capture the sign flip, so in later folds permuting
    # it away actively helps (negative importance), even though it mattered a lot early on.
    early = np.arange(n) < n // 5
    late = ~early
    y[early] += 4.0 * regime_feature[early]
    y[late] -= 1.0 * regime_feature[late]

    X = pd.DataFrame({"stable_feature": stable_feature, "regime_feature": regime_feature})
    return X, y


def _naive_mean_importance_survivors(X, y, estimator_factory, cv_splits, n_repeats, random_state):
    sk_scorer = make_scorer(_scoring, greater_is_better=True)
    per_fold = []
    for train_idx, val_idx in cv_splits:
        model = estimator_factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        result = permutation_importance(model, X.iloc[val_idx], y[val_idx], scoring=sk_scorer, n_repeats=n_repeats, random_state=random_state)
        per_fold.append(result.importances_mean)
    mean_importance = np.mean(per_fold, axis=0)
    return [name for name, m in zip(X.columns, mean_importance) if m >= 0]


def test_biz_val_unanimous_prune_keeps_regime_reversing_feature_naive_mean_would_drop():
    X, y = _make_regime_reversing_dataset(n=1200, seed=1)
    cv_splits = list(TimeSeriesSplit(n_splits=5).split(X))

    naive_survivors = _naive_mean_importance_survivors(X, y, lambda: Ridge(alpha=1.0), cv_splits, n_repeats=10, random_state=0)
    unanimous_survivors = unanimous_permutation_prune(X, y, lambda: Ridge(alpha=1.0), cv_splits, n_repeats=10, max_iterations=1, random_state=0)

    assert "regime_feature" not in naive_survivors, f"expected the naive mean-importance baseline to wrongly drop the regime-reversing feature (negative average importance), got {naive_survivors}"
    assert "regime_feature" in unanimous_survivors, f"expected unanimous pruning to correctly RETAIN the regime-reversing feature (genuinely useful in at least one fold), got {unanimous_survivors}"
    assert "stable_feature" in unanimous_survivors and "stable_feature" in naive_survivors


def _make_mostly_noise_one_spurious_fold_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    real_feature = rng.normal(size=n)
    noise_feature = rng.normal(size=n)
    y = 2.0 * real_feature + 0.15 * rng.standard_normal(n)
    y = y.copy()

    # noise_feature carries a spurious relationship to y ONLY in the earliest chunk (indices [0, n//6)),
    # which is part of every fold's TRAIN set under TimeSeriesSplit -- so Ridge learns a small nonzero
    # coefficient for it everywhere, and in every fold whose VALIDATION window lacks that relationship,
    # permuting it away correctly IMPROVES held-out RMSE (negative importance, unanimity-eligible for
    # pruning). Injecting the SAME relationship into the third fold's own validation window (indices
    # [3*n//6, 4*n//6)) makes the learned coefficient briefly correct THERE, so permuting it away hurts
    # (positive importance) -- one spurious fold blocking strict unanimity even though the feature is
    # genuinely noise everywhere else.
    train_leak = slice(0, n // 6)
    spurious_val_fold = slice(3 * n // 6, 4 * n // 6)
    y[train_leak] += 4.0 * noise_feature[train_leak]
    y[spurious_val_fold] += 4.0 * noise_feature[spurious_val_fold]

    X = pd.DataFrame({"real_feature": real_feature, "noise_feature": noise_feature})
    return X, y


def test_biz_val_unanimous_prune_kof_n_threshold_prunes_feature_one_spurious_fold_blocks_unanimity():
    X, y = _make_mostly_noise_one_spurious_fold_dataset(n=1500, seed=3)
    cv_splits = list(TimeSeriesSplit(n_splits=5).split(X))

    strict_survivors = unanimous_permutation_prune(X, y, lambda: Ridge(alpha=1.0), cv_splits, n_repeats=10, max_iterations=1, random_state=0)
    kofn_survivors = unanimous_permutation_prune(
        X, y, lambda: Ridge(alpha=1.0), cv_splits, n_repeats=10, max_iterations=1, random_state=0, min_fold_agreement_fraction=0.6
    )

    assert "noise_feature" in strict_survivors, f"expected strict unanimity to be blocked by the one spurious fold, got {strict_survivors}"
    assert "noise_feature" not in kofn_survivors, f"expected the k-of-n threshold (0.6) to prune the noise feature despite the spurious fold, got {kofn_survivors}"
    assert "real_feature" in kofn_survivors and "real_feature" in strict_survivors
    assert len(kofn_survivors) < len(strict_survivors)

    # default (omitted min_fold_agreement_fraction) must remain bit-identical to explicit fraction=None / unchanged strict behavior
    default_survivors = unanimous_permutation_prune(X, y, lambda: Ridge(alpha=1.0), cv_splits, n_repeats=10, max_iterations=1, random_state=0)
    assert default_survivors == strict_survivors


def test_unanimous_prune_keeps_all_when_no_feature_unanimously_hurts():
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = X["a"].to_numpy() * 2.0 + X["b"].to_numpy() * 2.0 + 0.1 * rng.standard_normal(n)
    cv_splits = list(TimeSeriesSplit(n_splits=3).split(X))

    surviving = unanimous_permutation_prune(X, y, lambda: Ridge(alpha=1.0), cv_splits, n_repeats=5, random_state=1)
    assert set(surviving) == {"a", "b"}
