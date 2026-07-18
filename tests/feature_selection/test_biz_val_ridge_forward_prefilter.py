"""biz_value test for ``feature_selection.ridge_forward_prefilter.ridge_coefficient_prefilter``.

The win (Bojan's 1st_home-credit-default-risk.md pattern): with many raw candidate features, most of them
noise, a cheap Ridge-coefficient ranking + a small CV sweep over log-spaced pool sizes should prune down to a
SMALL fraction of the original feature count while keeping CV score close to (not meaningfully worse than)
the full-feature-set score -- "almost no CV loss" with a materially smaller downstream MRMR/RFECV candidate
pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.ridge_forward_prefilter import ridge_coefficient_prefilter


def _make_noisy_dataset(n: int, d_informative: int, d_noise: int, seed: int):
    """Make noisy dataset."""
    rng = np.random.default_rng(seed)
    X_info = rng.normal(size=(n, d_informative))
    X_noise = rng.normal(size=(n, d_noise))
    w = rng.normal(size=d_informative)
    y = X_info @ w + rng.normal(scale=0.5, size=n)
    cols = [f"info{i}" for i in range(d_informative)] + [f"noise{i}" for i in range(d_noise)]
    X = pd.DataFrame(np.concatenate([X_info, X_noise], axis=1), columns=cols)
    return X, y


def test_biz_val_ridge_prefilter_prunes_features_with_minimal_cv_loss():
    """Biz val ridge prefilter prunes features with minimal cv loss."""
    X, y = _make_noisy_dataset(n=400, d_informative=8, d_noise=792, seed=0)  # 800 total features

    full_score = float(np.mean(cross_val_score(Ridge(alpha=1.0), X.to_numpy(), y, cv=3, scoring="r2")))

    selected = ridge_coefficient_prefilter(X.to_numpy(), y, list(X.columns), cv=3, tol=0.02, alpha=1.0)
    reduction = 1.0 - len(selected) / X.shape[1]

    pruned_score = float(np.mean(cross_val_score(Ridge(alpha=1.0), X[selected].to_numpy(), y, cv=3, scoring="r2")))

    assert reduction > 0.8, f"expected >80% feature-count reduction (800 raw features, mostly noise), got {reduction:.4f} ({len(selected)} kept)"
    assert (
        pruned_score >= full_score - 0.03
    ), f"expected pruned-pool CV score close to full-feature-set score, got pruned={pruned_score:.4f} vs full={full_score:.4f}"

    n_informative_kept = sum(1 for f in selected if f.startswith("info"))
    assert n_informative_kept >= 7, f"expected nearly all 8 informative features to survive the prefilter, kept {n_informative_kept}"


def test_ridge_prefilter_smallest_pool_within_tolerance_is_selected():
    """Ridge prefilter smallest pool within tolerance is selected."""
    X, y = _make_noisy_dataset(n=300, d_informative=4, d_noise=124, seed=1)  # 128 total features
    selected_tight = ridge_coefficient_prefilter(X.to_numpy(), y, list(X.columns), cv=3, tol=0.005, alpha=1.0)
    selected_loose = ridge_coefficient_prefilter(X.to_numpy(), y, list(X.columns), cv=3, tol=0.2, alpha=1.0)
    assert len(selected_loose) <= len(
        selected_tight
    ), f"a looser tolerance should select an equal-or-smaller pool, got loose={len(selected_loose)} tight={len(selected_tight)}"


def test_ridge_prefilter_classification_returns_valid_feature_names():
    """Ridge prefilter classification returns valid feature names."""
    rng = np.random.default_rng(2)
    n, d = 300, 60
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    names = [f"f{i}" for i in range(d)]
    selected = ridge_coefficient_prefilter(X, y, names, cv=3, tol=0.05, is_classifier=True, alpha=1.0)
    assert 0 < len(selected) <= d
    assert set(selected).issubset(set(names))


def _make_collinear_pair_dataset(n: int, d_noise: int, seed: int):
    """Two near-duplicate informative features sharing one latent signal (correlated ~0.9 with each other),
    plus 6 independently-informative features and ``d_noise`` pure-noise features. A single small-alpha
    Ridge fit on standardized data splits the coefficient weight between the duplicate pair somewhat
    arbitrarily (driven by each column's own noise realization), so at a tight pool size ONE of the two
    duplicates is dropped in a meaningful fraction of random draws even though both are genuinely useful.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    dup_a = signal + rng.normal(scale=0.5, size=n)
    dup_b = signal + rng.normal(scale=0.5, size=n)
    other_info = rng.normal(size=(n, 6))
    w_other = rng.normal(size=6)
    X_noise = rng.normal(size=(n, d_noise))
    y = signal * 1.5 + other_info @ w_other + rng.normal(scale=0.5, size=n)
    X = np.concatenate([dup_a.reshape(-1, 1), dup_b.reshape(-1, 1), other_info, X_noise], axis=1)
    names = ["dup_a", "dup_b"] + [f"other{i}" for i in range(6)] + [f"noise{i}" for i in range(d_noise)]
    return pd.DataFrame(X, columns=names), y


def test_biz_val_ridge_prefilter_bootstrap_stability_recovers_dropped_collinear_feature():
    """A single noisy Ridge fit drops one half of a genuinely-useful collinear duplicate pair from a tight
    pool in a meaningful fraction of random seeds (arbitrary coefficient split under collinearity). The
    opt-in ``n_bootstrap`` stability-selection mode re-fits across bootstrap row resamples and keeps the
    duplicate if it lands in the top pool often enough, recovering the full pair reliably more often than
    the single-fit baseline.

    Measured over 40 independent seeds (n=150, 306 candidate features, pool size fixed at 4):
    single-fit retains BOTH duplicates in 65% of seeds; bootstrap stability-selection (B=150,
    threshold=0.35) retains both in 75% of seeds -- thresholds below set ~10pp/~15% relative below the
    measured values to absorb run-to-run noise while still proving the recovery effect.
    """
    n_seeds = 40
    single_both = 0
    boot_both = 0
    for seed in range(n_seeds):
        X, y = _make_collinear_pair_dataset(n=150, d_noise=300, seed=seed)
        names = list(X.columns)
        Xv = X.to_numpy()
        single = ridge_coefficient_prefilter(Xv, y, names, candidate_sizes=[4], cv=3, tol=0.003, alpha=1.0, random_state=seed)
        boot = ridge_coefficient_prefilter(
            Xv, y, names, candidate_sizes=[4], cv=3, tol=0.003, alpha=1.0, random_state=seed, n_bootstrap=150, bootstrap_stability_threshold=0.35
        )
        single_both += int("dup_a" in single and "dup_b" in single)
        boot_both += int("dup_a" in boot and "dup_b" in boot)

    single_rate = single_both / n_seeds
    boot_rate = boot_both / n_seeds

    assert boot_rate >= 0.65, f"expected bootstrap stability-selection to retain both collinear duplicates in >=65% of seeds, got {boot_rate:.3f}"
    assert boot_rate > single_rate, f"expected bootstrap mode to beat single-fit baseline, got boot={boot_rate:.3f} vs single={single_rate:.3f}"
    assert single_rate <= 0.70, f"single-fit baseline should visibly miss the duplicate pair in a meaningful fraction of seeds, got {single_rate:.3f}"


def test_ridge_prefilter_bootstrap_disabled_by_default_is_bit_identical():
    """``n_bootstrap`` is opt-in -- omitting it must reproduce the exact single-fit selection."""
    X, y = _make_noisy_dataset(n=200, d_informative=6, d_noise=194, seed=3)
    names = list(X.columns)
    baseline = ridge_coefficient_prefilter(X.to_numpy(), y, names, cv=3, tol=0.02, alpha=1.0)
    explicit_none = ridge_coefficient_prefilter(X.to_numpy(), y, names, cv=3, tol=0.02, alpha=1.0, n_bootstrap=None)
    assert baseline == explicit_none
