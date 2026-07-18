"""biz_value tests for ``BaggedCompositeEstimator``.

Two quantitative wins:

1. On a NOISY composite target, bagging N members has LOWER OOS RMSE than a
   single composite (bootstrap averaging cancels per-member variance).
2. ``predict_std`` is LARGER in an EXTRAPOLATION region (few/no nearby train
   rows) than in a DENSE region -- the hallmark of an epistemic signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.bagging import BaggedCompositeEstimator


def _proto():
    # High-variance learner (deep tree) so bagging has variance to cancel.
    """Proto."""
    return CompositeTargetEstimator(
        base_estimator=DecisionTreeRegressor(max_depth=None, random_state=0),
        transform_name="diff",
        base_column="lag",
    )


def _rmse(a, b):
    """Rmse."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_biz_val_bagging_lowers_oos_rmse_on_noisy_target():
    """Biz val bagging lowers oos rmse on noisy target."""
    rng = np.random.RandomState(0)
    n = 1200
    base = rng.uniform(0.0, 10.0, size=n)
    feat = rng.uniform(-3.0, 3.0, size=n)
    signal = base + np.sin(feat) + 0.4 * feat**2
    y = signal + rng.normal(0.0, 1.5, size=n)  # heavy observation noise
    X = pd.DataFrame({"lag": base, "feat": feat})

    tr = slice(0, 800)
    te = slice(800, n)
    X_tr, y_tr = X.iloc[tr], y[tr]
    X_te = X.iloc[te]
    # OOS truth = the noiseless signal so we measure variance, not noise floor.
    truth_te = signal[te]

    single = _proto().fit(X_tr, y_tr)
    rmse_single = _rmse(single.predict(X_te), truth_te)

    bag = BaggedCompositeEstimator(
        base_estimator=_proto(),
        n_estimators=25,
        random_state=7,
    ).fit(X_tr, y_tr)
    rmse_bag = _rmse(bag.predict(X_te), truth_te)

    # Measured ~25-40% reduction; floor at a conservative 8% so seed noise
    # never trips but a regression that breaks averaging fails.
    assert rmse_bag <= 0.92 * rmse_single, f"bagging should lower OOS RMSE: single={rmse_single:.4f} bag={rmse_bag:.4f}"


def test_biz_val_predict_std_larger_in_sparse_region():
    # Epistemic signal: where training rows are SPARSE, bootstrap resamples
    # land different boundary leaves -> members disagree -> std grows. Where
    # rows are DENSE, every resample reconstructs the same local fit -> members
    # agree -> std shrinks. We make feat dense in [-2, 2] and SPARSE in [4, 6]
    # (only a handful of rows there), then query both bands inside support.
    """Biz val predict std larger in sparse region."""
    rng = np.random.RandomState(1)
    n_dense = 1500
    n_sparse = 15  # the under-sampled band
    feat_dense = rng.uniform(-2.0, 2.0, size=n_dense)
    feat_sparse = rng.uniform(4.0, 6.0, size=n_sparse)
    feat = np.concatenate([feat_dense, feat_sparse])
    base = rng.uniform(0.0, 10.0, size=feat.size)
    y = base + feat**3 + rng.normal(0.0, 0.5, size=feat.size)
    X = pd.DataFrame({"lag": base, "feat": feat})

    bag = BaggedCompositeEstimator(
        base_estimator=_proto(),
        n_estimators=40,
        random_state=3,
    ).fit(X, y)

    dense_q = pd.DataFrame(
        {
            "lag": rng.uniform(2.0, 8.0, size=200),
            "feat": rng.uniform(-1.5, 1.5, size=200),
        }
    )
    sparse_q = pd.DataFrame(
        {
            "lag": rng.uniform(2.0, 8.0, size=200),
            "feat": rng.uniform(4.5, 5.5, size=200),
        }
    )

    std_dense = float(np.mean(bag.predict_std(dense_q)))
    std_sparse = float(np.mean(bag.predict_std(sparse_q)))

    # Spread in the sparse band should clearly exceed the dense band (members
    # disagree where there is little training data). Floor at 1.5x.
    assert std_sparse >= 1.5 * std_dense, f"epistemic std should grow in the sparse region: dense={std_dense:.4f} sparse={std_sparse:.4f}"


def _bag(aggregation, seed, n_estimators=25, trim_fraction=0.2):
    """Bag."""
    return BaggedCompositeEstimator(
        base_estimator=DecisionTreeRegressor(max_depth=6, random_state=0),
        n_estimators=n_estimators,
        random_state=seed,
        aggregation=aggregation,
        trim_fraction=trim_fraction,
    )


def _outlier_contam_data(seed, n=1200, p=8):
    """Outlier contam data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    truth = 2.0 * X[:, 0] + 1.5 * X[:, 1] * X[:, 2] - 1.0 * X[:, 3] + 0.7 * np.sin(2.0 * X[:, 4])
    noise = rng.randn(n) * 0.5
    mask = rng.rand(n) < 0.05
    noise[mask] += rng.randn(mask.sum()) * 12.0
    y = truth + noise
    cut = int(n * 0.7)
    return X[:cut], y[:cut], X[cut:], truth[cut:]


def test_biz_val_bagging_trimmed_mean_beats_mean_on_outlier_contamination():
    """Default ``trimmed_mean`` aggregation must lower honest-holdout RMSE vs legacy ``mean`` on outlier-contaminated targets.

    Bench bench_bagging_aggregation_qual14: trimmed-mean wins RMSE+MAE on 7/7 seeds for outlier-contam, mean rel RMSE gain
    -4.7%. Floor here is a >=2% RMSE reduction on a majority (>=4/5) of seeds -- ~5-15% below the measured win, so seed noise
    does not trip it but a regression that disables the robust aggregator (e.g. default silently reverting to mean) does.
    """
    wins = 0
    seeds = [0, 1, 2, 3, 4]
    for sd in seeds:
        Xtr, ytr, Xte, truth_te = _outlier_contam_data(sd)
        rmse_mean = _rmse(_bag("mean", sd).fit(Xtr, ytr).predict(Xte), truth_te)
        rmse_trim = _rmse(_bag("trimmed_mean", sd).fit(Xtr, ytr).predict(Xte), truth_te)
        if rmse_trim <= 0.98 * rmse_mean:
            wins += 1
    assert wins >= 4, f"trimmed_mean should beat mean by >=2% RMSE on a majority of outlier-contaminated seeds; won {wins}/{len(seeds)}"


def test_biz_val_bagging_trimmed_mean_default_near_mean_on_clean_data():
    """Default ``trimmed_mean`` must NOT materially regress clean Gaussian data vs ``mean`` (efficiency-preserving robustness).

    Bench: clean-data mean rel RMSE delta(trim-mean)/mean = +0.42%, worst +1.16%. Guard a generous 5% ceiling so the default
    flip is safe on the no-contamination case.
    """
    rng = np.random.RandomState(7)
    n, p = 1200, 8
    X = rng.randn(n, p)
    truth = 2.0 * X[:, 0] + 1.5 * X[:, 1] * X[:, 2] - 1.0 * X[:, 3] + 0.7 * np.sin(2.0 * X[:, 4])
    y = truth + rng.randn(n) * 0.5
    cut = int(n * 0.7)
    Xtr, ytr, Xte, truth_te = X[:cut], y[:cut], X[cut:], truth[cut:]
    rmse_mean = _rmse(_bag("mean", 7).fit(Xtr, ytr).predict(Xte), truth_te)
    rmse_trim = _rmse(_bag("trimmed_mean", 7).fit(Xtr, ytr).predict(Xte), truth_te)
    assert rmse_trim <= 1.05 * rmse_mean, f"trimmed_mean must stay within 5% of mean on clean data: mean={rmse_mean:.4f} trim={rmse_trim:.4f}"


def test_biz_val_bagging_default_aggregation_is_trimmed_mean_not_mean():
    """Pin the default flip: a default-constructed bag must aggregate by trimmed-mean, NOT the legacy plain mean.

    On contaminated data the trimmed-mean point estimate differs from ``members.mean(axis=0)``; if the default ever silently
    reverts to ``mean`` this sensor catches it (default.predict would then equal the member mean exactly).
    """
    Xtr, ytr, Xte, _ = _outlier_contam_data(0)
    default_bag = BaggedCompositeEstimator(
        base_estimator=DecisionTreeRegressor(max_depth=6, random_state=0),
        n_estimators=25,
        random_state=0,
    ).fit(Xtr, ytr)
    assert default_bag.aggregation == "trimmed_mean"
    members = default_bag._member_predictions(Xte)
    assert not np.array_equal(default_bag.predict(Xte), members.mean(axis=0)), "default aggregation must be trimmed_mean (robust), not the legacy plain mean"


def _contam_data(seed, frac, n=1200, p=8):
    """Contam data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    truth = 2.0 * X[:, 0] + 1.5 * X[:, 1] * X[:, 2] - 1.0 * X[:, 3] + 0.7 * np.sin(2.0 * X[:, 4])
    noise = rng.randn(n) * 0.5
    mask = rng.rand(n) < frac
    noise[mask] += rng.randn(mask.sum()) * 12.0
    y = truth + noise
    cut = int(n * 0.7)
    return X[:cut], y[:cut], X[cut:], truth[cut:]


def test_biz_val_bagging_trim_fraction_default_is_0_2():
    """Pin the qual-17 default flip: trim_fraction defaults to 0.2 (the robust knee), NOT the earlier 0.1.

    A silent revert to 0.1 (or to mean=0.0) under-protects against heavy outlier contamination; this sensor catches it.
    """
    bag = BaggedCompositeEstimator(base_estimator=DecisionTreeRegressor(max_depth=6, random_state=0))
    assert bag.trim_fraction == 0.2, f"default trim_fraction must be 0.2, got {bag.trim_fraction}"


def test_biz_val_bagging_trim_0_2_beats_0_1_on_heavy_contamination():
    """The qual-17 default trim=0.2 lowers honest-holdout RMSE vs the earlier trim=0.1 under 15-30% outlier contamination.

    Bench bench_bagging_trim_fraction_qual17: trim 0.2 wins 7/7 seeds on every contaminated scenario (5/15/30%), ~5-9% RMSE
    drop. Floor here: 0.2 beats 0.1 on a majority (>=4/5) of seeds across two heavy-contamination levels, ~5-15% below measured.
    """
    for frac in (0.15, 0.30):
        wins = 0
        seeds = [0, 1, 2, 3, 4]
        for sd in seeds:
            Xtr, ytr, Xte, truth_te = _contam_data(sd, frac)
            rmse_01 = _rmse(_bag("trimmed_mean", sd, trim_fraction=0.1).fit(Xtr, ytr).predict(Xte), truth_te)
            rmse_02 = _rmse(_bag("trimmed_mean", sd, trim_fraction=0.2).fit(Xtr, ytr).predict(Xte), truth_te)
            if rmse_02 < rmse_01:
                wins += 1
        assert wins >= 4, f"trim=0.2 should beat trim=0.1 on a majority of seeds at {frac:.0%} contamination; won {wins}/{len(seeds)}"


def test_biz_val_bagging_trim_0_2_near_0_1_on_clean_data():
    """The qual-17 trim=0.2 default must NOT materially regress clean Gaussian data vs the earlier trim=0.1.

    Bench: clean-data mean RMSE delta(0.2 vs 0.1) ~ +0.3% over 20 seeds. Guard a generous 3% ceiling so the flip is safe on
    the no-contamination case.
    """
    rng = np.random.RandomState(11)
    n, p = 1200, 8
    X = rng.randn(n, p)
    truth = 2.0 * X[:, 0] + 1.5 * X[:, 1] * X[:, 2] - 1.0 * X[:, 3] + 0.7 * np.sin(2.0 * X[:, 4])
    y = truth + rng.randn(n) * 0.5
    cut = int(n * 0.7)
    Xtr, ytr, Xte, truth_te = X[:cut], y[:cut], X[cut:], truth[cut:]
    rmse_01 = _rmse(_bag("trimmed_mean", 11, trim_fraction=0.1).fit(Xtr, ytr).predict(Xte), truth_te)
    rmse_02 = _rmse(_bag("trimmed_mean", 11, trim_fraction=0.2).fit(Xtr, ytr).predict(Xte), truth_te)
    assert rmse_02 <= 1.03 * rmse_01, f"trim=0.2 must stay within 3% of trim=0.1 on clean data: 0.1={rmse_01:.4f} 0.2={rmse_02:.4f}"
