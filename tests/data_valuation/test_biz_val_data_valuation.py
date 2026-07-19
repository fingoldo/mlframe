"""biz_val + unit tests for gt_04's ``mlframe.data_valuation`` package (KNN-Shapley / TMC / Banzhaf / weights).

See ``research/gt_04_data_shapley_valuation.md`` for the full design. The scientific-correctness proof
(closed-form recursion vs brute-force exact Shapley) lives in ``test_knn_shapley_recursion.py``; this
file covers the plan's 4 biz_val scenarios plus the remaining unit-level contracts (weight transforms,
clone-free API sanity, njit-vs-numpy bit identity is implicit -- there is only one numpy path here,
unlike gt_03's dual-backend banzhaf kernel).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from mlframe.data_valuation import data_banzhaf, knn_shapley, propagate_subsample_values, tmc_shapley, valuation_sample_weight


def test_biz_val_knn_shapley_flags_label_noise():
    """10% flipped train labels: >=90% of flipped rows fall below the clean-row value median, AUROC(-value) >= 0.85."""
    from sklearn.datasets import make_blobs
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    X, y = make_blobs(n_samples=1200, centers=2, cluster_std=1.5, random_state=0)
    y = y.astype(np.int64)
    n = len(y)
    flip_idx = rng.choice(n, size=int(n * 0.10), replace=False)
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

    X_train, y_train_noisy = X[:1000], y_noisy[:1000]
    X_val, y_val = X[1000:], y[1000:]
    values = knn_shapley(X_train, y_train_noisy, X_val, y_val, k=5)

    flip_mask = np.zeros(1000, dtype=bool)
    flip_mask[flip_idx[flip_idx < 1000]] = True
    clean_median = np.median(values[~flip_mask])
    frac_below = float(np.mean(values[flip_mask] < clean_median))
    assert frac_below >= 0.90, f"only {frac_below:.2%} of flipped rows below clean-row median"

    auroc = roc_auc_score(flip_mask.astype(int), -values)
    assert auroc >= 0.85, f"noise-detector AUROC {auroc:.4f} < 0.85"


def test_biz_val_valuation_weights_improve_downstream_auc():
    """sample_weight=valuation_sample_weight(knn_shapley values) beats unweighted training on a noisy-label bed."""
    from sklearn.datasets import make_blobs
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    rng = np.random.default_rng(1)
    X, y = make_blobs(n_samples=1500, centers=2, cluster_std=1.6, random_state=1)
    y = y.astype(np.int64)
    n = len(y)
    flip_idx = rng.choice(n, size=int(n * 0.12), replace=False)
    y_noisy = y.copy()
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

    idx_all = np.arange(n)
    idx_train, idx_test = train_test_split(idx_all, test_size=0.3, random_state=1, stratify=y)
    X_train, y_train_noisy = X[idx_train], y_noisy[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    X_val_sub, y_val_sub = X_train[:200], y_train_noisy[:200]  # in-sample proxy val split for the valuation itself
    values = knn_shapley(X_train, y_train_noisy, X_val_sub, y_val_sub, k=5)
    weights = valuation_sample_weight(values)

    clf_unweighted = XGBClassifier(n_estimators=150, random_state=0, eval_metric="logloss")
    clf_unweighted.fit(X_train, y_train_noisy)
    auc_unweighted = roc_auc_score(y_test, clf_unweighted.predict_proba(X_test)[:, 1])

    clf_weighted = XGBClassifier(n_estimators=150, random_state=0, eval_metric="logloss")
    clf_weighted.fit(X_train, y_train_noisy, sample_weight=weights)
    auc_weighted = roc_auc_score(y_test, clf_weighted.predict_proba(X_test)[:, 1])

    assert auc_weighted >= auc_unweighted + 0.01, f"weighted AUC {auc_weighted:.4f} did not beat unweighted {auc_unweighted:.4f} by >= 0.01"


def test_biz_val_tmc_matches_knn_on_small_n():
    """TMC-Shapley (logistic-regression utility) and KNN-Shapley values correlate Spearman >= 0.45 on a
    small bed (sign/scale guard, not exact-value equivalence -- they are different games: logistic
    regression on arbitrary subsets vs a fixed K-NN closed form). Threshold calibrated on first
    measurement (0.5159, seed 0) per the plan's own instruction ("calibrate the floor on first run");
    0.45 sits a comfortable margin below the measured value while still ruling out a sign/scale bug."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=0)
    X_train, y_train = X[:160], y[:160]
    X_val, y_val = X[160:], y[160:]

    def utility_fn(idx: np.ndarray) -> float:
        """Validation AUC of a logistic regression fit on the given training-row subset (or 0.5 if degenerate)."""
        if idx.size < 2 or len(np.unique(y_train[idx])) < 2:
            return 0.5
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train[idx], y_train[idx])
        return float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))

    rng = np.random.default_rng(0)
    tmc_values, _info = tmc_shapley(utility_fn, n_rows=160, n_permutations=200, truncation_tol=1e-3, rng=rng)
    knn_values = knn_shapley(X_train, y_train, X_val, y_val, k=5)

    rho, _p = spearmanr(tmc_values, knn_values)
    assert rho >= 0.45, f"TMC-vs-KNN Spearman={rho:.4f} < 0.45"


def test_biz_val_duplicate_rows_share_credit():
    """5 exact copies of a high-value row: exact symmetry (all 6 copies get IDENTICAL value -- a real
    Shapley axiom, verified to float precision) plus a bounded-credit check.

    The plan's original hypothesis ("each copy's value ~= solo_value / 6, i.e. equal division among
    the 6 duplicates") does NOT hold for this K-normalized KNN utility -- measured at k=5: each copy
    gets ~0.71x the solo value (not ~0.17x = 1/6); at k=20 ~0.91x; at k=50 ~0.98x. Increasing k made
    the discrepancy from 1/6 LARGER, not smaller, ruling out "too few duplicates relative to k" as the
    explanation. The real mechanism: with num_duplicates < K, the duplicated points don't fully
    saturate every validation point's K-nearest slot, so each copy still carries most of its original
    informativeness rather than being diluted by 1/(m+1); as K grows the duplicates' share of the
    K-neighborhood shrinks further, so PER-copy value converges toward (not away from) the original
    solo value. This is a genuine property of the K-fixed-normalization utility, not a bug -- the
    exact symmetry axiom (below) is what the recursion's correctness actually guarantees; the specific
    dilution ratio is data/k-dependent and was never a scientific claim needing a fixed tolerance.
    """
    rng = np.random.default_rng(2)
    n = 300
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    X_val = rng.standard_normal((60, 4))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int64)

    values_solo = knn_shapley(X, y, X_val, y_val, k=5)
    target_row = int(np.argmax(values_solo))
    solo_value = values_solo[target_row]

    X_dup = np.vstack([X, np.tile(X[target_row], (5, 1))])
    y_dup = np.concatenate([y, np.full(5, y[target_row])])
    values_dup = knn_shapley(X_dup, y_dup, X_val, y_val, k=5)

    copy_indices = [target_row, *range(n, n + 5)]
    copy_values = values_dup[copy_indices]

    # Exact symmetry: exchangeable players (identical rows) MUST get identical Shapley values.
    np.testing.assert_allclose(copy_values, copy_values[0], atol=1e-12)

    # Bounded credit: no individual copy exceeds what the row was worth entirely alone, and every
    # copy still carries positive value (a genuinely informative row stays informative when duplicated).
    assert 0.0 < copy_values[0] <= solo_value * 1.05, f"copy value {copy_values[0]:.6f} not in (0, solo_value={solo_value:.6f}] within a 5% margin"


def test_data_banzhaf_correlates_with_knn_shapley_small_n():
    """MSR-Banzhaf over the same retraining utility roughly agrees in DIRECTION with KNN-Shapley on a
    tiny bed (sign/scale guard). Threshold calibrated on first measurement (0.2654, seed 3) -- loose
    on purpose, these are different games/engines entirely."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = make_classification(n_samples=120, n_features=5, n_informative=3, random_state=3)
    X_train, y_train = X[:90], y[:90]
    X_val, y_val = X[90:], y[90:]

    def utility_fn(idx: np.ndarray) -> float:
        """Validation AUC of a logistic regression fit on the given training-row subset (or 0.5 if degenerate)."""
        if idx.size < 2 or len(np.unique(y_train[idx])) < 2:
            return 0.5
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train[idx], y_train[idx])
        return float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))

    rng = np.random.default_rng(3)
    beta, info = data_banzhaf(utility_fn, n_rows=90, n_coalitions=1024, rng=rng)
    knn_values = knn_shapley(X_train, y_train, X_val, y_val, k=5)

    rho, _p = spearmanr(beta, knn_values)
    assert rho >= 0.20, f"Banzhaf-vs-KNN Spearman={rho:.4f} < 0.20 (loose sign/scale guard)"
    assert info["degenerate_rows"] == []


def test_propagate_subsample_values_recovers_subsample_values_exactly():
    """A subsample row's propagated value equals its own (0-distance nearest-neighbor to itself)."""
    rng = np.random.default_rng(4)
    X_full = rng.standard_normal((50, 3))
    sub_idx = rng.choice(50, size=15, replace=False)
    X_sub = X_full[sub_idx]
    sub_values = rng.standard_normal(15)

    propagated = propagate_subsample_values(X_full, X_sub, sub_values, k=1)
    np.testing.assert_allclose(propagated[sub_idx], sub_values, atol=1e-10)


def test_valuation_sample_weight_clip_negative_contract():
    """clip_negative mode: nonneg, mean ~= 1, no NaN, negative inputs floored to 0."""
    values = np.array([-2.0, -0.5, 0.1, 1.0, 3.0])
    w = valuation_sample_weight(values, mode="clip_negative")
    assert np.all(w >= 0.0)
    assert not np.any(np.isnan(w))
    assert w[0] == 0.0 and w[1] == 0.0  # both negative values clipped to the floor
    assert w.mean() == pytest.approx(1.0, abs=1e-9)


def test_valuation_sample_weight_rank_and_softmax_contracts():
    """rank/softmax modes: nonneg, mean ~= 1, no NaN."""
    values = np.array([-2.0, -0.5, 0.1, 1.0, 3.0, 10.0])
    for mode in ("rank", "softmax"):
        w = valuation_sample_weight(values, mode=mode, temperature=2.0)
        assert np.all(w >= 0.0), mode
        assert not np.any(np.isnan(w)), mode
        assert w.mean() == pytest.approx(1.0, abs=1e-6), mode


def test_valuation_sample_weight_all_negative_falls_back_to_uniform():
    """When every value is below the floor, the degenerate all-zero-sum case falls back to uniform weight instead of NaN."""
    values = np.array([-3.0, -2.0, -1.0])
    w = valuation_sample_weight(values, mode="clip_negative", floor=0.0)
    np.testing.assert_allclose(w, np.ones(3))


def test_valuation_sample_weight_rejects_bad_mode():
    """An unrecognized mode raises ValueError."""
    with pytest.raises(ValueError):
        valuation_sample_weight(np.array([1.0, 2.0]), mode="not_a_real_mode")
