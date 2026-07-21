"""biz_val for ``knn_shapley_regression_binarize`` -- does median-split binarization + exact
classification KNN-Shapley work as a cheap proxy for CONTINUOUS-target row valuation?

This is an open empirical question (see the module's own docstring): a row whose true y is wrong but
happens to land on the correct side of the median split is invisible to this proxy. The tests below
measure the real effect on a mislabeled-row detection task and a downstream sample_weight/RMSE task,
using the same three-way train/val/test discipline as the classification adapter (valuation computed
on VAL, label corruption injected only into TRAIN, effect measured on held-out TEST).
"""

from __future__ import annotations

import numpy as np


def _regression_noisy_bed(n=3000, corrupt_frac=0.12, seed=0):
    """Linear-signal regression bed with a fraction of TRAIN rows' y reassigned to another row's true value.

    Mirrors the classification label-flip fixture's mislabeling mechanism (a genuinely wrong, but
    plausible-looking, target) -- reassigning to ANOTHER real row's y (not a random unrelated value)
    means some corrupted rows can still land on the correct side of the median by chance, which is
    exactly the failure mode this proxy is expected to sometimes miss.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    y_true = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(n) * 0.5
    y_noisy = y_true.copy()
    corrupt_idx = rng.choice(n, size=int(n * corrupt_frac), replace=False)
    shuffle_source = rng.permutation(corrupt_idx)
    y_noisy[corrupt_idx] = y_true[shuffle_source]
    return X, y_true, y_noisy, corrupt_idx


def test_biz_val_regression_binarize_flags_mislabeled_rows():
    """Corrupted TRAIN rows should skew toward lower values than clean rows -- measures the REAL
    effect size rather than asserting a specific pre-guessed threshold."""
    from mlframe.data_valuation import knn_shapley_regression_binarize
    from sklearn.metrics import roc_auc_score

    n = 3000
    X, y_true, y_noisy, corrupt_idx = _regression_noisy_bed(n=n)
    idx = np.arange(n)
    rng = np.random.default_rng(1)
    rng.shuffle(idx)
    n_train, n_val = int(n * 0.6), int(n * 0.2)
    idx_train, idx_val = idx[:n_train], idx[n_train : n_train + n_val]

    X_train, y_train_noisy = X[idx_train], y_noisy[idx_train]
    X_val, y_val_true = X[idx_val], y_true[idx_val]

    values, info = knn_shapley_regression_binarize(X_train, y_train_noisy, X_val, y_val_true, k=5)

    corrupt_mask_train = np.isin(idx_train, corrupt_idx)
    n_corrupt_in_train = int(corrupt_mask_train.sum())
    assert n_corrupt_in_train >= 20, "too few corrupted rows landed in train for a meaningful measurement"

    clean_median = np.median(values[~corrupt_mask_train])
    frac_below = float(np.mean(values[corrupt_mask_train] < clean_median))
    auroc = float(roc_auc_score(corrupt_mask_train.astype(int), -values))

    print(f"\nregression-binarize noise detection: frac_below_clean_median={frac_below:.4f}, AUROC={auroc:.4f}, train_positive_frac={info['train_positive_frac']:.4f}")

    # A weak-but-real signal is still useful; report the honest number either way. >= 0.55 AUROC (a
    # modest but genuine improvement over chance 0.5) is the bar for "this proxy is worth using at
    # all" -- the classification-native version clears 0.85 on an analogous bed, so this is a
    # deliberately loose floor acknowledging the proxy's known blind spot (same-side-of-median noise).
    assert auroc >= 0.55, f"regression-binarize noise-detection AUROC {auroc:.4f} did not clear the 0.55 floor -- the proxy shows no real signal on this bed"


def test_biz_val_regression_binarize_weights_improve_test_rmse():
    """sample_weight from the median-split proxy, computed on VAL (label noise only in TRAIN),
    measured for RMSE improvement on a separate held-out TEST split."""
    from mlframe.data_valuation import knn_shapley_regression_binarize, valuation_sample_weight
    from sklearn.metrics import mean_squared_error
    from xgboost import XGBRegressor

    n = 4000
    X, y_true, y_noisy, _corrupt_idx = _regression_noisy_bed(n=n, corrupt_frac=0.15, seed=2)
    idx = np.arange(n)
    rng = np.random.default_rng(3)
    rng.shuffle(idx)
    n_train, n_val = int(n * 0.5), int(n * 0.2)
    idx_train, idx_val, idx_test = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    X_train, y_train_noisy = X[idx_train], y_noisy[idx_train]
    X_val, y_val_true = X[idx_val], y_true[idx_val]
    X_test, y_test_true = X[idx_test], y_true[idx_test]

    values, _info = knn_shapley_regression_binarize(X_train, y_train_noisy, X_val, y_val_true, k=5)
    weights = valuation_sample_weight(values)

    reg_unweighted = XGBRegressor(n_estimators=150, random_state=0)
    reg_unweighted.fit(X_train, y_train_noisy)
    rmse_unweighted = float(np.sqrt(mean_squared_error(y_test_true, reg_unweighted.predict(X_test))))

    reg_weighted = XGBRegressor(n_estimators=150, random_state=0)
    reg_weighted.fit(X_train, y_train_noisy, sample_weight=weights)
    rmse_weighted = float(np.sqrt(mean_squared_error(y_test_true, reg_weighted.predict(X_test))))

    print(f"\nregression-binarize downstream RMSE: unweighted={rmse_unweighted:.4f}, weighted={rmse_weighted:.4f}")

    # Report the real number; only assert it's not a REGRESSION (weighted must not be meaningfully
    # worse than unweighted) -- whether it's a genuine win is exactly what this test measures and
    # documents, not something to force.
    assert rmse_weighted <= rmse_unweighted * 1.02, (
        f"weighted RMSE ({rmse_weighted:.4f}) regressed more than 2% above unweighted ({rmse_unweighted:.4f}) -- "
        "the proxy made downstream regression worse, not just neutral"
    )
