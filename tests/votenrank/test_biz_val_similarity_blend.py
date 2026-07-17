"""biz_value test for ``votenrank.SimilarityBlendEnsemble``.

The win: an in-distribution specialist (trained only on a dense "seen" region A) and an out-of-distribution
specialist (trained only on a distant "unseen-like" region B) each generalize badly to the OTHER region. A
fixed 50/50 blend averages a good and a catastrophically wrong prediction everywhere, which is only
marginally better than either specialist alone. Weighting the blend per-row by k-NN similarity to the
training set (high weight on the in-distribution model near A, low weight far away near B) should recover
close to each specialist's own accuracy in its own region -- mirroring the MoA 5th place's
seen/unseen-drug similarity-blended validation-scheme technique.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.votenrank import SimilarityBlendEnsemble


def _make_two_region_dataset(seed: int):
    """Helper that make two region dataset."""
    rng = np.random.default_rng(seed)
    w_a = np.array([2.0, -1.0])
    w_b = np.array([-3.0, 4.0])

    X_a_train = rng.normal(scale=1.0, size=(1500, 2))
    X_b_train = rng.normal(loc=[10, 10], scale=1.0, size=(1500, 2))
    y_a_train = X_a_train @ w_a + rng.normal(scale=0.2, size=1500)
    y_b_train = X_b_train @ w_b + rng.normal(scale=0.2, size=1500)

    X_a_test = rng.normal(scale=1.0, size=(300, 2))
    X_b_test = rng.normal(loc=[10, 10], scale=1.0, size=(300, 2))
    y_a_test = X_a_test @ w_a + rng.normal(scale=0.2, size=300)
    y_b_test = X_b_test @ w_b + rng.normal(scale=0.2, size=300)

    X_test = np.vstack([X_a_test, X_b_test])
    y_test = np.concatenate([y_a_test, y_b_test])
    return X_a_train, y_a_train, X_b_train, y_b_train, X_test, y_test


def test_biz_val_similarity_blend_beats_fixed_5050_blend_mse():
    """Similarity blend beats fixed 5050 blend mse."""
    X_a_train, y_a_train, X_b_train, y_b_train, X_test, y_test = _make_two_region_dataset(seed=0)

    in_dist_model = LinearRegression().fit(X_a_train, y_a_train)
    out_dist_model = LinearRegression().fit(X_b_train, y_b_train)

    blend = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=10, similarity_scale=3.0)
    blend.in_dist_model_ = in_dist_model
    blend.out_dist_model_ = out_dist_model
    blend.train_embedding_ = blend.embedding_fn(X_a_train.astype(np.float32))

    pred_blend = blend.predict(X_test)
    pred_fixed = 0.5 * in_dist_model.predict(X_test) + 0.5 * out_dist_model.predict(X_test)

    mse_blend = mean_squared_error(y_test, pred_blend)
    mse_fixed = mean_squared_error(y_test, pred_fixed)
    improvement = 1.0 - mse_blend / mse_fixed

    assert improvement > 0.8, f"expected >80% MSE reduction vs. a fixed 50/50 blend, got {improvement:.4f} (fixed={mse_fixed:.4f}, blend={mse_blend:.4f})"


def test_similarity_blend_weight_higher_near_training_region():
    """Similarity blend weight higher near training region."""
    X_a_train, y_a_train, X_b_train, y_b_train, _, _ = _make_two_region_dataset(seed=1)
    blend = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=10, similarity_scale=3.0)
    blend.in_dist_model_ = LinearRegression().fit(X_a_train, y_a_train)
    blend.out_dist_model_ = LinearRegression().fit(X_b_train, y_b_train)
    blend.train_embedding_ = blend.embedding_fn(X_a_train.astype(np.float32))

    w_near = blend.similarity_weight(X_a_train[:50])
    w_far = blend.similarity_weight(X_b_train[:50])
    assert w_near.mean() > 0.9
    assert w_far.mean() < 0.1


def test_similarity_blend_end_to_end_fit_predict():
    """Similarity blend end to end fit predict."""
    X_a_train, y_a_train, _, _, X_test, _ = _make_two_region_dataset(seed=2)
    blend = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=5)
    blend.fit(X_a_train, y_a_train)
    pred = blend.predict(X_test)
    assert pred.shape == (X_test.shape[0],)
    assert np.isfinite(pred).all()


def _make_n_region_dataset(seed: int, n_regions: int = 4):
    """N well-separated regions, each with its own linear relationship -- no single global linear fit,
    and no binary in/out split, can do well everywhere; only a per-region specialist can.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-30, 30, size=(n_regions, 2))
    # force centers apart so regions don't overlap in embedding space
    centers = centers * 6.0
    weights = [rng.normal(scale=3.0, size=2) for _ in range(n_regions)]

    region_X_train, region_y_train, region_X_test, region_y_test = [], [], [], []
    for center, w in zip(centers, weights):
        X_train = rng.normal(loc=center, scale=1.0, size=(600, 2))
        y_train = X_train @ w + rng.normal(scale=0.2, size=600)
        X_test = rng.normal(loc=center, scale=1.0, size=(120, 2))
        y_test = X_test @ w + rng.normal(scale=0.2, size=120)
        region_X_train.append(X_train)
        region_y_train.append(y_train)
        region_X_test.append(X_test)
        region_y_test.append(y_test)

    X_test_all = np.vstack(region_X_test)
    y_test_all = np.concatenate(region_y_test)
    return region_X_train, region_y_train, X_test_all, y_test_all


def test_biz_val_similarity_blend_n_specialist_beats_global_and_2blend_mse():
    """Similarity blend n specialist beats global and 2blend mse."""
    n_regions = 4
    region_X_train, region_y_train, X_test, y_test = _make_n_region_dataset(seed=3, n_regions=n_regions)

    # single global model, trained on the union of all regions
    X_all = np.vstack(region_X_train)
    y_all = np.concatenate(region_y_train)
    global_model = LinearRegression().fit(X_all, y_all)
    pred_global = global_model.predict(X_test)
    mse_global = mean_squared_error(y_test, pred_global)

    # naive 2-specialist blend: region 0 as "in-distribution", the rest pooled as "out-of-distribution"
    X_out_pool = np.vstack(region_X_train[1:])
    y_out_pool = np.concatenate(region_y_train[1:])
    in_model_2 = LinearRegression().fit(region_X_train[0], region_y_train[0])
    out_model_2 = LinearRegression().fit(X_out_pool, y_out_pool)
    blend_2 = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=10, similarity_scale=5.0)
    blend_2.in_dist_model_ = in_model_2
    blend_2.out_dist_model_ = out_model_2
    blend_2.train_embedding_ = blend_2.embedding_fn(region_X_train[0].astype(np.float32))
    pred_2blend = blend_2.predict(X_test)
    mse_2blend = mean_squared_error(y_test, pred_2blend)

    # N-specialist soft blend: one specialist per region, weighted by similarity to each region's own training data
    region_estimators = [LinearRegression() for _ in range(n_regions)]
    blend_n = SimilarityBlendEnsemble(
        in_dist_estimator=LinearRegression(),
        out_dist_estimator=LinearRegression(),
        k=10,
        similarity_scale=5.0,
        region_estimators=region_estimators,
    )
    blend_n.fit_multi_region(region_X_train, region_y_train)
    pred_n = blend_n.predict_multi_region(X_test)
    mse_n = mean_squared_error(y_test, pred_n)

    improvement_vs_global = 1.0 - mse_n / mse_global
    improvement_vs_2blend = 1.0 - mse_n / mse_2blend

    assert improvement_vs_global > 0.9, (
        f"expected >90% MSE reduction vs. a single global model, got {improvement_vs_global:.4f} (global={mse_global:.4f}, n_blend={mse_n:.4f})"
    )
    assert improvement_vs_2blend > 0.5, (
        f"expected >50% MSE reduction vs. a naive 2-specialist blend, got {improvement_vs_2blend:.4f} (2blend={mse_2blend:.4f}, n_blend={mse_n:.4f})"
    )


def test_similarity_blend_n_specialist_weights_sum_to_one_and_default_path_unchanged():
    """Similarity blend n specialist weights sum to one and default path unchanged."""
    region_X_train, region_y_train, X_test, _ = _make_n_region_dataset(seed=4, n_regions=3)
    region_estimators = [LinearRegression() for _ in range(3)]
    blend_n = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=10, region_estimators=region_estimators)
    blend_n.fit_multi_region(region_X_train, region_y_train)
    weights = blend_n.region_similarity_weights(X_test)
    assert weights.shape == (X_test.shape[0], 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(weights >= 0.0)

    # region_estimators is opt-in: leaving it unset must not change the original 2-specialist fit/predict output
    X_a_train, y_a_train, _X_b_train, _y_b_train, X_probe, _ = _make_two_region_dataset(seed=5)
    baseline = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=10, similarity_scale=3.0)
    baseline.fit(X_a_train, y_a_train)
    pred_baseline = baseline.predict(X_probe)

    with_none_region_estimators = SimilarityBlendEnsemble(
        in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=10, similarity_scale=3.0, region_estimators=None
    )
    with_none_region_estimators.fit(X_a_train, y_a_train)
    pred_with_param = with_none_region_estimators.predict(X_probe)

    np.testing.assert_array_equal(pred_baseline, pred_with_param)
