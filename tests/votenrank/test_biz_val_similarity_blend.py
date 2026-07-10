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
    X_a_train, y_a_train, _, _, X_test, _ = _make_two_region_dataset(seed=2)
    blend = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=5)
    blend.fit(X_a_train, y_a_train)
    pred = blend.predict(X_test)
    assert pred.shape == (X_test.shape[0],)
    assert np.isfinite(pred).all()
