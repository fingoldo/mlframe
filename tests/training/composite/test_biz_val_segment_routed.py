"""biz_value test for ``training.composite.SegmentRoutedEstimator``.

The win: when a subset of engineered feature columns carries real signal for the MAJORITY of rows but is
pure noise for a data-sparse segment (e.g. aggregate features computed from too little history to be
meaningful, as in the AMEX <=2-statements case), a single global model fit on all rows learns to weight
those columns heavily (since they're genuinely informative for most of the data) -- which then actively HURTS
its ranking quality on the sparse segment, where those same columns are noise. A specialist model trained
only on the sparse segment's rows, restricted to the small reliable feature subset, should rank that segment
far better. ``SegmentRoutedEstimator`` should recover close to the specialist's own ranking quality via its
rank-splice combination.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from mlframe.training.composite import SegmentRoutedEstimator


def _make_sparse_segment_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    n_reliable = 2
    n_noise = 30
    reliable = rng.normal(size=(n, n_reliable))
    noise_cols = rng.normal(size=(n, n_noise))
    hist_len = rng.integers(1, 20, n)
    X = np.concatenate([reliable, noise_cols, hist_len.reshape(-1, 1)], axis=1)

    is_sparse = hist_len <= 2
    y = np.zeros(n)
    y[~is_sparse] = reliable[~is_sparse, 0] + 0.5 * reliable[~is_sparse, 1] + noise_cols[~is_sparse].sum(axis=1) + rng.normal(scale=0.3, size=(~is_sparse).sum())
    y[is_sparse] = reliable[is_sparse, 0] + 0.5 * reliable[is_sparse, 1] + rng.normal(scale=0.3, size=is_sparse.sum())
    return X, y, is_sparse


def _segment_predicate(X: np.ndarray) -> np.ndarray:
    return X[:, -1] <= 2


def test_biz_val_segment_routed_estimator_beats_global_model_within_sparse_segment():
    X, y, is_sparse = _make_sparse_segment_dataset(n=5000, seed=0)

    est = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0),
        specialist_estimator=Ridge(alpha=1.0),
        segment_predicate=_segment_predicate,
        specialist_features=[0, 1],
    )
    est.fit(X, y)
    routed_pred = est.predict(X)
    main_only_pred = est.main_model_.predict(X)

    rho_routed, _ = spearmanr(routed_pred[is_sparse], y[is_sparse])
    rho_main_only, _ = spearmanr(main_only_pred[is_sparse], y[is_sparse])

    assert rho_routed > 0.9, f"expected the routed specialist to rank the sparse segment well, got rho={rho_routed:.4f}"
    assert rho_routed - rho_main_only > 0.5, (
        f"expected routing to beat the global model's within-segment ranking by >0.5 Spearman rho, "
        f"got routed={rho_routed:.4f} vs main_only={rho_main_only:.4f}"
    )


def test_segment_routed_estimator_rank_splice_preserves_segment_score_multiset():
    """The rank-splice combination must reuse the segment's own main-model score VALUES (only permuted),
    not introduce the specialist's raw score scale -- this is what keeps the majority segment's global
    calibration untouched."""
    X, y, is_sparse = _make_sparse_segment_dataset(n=1000, seed=1)
    est = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0), specialist_estimator=Ridge(alpha=1.0), segment_predicate=_segment_predicate, specialist_features=[0, 1],
    )
    est.fit(X, y)
    routed_pred = est.predict(X)
    main_only_pred = est.main_model_.predict(X)

    np.testing.assert_allclose(np.sort(routed_pred[is_sparse]), np.sort(main_only_pred[is_sparse]))
    # Majority segment is untouched by routing.
    np.testing.assert_array_equal(routed_pred[~is_sparse], main_only_pred[~is_sparse])


def test_biz_val_segment_routed_estimator_auto_segment_column_matches_manual_predicate():
    """Regression: with segment_predicate given (the pre-existing path), behavior is bit-identical to before
    auto-discovery was added -- auto_segment_column defaults to None and must never engage."""
    X, y, is_sparse = _make_sparse_segment_dataset(n=2000, seed=3)
    est = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0), specialist_estimator=Ridge(alpha=1.0), segment_predicate=_segment_predicate, specialist_features=[0, 1],
    )
    est.fit(X, y)
    pred_manual = est.predict(X)
    assert est.segment_threshold_ is None

    # hist_len column is index -1; is_sparse is exactly hist_len<=2, i.e. the bottom quantile of hist_len.
    auto_quantile = float(is_sparse.mean())
    est_auto = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0),
        specialist_estimator=Ridge(alpha=1.0),
        auto_segment_column=-1,
        auto_segment_quantile=auto_quantile,
        auto_segment_direction="low",
        specialist_features=[0, 1],
    )
    est_auto.fit(X, y)
    pred_auto = est_auto.predict(X)

    np.testing.assert_array_equal(est_auto._resolve_segment_mask(X, fitting=False), is_sparse)
    np.testing.assert_allclose(pred_auto, pred_manual)


def test_biz_val_segment_routed_estimator_auto_segment_discovers_sparse_segment_without_predicate():
    """The core new capability: auto_segment_column discovers the sparse segment from a threshold alone (no
    hand-written segment_predicate), and the routed specialist still beats the global model within it -- on a
    held-out (out-of-sample) split, proving the discovered threshold generalizes."""
    X_train, y_train, _ = _make_sparse_segment_dataset(n=4000, seed=10)
    X_test, y_test, is_sparse_test = _make_sparse_segment_dataset(n=2000, seed=11)

    est = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0),
        specialist_estimator=Ridge(alpha=1.0),
        auto_segment_column=-1,
        auto_segment_quantile=0.1,  # hist_len in [1,20) uniform int -> bottom decile ~= <=2, matching is_sparse
        auto_segment_direction="low",
        specialist_features=[0, 1],
    )
    est.fit(X_train, y_train)

    discovered_mask_test = est._resolve_segment_mask(X_test, fitting=False)
    precision = float((discovered_mask_test & is_sparse_test).sum() / max(1, discovered_mask_test.sum()))
    assert precision > 0.9, f"expected the auto-discovered segment to overlap the true sparse segment, got precision={precision:.4f}"

    routed_pred = est.predict(X_test)
    main_only_pred = est.main_model_.predict(X_test)

    rho_routed, _ = spearmanr(routed_pred[discovered_mask_test], y_test[discovered_mask_test])
    rho_main_only, _ = spearmanr(main_only_pred[discovered_mask_test], y_test[discovered_mask_test])

    assert rho_routed > 0.85, f"expected auto-routed specialist to rank the discovered segment well OOS, got rho={rho_routed:.4f}"
    assert rho_routed - rho_main_only > 0.4, (
        f"expected auto-discovery routing to beat the global model's OOS within-segment ranking by >0.4 Spearman rho, "
        f"got routed={rho_routed:.4f} vs main_only={rho_main_only:.4f}"
    )


def test_segment_routed_estimator_rejects_both_predicate_and_auto_column():
    est = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0),
        specialist_estimator=Ridge(alpha=1.0),
        segment_predicate=_segment_predicate,
        auto_segment_column=-1,
    )
    import pytest
    with pytest.raises(ValueError, match="exactly one"):
        est.fit(*( _make_sparse_segment_dataset(n=50, seed=4)[:2] ))


def test_segment_routed_estimator_no_sparse_rows_falls_back_to_main_model():
    X, y, _ = _make_sparse_segment_dataset(n=200, seed=2)
    est = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0), specialist_estimator=Ridge(alpha=1.0), segment_predicate=lambda X: np.zeros(X.shape[0], dtype=bool),
    )
    est.fit(X, y)
    pred = est.predict(X)
    np.testing.assert_allclose(pred, est.main_model_.predict(X))
    assert est.segment_rate_ == 0.0
