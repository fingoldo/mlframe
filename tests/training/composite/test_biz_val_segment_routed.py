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


def test_segment_routed_estimator_no_sparse_rows_falls_back_to_main_model():
    X, y, _ = _make_sparse_segment_dataset(n=200, seed=2)
    est = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0), specialist_estimator=Ridge(alpha=1.0), segment_predicate=lambda X: np.zeros(X.shape[0], dtype=bool),
    )
    est.fit(X, y)
    pred = est.predict(X)
    np.testing.assert_allclose(pred, est.main_model_.predict(X))
    assert est.segment_rate_ == 0.0
