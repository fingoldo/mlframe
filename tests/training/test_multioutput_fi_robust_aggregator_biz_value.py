"""biz_value test for the MultiOutput-FI robust-aggregator upgrade (iter577).

Before iter577 the aggregator was ``np.mean(per_child, axis=0)`` for both
``feature_importances_`` and ``coef_`` paths. ``coef_`` is signed; for a
feature whose direction reverses across labels the mean cancels out and
the genuinely-important feature gets pushed to the bottom of the FI report.

This biz_value test plants exactly that failure mode:

  Synthetic regression-multilabel: 4 labels, 10 features.
  Feature ``f3`` is the TRUE informative feature for every label, but each
  label uses a different sign for f3 (label_0 = +2 * f3 + noise, label_1 =
  -2 * f3 + noise, label_2 = +2 * f3 + noise, label_3 = -2 * f3 + noise).
  Every label is independent of f0..f2 and f4..f9 (pure noise features).

  MultiOutputRegressor(Ridge()) will fit each label separately and produce
  per-child coef_ vectors where coef_[3] is +large for labels 0, 2 and
  -large for labels 1, 3.

  EXPECTED:
    * The MEAN-of-signed aggregator (pre-iter577) returns aggregate
      coef_[3] ~= 0 -- the +2 and -2 entries cancel exactly. f3 ranks
      BELOW noise features whose tiny non-cancelling random aggregate
      happens to land above zero.
    * The MEDIAN-of-|coef| aggregator (iter577) returns aggregate ~= 2
      for f3 and small values for noise features. f3 correctly ranks #1.

The biz_value here is the report quality: an operator looking at the FI
chart for a 4-label model where every label depends strongly on the same
feature would expect that feature on top. Pre-fix they got noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from mlframe.training._feature_importances import get_model_feature_importances


def _build_planted_data(n=500, n_features=10, sign_flip_label=True, seed=42):
    """4 labels, all depending on f3. If ``sign_flip_label``, half the
    labels invert the sign so mean-of-signed-coef would cancel."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    noise = rng.standard_normal((n, 4)) * 0.2
    f3 = X[:, 3]
    if sign_flip_label:
        signs = np.array([+1.0, -1.0, +1.0, -1.0])
    else:
        signs = np.array([+1.0, +1.0, +1.0, +1.0])
    y = np.column_stack([signs[k] * 2.0 * f3 for k in range(4)]) + noise
    cols = [f"f{i}" for i in range(n_features)]
    return X, y, cols


def _rank_of(fi: np.ndarray, j: int) -> int:
    """1-based rank of feature ``j`` in ``fi`` by descending magnitude."""
    order = np.argsort(np.abs(fi))[::-1]
    return int(np.where(order == j)[0][0]) + 1


def test_biz_value_robust_aggregator_ranks_sign_flipped_planted_feature_first():
    """The planted f3 (true importance for all labels, mixed signs)
    must rank in the TOP 2 with the iter577 median-|coef| aggregator,
    despite the +sign/-sign cancellation that would otherwise hide it.

    Top-2 (not strict #1) acknowledges that with only n=500 samples and
    Ridge regularisation a single noise feature may occasionally land
    above f3 by random chance; the planted feature should be reliably
    near the top, which is the operator-facing biz contract.
    """
    X, y, cols = _build_planted_data(seed=20260530)

    model = MultiOutputRegressor(Ridge()).fit(X, y)
    fi = get_model_feature_importances(model, cols, X=X, y=y)

    assert fi is not None
    assert fi.shape == (len(cols),)

    rank_f3 = _rank_of(fi, 3)
    assert rank_f3 <= 2, f"Planted f3 (sign-flipped across labels) should rank top-2 with median-|coef| aggregator; got rank {rank_f3}. FI values: {fi}"
    # Magnitude check: f3 should be MUCH bigger than median noise feature.
    noise_indices = [i for i in range(len(cols)) if i != 3]
    noise_median = float(np.median(np.abs(fi[noise_indices])))
    f3_mag = float(np.abs(fi[3]))
    assert (
        f3_mag > 3.0 * noise_median
    ), f"Planted f3 magnitude {f3_mag:.4f} should exceed 3x the noise median |fi| {noise_median:.4f}; got ratio {f3_mag / max(noise_median, 1e-12):.2f}"


def test_mean_signed_aggregator_would_have_failed_on_same_data():
    """Documentation test: prove the same data with the LEGACY mean-of-
    signed-coef aggregator would have HIDDEN f3 (cancelled by sign flip).

    This pins the rationale for iter577. If sklearn's Ridge ever changes
    such that mean-of-signed-coef stops cancelling, this test should
    fail loudly so the iter577 rationale is re-examined.
    """
    X, y, _cols = _build_planted_data(seed=20260530)

    model = MultiOutputRegressor(Ridge()).fit(X, y)
    # Manually compute legacy mean-of-signed-coef aggregator.
    per_child = []
    for child in model.estimators_:
        coef = np.asarray(child.coef_)
        per_child.append(coef if coef.ndim == 1 else coef[-1, :])
    legacy_mean_signed = np.mean(per_child, axis=0)

    # f3's legacy aggregate should be ~0 (sign cancellation).
    f3_legacy = float(np.abs(legacy_mean_signed[3]))
    # The actual TRUE per-child |coef| is around 2.0 each; mean-of-signed
    # should be MUCH smaller than the |coef| of any single label.
    median_individual = float(np.median([np.abs(c[3]) for c in per_child]))
    assert f3_legacy < 0.3 * median_individual, (
        f"Legacy mean-signed |aggregate| {f3_legacy:.4f} on f3 should be "
        f"<30% of the per-label median |coef| {median_individual:.4f} -- "
        f"sign-cancellation is the entire point of the iter577 fix."
    )

    # And the legacy ranking would have demoted f3 below at least one noise feature.
    legacy_rank_f3 = _rank_of(legacy_mean_signed, 3)
    assert legacy_rank_f3 >= 3, (
        f"Legacy mean-signed aggregator should rank f3 below at least 2 "
        f"noise features; got rank {legacy_rank_f3}. This means the bug "
        f"the iter577 fix targets is not reproducible on this seed; either "
        f"adjust the seed or re-examine the rationale."
    )


def test_unsigned_path_still_uses_mean():
    """Sanity: when children use ``feature_importances_`` (always
    non-negative, e.g. CB / XGB / LGB), the aggregator stays at
    ``np.mean`` -- median for non-negative data wouldn't change the
    qualitative ranking but would lose tied-rank precision the
    historical FI report consumers rely on (mean preserves the
    "average importance across labels" interpretation that operators
    are used to)."""
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(42)
    n, n_feats, n_labels = 300, 8, 3
    X = rng.standard_normal((n, n_feats))
    y = (rng.random((n, n_labels)) > 0.5).astype(int)
    cols = [f"f{i}" for i in range(n_feats)]

    from sklearn.multioutput import MultiOutputClassifier

    model = MultiOutputClassifier(catboost.CatBoostClassifier(iterations=10, verbose=False)).fit(X, y)
    fi = get_model_feature_importances(model, cols, X=X, y=y)

    assert fi is not None
    # Aggregate matches np.mean of per-child native feature_importances_.
    manual_mean = np.mean(
        [np.asarray(est.feature_importances_) for est in model.estimators_],
        axis=0,
    )
    np.testing.assert_allclose(fi, manual_mean, atol=0.0)
