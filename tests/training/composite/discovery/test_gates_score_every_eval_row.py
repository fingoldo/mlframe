"""A partly collapsing reconstruction is scored on the whole holdout, not on the rows where it behaved (DSC-08).

Every y-scale gate rejected a spec only when fewer than half its inverse predictions were finite, and then scored the
survivors as ``rmse(y_eval[finite], y_hat[finite])``. A spec whose inverse blows up on 40% of the holdout - the unseen-base
tail the gates exist to catch - was therefore judged on the 60% where it worked. The shipped estimator has no such option:
``predict`` fills a collapsed row with the train median. The gates now score that same vector over every row.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._yscale_scoring import median_filled_predictions


def test_a_collapsed_row_is_filled_with_the_fit_median():
    """The fill matches the estimator's fallback, and the finite entries are untouched."""
    y_fit = np.array([1.0, 3.0, 5.0, 7.0, np.nan])
    y_hat = np.array([2.0, np.inf, np.nan, -np.inf, 8.0])
    out = median_filled_predictions(y_hat, y_fit)
    np.testing.assert_allclose(out, [2.0, 4.0, 4.0, 4.0, 8.0])
    assert np.isfinite(out).all()


def test_nothing_is_copied_back_into_the_caller_and_an_all_finite_vector_is_unchanged():
    """The helper returns its own array, so a gate's later use cannot mutate the model's predictions."""
    y_hat = np.array([1.0, 2.0, 3.0])
    out = median_filled_predictions(y_hat, np.array([0.0, 10.0]))
    np.testing.assert_allclose(out, y_hat)
    out[0] = 99.0
    assert y_hat[0] == 1.0


def test_an_all_non_finite_fit_fold_leaves_the_entries_alone():
    """With no median to fall back on, the collapsed entries stay non-finite and the gate's finite checks still see them."""
    out = median_filled_predictions(np.array([np.nan, 1.0]), np.array([np.nan, np.inf]))
    assert not np.isfinite(out[0]) and out[1] == 1.0


def test_the_score_of_a_partial_collapse_is_no_better_than_the_median_fill():
    """The measured property: a 40%-collapsing spec is scored worse than a spec that reconstructs every row.

    Scored on finite rows only, the collapsing spec looked near-perfect; with the fill it carries the median error on the
    rows it could not invert, which is what the deployed model would pay there.
    """
    rng = np.random.default_rng(0)
    y_eval = rng.normal(loc=10.0, scale=5.0, size=500)
    y_fit = rng.normal(loc=10.0, scale=5.0, size=500)

    good = y_eval + rng.normal(scale=0.1, size=y_eval.size)
    collapsing = good.copy()
    collapsing[rng.permutation(y_eval.size)[:200]] = np.nan  # 40% of the holdout

    def rmse(pred):
        """RMSE of a prediction vector against the eval target."""
        return float(np.sqrt(np.mean((np.asarray(pred) - y_eval) ** 2)))

    finite = np.isfinite(collapsing)
    old_score = float(np.sqrt(np.mean((collapsing[finite] - y_eval[finite]) ** 2)))
    new_score = rmse(median_filled_predictions(collapsing, y_fit))
    assert old_score < 0.2, "the old scoring saw a near-perfect spec"
    assert new_score > 3.0, "the collapsed rows must cost what the median fill costs"
    assert new_score > rmse(median_filled_predictions(good, y_fit))
