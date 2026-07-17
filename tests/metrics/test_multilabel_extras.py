"""Tests for mlframe.metrics._multilabel_extras."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import (
    label_ranking_average_precision,
    coverage_error,
    label_ranking_loss,
    one_error,
    multilabel_f1_macro,
    multilabel_f1_micro,
    multilabel_f1_weighted,
    fast_multilabel_classification_metrics_block,
    hamming_loss,
    subset_accuracy,
    jaccard_score_multilabel,
)


def _rand_ml(N=200, K=5, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=(N, K)) > 0.7).astype(np.int64)
    p = (rng.uniform(size=(N, K)) > 0.7).astype(np.int64)
    s = rng.uniform(size=(N, K))
    return y, p, s


def test_lrap_matches_sklearn():
    from sklearn.metrics import label_ranking_average_precision_score

    y, _, s = _rand_ml(seed=1)
    # sklearn skips rows where n_true is 0 or K - we must filter to compare.
    mask = (y.sum(axis=1) > 0) & (y.sum(axis=1) < y.shape[1])
    if mask.sum() < 10:
        return  # not enough comparable rows; sample again would be flaky
    expected = label_ranking_average_precision_score(y[mask], s[mask])
    assert label_ranking_average_precision(y, s) == pytest.approx(expected, abs=1e-10)


def test_coverage_error_matches_sklearn():
    from sklearn.metrics import coverage_error as skl_cov

    y, _, s = _rand_ml(seed=2)
    mask = y.sum(axis=1) > 0
    expected = skl_cov(y[mask], s[mask])
    assert coverage_error(y, s) == pytest.approx(expected, abs=1e-10)


def test_ranking_loss_matches_sklearn():
    from sklearn.metrics import label_ranking_loss as skl_rl

    y, _, s = _rand_ml(seed=3)
    mask = (y.sum(axis=1) > 0) & (y.sum(axis=1) < y.shape[1])
    if mask.sum() < 10:
        return
    expected = skl_rl(y[mask], s[mask])
    assert label_ranking_loss(y, s) == pytest.approx(expected, abs=1e-10)


def test_one_error_matches_sklearn_formula():
    """One-error = 1 - top1-prediction-in-true-labels rate."""
    rng = np.random.default_rng(4)
    N, K = 100, 5
    y = (rng.uniform(size=(N, K)) > 0.5).astype(np.int64)
    s = rng.uniform(size=(N, K))
    # Manual: top-1 label = argmax(s, axis=1); one-error = mean(y[i, top1[i]] == 0)
    top1 = s.argmax(axis=1)
    expected = float(np.mean(y[np.arange(N), top1] == 0))
    assert one_error(y, s) == pytest.approx(expected, abs=1e-12)


def test_f1_macro_micro_weighted_match_sklearn():
    from sklearn.metrics import f1_score

    y, p, _ = _rand_ml(seed=5)
    assert multilabel_f1_macro(y, p) == pytest.approx(
        f1_score(y, p, average="macro", zero_division=0),
        abs=1e-12,
    )
    assert multilabel_f1_micro(y, p) == pytest.approx(
        f1_score(y, p, average="micro", zero_division=0),
        abs=1e-12,
    )
    assert multilabel_f1_weighted(y, p) == pytest.approx(
        f1_score(y, p, average="weighted", zero_division=0),
        abs=1e-12,
    )


def test_multilabel_fused_block_agrees_with_individual_metrics():
    y, p, _ = _rand_ml(seed=6)
    block = fast_multilabel_classification_metrics_block(y, p)
    assert block["hamming_loss"] == pytest.approx(hamming_loss(y, p), abs=1e-12)
    assert block["subset_accuracy"] == pytest.approx(subset_accuracy(y, p), abs=1e-12)
    assert block["f1_macro"] == pytest.approx(multilabel_f1_macro(y, p), abs=1e-12)
    assert block["f1_micro"] == pytest.approx(multilabel_f1_micro(y, p), abs=1e-12)
    assert block["f1_weighted"] == pytest.approx(multilabel_f1_weighted(y, p), abs=1e-12)


def test_multilabel_fused_block_jaccard_matches():
    """Block's jaccard_macro should match the sample-mean of per-label
    Jaccard with the existing jaccard_score_multilabel (uses 1-D
    intersection-over-union per sample, so direct comparison isn't 1:1;
    instead verify the block jaccard sits inside [0, 1] and matches
    sklearn's per-label macro-jaccard)."""
    from sklearn.metrics import jaccard_score

    y, p, _ = _rand_ml(seed=7)
    block = fast_multilabel_classification_metrics_block(y, p)
    expected = jaccard_score(y, p, average="macro", zero_division=0)
    assert block["jaccard_macro"] == pytest.approx(expected, abs=1e-12)


def test_multilabel_fused_block_per_label_internals():
    """The block exposes _per_label_precision/recall/f1/jaccard for
    downstream callers; check they are length-K arrays of finite floats."""
    y, p, _ = _rand_ml(N=50, K=3, seed=8)
    block = fast_multilabel_classification_metrics_block(y, p)
    for key in (
        "_per_label_precision",
        "_per_label_recall",
        "_per_label_f1",
        "_per_label_jaccard",
    ):
        arr = block[key]
        assert arr.shape == (3,)
        assert np.all(np.isfinite(arr))
