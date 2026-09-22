"""Edge-case contracts three metrics did not keep.

MET-11: pinball / coverage / Winkler returned 0.0 on empty input - the BEST value of a lower-is-better loss - so an
empty slice won every min() selection. The numpy fallbacks already returned NaN.
MET-12: `fast_roc_auc` / `fast_aucs` counted `tps += y` / `fps += 1 - y`, so {-1, 1} labels gave NaN where sklearn
returns the real AUC.
MET-15: `lift_at_k` picked the tied rows at the cutoff with `np.argpartition`, arbitrarily, so tie-heavy scores gave
different lift on identical data.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.metrics.classification._classification_extras import lift_at_k
from mlframe.metrics.core import fast_aucs, fast_roc_auc
from mlframe.metrics.quantile import coverage, pinball_loss, winkler_score


def test_empty_quantile_losses_are_nan_not_perfect():
    e = np.array([], dtype=np.float64)
    assert np.isnan(pinball_loss(e, e, 0.5))
    assert np.isnan(coverage(e, e, e))
    assert np.isnan(winkler_score(e, e, e, 0.2))


def test_minus_one_plus_one_labels_score_like_sklearn():
    rng = np.random.default_rng(0)
    n = 1000
    y = (rng.random(n) < 0.4).astype(int)
    s = y * 0.5 + rng.random(n)
    y_pm = np.where(y == 1, 1, -1)
    expected = roc_auc_score(y_pm, s)
    assert fast_roc_auc(y_pm, s) == pytest.approx(expected, rel=1e-12)
    assert fast_aucs(y_pm, s)[0] == pytest.approx(expected, rel=1e-12)
    assert fast_roc_auc(y, s) == pytest.approx(expected, rel=1e-12), "{0,1} labels must be unaffected"


def test_lift_on_tied_scores_is_deterministic_and_equals_the_tie_expectation():
    """Four rows tied at the cutoff, two slots left: the tied run contributes its positive rate times two."""
    y_score = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    y_true = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    # k=30% of 10 -> 3 slots: the 0.9 row plus 2 of the four 0.5 rows (2 positives among 4 -> 1.0 expected)
    expected_captured = 1 + 2 * (2 / 4)
    expected = (expected_captured / 3) / (4 / 10)
    values = {lift_at_k(y_true, y_score, 30) for _ in range(5)}
    assert len(values) == 1, f"identical inputs gave {sorted(values)}"
    assert values.pop() == pytest.approx(expected, rel=1e-12)


def test_lift_without_ties_is_unchanged():
    rng = np.random.default_rng(1)
    ys = rng.random(1000)
    yt = (rng.random(1000) < ys).astype(int)
    top = np.argsort(-ys)[:100]
    assert lift_at_k(yt, ys, 10) == pytest.approx(yt[top].mean() / yt.mean())
