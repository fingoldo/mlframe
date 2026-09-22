"""The default selection metric must score a multiclass blend on every class, not on class 1 alone.

`_score_blend` took `blend[:, 1]` for any (N, K) input, so a 5-class stack was judged purely by how well each
candidate separated class 1; a member excellent on the other four classes was never picked, and the weights learned
that way were then applied to all five columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling.selection import _score_blend


def _probs_good_on(classes_good: set, n: int = 600, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    p = rng.random((n, k))
    for c in classes_good:
        p[:, c] += 3.0 * (y == c)
    p /= p.sum(axis=1, keepdims=True)
    return y, p


def test_a_member_good_on_most_classes_outscores_one_good_on_class_one_only():
    y, good_on_three = _probs_good_on({0, 2, 3})
    _, good_on_one = _probs_good_on({1}, seed=0)
    assert _score_blend(good_on_three, y, None) > _score_blend(good_on_one, y, None)


def test_binary_behaviour_is_unchanged():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p1 = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 500), 0, 1)
    blend = np.stack([1 - p1, p1], axis=1)
    from mlframe.metrics.core import fast_roc_auc

    assert _score_blend(blend, y, None) == pytest.approx(fast_roc_auc(y, p1))
