"""The default blend metric required strictly 0/1 labels and said nothing when it did not get them.

`_score_blend`'s `metric=None` path calls `fast_roc_auc`, whose kernel accumulates `tps += y_true[i]` and is
therefore only defined on 0/1. The call cast with `astype(np.int64)`, which changes the dtype and leaves the
ENCODING: on `{1, 2}` or `{-1, +1}` every AUC came back NaN. Because `nan > best` is always False, the greedy
walk then accepted no candidate at all and returned whichever model index happened to come first -- a
well-formed `CaruanaSelectionResult` built from zero comparisons.

Both halves are pinned here: the metric now binarises against the larger label, and a non-finite starting score
raises instead of degenerating.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling.selection import caruana_greedy_selection


def _members(seed: int = 0, n: int = 800):
    """One informative member and one pure-noise member, so the correct pick is unambiguous."""
    rng = np.random.default_rng(seed)
    y01 = (rng.random(n) < 0.4).astype(np.int64)
    signal = y01 * 0.9 + rng.random(n) * 0.1
    good = np.column_stack([1.0 - signal, signal])
    noise = rng.random(n)
    bad = np.column_stack([1.0 - noise, noise])
    return y01, [good, bad]


class TestEveryBinaryEncodingSelectsTheSameModel:
    """A relabelling is not a different problem; the selection must not notice it."""

    @pytest.mark.parametrize("encode", [lambda y: y, lambda y: y + 1, lambda y: 2 * y - 1, lambda y: y * 5 + 3])
    def test_the_informative_member_is_picked(self, encode):
        """`{0,1}`, `{1,2}`, `{-1,+1}` and an arbitrary `{3,8}` must all reach the same answer."""
        y01, members = _members()
        result = caruana_greedy_selection(members, encode(y01), max_picks=10)
        assert np.isfinite(result.score), f"score came back {result.score!r}"
        assert result.weights[0] > result.weights[1], "the informative member must outweigh the noise member"

    def test_the_score_is_identical_across_encodings(self):
        """The discriminating assertion: pre-fix, only the 0/1 arm produced a number at all."""
        y01, members = _members()
        scores = [caruana_greedy_selection(members, enc(y01), max_picks=10).score for enc in (lambda y: y, lambda y: y + 1, lambda y: 2 * y - 1)]
        assert all(np.isfinite(s) for s in scores), f"non-finite scores: {scores}"
        assert max(scores) - min(scores) < 1e-12, f"the same problem scored differently per encoding: {scores}"


class TestANonFiniteScoreIsLoud:
    """Degenerating to the starting set while reporting a plausible result is the failure being prevented."""

    def test_a_metric_returning_nan_raises(self):
        """Silence here is what let the encoding bug live: the walk simply accepted nothing."""
        y01, members = _members()
        with pytest.raises(ValueError, match="no candidate can ever beat it"):
            caruana_greedy_selection(members, y01, max_picks=5, metric=lambda y, blend: float("nan"))

    def test_a_working_custom_metric_is_untouched(self):
        """The guard must not fire on a finite metric, however unusual its scale."""
        y01, members = _members()
        result = caruana_greedy_selection(members, y01, max_picks=5, metric=lambda y, blend: -float(np.mean((y - blend[:, 1]) ** 2)))
        assert np.isfinite(result.score)
        assert result.weights[0] > result.weights[1]
