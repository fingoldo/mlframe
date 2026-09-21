"""Components that share one fitted pre-pipeline must have each fold's slices transformed once, not once per component.

Every component built from one strategy holds the same fitted ``pre_pipeline`` object, and the OOF loops re-ran the same
transform over the same fold slices for each of them: on the composite integration suite half of all transform calls
repeated an earlier one. The pair is now cached per fold for fitted pipelines; unfitted ones are fold-fit clones and are
never shared.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from mlframe.training.composite.ensemble import _transform_pair_cached


class _CountingScaler(StandardScaler):
    """A scaler that counts ``transform`` calls."""

    calls = 0

    def transform(self, X, copy=None):
        """Count, then scale."""
        type(self).calls += 1
        return super().transform(X, copy=copy)


def _slices(seed: int = 0):
    """A fold's stack and holdout slices."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(200, 4)), rng.normal(size=(50, 4))


def test_a_shared_fitted_pipeline_transforms_each_slice_once_per_fold():
    """Two components on one fitted pipeline: 2 transforms per fold (stack + holdout), not 4."""
    x_stack, x_hold = _slices()
    pp = _CountingScaler().fit(x_stack)
    _CountingScaler.calls = 0
    memo: dict = {}
    first = _transform_pair_cached(memo, pp, x_stack, x_hold)
    second = _transform_pair_cached(memo, pp, x_stack, x_hold)
    assert _CountingScaler.calls == 2, f"expected one transform per slice, got {_CountingScaler.calls}"
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_a_new_fold_transforms_again():
    """The cache is per fold: new slices are new keys, so nothing leaks across folds."""
    x_stack, x_hold = _slices()
    pp = _CountingScaler().fit(x_stack)
    _CountingScaler.calls = 0
    _transform_pair_cached({}, pp, x_stack, x_hold)
    x2, h2 = _slices(1)
    _transform_pair_cached({}, pp, x2, h2)
    assert _CountingScaler.calls == 4


def test_an_unfitted_pipeline_is_never_shared():
    """An unfitted pipeline is fit as a clone per component on the fold's train slice; caching it would be wrong."""
    x_stack, x_hold = _slices()
    pp = _CountingScaler()
    memo: dict = {}
    _CountingScaler.calls = 0
    _transform_pair_cached(memo, pp, x_stack, x_hold)
    _transform_pair_cached(memo, pp, x_stack, x_hold)
    assert not memo, "an unfitted pipeline must not enter the cache"
    assert _CountingScaler.calls == 4


def test_the_cached_result_equals_the_direct_transform():
    """Caching changes nothing about the values: they equal a plain transform of each slice."""
    x_stack, x_hold = _slices()
    pp = StandardScaler().fit(x_stack)
    got_stack, got_hold = _transform_pair_cached({}, pp, x_stack, x_hold)
    np.testing.assert_array_equal(got_stack, pp.transform(x_stack))
    np.testing.assert_array_equal(got_hold, pp.transform(x_hold))
