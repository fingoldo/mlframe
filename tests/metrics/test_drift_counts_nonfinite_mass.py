"""A change in the NaN rate is drift, and PSI / KL / JS must see it.

The binner used `np.histogram`, which silently drops non-finite rows, and every divergence renormalised the
remaining mass to 1: a target window where half the predictions had become NaN reported PSI 0.0 ("no significant
change") - the most serious drift there is looked like none.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._drift import js_divergence, kl_divergence, population_stability_index


def _pair(seed: int = 0, n: int = 5000):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=n)


@pytest.mark.parametrize("fn, floor", [(population_stability_index, 0.25), (kl_divergence, 0.1), (js_divergence, 0.05)])
def test_a_jump_in_the_nan_rate_is_reported_as_drift(fn, floor):
    ref, tgt = _pair()
    tgt = tgt.copy()
    tgt[: len(tgt) // 2] = np.nan
    assert fn(ref, tgt) > floor, f"{fn.__name__}: half the target turned NaN and the divergence did not move"


@pytest.mark.parametrize("fn", [population_stability_index, kl_divergence, js_divergence])
def test_the_same_nan_rate_on_both_sides_is_not_drift(fn):
    ref, tgt = _pair()
    ref, tgt = ref.copy(), tgt.copy()
    ref[::10] = np.nan
    tgt[::10] = np.nan
    assert fn(ref, tgt) < 0.05


@pytest.mark.parametrize("fn", [population_stability_index, kl_divergence, js_divergence])
def test_finite_samples_score_as_before(fn):
    """No non-finite rows means an empty extra bin on both sides, which contributes nothing."""
    ref, tgt = _pair(seed=3)
    small = fn(ref, tgt)
    assert np.isfinite(small) and small < 0.05
