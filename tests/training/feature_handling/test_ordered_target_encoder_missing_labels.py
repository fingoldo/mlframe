"""Ordered target encoding with rows that have no label: they are encoded, but feed nothing forward.

The oracle is the definition written as a plain loop: in causal order, a row's encoding is its category's sum and count
of strictly earlier LABELLED rows, smoothed toward the prior (the labelled mean, or the expanding labelled mean of
strictly earlier rows, falling back to 0.0 when there is none).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode, ordered_target_encode_batch


def _oracle(cats, y, order, smoothing, causal_prior):
    """Reference ordered target encoding computed row by row over the sorted order."""
    sort_idx = np.argsort(order, kind="mergesort")
    labelled_mean = float(np.nanmean(y))
    out = np.empty(len(y))
    sums, counts = {}, {}
    g_sum, g_count = 0.0, 0
    for i in sort_idx:
        prior = (g_sum / g_count if g_count else 0.0) if causal_prior else labelled_mean
        c = cats[i]
        out[i] = (sums.get(c, 0.0) + smoothing * prior) / (counts.get(c, 0) + smoothing)
        if not np.isnan(y[i]):
            sums[c] = sums.get(c, 0.0) + y[i]
            counts[c] = counts.get(c, 0) + 1
            g_sum += y[i]
            g_count += 1
    return out


@pytest.mark.parametrize("causal_prior", [False, True])
@pytest.mark.parametrize("missing_share", [0.0, 0.3])
def test_the_encoding_matches_its_definition(causal_prior, missing_share):
    """The encoding matches its definition."""
    rng = np.random.default_rng(1)
    n = 500
    cats = rng.choice(list("abcde"), n)
    y = rng.normal(size=n) + (cats == "a") * 2.0
    y[rng.random(n) < missing_share] = np.nan
    order = rng.permutation(n).astype(float)
    expected = _oracle(cats, y, order, smoothing=1.0, causal_prior=causal_prior)
    got = ordered_target_encode(cats, y, order=order, smoothing=1.0, causal_prior=causal_prior)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
    assert np.isfinite(got).all(), "an unlabelled row is still encoded from the labelled rows before it"
    batch = ordered_target_encode_batch({"c": cats}, y, order=order, smoothing=1.0, causal_prior=causal_prior)["c"]
    np.testing.assert_allclose(batch, expected, rtol=1e-10, atol=1e-12)
