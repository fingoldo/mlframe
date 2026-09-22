"""The geometric-mean blend must clip member probabilities to [0, 1] before blending, like every other flavour."""

from __future__ import annotations

import numpy as np

from mlframe.models.ensembling.base import combine_probs


def test_an_out_of_range_member_does_not_inflate_the_geometric_mean():
    good = np.array([[0.2, 0.8], [0.6, 0.4]])
    leaky = np.array([[0.2, 1.4], [0.6, 0.4]])  # a raw margin leaking through as a "probability"
    clipped = np.clip(leaky, 0.0, 1.0)
    got = combine_probs(np.stack([good, leaky]), "geo")
    expected = combine_probs(np.stack([good, clipped]), "geo")
    np.testing.assert_allclose(got, expected, rtol=1e-12)
