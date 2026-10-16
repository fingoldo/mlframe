"""The presorted MDLP cut search of the honest null returns exactly the cuts of ``mdlp_inner_cuts`` for every permuted target."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._mdlp_honest_njit import _mdlp_cuts_presorted, _presort_columns, mdlp_inner_cuts


@pytest.mark.parametrize("n_classes", [2, 3])
def test_presorted_cuts_equal_reference_on_ties_and_discrete_columns(n_classes):
    rng = np.random.default_rng(0)
    n = 3000
    raw = np.ascontiguousarray(
        np.column_stack([rng.standard_normal(n), np.round(rng.standard_normal(n), 1), rng.integers(0, 7, n).astype(float), np.exp(2 * rng.standard_normal(n))])
    )
    sig = (raw[:, 0] + rng.standard_normal(n) > 0).astype(np.int64) if n_classes == 2 else rng.integers(0, 3, n)
    order = _presort_columns(raw)
    for p in range(15):
        y = sig if p == 0 else sig[rng.permutation(n)]
        for j in range(raw.shape[1]):
            ref = mdlp_inner_cuts(raw[:, j], y, n_classes, 50, 8, 0.05, 99, 7, 1000, 5.0)
            got = _mdlp_cuts_presorted(raw[order[j], j], y[order[j]], n_classes, 50, 8, 0.05, 99, 7, 1000, 5.0)
            np.testing.assert_array_equal(ref, got)
