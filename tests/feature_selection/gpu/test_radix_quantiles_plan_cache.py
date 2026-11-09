"""The radix quantiles equal ``cp.percentile`` bit for bit and a repeated quantile set reuses its cached plan."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_select as sel


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_equals_percentile_and_reuses_the_plan(dtype):
    """Same values as ``cp.percentile`` on two shapes and one quantile set; the plan table gains one entry per distinct (rows, quantiles), not one per call."""
    r = np.random.default_rng(0)
    q = [0.25, 0.5, 0.75]
    for n in (3001, 3001, 4500):
        x = cp.asarray(r.standard_normal((n, 7)).astype(dtype))
        got = sel._radix_quantiles(x, q)
        want = cp.percentile(x.astype(cp.float64), [v * 100 for v in q], axis=0)
        assert got is not None
        np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(want))
    keys = [k for k in sel._QUANTILE_PLANS if k[0] in (3001, 4500) and k[1] == tuple(q)]
    assert len(keys) == 2
