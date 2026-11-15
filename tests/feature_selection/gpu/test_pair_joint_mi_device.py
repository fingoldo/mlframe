"""The device per-pair joint MI reduction equals the host reference over the same flat count buffer."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
from mlframe.feature_selection.filters import _gpu_pairs as g
from tests.feature_selection.gpu._gpu_support import has_device

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("nb_y", [2, 5, 33])
def test_device_reduction_equals_host_reference(nb_y):
    """Random multinomial tables of unequal sizes: device and host MI agree to 1e-12, and the MI values are not all zero."""
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    rng = np.random.default_rng(nb_y)
    n_pairs, n = 200, 5000
    merged = rng.integers(2, 40, n_pairs).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(merged * nb_y)[:-1]]).astype(np.int32)
    counts = np.zeros(int((merged * nb_y).sum()), np.int32)
    for k in range(n_pairs):
        cells = int(merged[k]) * nb_y
        p = rng.dirichlet(np.ones(cells) * 0.3)
        counts[off[k] : off[k] + cells] = rng.multinomial(n, p)
    ref = np.zeros(n_pairs)
    g.joint_mi_from_flat_counts(counts, off, merged, nb_y, n_pairs, float(n), ref)
    out = g.joint_mi_on_device(cp, cp.asarray(counts), cp.asarray(off), merged, nb_y, n_pairs, float(n))
    assert float(ref.max()) > 0
    assert np.abs(out - ref).max() < 1e-12
