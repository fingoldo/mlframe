"""The GPU external-validation binner returns the host binner's codes for candidates holding NaN (edges over the finite rows only; tracker 2026-11-02 wave 13, PAR13-1) and still hands infinite candidates to the host."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_extval as ev
from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch

pytestmark = pytest.mark.skipif(not cp.cuda.is_available(), reason="needs a CUDA device")


def _host_codes(a, bs, ops, nbins):
    """The host path: numpy ops of the external-validation sweep, then the host quantile binner."""
    cols = []
    for b in bs:
        for o in ops:
            with np.errstate(all="ignore"):
                cols.append({0: a * b, 1: a + b, 2: a - b}[int(o)])
    return discretize_2d_quantile_batch(np.stack(cols, axis=1), nbins, np.int8)


def test_nan_in_an_external_factor_gives_the_host_codes():
    """30% NaN in the external factor: the codes equal the host binner's (NaN to the top bin), for several ops and a second finite factor."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    b_nan = b.copy()
    b_nan[:300] = np.nan
    ops = np.array([0, 1, 2], dtype=np.int64)
    got = ev.gpu_materialise_extval_codes_host(a, [b_nan, b], ops, 10)
    assert got is not None
    assert np.array_equal(got, _host_codes(a, [b_nan, b], ops, 10), equal_nan=True)
    assert ev.gpu_materialise_extval_codes_host(a, [b], ops, 10) is not None


def test_an_infinite_candidate_is_handed_to_the_host():
    """An infinite entry makes the interpolated edge undefined: the function declines (None) and the caller keeps the host binner."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=1000)
    b = rng.normal(size=1000)
    b[5] = np.inf
    assert ev.gpu_materialise_extval_codes_host(a, [b], np.array([1], dtype=np.int64), 10) is None
