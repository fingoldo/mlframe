"""The rescue's pair-MI probe widens the class-id workspace to the product of the two columns' bin counts."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._finalise import pair_probe_dtype
from mlframe.feature_selection.filters.info_theory import mi


def test_int8_request_is_widened_for_a_wide_pair():
    """30 x 30 bins need 900 class ids: int8 is widened, a small pair keeps int8."""
    nbins = np.array([30, 30, 4, 5])
    assert np.iinfo(pair_probe_dtype(np.int8, nbins, 0, 1)).max >= 900
    assert pair_probe_dtype(np.int8, nbins, 2, 3) is np.int8


def test_wide_pair_probe_runs_where_int8_raises():
    """The same pair through ``mi``: the requested int8 raises, the widened dtype gives a finite MI."""
    rng = np.random.default_rng(0)
    n = 4000
    data = np.stack([rng.integers(0, 30, n), rng.integers(0, 30, n), rng.integers(0, 2, n)], axis=1).astype(np.int8)
    nbins = np.array([30, 30, 2])
    kwargs = dict(factors_data=data, x=np.array([0], dtype=np.int64), y=np.array([1], dtype=np.int64), factors_nbins=nbins)
    try:
        mi(dtype=np.int8, **kwargs)
        raised = False
    except ValueError:
        raised = True
    assert raised
    value = float(mi(dtype=pair_probe_dtype(np.int8, nbins, 0, 1), **kwargs))
    assert np.isfinite(value) and value >= 0.0
