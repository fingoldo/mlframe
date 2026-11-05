"""The thread-parallel Fleuret confidence equals the serial kernel's ``(nfailed, nchecked)`` on the compacted inputs, early stop included."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._confirm_predictor_helpers import _compact_confidence_inputs
from mlframe.feature_selection.filters._fleuret_confidence_prange import fleuret_confidence_prange
from mlframe.feature_selection.filters._numba_utils import get_fleuret_criteria_confidence

from tests.feature_selection.filters.test_fleuret_confidence_compact import _inputs, _typed


@pytest.mark.parametrize("selected,x", [([3], (12,)), ([3, 7], (12,)), ([3], (12, 7))])
@pytest.mark.parametrize("max_failed", [1000, 3])
@pytest.mark.parametrize("block", [1, 3, 8])
def test_prange_equals_serial(selected, x, max_failed, block):
    """Same counts at thresholds through the null (partial failure counts and early stops) for several block sizes."""
    data, nbins = _inputs()
    y = np.array([data.shape[1] - 1], dtype=np.int64)
    for thr in np.linspace(0.0, 0.02, 7):
        kw = dict(npermutations=40, bootstrapped_gain=float(thr), max_failed=max_failed, nexisting=0, base_seed=np.uint64(77),
                  mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1, extra_x_shuffling=True)
        cd, cnb, cx, cy, cs, (ce, cc) = _compact_confidence_inputs(data, nbins, x, y, list(selected), _typed({}))
        serial = get_fleuret_criteria_confidence(data_copy=cd.copy(), factors_nbins=cnb, x=cx, y=cy, selected_vars=cs, cached_cond_MIs=cc, entropy_cache=ce, **kw)
        par = fleuret_confidence_prange(cd, cnb, cx, cy, cs, block=block, **kw)
        assert tuple(par) == tuple(serial), (thr, par, serial)
