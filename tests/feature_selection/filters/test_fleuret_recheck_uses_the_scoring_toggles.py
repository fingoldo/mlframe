"""The Fleuret recheck confirms a candidate with the criterion its gain was scored with (SU, Miller-Madow, JMIM), whatever ``n_workers`` is.

The toggles used to reach the confirmation only on the multi-worker route; with the default ``n_workers=1`` the permutation null was the plug-in conditional MI while the expected gain it was
compared with was scored with the toggle (a unit mismatch on the SU scale). The recheck now reads the toggles once per screen and forwards them on every route.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters._confirm_predictor_helpers import _compact_confidence_inputs, _host_recheck_counts, _recheck_counts
from mlframe.feature_selection.filters._numba_utils import get_fleuret_criteria_confidence


def _ctx(toggles, n_workers=1, permutations=12):
    """A screen-context stand-in with the fields the host recheck reads and the toggles preset (as the walk presets them in the fitting thread)."""
    rng = np.random.default_rng(3)
    n = 5000
    cols = [rng.integers(0, 4, n) for _ in range(5)]
    y = (cols[0] + cols[1] + rng.integers(0, 2, n)) % 3
    data = np.column_stack([*cols, y]).astype(np.int8)
    ctx = SimpleNamespace(
        data_copy=data, factors_nbins=np.asarray([4] * 5 + [3], dtype=np.int64), y=(5,), selected_vars=[0, 1], entropy_cache=None, full_npermutations=permutations, max_failed=3,
        mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1, extra_x_shuffling=True, n_workers=n_workers,
        use_simple_mode=False, random_seed=0,
    )
    ctx.__dict__["fleuret_toggles"] = toggles
    return ctx


def _direct(ctx, X, gain, seed, toggles):
    """The serial kernel called with the toggles, as the recheck has to."""
    cd, cnb, cx, cy, csel, cent = _compact_confidence_inputs(ctx.data_copy.copy(), ctx.factors_nbins, X, ctx.y, ctx.selected_vars, None)
    return get_fleuret_criteria_confidence(
        data_copy=cd, factors_nbins=cnb, x=cx, y=cy, selected_vars=csel, bootstrapped_gain=float(gain), npermutations=ctx.full_npermutations, max_failed=ctx.max_failed, nexisting=0,
        mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1, cached_cond_MIs=cent[1], entropy_cache=cent[0], extra_x_shuffling=True,
        base_seed=np.uint64(seed), use_su=toggles[0], use_mm=toggles[1], use_jmim=toggles[2], jmim_discount_only=toggles[3],
    )


@pytest.mark.parametrize("toggles", [(True, False, False, False), (False, True, False, False), (False, False, True, False)])
def test_single_worker_recheck_forwards_the_toggles(toggles):
    """With one worker the recheck counts equal the serial kernel run WITH the toggles, and differ from the plug-in statistic for at least one gain."""
    ctx = _ctx(toggles)
    plain = _ctx((False, False, False, False))
    differs = False
    for gain in (0.0, *np.geomspace(1e-5, 0.5, 40)):
        got = tuple(_host_recheck_counts(ctx, (4,), gain, 0, 777))
        assert got == tuple(_direct(ctx, (4,), gain, 777, toggles)), (toggles, gain)
        differs |= got != tuple(_host_recheck_counts(plain, (4,), gain, 0, 777))
    assert differs, "the toggle changed no count over the gains tried: the comparison proves nothing"


def test_the_segmented_route_forwards_the_same_toggles():
    """With several workers the segmented route (``fleuret_segment_counts``, no card needed) uses the same toggles, so the routes agree on the statistic."""
    toggles = (True, False, False, False)
    one = tuple(_recheck_counts(_ctx(toggles, n_workers=1), (4,), 0.002, 0, 777))
    assert one == tuple(_direct(_ctx(toggles), (4,), 0.002, 777, toggles))
    seg = tuple(_recheck_counts(_ctx(toggles, n_workers=3), (4,), 0.002, 0, 777))
    assert 1 <= seg[1] and seg[0] <= seg[1]
