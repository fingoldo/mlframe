"""The Fleuret confidence kernel on the compacted columns counts the same failures as on the full matrix, permutation by permutation."""

from __future__ import annotations

import numba
import numpy as np
import pytest
from numba import types

from mlframe.feature_selection.filters._confirm_predictor_helpers import _compact_confidence_inputs
from mlframe.feature_selection.filters._numba_utils import get_fleuret_criteria_confidence


def _typed(src: dict) -> "numba.typed.Dict":
    """A unicode -> float64 typed dict holding ``src``."""
    out = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for k, v in src.items():
        out[k] = v
    return out


def _inputs(n: int = 6000, p: int = 40):
    """Binned columns with a target driven by columns 3 and 7, the candidate column 12 related to both."""
    rng = np.random.default_rng(2)
    data = rng.integers(0, 5, size=(n, p)).astype(np.int32)
    data[:, 12] = (data[:, 3] + rng.integers(0, 2, n)) % 5
    data[:, p - 1] = ((data[:, 3] + data[:, 7] + data[:, 12]) % 2).astype(np.int32)
    nbins = np.full(p, 5, dtype=np.int32)
    nbins[p - 1] = 2
    return data, nbins


@pytest.mark.parametrize("selected", [[3], [3, 7]])
def test_failure_counts_equal_on_compacted_columns(selected):
    """At thresholds through the null distribution (partial failure counts) both paths count the same failures; ``data_copy`` comes back unchanged."""
    data, nbins = _inputs()
    x, y = (12,), np.array([data.shape[1] - 1], dtype=np.int64)
    entropy: dict = {}
    counts = []
    for thr in np.linspace(0.0, 0.02, 9):
        kw = dict(
            npermutations=40, bootstrapped_gain=np.float32(thr), max_failed=1000, nexisting=0, base_seed=np.uint64(77),
            mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1, extra_x_shuffling=True,
        )
        before = data.copy()
        full = get_fleuret_criteria_confidence(
            data_copy=data, factors_nbins=nbins, x=x, y=y, selected_vars=list(selected), cached_cond_MIs=_typed({}), entropy_cache=_typed(entropy), **kw
        )
        assert np.array_equal(data, before)
        cd, cnb, cx, cy, cs, (ce, cc) = _compact_confidence_inputs(data, nbins, x, y, list(selected), _typed(entropy))
        compact = get_fleuret_criteria_confidence(data_copy=cd, factors_nbins=cnb, x=cx, y=cy, selected_vars=cs, cached_cond_MIs=cc, entropy_cache=ce, **kw)
        assert compact == full, thr
        counts.append(full[0])
    assert any(0 < c < 40 for c in counts), counts  # the grid reaches the null: some thresholds give partial counts


def test_compacted_x_has_one_type_whatever_its_arity():
    """A tuple ``X`` is typed by its length in numba (``UniTuple(int64, k)``), so each new arity compiled the kernel again inside the fit; the compacted
    ``X`` is an int64 array of any length, and a two-column tuple gives the same verdict as on the full matrix."""
    data, nbins = _inputs()
    y = np.array([data.shape[1] - 1], dtype=np.int64)
    kw = dict(
        npermutations=30, bootstrapped_gain=np.float32(0.005), max_failed=1000, nexisting=0, base_seed=np.uint64(5),
        mrmr_relevance_algo="fleuret", mrmr_redundancy_algo="fleuret", max_veteranes_interactions_order=1, extra_x_shuffling=True,
    )
    types_seen = set()
    for x in ((12,), (12, 7), (12, 7, 20)):
        cd, cnb, cx, cy, cs, (ce, cc) = _compact_confidence_inputs(data, nbins, x, y, [3], _typed({}))
        types_seen.add((type(cx), cx.dtype, cx.ndim))
        if len(x) == 2:
            full = get_fleuret_criteria_confidence(data_copy=data, factors_nbins=nbins, x=x, y=y, selected_vars=[3], cached_cond_MIs=_typed({}), entropy_cache=_typed({}), **kw)
            compact = get_fleuret_criteria_confidence(data_copy=cd, factors_nbins=cnb, x=cx, y=cy, selected_vars=cs, cached_cond_MIs=cc, entropy_cache=ce, **kw)
            assert compact == full
    assert types_seen == {(np.ndarray, np.dtype(np.int64), 1)}, types_seen
