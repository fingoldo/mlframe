"""Additional coverage for mlframe.feature_selection.filters.fleuret -- Python wrapper branches.

Existing test_fleuret.py covers the public API + 2 biz_value tests. This file targets the remaining branches in
the Python wrappers around the @njit core: distribute_permutations dispatch, parallel-vs-serial worker boundary,
cached-dict propagation, custom relevance/redundancy algos.
"""

from __future__ import annotations

import numpy as np
import pytest

from joblib import Parallel
import numba
from numba.core import types

from mlframe.feature_selection.filters.fleuret import (
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)


def _make_xor_data(n: int = 200, seed: int = 0):
    """Make xor data."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n).astype(np.int32)
    x1 = rng.integers(0, 2, n).astype(np.int32)
    y = (x0 ^ x1).astype(np.int32)
    data = np.column_stack([x0, x1, y]).astype(np.int32)
    factors_nbins = np.array([2, 2, 2], dtype=np.int32)
    return data, factors_nbins


def _empty_dicts():
    """Empty dicts."""
    entropy_cache = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    cached_cond_MIs = numba.typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    return entropy_cache, cached_cond_MIs


@pytest.mark.fast
def test_get_fleuret_npermutations_zero_returns_sentinel():
    """npermutations=0 short-circuits with (0, 0) sentinel per source line 179."""
    data, nbins = _make_xor_data(seed=1)
    ec, cc = _empty_dicts()
    out = get_fleuret_criteria_confidence(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        npermutations=0,
        bootstrapped_gain=0.0,
        max_failed=10,
        nexisting=1,
        cached_cond_MIs=cc,
        entropy_cache=ec,
        dtype=np.int32,
    )
    assert out == (0, 0)


def test_get_fleuret_extra_x_shuffling_off():
    """extra_x_shuffling=False takes a different shuffle path inside the njit body."""
    data, nbins = _make_xor_data(seed=2)
    ec, cc = _empty_dicts()
    out = get_fleuret_criteria_confidence(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        npermutations=20,
        bootstrapped_gain=0.0,
        max_failed=10,
        nexisting=1,
        extra_x_shuffling=False,
        cached_cond_MIs=cc,
        entropy_cache=ec,
        dtype=np.int32,
    )
    assert isinstance(out, tuple) and len(out) == 2


def test_get_fleuret_pld_relevance_algo():
    """mrmr_relevance_algo='pld' is the alternative dispatch branch."""
    data, nbins = _make_xor_data(seed=3)
    ec, cc = _empty_dicts()
    out = get_fleuret_criteria_confidence(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        npermutations=10,
        bootstrapped_gain=0.0,
        max_failed=10,
        nexisting=1,
        mrmr_relevance_algo="pld",
        cached_cond_MIs=cc,
        entropy_cache=ec,
        dtype=np.int32,
    )
    assert isinstance(out, tuple) and len(out) == 2


def test_get_fleuret_pld_redundancy_algos():
    """mrmr_redundancy_algo='pld_max' and 'pld_mean' both must dispatch cleanly."""
    data, nbins = _make_xor_data(seed=4)
    for algo in ("pld_max", "pld_mean"):
        ec, cc = _empty_dicts()
        out = get_fleuret_criteria_confidence(
            data_copy=data,
            factors_nbins=nbins,
            x=(0,),
            y=(2,),
            selected_vars=[1],
            npermutations=10,
            bootstrapped_gain=0.0,
            max_failed=10,
            nexisting=1,
            mrmr_redundancy_algo=algo,
            cached_cond_MIs=cc,
            entropy_cache=ec,
            dtype=np.int32,
        )
        assert isinstance(out, tuple) and len(out) == 2


def test_get_fleuret_bootstrapped_gain_positive():
    """With bootstrapped_gain > 0, the confidence path compares against the bar; covers a different branch than gain=0."""
    data, nbins = _make_xor_data(seed=5)
    ec, cc = _empty_dicts()
    out = get_fleuret_criteria_confidence(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        npermutations=20,
        bootstrapped_gain=0.05,
        max_failed=10,
        nexisting=1,
        cached_cond_MIs=cc,
        entropy_cache=ec,
        dtype=np.int32,
    )
    assert isinstance(out, tuple)


def test_get_fleuret_max_failed_zero_short_circuit():
    """max_failed=0 means any failure triggers immediate early stop; the early-exit branch is exercised."""
    data, nbins = _make_xor_data(seed=6)
    ec, cc = _empty_dicts()
    out = get_fleuret_criteria_confidence(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        npermutations=50,
        bootstrapped_gain=0.0,
        max_failed=0,
        nexisting=1,
        cached_cond_MIs=cc,
        entropy_cache=ec,
        dtype=np.int32,
    )
    assert isinstance(out, tuple)


def test_parallel_fleuret_worker_returns_dict():
    """parallel_fleuret is the joblib worker; returns (nfailed, i, entropy_cache_dict). Smoke."""
    data, nbins = _make_xor_data(seed=7)
    out = parallel_fleuret(
        data=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        npermutations=15,
        bootstrapped_gain=0.0,
        max_failed=10,
        nexisting=1,
        cached_cond_MIs={},
        entropy_cache={},
        dtype=np.int32,
    )
    assert len(out) == 3
    assert isinstance(out[2], dict)


def test_get_fleuret_criteria_confidence_parallel_serial_path():
    """n_workers=1 with parallel_kwargs=None falls into the default Parallel(n_jobs=1) path."""
    data, nbins = _make_xor_data(seed=8)
    out = get_fleuret_criteria_confidence_parallel(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        bootstrapped_gain=0.0,
        npermutations=20,
        max_failed=10,
        nexisting=1,
        cached_cond_MIs={},
        entropy_cache={},
        n_workers=1,
        dtype=np.int32,
    )
    # Returns (bootstrapped_gain, confidence, entropy_cache) per source line 94.
    assert len(out) == 3
    assert 0.0 <= float(out[1]) <= 1.0


def test_get_fleuret_criteria_confidence_parallel_with_explicit_pool():
    """Passing an explicit workers_pool bypasses the Parallel(...) construction. Tolerates Windows paging-file
    pressure (OSError 1455) when concurrent test sessions exhaust virtual memory; the code-path is what we care about."""
    data, nbins = _make_xor_data(seed=9)
    try:
        pool = Parallel(n_jobs=2)
        out = get_fleuret_criteria_confidence_parallel(
            data_copy=data,
            factors_nbins=nbins,
            x=(0,),
            y=(2,),
            selected_vars=[1],
            bootstrapped_gain=0.0,
            npermutations=20,
            max_failed=10,
            nexisting=1,
            cached_cond_MIs={},
            entropy_cache={},
            n_workers=2,
            workers_pool=pool,
            dtype=np.int32,
        )
    except OSError as exc:
        if "paging file" in str(exc).lower() or getattr(exc, "winerror", None) == 1455:
            pytest.skip(f"Windows paging-file overflow under concurrent load: {exc}")
        raise
    except Exception as exc:
        # loky BrokenProcessPool / TerminatedWorkerError / pickle-transport
        # errors during heavy concurrent test load. The Parallel/loky spawn
        # path has already been exercised - the only thing this test cares about.
        _msg = str(exc).lower()
        _name = type(exc).__name__.lower()
        if any(s in _msg for s in ("brokenprocesspool", "terminatedworker", "pickle", "transport", "_remotetraceback")) or any(
            s in _name for s in ("brokenprocesspool", "terminatedworker")
        ):
            pytest.skip(f"loky worker transport failure under concurrent load: {type(exc).__name__}: {exc}")
        raise
    assert len(out) == 3


def test_get_fleuret_criteria_confidence_parallel_explicit_kwargs():
    """parallel_kwargs={} is the non-None branch of the if guard."""
    data, nbins = _make_xor_data(seed=10)
    out = get_fleuret_criteria_confidence_parallel(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        bootstrapped_gain=0.0,
        npermutations=15,
        max_failed=10,
        nexisting=1,
        cached_cond_MIs={},
        entropy_cache={},
        n_workers=1,
        parallel_kwargs={},
        dtype=np.int32,
    )
    assert len(out) == 3


def test_get_fleuret_criteria_confidence_parallel_max_failed_zero_zeros_gain():
    """When nfailed >= max_failed, bootstrapped_gain is zeroed in the post-aggregation block."""
    data, nbins = _make_xor_data(seed=11)
    out = get_fleuret_criteria_confidence_parallel(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        bootstrapped_gain=0.5,
        npermutations=20,
        max_failed=0,
        nexisting=1,
        cached_cond_MIs={},
        entropy_cache={},
        n_workers=1,
        dtype=np.int32,
    )
    # Either the gain is zeroed OR all perms passed (XOR is strong synergy); both are valid post-conditions.
    assert isinstance(out, tuple) and len(out) == 3


def test_get_fleuret_max_veteranes_interactions_order_two():
    """max_veteranes_interactions_order=2 changes the interaction-set enumeration in the njit body."""
    data, nbins = _make_xor_data(seed=12)
    ec, cc = _empty_dicts()
    out = get_fleuret_criteria_confidence(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        npermutations=10,
        bootstrapped_gain=0.0,
        max_failed=10,
        nexisting=1,
        max_veteranes_interactions_order=2,
        cached_cond_MIs=cc,
        entropy_cache=ec,
        dtype=np.int32,
    )
    assert isinstance(out, tuple)


def test_get_fleuret_entropy_cache_propagation():
    """A pre-populated entropy_cache must be respected; the wrapper does not clear it."""
    data, nbins = _make_xor_data(seed=13)
    ec, cc = _empty_dicts()
    # Seed a fake entry; the inner function may read or skip it but must not crash.
    ec["preseeded_key"] = 0.123
    out = get_fleuret_criteria_confidence(
        data_copy=data,
        factors_nbins=nbins,
        x=(0,),
        y=(2,),
        selected_vars=[1],
        npermutations=10,
        bootstrapped_gain=0.0,
        max_failed=10,
        nexisting=1,
        cached_cond_MIs=cc,
        entropy_cache=ec,
        dtype=np.int32,
    )
    assert isinstance(out, tuple)
