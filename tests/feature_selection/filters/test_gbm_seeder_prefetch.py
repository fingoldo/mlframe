"""The surrogate-seeder prefetch hands its result only to the identical call, and the seeder falls back otherwise."""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._gbm_seeder_prefetch as P
import mlframe.feature_selection.filters._surrogate_interaction_seeder as S


class _Est:
    """The estimator attributes the prefetch reads."""

    fe_gbm_seeder_enable = True
    fe_gbm_seeder_min_features = 2
    factors_to_use = None
    factors_names_to_use = None
    random_seed = 3

    def __init__(self, names):
        self.feature_names_in_ = np.array(names)


def _setup(monkeypatch):
    """A recording stub for the seeder, a small frame, its codes and target."""
    calls = []

    def stub(data, y, cand, **kw):
        """Record the call and return a value derived from its arguments."""
        calls.append((id(data), tuple(cand)))
        return [(cand[0], cand[1])], [], {"n": len(cand)}

    monkeypatch.setattr(S, "surrogate_gbm_interaction_seeds", stub)
    cols = ["a", "b", "c", "y"]
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0], "y": [0, 1]})
    data = np.zeros((2, 4), dtype=np.int32)
    y = np.array([0, 1])
    return calls, cols, X, data, y


def test_identical_call_takes_the_prefetched_result(monkeypatch):
    """The prefetched call runs once; the seeder's identical call gets its result without calling again."""
    calls, cols, X, data, y = _setup(monkeypatch)
    est = _Est(cols[:3])
    P.prefetch_surrogate_seeds(est, num_fs_steps=0, data=data, cols=cols, X=X, target_indices=[3], categorical_vars=[], classes_y=y)
    cand = P.seeder_candidates(est, cols, [3], [], set())
    got = P.take_prefetched(est, data, y, cand, P.seeder_kwargs(est))
    assert got == ([(0, 1)], [], {"n": 3}) and calls == [(id(data), (0, 1, 2))]


def test_other_arguments_fall_back(monkeypatch):
    """A different candidate list (or matrix) gets None: the seeder computes its own call."""
    _calls, cols, X, data, y = _setup(monkeypatch)
    est = _Est(cols[:3])
    P.prefetch_surrogate_seeds(est, num_fs_steps=0, data=data, cols=cols, X=X, target_indices=[3], categorical_vars=[], classes_y=y)
    assert P.take_prefetched(est, data, y, [0, 1], P.seeder_kwargs(est)) is None
    assert P.take_prefetched(est, data, y, [0, 1, 2], P.seeder_kwargs(est)) is None  # the entry was consumed by the first take


def test_off_switch_and_later_steps(monkeypatch):
    """No prefetch on a later FE step or with MLFRAME_GBM_SEEDER_PREFETCH=0."""
    calls, cols, X, data, y = _setup(monkeypatch)
    est = _Est(cols[:3])
    P.prefetch_surrogate_seeds(est, num_fs_steps=1, data=data, cols=cols, X=X, target_indices=[3], categorical_vars=[], classes_y=y)
    monkeypatch.setenv("MLFRAME_GBM_SEEDER_PREFETCH", "0")
    P.prefetch_surrogate_seeds(est, num_fs_steps=0, data=data, cols=cols, X=X, target_indices=[3], categorical_vars=[], classes_y=y)
    assert calls == [] and P.take_prefetched(est, data, y, [0, 1, 2], P.seeder_kwargs(est)) is None


def test_same_values_in_a_new_matrix_object_take_the_result(monkeypatch):
    """The match is by value: the screening may hand the step a new matrix object with the same candidate columns and target, and the result is used."""
    calls, cols, X, data, y = _setup(monkeypatch)
    est = _Est(cols[:3])
    P.prefetch_surrogate_seeds(est, num_fs_steps=0, data=data, cols=cols, X=X, target_indices=[3], categorical_vars=[], classes_y=y)
    cand = P.seeder_candidates(est, cols, [3], [], set())
    wider = np.hstack([data, np.ones((2, 2), dtype=np.int32)])  # extra engineered columns after the raw ones
    got = P.take_prefetched(est, wider, y.copy(), cand, P.seeder_kwargs(est))
    assert got == ([(0, 1)], [], {"n": 3}) and len(calls) == 1


def test_changed_candidate_values_or_target_fall_back(monkeypatch):
    """A matrix whose candidate columns differ, or another target, gets None."""
    _calls, cols, X, data, y = _setup(monkeypatch)
    est = _Est(cols[:3])
    cand = P.seeder_candidates(est, cols, [3], [], set())
    for other_data, other_y in ((data + 1, y), (data, y[::-1].copy() + 1)):
        P.prefetch_surrogate_seeds(est, num_fs_steps=0, data=data, cols=cols, X=X, target_indices=[3], categorical_vars=[], classes_y=y)
        assert P.take_prefetched(est, other_data, other_y, cand, P.seeder_kwargs(est)) is None


def test_the_step_entry_call_does_not_resubmit_an_early_start(monkeypatch):
    """The early start and the step-entry call carry the same values: the fits run once."""
    calls, cols, X, data, y = _setup(monkeypatch)
    est = _Est(cols[:3])
    P.prefetch_surrogate_seeds(est, num_fs_steps=0, data=data, cols=cols, X=X, target_indices=[3], categorical_vars=[], classes_y=y)
    P.prefetch_surrogate_seeds(est, num_fs_steps=0, data=data.copy(), cols=cols, X=X, target_indices=[3], categorical_vars=[], classes_y=y.copy())
    cand = P.seeder_candidates(est, cols, [3], [], set())
    assert P.take_prefetched(est, data, y, cand, P.seeder_kwargs(est)) is not None and len(calls) == 1
