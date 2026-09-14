"""A SIS-screened fit must describe the caller's input, not the survivor subset (mrmr_audit_2026-09-14 FIT_IMPL-2).

At or above ``sis_screen_threshold`` columns, ``fit`` narrows X to the screen's survivors before the rest of the fit runs, so
``n_features_in_``, ``feature_names_in_`` and ``support_`` all came out in SURVIVOR space. ``transform`` checks an ndarray's width against
``n_features_in_``, so ``fit(X).transform(X)`` raised on the caller's own training matrix, and on ndarray input the synthesized names were
subset positions (``feature_3`` meant "the 4th survivor"), so ``support_`` pointed at the wrong input columns. Named frames survived only
because the width check is skipped for them. The gate is on by default, so every wide enough ndarray fit hit this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._mrmr_sis_screen as sis
from mlframe.feature_selection.filters.mrmr import MRMR

_P, _N = 60, 800
_SIGNAL = (41, 53)  # deliberately far from the front, so subset positions and input positions cannot coincide
_SURVIVORS = 12


@pytest.fixture(autouse=True)
def _narrow_screen(monkeypatch):
    """Keep 12 survivors. The real survivor floor is at least 1000 columns, which would make this fixture slow for no extra coverage:
    everything under test happens after the screen has chosen, whatever the count."""
    monkeypatch.setattr(sis, "survivor_count", lambda fused, **kwargs: _SURVIVORS)


def _data(seed=0):
    """A wide matrix whose target depends on two late columns."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(_N, _P))
    y = ((X[:, _SIGNAL[0]] + X[:, _SIGNAL[1]] + 0.3 * rng.normal(size=_N)) > 0).astype(int)
    return X, y


def _mrmr():
    """A light, deterministic selector whose SIS gate fires at this fixture's width."""
    return MRMR(verbose=0, random_seed=0, fe_max_steps=0, fe_hinge_enable=False, dcd_enable=False, build_friend_graph=False, sis_screen_threshold=20)


def test_sis_screen_ran_so_this_test_observes_the_gate():
    """Precondition for the tests below: the screen must actually narrow the pool on this fixture."""
    X, y = _data()
    m = _mrmr().fit(X, y)
    assert getattr(m, "sis_survivors_", None) is not None, "the SIS gate did not run"
    assert len(m.sis_survivors_) < _P, "the SIS gate kept every column, so input and survivor space coincide"


def test_ndarray_fit_then_transform_on_the_same_matrix():
    """The caller's own training matrix must round-trip through transform."""
    X, y = _data()
    m = _mrmr().fit(X, y)
    assert m.n_features_in_ == _P, f"n_features_in_ is {m.n_features_in_}, the survivor count, not the input width {_P}"
    out = m.transform(X)
    assert out.shape == (_N, m.n_features_), f"transform returned {out.shape}"


def test_ndarray_support_indexes_the_input_columns():
    """support_ must point at the input columns that carry the signal, and the returned values must be those columns."""
    X, y = _data()
    m = _mrmr().fit(X, y)
    assert set(_SIGNAL) <= set(int(i) for i in m.support_), f"support_ {sorted(int(i) for i in m.support_)} misses the signal columns {_SIGNAL}"
    assert list(m.feature_names_in_) == [f"feature_{i}" for i in range(_P)]
    assert all(isinstance(n, str) for n in m.feature_names_in_)
    out = m.transform(X)
    np.testing.assert_array_equal(out, X[:, np.asarray(m.support_)])


def test_named_frame_reports_the_input_width_and_selects_the_same_columns():
    """Named frames already transformed by name; they must now also report input-space attributes."""
    X, y = _data()
    df = pd.DataFrame(X, columns=[f"c{i}" for i in range(_P)])
    m = _mrmr().fit(df, y)
    assert m.n_features_in_ == _P
    assert list(m.feature_names_in_) == list(df.columns)
    selected = [m.feature_names_in_[int(i)] for i in m.support_]
    assert {f"c{i}" for i in _SIGNAL} <= set(selected)
    assert list(m.transform(df).columns) == selected
