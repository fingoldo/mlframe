"""Proposer and held-out gate of the target-transform search (PLAN_target_transform_in_mrmr, step 2; the functions are not wired into MRMR yet).

The cases are the ones of ``_benchmarks/binning_scheme/target_transform_mi_ext.py``: a cubic target that cbrt linearises, a lognormal target that log linearises, and a plain linear target on which the proposer is known
to rank ``signed_sqrt`` first (a false positive the gate must reject).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._target_transform_search import (
    TargetTransform,
    default_transforms,
    gate_target_transform,
    propose_target_transforms,
    search_target_transform,
)

N, P = 6000, 6


def _data(kind: str, seed: int = 0):
    """Features and a target of the given kind."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, P))
    noise = rng.standard_normal(N)
    if kind == "cube":
        y = (X[:, 0] + 0.7 * X[:, 1] + 0.3 * noise) ** 3
    elif kind == "lognormal":
        y = np.exp(0.9 * X[:, 0] + 0.6 * X[:, 1] + 0.4 * noise)
    else:
        y = X[:, 0] + 0.7 * X[:, 1] + 0.3 * noise
    return X, y


def _by_name(name: str) -> TargetTransform:
    """The default transform called ``name``."""
    return next(t for t in default_transforms() if t.name == name)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cubic_target_cbrt_is_proposed_and_accepted(seed):
    """The proposer ranks cbrt first for a cubic target and the gate accepts it; the search returns it."""
    X, y = _data("cube", seed)
    assert propose_target_transforms(X, y)[0][0] == "cbrt"
    assert gate_target_transform(X, y, _by_name("cbrt"), seed=seed).accepted
    assert search_target_transform(X, y, seed=seed)[0].name == "cbrt"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lognormal_target_log_is_accepted(seed):
    """For a lognormal target the gate accepts log, and the search returns an accepted transform that lowers the error."""
    X, y = _data("lognormal", seed)
    assert gate_target_transform(X, y, _by_name("log"), seed=seed).accepted
    chosen, results = search_target_transform(X, y, seed=seed)
    assert chosen.name != "identity" and any(r.accepted for r in results)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_target_the_signed_sqrt_false_positive_is_rejected(seed):
    """On a plain linear target the proposer still ranks signed_sqrt first (the measured bias) and the gate rejects it: the search returns the identity."""
    X, y = _data("linear", seed)
    assert propose_target_transforms(X, y)[0][0] == "signed_sqrt"
    assert not gate_target_transform(X, y, _by_name("signed_sqrt"), seed=seed).accepted
    assert search_target_transform(X, y, seed=seed)[0].name == "identity"


def test_log_is_skipped_for_a_target_that_is_not_positive():
    """A target with non-positive values never gets log: it is absent from the proposals and the gate refuses it."""
    X, y = _data("linear")
    assert "log" not in {name for name, _ in propose_target_transforms(X, y)}
    assert not gate_target_transform(X, y, _by_name("log")).accepted


def test_the_identity_is_never_accepted_and_non_finite_input_rows_are_dropped():
    """The identity has a zero difference to itself; NaN rows in X or y are removed before anything is fitted."""
    X, y = _data("cube")
    assert not gate_target_transform(X, y, _by_name("identity")).accepted
    X2, y2 = X.copy(), y.copy()
    X2[5, 0] = np.nan
    y2[9] = np.inf
    assert gate_target_transform(X2, y2, _by_name("cbrt")).accepted
