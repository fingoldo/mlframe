"""Pickle round-trip bit-identity gate for :class:`CompositeTargetEstimator`.

A fitted estimator that is pickled + unpickled must predict BIT-IDENTICALLY
pre/post pickle, and its ``fitted_params_`` (the reproducible-inversion dict)
must survive the round trip unchanged. The risk this gate pins: any
runtime-cache exclusion from ``__getstate__`` (added for the 100+GB no-copy
contract -- a ``__getstate__`` must drop the pinned ``_df_ref`` / live buffers,
never a fitted param) silently dropping a param that ``predict`` / ``inverse``
needs. A dropped ``alpha`` / ``p`` / per-group coefficient would not crash --
it would silently change predictions, which only a bit-identity assertion
catches.

Covered transforms span the structurally-distinct inverse families:

* ``linear_residual`` -- two scalar params (alpha, beta).
* ``ratio`` -- multiplicative inverse, domain-sensitive base.
* ``ewma_residual`` -- left-recurrent, carries a ``tail_anchor`` / ``anchor``.
* ``signed_power_y`` -- base-free unary, fitted exponent ``p``.
* ``linear_residual_grouped`` -- per-group coefficient table (a dict-valued
  fitted param, the easiest one to lose to a careless ``__getstate__``).
"""
from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


# ----------------------------------------------------------------------
# Synthetic frames -- one per transform family, all deterministic.
# ----------------------------------------------------------------------


def _base_frame(n: int = 800, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = np.linspace(1.0, 20.0, n) + rng.normal(0.0, 0.05, n)
    feat = rng.normal(0.0, 1.0, n)
    y = 0.7 * base + 0.4 * feat + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"base": base, "feat": feat}), y


def _grouped_frame(n: int = 900, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)
    base = rng.normal(5.0, 1.0, n)
    feat = rng.normal(0.0, 1.0, n)
    # Per-group slope so the grouped fit produces a non-trivial coefficient table.
    slope = np.array([0.5, 1.5, 2.5])[g]
    y = slope * base + 0.3 * feat + rng.normal(0.0, 0.1, n)
    return pd.DataFrame({"base": base, "feat": feat, "grp": g}), y


def _make_estimator(transform: str) -> CompositeTargetEstimator:
    if transform == "linear_residual_grouped":
        return CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name=transform,
            base_column="base",
            group_column="grp",
        )
    return CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name=transform,
        base_column="base",
    )


def _fit(transform: str) -> tuple[CompositeTargetEstimator, pd.DataFrame]:
    if transform == "linear_residual_grouped":
        X, y = _grouped_frame()
    else:
        X, y = _base_frame()
    est = _make_estimator(transform).fit(X, y)
    return est, X


_TRANSFORMS = [
    "linear_residual",
    "ratio",
    "ewma_residual",
    "signed_power_y",
    "linear_residual_grouped",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _params_equal(a: dict, b: dict) -> bool:
    """Deep numeric-aware equality for two ``fitted_params_`` dicts."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, dict):
            if not isinstance(vb, dict) or not _params_equal(va, vb):
                return False
        elif isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not np.array_equal(np.asarray(va), np.asarray(vb), equal_nan=True):
                return False
        elif isinstance(va, float):
            if not (va == vb or (np.isnan(va) and np.isnan(vb))):
                return False
        else:
            if va != vb:
                return False
    return True


# ----------------------------------------------------------------------
# Pickle round-trip bit-identity
# ----------------------------------------------------------------------


@pytest.mark.parametrize("transform", _TRANSFORMS)
def test_predict_bit_identical_after_pickle(transform: str) -> None:
    """``predict`` is byte-for-byte identical before and after a pickle
    round trip. Any runtime-cache exclusion that drops a needed fitted param
    would shift at least one prediction and fail ``assert_array_equal``."""
    est, X = _fit(transform)
    pred_before = est.predict(X)

    blob = pickle.dumps(est)
    est2 = pickle.loads(blob)
    pred_after = est2.predict(X)

    np.testing.assert_array_equal(
        pred_before, pred_after,
        err_msg=f"predict not bit-identical after pickle for {transform}",
    )
    # Also confirm a second predict on the restored estimator is itself stable
    # (no lazily-rebuilt runtime cache introduces drift on the first call).
    np.testing.assert_array_equal(pred_after, est2.predict(X))


@pytest.mark.parametrize("transform", _TRANSFORMS)
def test_fitted_params_survive_pickle(transform: str) -> None:
    """``fitted_params_`` survives the round trip with identical keys + values.
    This is the direct guard on the ``__getstate__`` runtime-cache exclusion:
    it must drop only live runtime state (pinned frame ref / streaming buffer),
    never a param the inverse depends on."""
    est, _ = _fit(transform)
    before = copy.deepcopy(est.fitted_params_)

    est2 = pickle.loads(pickle.dumps(est))

    assert hasattr(est2, "fitted_params_"), (
        f"fitted_params_ missing after unpickle for {transform}"
    )
    assert _params_equal(before, est2.fitted_params_), (
        f"fitted_params_ changed across pickle for {transform}: "
        f"{before} != {est2.fitted_params_}"
    )


def test_grouped_coefficient_table_preserved() -> None:
    """The grouped transform stores a per-group coefficient mapping -- the
    nested dict-valued param most likely to be lost to a careless
    ``__getstate__`` that only whitelists scalar params. Round-trip it and
    assert every group key + coefficient survives, and predictions match."""
    est, X = _fit("linear_residual_grouped")
    fp = est.fitted_params_
    # The grouped fit must carry a non-scalar (table-shaped) param.
    has_table = any(
        isinstance(v, (dict, list, tuple, np.ndarray)) for v in fp.values()
    )
    assert has_table, f"grouped fitted_params_ has no table-shaped entry: {fp}"

    est2 = pickle.loads(pickle.dumps(est))
    assert _params_equal(fp, est2.fitted_params_)
    np.testing.assert_array_equal(est.predict(X), est2.predict(X))
