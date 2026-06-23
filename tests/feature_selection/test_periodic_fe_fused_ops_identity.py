"""Regression: fused 3-op modular kernel in generate_modular_features stays bit-identical to three separate passes.

generate_modular_features computes the residue ONCE per (col, period) via _modular_all_ops_njit and emits mod/sin/cos
together. This must match three back-to-back single-op _modular_njit calls exactly (same residue formula, same NaN/inf
scrub, same trig of the same phase). A future "just call apply_modular three times" or a residue-formula drift in
either kernel would break this pin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._periodic_fe import (
    _modular_njit,
    _modular_all_ops_njit,
    _VALID_OPS,
    engineered_name_modular,
    DEFAULT_PERIODS,
    generate_modular_features,
)


def _legacy_generate(X, periods):
    out = {}
    for c in X.columns:
        x = np.ascontiguousarray(X[c].to_numpy(), dtype=np.float64)
        for p in periods:
            for op in _VALID_OPS:
                oc = 0 if op == "mod" else (1 if op == "sin" else 2)
                out[engineered_name_modular(c, p, op)] = _modular_njit(x, float(p), oc)
    return pd.DataFrame(out, index=X.index)


def test_fused_modular_ops_bit_identical_to_separate_passes():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({f"f{j}": rng.normal(0, 50, 4096) for j in range(5)})
    X.iloc[::101, 0] = np.nan
    X.iloc[::103, 1] = np.inf
    X.iloc[::107, 2] = -np.inf

    legacy = _legacy_generate(X, DEFAULT_PERIODS)
    fused = generate_modular_features(X, periods=DEFAULT_PERIODS)

    assert list(legacy.columns) == list(fused.columns)
    assert np.array_equal(legacy.to_numpy(), fused.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("period", [7.0, 24.0, 0.5, 365.0])
def test_fused_kernel_matches_single_op_kernel(period):
    rng = np.random.default_rng(11)
    arr = np.ascontiguousarray(rng.normal(0, 30, 2048), dtype=np.float64)
    arr[5] = np.nan
    arr[9] = np.inf

    mod_v, sin_v, cos_v = _modular_all_ops_njit(arr, period)
    assert np.array_equal(mod_v, _modular_njit(arr, period, 0), equal_nan=True)
    assert np.array_equal(sin_v, _modular_njit(arr, period, 1), equal_nan=True)
    assert np.array_equal(cos_v, _modular_njit(arr, period, 2), equal_nan=True)
