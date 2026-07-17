"""Master switch for the discrete structural FE operator suite.

``fe_discrete_structural_operators_enable`` (default True) toggles all four discrete operators -- pairwise-modular,
integer-lattice, row-argmax, conditional-gate -- at once. False disables every one regardless of the individual
``fe_*_enable`` flags (pure classical FE); True lets the per-operator flags govern. This is the single UX knob a caller
flips to opt the whole family out without hunting down four separate flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

pytestmark = pytest.mark.timeout(60)  # untimed biz_val real-fit tier: surface a hang fast (global --timeout=600 is a coarse backstop)

_DISCRETE_PREFIXES = ("il_", "pmod_", "argmax_", "gate_")


def _gcd_frame(seed: int = 1, n: int = 2000):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 40, n)
    b = rng.integers(1, 40, n)
    y = (np.gcd(a, b) >= 4).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "noise0": rng.integers(0, 50, n), "noise1": rng.normal(size=n)})
    return X, y


def _discrete_cols(estimator, X):
    out = estimator.transform(X)
    return [c for c in out.columns if any(str(c).startswith(p) for p in _DISCRETE_PREFIXES)]


def test_master_switch_defaults_on():
    assert MRMR(n_workers=1).fe_discrete_structural_operators_enable is True


def test_master_on_emits_discrete_operator_feature():
    X, y = _gcd_frame()
    sel = MRMR(n_workers=1, max_runtime_mins=1, quantization_nbins=8, verbose=0, fe_discrete_structural_operators_enable=True)
    sel.fit(X, y)
    assert _discrete_cols(sel, X), "master ON should let the gcd structure surface a discrete-operator feature."


def test_master_off_suppresses_all_four_operators():
    X, y = _gcd_frame()
    sel = MRMR(n_workers=1, max_runtime_mins=1, quantization_nbins=8, verbose=0, fe_discrete_structural_operators_enable=False)
    sel.fit(X, y)
    assert _discrete_cols(sel, X) == [], "master OFF must suppress every discrete operator regardless of individual flags."


def test_master_off_overrides_individual_enables():
    """Master OFF wins even when an individual operator flag is explicitly True."""
    X, y = _gcd_frame()
    sel = MRMR(
        n_workers=1,
        max_runtime_mins=1,
        quantization_nbins=8,
        verbose=0,
        fe_discrete_structural_operators_enable=False,
        fe_integer_lattice_enable=True,
        fe_pairwise_modular_enable=True,
        fe_row_argmax_enable=True,
        fe_conditional_gate_enable=True,
    )
    sel.fit(X, y)
    assert _discrete_cols(sel, X) == [], "an individual fe_*_enable=True must not override the master OFF switch."
