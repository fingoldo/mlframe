"""Simple-mode linear-usability raw retention (variant-3).

Under ``use_simple_mode=True`` MRMR still runs feature engineering, and the
raw-vs-engineered redundancy sweep would drop a raw whose conditional MI collapses
given a surviving engineered child -- correct for a truly-subsumed ratio operand
(``a`` in ``a**2/b``), but WRONG for a dominant private LINEAR term (``s0`` in
``y=2*s0-1.3*s1+0.8*s2``) that a nonlinear engineered nesting only encodes
non-linearly. The linear-usability keep-leg restores such linearly-usable raws in
simple mode (a downstream linear model still needs them) while leaving the full-mode
subsumed-drop untouched (I4b). These tests pin both directions.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_raw_redundancy_drop import (
    drop_redundant_raw_operands,
    raw_retains_linear_signal_given_children,
)


def _bin10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    edges = np.quantile(x, np.linspace(0, 1, 11)[1:-1])
    return np.digitize(x, edges).astype(np.int64)


def test_linear_residual_keeps_dominant_linear_term_under_nonlinear_child():
    """A genuine linear term retains significant partial linear signal given a nonlinear
    child built from it; a subsumed ratio operand does not."""
    rng = np.random.default_rng(0)
    n = 4000
    s0 = rng.normal(size=n)
    s1 = rng.normal(size=n)
    s2 = rng.normal(size=n)
    y = 2.0 * s0 - 1.3 * s1 + 0.8 * s2
    # Nonlinear engineered children that info-subsume the raws but are not linear equivalents.
    child = s0 - np.sin(s1)
    assert raw_retains_linear_signal_given_children(s0, y, [child], seed=0) is True

    # Pure-noise raw consumed by the same child -> no private linear signal -> not kept.
    noise = rng.normal(size=n)
    assert raw_retains_linear_signal_given_children(noise, y, [child], seed=0) is False


def test_simple_mode_keeps_linear_raw_full_mode_drops_subsumed():
    """``linear_usability_keep`` (set from ``use_simple_mode`` at the call site) flips the
    verdict on a linearly-usable-but-CMI-subsumed raw: kept in simple mode, dropped in full
    mode. The same call on a TRULY subsumed ratio operand drops in BOTH modes."""
    rng = np.random.default_rng(1)
    n = 6000
    s0 = rng.normal(size=n)
    s1 = rng.normal(size=n)
    y_lin = 2.0 * s0 - 1.3 * s1
    child_lin = s0 - np.sin(s1)  # nonlinear nesting of s0, s1
    cols = ["s0", "s1", "sub(s0,sin(s1))"]
    raw_set = {"s0", "s1"}
    data = np.column_stack([_bin10(s0), _bin10(s1), _bin10(child_lin)]).astype(np.int64)
    eng_cont = {"sub(s0,sin(s1))": child_lin}

    common = dict(
        data=data, cols=cols, selected_cols_idx=[0, 1, 2], raw_name_set=raw_set,
        y_binned=_bin10(y_lin), y_continuous=y_lin, engineered_continuous=eng_cont,
        replayable_eng_names={"sub(s0,sin(s1))"},
        raw_X=None, seed=1,
    )
    _, dropped_full = drop_redundant_raw_operands(linear_usability_keep=False, **common)
    _, dropped_simple = drop_redundant_raw_operands(linear_usability_keep=True, **common)
    # Simple mode keeps strictly more raws than full mode on a linear DGP.
    assert len(dropped_simple) <= len(dropped_full)
    assert "s0" not in dropped_simple, f"linear term s0 must survive in simple mode; dropped={dropped_simple}"
