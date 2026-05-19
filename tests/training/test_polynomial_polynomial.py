"""Regression tests for the polynomial auto-tune behaviour in
``PolynomialFeatureExpander``. Each test covers one auto-tune tier: flip
interaction_only, decrement degree, full-skip. The expander must never
raise on cap exceedance -- the user's explicit decision (audit
disposition 2026-05-16) was "auto-tune, not ValueError".
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.feature_handling.polynomial import (
    PolynomialFeatureExpander,
    _projected_output_cols,
)


def _make_X(n_rows: int, n_cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_rows, n_cols)).astype(np.float32)


def test_no_cap_leaves_config_untouched():
    """``max_features_out=None`` preserves legacy behaviour: no auto-tune."""
    X = _make_X(50, 10)
    exp = PolynomialFeatureExpander(degree=2, interaction_only=False, max_features_out=None)
    exp.fit(X)
    assert exp._fitted
    assert not exp._skipped
    assert exp.effective_degree == 2
    assert exp.effective_interaction_only is False


def test_cap_zero_disables_autotune():
    """``max_features_out=0`` also disables auto-tune."""
    X = _make_X(50, 10)
    exp = PolynomialFeatureExpander(degree=2, interaction_only=False, max_features_out=0)
    exp.fit(X)
    assert exp.effective_degree == 2
    assert exp.effective_interaction_only is False


def test_tier1_flip_interaction_only_when_full_exceeds_cap(caplog):
    """When degree=2, interaction_only=False would exceed cap but interaction-only fits:
    expect interaction_only flipped to True. ``n=20, degree=2``:
    full = C(20,1)+C(21,2) = 20+210=230; interaction = C(20,1)+C(20,2) = 20+190=210.
    Set cap=215 so the full path overshoots but interaction-only path fits.
    """
    n_features = 20
    X = _make_X(40, n_features)
    cap = 215
    full_projected = _projected_output_cols(n_features, 2, False)
    inter_projected = _projected_output_cols(n_features, 2, True)
    assert full_projected > cap > inter_projected, (
        f"sanity: full={full_projected}, inter={inter_projected}, cap={cap}"
    )

    exp = PolynomialFeatureExpander(degree=2, interaction_only=False, max_features_out=cap)
    with caplog.at_level("WARNING"):
        exp.fit(X)
    assert exp._fitted
    assert not exp._skipped
    assert exp.effective_degree == 2
    assert exp.effective_interaction_only is True
    out = exp.transform(X)
    assert out.shape[1] == inter_projected
    assert any("flipping interaction_only" in rec.getMessage() for rec in caplog.records)


def test_tier2_decrement_degree_when_interaction_still_exceeds(caplog):
    """When interaction-only at degree=3 still busts the cap but degree=2 fits: expect
    degree decremented. ``n=20, degree=3, interaction_only=True``:
    interaction-3 = C(20,1)+C(20,2)+C(20,3) = 20+190+1140=1350.
    interaction-2 = 210. Cap between 211 and 1350.
    """
    n_features = 20
    X = _make_X(40, n_features)
    cap = 250
    proj_d3_io = _projected_output_cols(n_features, 3, True)
    proj_d2_io = _projected_output_cols(n_features, 2, True)
    assert proj_d3_io > cap > proj_d2_io

    exp = PolynomialFeatureExpander(degree=3, interaction_only=True, max_features_out=cap)
    with caplog.at_level("WARNING"):
        exp.fit(X)
    assert exp._fitted
    assert not exp._skipped
    assert exp.effective_degree == 2
    assert exp.effective_interaction_only is True
    out = exp.transform(X)
    assert out.shape[1] == proj_d2_io
    assert any("decrementing degree" in rec.getMessage() for rec in caplog.records)


def test_tier3_full_skip_when_degree1_exceeds_cap(caplog):
    """When even degree=1 (which is just identity n_features) exceeds the cap: expect full skip.
    ``n=100, cap=50``: degree=1 projects to 100, > 50, so skip entirely.
    """
    n_features = 100
    X = _make_X(20, n_features)
    cap = 50

    exp = PolynomialFeatureExpander(degree=3, interaction_only=False, max_features_out=cap)
    with caplog.at_level("WARNING"):
        exp.fit(X)
    assert exp._fitted
    assert exp._skipped
    assert any("skipping polynomial expansion entirely" in rec.getMessage() for rec in caplog.records)
    # transform returns input unchanged
    out = exp.transform(X)
    assert out.shape == X.shape
    np.testing.assert_array_equal(out, X)


def test_never_raises_on_cap_exceedance():
    """Regardless of how absurd the degree/cap combination is, fit MUST NOT raise."""
    X = _make_X(20, 50)
    # Degree 5 on 50 features yields combinatorial blow-up; cap=10 forces full-skip.
    exp = PolynomialFeatureExpander(degree=5, interaction_only=False, max_features_out=10)
    # Must not raise:
    exp.fit(X)
    assert exp._fitted


def test_within_cap_fits_normally():
    """When projected <= cap, no auto-tune fires."""
    n_features = 5
    X = _make_X(30, n_features)
    cap = 10_000
    exp = PolynomialFeatureExpander(degree=2, interaction_only=False, max_features_out=cap)
    exp.fit(X)
    expected = _projected_output_cols(n_features, 2, False)
    out = exp.transform(X)
    assert out.shape[1] == expected
    assert exp.effective_degree == 2
    assert exp.effective_interaction_only is False
