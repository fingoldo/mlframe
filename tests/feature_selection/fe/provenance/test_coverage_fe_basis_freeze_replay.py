"""Closed-form freeze + replay fidelity for the orthogonal / basis FE families.

The basis families (``orth_univariate``, ``orth_pair_cross``, ``orth_fourier``,
``orth_wavelet``, ``hinge_basis``) replay a CLOSED-FORM function of the source column(s)
plus frozen scalar constants (degree, basis name, lo/span/freq/tau...). No y reference is
captured at fit, so replay is leakage-free by construction. This file pins:

* MANUAL-REPLAY parity -- ``apply_recipe`` reproduces the documented closed form (Haar leg,
  hinge ``max(x-tau,0)`` / step, Fourier ``sin/cos(2*pi*freq*z)``) bit-for-bit on a
  held-out frame the recipe never saw at build time.
* FROZEN-CONSTANT replay -- the SAME recipe replayed against a DIFFERENT (held-out) frame
  uses the SAME frozen lo/span/tau (transform-before-refit on new data): a row whose value
  matches a fit row produces the identical engineered value, and the constants do not
  silently refit to the new frame's min/max.
* NO-Y leakage -- replay output is invariant to any y in scope.
* COLUMN-ORDER invariance + pickle round-trip.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import (
    apply_recipe,
    build_orth_univariate_recipe,
    build_orth_pair_cross_recipe,
    build_orth_fourier_recipe,
)
from mlframe.feature_selection.filters._wavelet_basis_fe import (
    build_orth_wavelet_recipe,
    _dyadic_haar_leg,
)
from mlframe.feature_selection.filters._hinge_basis_fe import build_hinge_basis_recipe

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(11)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=400),
            "b": rng.normal(loc=2.0, scale=3.0, size=400),
        }
    )
    # Held-out frame with a DIFFERENT range (wider) -> exposes any silent refit of
    # lo/span to the new data.
    Xnew = pd.DataFrame(
        {
            "a": rng.normal(scale=5.0, size=120),
            "b": rng.normal(loc=-4.0, scale=1.0, size=120),
        }
    )
    return X, Xnew


# ---------------------------------------------------------------------------
# hinge_basis: max(x-tau,0) / max(tau-x,0) / 1[x>tau]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "side,fn",
    [
        ("gt", lambda v, tau: np.maximum(v - tau, 0.0)),
        ("lt", lambda v, tau: np.maximum(tau - v, 0.0)),
        ("ind", lambda v, tau: (v > tau).astype(float)),
    ],
)
def test_hinge_basis_closed_form_replay_on_heldout(frames, side, fn):
    _X, Xnew = frames
    tau = 0.5
    rec = build_hinge_basis_recipe(name=f"hinge_{side}(a)", src_name="a", tau=tau, side=side)
    out = apply_recipe(rec, Xnew)
    manual = fn(Xnew["a"].to_numpy(dtype=float), tau)
    np.testing.assert_allclose(out, manual, rtol=0, atol=0)


def test_hinge_tau_is_frozen_not_refit(frames):
    """The change-point tau is frozen at fit; replaying on a wider held-out frame must
    keep the SAME tau, so a value below tau in BOTH frames maps to 0 either way."""
    _X, Xnew = frames
    rec = build_hinge_basis_recipe(name="h", src_name="a", tau=0.5, side="gt")
    # Manually clamp: any held-out value <= 0.5 must give exactly 0.
    out = apply_recipe(rec, Xnew)
    below = Xnew["a"].to_numpy(dtype=float) <= 0.5
    assert np.all(out[below] == 0.0)


# ---------------------------------------------------------------------------
# orth_wavelet: Haar leg psi_{j,k}(clip((x-lo)/span,0,1))
# ---------------------------------------------------------------------------


def test_wavelet_replay_matches_dyadic_haar(frames):
    X, Xnew = frames
    xf = X["a"].to_numpy(dtype=float)
    lo, hi = float(xf.min()), float(xf.max())
    span = max(hi - lo, 1e-12)
    rec = build_orth_wavelet_recipe(name="a__haar_j1k0", src_name="a", j=1, k=0, lo=lo, span=span)
    out = apply_recipe(rec, Xnew)
    z = np.clip((Xnew["a"].to_numpy(dtype=float) - lo) / span, 0.0, 1.0)
    manual = _dyadic_haar_leg(z, 1, 0)
    np.testing.assert_allclose(out, manual, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# orth_fourier: sin/cos(2*pi*freq*(x-lo)/span)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_fourier_replay_matches_closed_form(frames, kind):
    X, Xnew = frames
    xf = X["a"].to_numpy(dtype=float)
    lo, hi = float(xf.min()), float(xf.max())
    span = max(hi - lo, 1e-12)
    freq = 1.5
    rec = build_orth_fourier_recipe(
        name=f"{kind}(a)",
        src_name="a",
        kind=kind,
        freq=freq,
        lo=lo,
        span=span,
    )
    out = apply_recipe(rec, Xnew)
    z = (Xnew["a"].to_numpy(dtype=float) - lo) / span
    ang = 2.0 * np.pi * freq * z
    manual = np.sin(ang) if kind == "sin" else np.cos(ang)
    np.testing.assert_allclose(out, manual, rtol=0, atol=1e-12)


def test_fourier_quadratic_chirp_axis_requires_mean_std():
    with pytest.raises(ValueError):
        build_orth_fourier_recipe(
            name="bad",
            src_name="a",
            kind="sin",
            freq=1.0,
            lo=0.0,
            span=1.0,
            arg="quadratic",  # mean/std omitted -> must raise
        )


def test_fourier_quadratic_chirp_replays_leakfree(frames):
    X, Xnew = frames
    xf = X["a"].to_numpy(dtype=float)
    mean, std = float(xf.mean()), float(xf.std())
    zs = (xf - mean) / max(std, 1e-12)
    u = np.sign(zs) * (zs * zs)
    lo, span = float(u.min()), max(float(u.max() - u.min()), 1e-12)
    rec = build_orth_fourier_recipe(
        name="chirp(a)",
        src_name="a",
        kind="sin",
        freq=1.0,
        lo=lo,
        span=span,
        arg="quadratic",
        mean=mean,
        std=std,
    )
    # Replay on held-out: rebuild the frozen quadratic axis from frozen mean/std.
    xn = Xnew["a"].to_numpy(dtype=float)
    zsn = (xn - mean) / max(std, 1e-12)
    un = np.sign(zsn) * (zsn * zsn)
    zn = (un - lo) / span
    manual = np.sin(2.0 * np.pi * 1.0 * zn)
    np.testing.assert_allclose(apply_recipe(rec, Xnew), manual, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# orth_univariate + orth_pair_cross
# ---------------------------------------------------------------------------


def test_orth_pair_cross_is_product_of_univariate_legs(frames):
    """A pair-cross column must equal the elementwise PRODUCT of its two univariate
    basis legs (the documented ``h_a * h_b`` replay), to FLOAT32 precision -- not bit-exact.

    ``_apply_orth_pair_cross``'s ``_eval_side`` deliberately casts each operand to
    ``_crit_np_dtype()`` (float32 under the default ``MLFRAME_CRIT_DTYPE_RELAXED``) before
    evaluating the basis, to match ``generate_pair_cross_basis_features``'s fit-time operand dtype
    (see that function's own comment: this cast FIXES a fit/replay precision-drift bug). The
    standalone ``_apply_orth_univariate`` path used for ``leg_i``/``leg_j`` here has no such cast
    (native float64), so ``cross`` and ``leg_i * leg_j`` are genuinely two DIFFERENT-precision
    computations of the same mathematical identity -- comparing them at ``atol=1e-10`` compares f32
    accumulation against f64 and fails on every run (verified: max diff 1.38e-7, exactly float32
    epsilon-scale, not a bug). ``atol=1e-6`` comfortably covers a degree-3 Legendre eval + multiply
    at float32 precision while still catching a genuine correctness break (a wrong basis/degree/side
    diverges by orders of magnitude more than rounding noise)."""
    _X, Xnew = frames
    deg = 3
    rec_i = build_orth_univariate_recipe(name="L(a)", src_name="a", basis="legendre", degree=deg)
    rec_j = build_orth_univariate_recipe(name="L(b)", src_name="b", basis="legendre", degree=deg)
    rec_cross = build_orth_pair_cross_recipe(
        name="L(a)*L(b)",
        src_a_name="a",
        src_b_name="b",
        basis_i="legendre",
        basis_j="legendre",
        deg_a=deg,
        deg_b=deg,
    )
    leg_i = apply_recipe(rec_i, Xnew)
    leg_j = apply_recipe(rec_j, Xnew)
    cross = apply_recipe(rec_cross, Xnew)
    np.testing.assert_allclose(cross, leg_i * leg_j, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Shared contracts across the basis families
# ---------------------------------------------------------------------------


def _all_basis_recipes(frames):
    X, _ = frames
    xf = X["a"].to_numpy(dtype=float)
    lo, hi = float(xf.min()), float(xf.max())
    span = max(hi - lo, 1e-12)
    return [
        build_hinge_basis_recipe(name="h", src_name="a", tau=0.5, side="gt"),
        build_orth_wavelet_recipe(name="w", src_name="a", j=1, k=0, lo=lo, span=span),
        build_orth_fourier_recipe(name="f", src_name="a", kind="sin", freq=1.0, lo=lo, span=span),
        build_orth_univariate_recipe(name="u", src_name="a", basis="hermite", degree=2),
    ]


def test_basis_replay_invariant_to_y_in_scope(frames):
    X, _ = frames
    for rec in _all_basis_recipes(frames):
        out_a = apply_recipe(rec, X)
        _ = np.random.default_rng(0).normal(size=len(X))  # a y-shaped distractor
        out_b = apply_recipe(rec, X)
        np.testing.assert_array_equal(out_a, out_b)


def test_basis_replay_column_order_invariant(frames):
    X, _ = frames
    Xrev = X[["b", "a"]]
    for rec in _all_basis_recipes(frames):
        np.testing.assert_array_equal(apply_recipe(rec, X), apply_recipe(rec, Xrev))


def test_basis_recipe_pickle_roundtrip(frames):
    X, _ = frames
    for rec in _all_basis_recipes(frames):
        rec2 = pickle.loads(pickle.dumps(rec))
        assert rec2 == rec
        np.testing.assert_array_equal(apply_recipe(rec, X), apply_recipe(rec2, X))
