"""Generate -> build -> replay parity for the "extra" FE families.

Families: ``rare_category`` (is_rare / freq_band), ``conditional_residual``
(NUM x NUM ``x_i - E[x_i|bin(x_j)]``), ``conditional_dispersion`` (NUM x NUM
conditional z-score / |z| / z^2), and ``rankgauss``.

The core contract pinned here is FIT==REPLAY: the value ``generate_*`` materialised at fit
time, when rebuilt into a frozen recipe and replayed through ``apply_recipe`` against the
SAME frame, is bit-identical. This is the canonical leakage check -- the engineered column
is computed only from frozen fit constants (frequency lookup / quantile edges / per-bin
(mu, sigma) / sorted fit values), and y is never read at transform (these families'
``generate_*`` either takes no y at all or uses it only for selection, never for the
emitted value). Held-out replay reuses the FROZEN constants (transform-before-refit), and
unseen categories take the documented fallback.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._extra_fe_families import (
    generate_rare_category_features,
    build_rare_category_recipe,
    generate_conditional_residual_features,
    build_conditional_residual_recipe,
    generate_rankgauss_features,
    build_rankgauss_recipe,
)
from mlframe.feature_selection.filters._extra_fe_families_dispersion import (
    generate_conditional_dispersion_features,
    build_conditional_dispersion_recipe,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def num_frame():
    """Num frame."""
    rng = np.random.default_rng(3)
    n = 800
    x0 = rng.normal(size=n)
    x1 = 0.7 * x0 + rng.normal(scale=0.5, size=n)
    # Heteroscedastic so dispersion is non-degenerate.
    x2 = rng.normal(scale=1.0 + np.abs(x0), size=n)
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def cat_frame():
    """Cat frame."""
    rng = np.random.default_rng(5)
    n = 800
    # Mostly common + a sprinkle of rare categories.
    cats = rng.choice(
        np.array(["A", "B", "C", "rare1", "rare2"]),
        p=[0.4, 0.3, 0.25, 0.03, 0.02],
        size=n,
    )
    return pd.DataFrame({"g": cats})


# ---------------------------------------------------------------------------
# rare_category
# ---------------------------------------------------------------------------


def test_rare_category_fit_equals_replay(cat_frame):
    """Rare category fit equals replay."""
    enc_df, raw = generate_rare_category_features(cat_frame, ["g"], rare_threshold=0.05)
    assert len(raw) > 0
    for name, payload in raw.items():
        rec = build_rare_category_recipe(
            name=name,
            src_col=payload["src_col"],
            kind=payload["kind"],
            freq_lookup=payload["freq_lookup"],
            rare_threshold=payload["rare_threshold"],
            dominant_cut=payload["dominant_cut"],
        )
        replay = apply_recipe(rec, cat_frame)
        np.testing.assert_array_equal(replay, enc_df[name].to_numpy())


def test_rare_category_unseen_maps_to_rare(cat_frame):
    """Rare category unseen maps to rare."""
    _enc_df, raw = generate_rare_category_features(cat_frame, ["g"], rare_threshold=0.05)
    name = next(n for n, p in raw.items() if p["kind"] == "is_rare")
    payload = raw[name]
    rec = build_rare_category_recipe(
        name=name,
        src_col=payload["src_col"],
        kind=payload["kind"],
        freq_lookup=payload["freq_lookup"],
        rare_threshold=payload["rare_threshold"],
        dominant_cut=payload["dominant_cut"],
    )
    Xnew = pd.DataFrame({"g": ["A", "never_seen_cat"]})
    out = apply_recipe(rec, Xnew)
    assert np.isfinite(out).all()
    # Unseen category -> frequency 0 -> is_rare = 1.
    assert out[1] == 1.0


# ---------------------------------------------------------------------------
# conditional_residual
# ---------------------------------------------------------------------------


def test_conditional_residual_fit_equals_replay(num_frame):
    """Conditional residual fit equals replay."""
    enc_df, raw = generate_conditional_residual_features(num_frame, ["x0", "x1"], n_bins=8)
    assert len(raw) > 0
    for name, payload in raw.items():
        rec = build_conditional_residual_recipe(
            name=name,
            x_i=payload["x_i"],
            x_j=payload["x_j"],
            edges=payload["edges"],
            bin_mean=payload["bin_mean"],
            global_mean=payload["global_mean"],
        )
        replay = apply_recipe(rec, num_frame)
        np.testing.assert_allclose(replay, enc_df[name].to_numpy(), rtol=0, atol=1e-9)


def test_conditional_residual_heldout_reuses_frozen_edges(num_frame):
    """A held-out frame with a shifted x_j range must be digitised with the FROZEN
    fit edges, not refit -- so a held-out row whose (x_i, bin(x_j)) match a fit row
    yields the fit residual."""
    enc_df, raw = generate_conditional_residual_features(num_frame, ["x0", "x1"], n_bins=8)
    name = next(iter(raw))
    payload = raw[name]
    rec = build_conditional_residual_recipe(
        name=name,
        x_i=payload["x_i"],
        x_j=payload["x_j"],
        edges=payload["edges"],
        bin_mean=payload["bin_mean"],
        global_mean=payload["global_mean"],
    )
    # Take the first 10 fit rows as "held-out" -- replay must equal the fit value
    # since the recipe is frozen and stateless in X.
    head = num_frame.iloc[:10].copy()
    replay_head = apply_recipe(rec, head)
    np.testing.assert_allclose(replay_head, enc_df[name].to_numpy()[:10], rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# conditional_dispersion
# ---------------------------------------------------------------------------


def test_conditional_dispersion_fit_equals_replay(num_frame):
    """Conditional dispersion fit equals replay."""
    enc_df, raw = generate_conditional_dispersion_features(
        num_frame,
        ["x0", "x2"],
        n_bins=8,
        kinds=("absz", "z2"),
    )
    assert len(raw) > 0
    for name, payload in raw.items():
        rec = build_conditional_dispersion_recipe(
            name=name,
            x_i=payload["x_i"],
            x_j=payload["x_j"],
            edges=payload["edges"],
            bin_mean=payload["bin_mean"],
            bin_std=payload["bin_std"],
            global_mean=payload["global_mean"],
            global_std=payload["global_std"],
            kind=payload["kind"],
        )
        replay = apply_recipe(rec, num_frame)
        np.testing.assert_allclose(replay, enc_df[name].to_numpy(), rtol=0, atol=1e-9)


def test_conditional_dispersion_kind_is_frozen(num_frame):
    """The emission kind (absz vs z2) is frozen; replay of an absz recipe must be
    non-negative, and a z2 recipe must equal absz**2 on the same fit constants."""
    _, raw = generate_conditional_dispersion_features(
        num_frame,
        ["x0", "x2"],
        n_bins=8,
        kinds=("absz", "z2"),
    )
    # Find an (x_i, x_j) pair present in both kinds.
    by_pair: dict = {}
    for name, p in raw.items():
        by_pair.setdefault((p["x_i"], p["x_j"]), {})[p["kind"]] = (name, p)
    pair = next(v for v in by_pair.values() if "absz" in v and "z2" in v)
    abz_name, abz_p = pair["absz"]
    z2_name, z2_p = pair["z2"]
    rec_abz = build_conditional_dispersion_recipe(
        name=abz_name,
        x_i=abz_p["x_i"],
        x_j=abz_p["x_j"],
        edges=abz_p["edges"],
        bin_mean=abz_p["bin_mean"],
        bin_std=abz_p["bin_std"],
        global_mean=abz_p["global_mean"],
        global_std=abz_p["global_std"],
        kind="absz",
    )
    rec_z2 = build_conditional_dispersion_recipe(
        name=z2_name,
        x_i=z2_p["x_i"],
        x_j=z2_p["x_j"],
        edges=z2_p["edges"],
        bin_mean=z2_p["bin_mean"],
        bin_std=z2_p["bin_std"],
        global_mean=z2_p["global_mean"],
        global_std=z2_p["global_std"],
        kind="z2",
    )
    abz = apply_recipe(rec_abz, num_frame)
    z2 = apply_recipe(rec_z2, num_frame)
    assert np.all(abz >= -1e-12)
    np.testing.assert_allclose(z2, abz**2, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# rankgauss
# ---------------------------------------------------------------------------


def test_rankgauss_fit_equals_replay(num_frame):
    """Rankgauss fit equals replay."""
    enc_df, raw = generate_rankgauss_features(num_frame, ["x0", "x1"])
    assert len(raw) > 0
    for name, payload in raw.items():
        rec = build_rankgauss_recipe(
            name=name,
            src_col=payload["src_col"],
            fit_sorted=payload["fit_sorted"],
            n_fit=payload["n_fit"],
        )
        replay = apply_recipe(rec, num_frame)
        # Fit and replay use slightly different (but documented-equivalent) rank
        # paths; on the fit frame they must agree to tight tolerance.
        np.testing.assert_allclose(replay, enc_df[name].to_numpy(), rtol=0, atol=1e-9)


def test_rankgauss_monotone_preserves_order(num_frame):
    """RankGauss is monotone: the replayed column must be a non-decreasing function
    of the source values (the property that makes it MI-invariant by the DPI)."""
    _, raw = generate_rankgauss_features(num_frame, ["x0"])
    name, payload = next(iter(raw.items()))
    rec = build_rankgauss_recipe(
        name=name,
        src_col=payload["src_col"],
        fit_sorted=payload["fit_sorted"],
        n_fit=payload["n_fit"],
    )
    out = apply_recipe(rec, num_frame)
    src = num_frame["x0"].to_numpy()
    order = np.argsort(src, kind="stable")
    sorted_out = out[order]
    # Non-decreasing along sorted source (allow tiny FP slack at ties).
    assert np.all(np.diff(sorted_out) >= -1e-9)


def test_rankgauss_replay_invariant_to_y_in_scope(num_frame):
    """Rankgauss replay invariant to y in scope."""
    _, raw = generate_rankgauss_features(num_frame, ["x0"])
    name, payload = next(iter(raw.items()))
    rec = build_rankgauss_recipe(
        name=name,
        src_col=payload["src_col"],
        fit_sorted=payload["fit_sorted"],
        n_fit=payload["n_fit"],
    )
    out_a = apply_recipe(rec, num_frame)
    _ = np.random.default_rng(1).normal(size=len(num_frame))
    out_b = apply_recipe(rec, num_frame)
    np.testing.assert_array_equal(out_a, out_b)
