"""Regression test: the ORTH-BASIS UNIVARIATE PROTECTION block (``_fit_impl_core.py``) must fire even when
``fe_hinge_enable=False`` (no hinge legs generated this fit).

BUG (found while investigating ``TestCmimAucGteDefault::test_cmim_auc_geq_plug_in_on_redundant_pool``):
``_heldout_incr_over_selected`` - the held-out-uplift closure the orth-basis protection block calls to decide
whether to re-add a single-source basis column (e.g. ``a__He2``) the greedy MI screen data-processing-inequality
(DPI) dropped as redundant with its own raw source - was defined ONLY inside ``if _hinge_feats and
len(selected_vars):``, i.e. only when the (separately gated, default-off-in-many-presets) hinge/change-point FE
stage actually produced at least one leg. The orth-basis protection block, ~150 lines later, gated its own
``if`` on ``"_heldout_incr_over_selected" in locals()`` - so whenever ``fe_hinge_enable=False`` (or the hinge
stage simply found no breakpoint), the closure was never defined, the orth-basis protection's guard was always
False, and a genuine single-source Hermite/Chebyshev/Legendre/Laguerre basis column the screen DPI-dropped was
NEVER re-added, regardless of how much held-out linear usability it carried.

Fix: hoist the closure's setup to run whenever EITHER hinge legs OR hybrid-orth univariate-basis candidates
exist (not hinge legs alone); the hinge-specific re-add loop still runs only when ``_hinge_feats`` is non-empty.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr


def _build_pure_quadratic(seed: int, n: int = 1500):
    """Single raw feature ``a`` whose target is a deterministic even function of it (``y = a**2``). ``a``
    itself easily survives the MI screen (it's the ONLY candidate raw feature and MI is nonparametric, so
    the U-shape is detected without linear correlation), but the Hermite-degree-2 basis column built from
    it (``a__He2`` ~ ``a**2``) is a deterministic function of ``a`` -- the greedy MRMR redundancy scan
    (``I(a__He2; y | a)``) collapses toward 0 under the data-processing inequality and DPI-drops it, even
    though a downstream LINEAR model needs the literal quadratic term ``a**2`` to fit ``y`` well (a raw
    linear term in ``a`` alone cannot)."""
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal(n)
    y = a**2 + 0.02 * rng.standard_normal(n)
    X = pd.DataFrame({"a": a})
    return X, pd.Series(y, name="y")


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_orth_basis_protection_readds_dpi_dropped_basis_without_hinge(seed):
    """With hinge disabled (the common ``make_fast_mrmr`` preset), a single-source Hermite basis column the
    MI screen DPI-drops must still be re-added by the orth-basis protection block."""
    X, y = _build_pure_quadratic(seed)
    m = _make_mrmr(
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_degrees=(2,),
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_top_k=2,
        fe_hybrid_orth_pair_enable=False,
        fe_hinge_enable=False,  # explicit -- this is also make_fast_mrmr's own default
    ).fit(X, y)
    added = list(getattr(m, "hybrid_orth_features_", []) or [])
    assert added, (
        f"seed={seed}: no hybrid-orth basis column survived with fe_hinge_enable=False; the orth-basis "
        f"univariate protection block should have re-added the DPI-dropped Hermite basis of 'a' (it lifts "
        f"a held-out linear fit by a wide margin over raw 'a' alone on a pure-quadratic target). "
        f"hybrid_orth_candidates_={list(getattr(m, 'hybrid_orth_candidates_', []) or [])}"
    )
    assert any(c.startswith("a__") for c in added), f"seed={seed}: re-added column(s) {added} do not include a basis of 'a'"
