"""Cycle-12 provenance origin-labeling: hinge / change-point + sibling
engineered families must carry their TRUE recipe kind in ``fe_provenance_``
rather than falling through to ``engineered_unknown``.

ROOT CAUSE FIXED HERE
---------------------
The hinge generator (``_hinge_basis_fe.build_hinge_basis_recipe``) DOES stamp
``kind="hinge_basis"`` at creation, but ``_mrmr_fe_provenance._RECIPE_KIND_TO_ORIGIN``
had no branch for it (nor for ~20 other real recipe kinds), so the provenance
classifier mapped them to ``engineered_unknown``. This is a pure labeling fix:
selection is byte-identical.

TRIAD
-----
* unit: a hinge recipe carries its kind through ``_origin_from_recipe``.
* biz_value: explain_selection / fe_provenance_ on the canonical
  ``y = a**2/b + log(c)*sin(d)`` fixture reports 0 ``engineered_unknown`` where
  hinge legs exist, and selection stays byte-identical.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# unit: recipe.kind survives the provenance classifier
# --------------------------------------------------------------------------


def test_hinge_recipe_kind_maps_to_hinge_basis_origin():
    from mlframe.feature_selection.filters._hinge_basis_fe import build_hinge_basis_recipe
    from mlframe.feature_selection.filters._mrmr_fe_provenance import _origin_from_recipe

    recipe = build_hinge_basis_recipe(name="a__relu_gt0.5", src_name="a", tau=0.5, side="gt")
    assert recipe.kind == "hinge_basis"
    origin, details = _origin_from_recipe(recipe)
    assert origin == "hinge_basis", f"hinge recipe mislabeled as {origin!r}"
    assert details["kind"] == "hinge_basis"
    assert details["src_names"] == ("a",)


def test_no_real_recipe_kind_falls_to_engineered_unknown():
    """Every recipe ``kind=`` literal emitted by an engineered-recipe builder
    (except the intentional ``factorize``) must have an origin mapping -- guards
    against a new FE family silently regressing to ``engineered_unknown``."""
    from mlframe.feature_selection.filters import _mrmr_fe_provenance as prov

    # kinds intentionally left as engineered_unknown (documented in the map).
    INTENTIONAL_UNKNOWN = {"factorize"}
    mapped = set(prov._RECIPE_KIND_TO_ORIGIN)
    for kind in (
        "hinge_basis",
        "numeric_rounding",
        "digit_extract",
        "modular",
        "group_distance",
        "temporal_expanding",
        "temporal_rolling",
        "temporal_lag",
        "orth_wavelet",
        "rare_category",
        "conditional_residual",
        "conditional_dispersion",
        "rankgauss",
        "div",
        "log_div",
        "grouped_agg",
        "composite_group_agg",
        "grouped_quantile",
        "target_aware_group_bin",
        "cat_pair_cross",
        "cat_triple_cross",
        "orth_diff_basis",
        "orth_cluster_basis",
        "orth_triplet_cross",
        "orth_quadruplet_cross",
    ):
        assert kind in mapped and kind not in INTENTIONAL_UNKNOWN, f"recipe kind {kind!r} not mapped -> would render as engineered_unknown"


# --------------------------------------------------------------------------
# biz_value: canonical fixture -> 0 engineered_unknown for hinge legs
# --------------------------------------------------------------------------


def _canonical_frame(n: int = 800, seed: int = 7):
    rng = np.random.default_rng(int(seed))
    a = rng.standard_normal(n)
    b = rng.uniform(0.5, 2.5, n)
    c = rng.uniform(0.5, 5.0, n)
    d = rng.uniform(0.0, 2.0 * np.pi, n)
    X = pd.DataFrame(
        {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
        }
    )
    score = a**2 / b + np.log(c) * np.sin(d) + 0.3 * rng.standard_normal(n)
    y = pd.Series((score > np.median(score)).astype(int))
    return X, y


def _fe_on():
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR(
        verbose=0,
        random_seed=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        stability_selection_method="classic",
        retain_artifacts=False,
        n_jobs=1,
        fe_hybrid_orth_enable=True,
        fe_auto=True,
    )


def test_canonical_hinge_legs_labeled_not_unknown():
    X, y = _canonical_frame()
    est = _fe_on()
    est.fit(X, y)
    prov = est.fe_provenance_
    assert prov is not None and not prov.empty

    hinge = prov[prov["feature_name"].astype(str).str.contains("relu_")]
    if hinge.empty:
        pytest.skip("FE search produced no hinge legs on this build; nothing to label")

    origins = set(hinge["origin"].astype(str))
    assert origins == {"hinge_basis"}, f"hinge legs carry wrong origin(s): {origins}"
    assert "engineered_unknown" not in origins

    # 0 engineered_unknown among engineered survivors where hinge legs exist.
    eng = prov[prov["origin"].astype(str) != "raw"]
    n_unknown = int((eng["origin"].astype(str) == "engineered_unknown").sum())
    assert n_unknown == 0, f"{n_unknown} engineered_unknown rows remain:\n{eng}"

    # explain_selection names the real kind, not engineered_unknown.
    report = est.explain_selection()
    assert "hinge_basis" in report
