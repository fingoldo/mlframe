"""Regression sensors for the fuzz-surfaced KeyError in the recipe-based FE
families ``conditional_residual`` / ``conditional_dispersion`` / ``grouped_quantile``.

Failure mode
------------
``KeyError("apply_conditional_residual/apply_conditional_dispersion/
apply_grouped_quantile: missing column(s) ... from X_test")`` at ``transform()``.

Root cause (one shared bug across the three): in ``MRMR.fit`` each family
auto-detects its source ``num_cols`` / ``group_cols`` over the X that has
ALREADY been augmented with engineered intermediates from prior FE stages
(poly / fourier / ratio / grouped-agg / ...). A recipe built on an engineered
source cannot be replayed at ``transform()`` -- ``apply_recipe`` regenerates each
recipe independently against the RAW input X, where the engineered parent is
absent -> KeyError. The fix scopes every auto-detected source to the raw pre-FE
column snapshot (the same ``_raw_input_cols_pre_fe`` ledger cat_pair / cat_triple
already use).

Two layers of sensor:

* GENERATOR layer (``test_*_recipe_on_engineered_source_keyerrors_*``): pins the
  bug class -- when these generators are handed an augmented frame with
  ``num_cols=None`` they DO build engineered-source recipes that KeyError on raw
  replay. This is the exact path the fuzz hit; it documents WHY the fit-stage
  scoping is required.
* FIT layer (``test_fit_*_only_passes_raw_columns``): pins the FIX -- ``MRMR.fit``
  must hand each family only raw columns, so no engineered-source recipe is ever
  built and ``transform`` round-trips cleanly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters._extra_fe_families import (
    hybrid_conditional_dispersion_fe,
    hybrid_conditional_residual_fe,
)
from mlframe.feature_selection.filters._grouped_quantile_fe import (
    hybrid_grouped_quantile_fe,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _augmented_num_frame(n: int = 600, seed: int = 1):
    """Raw {a, b} plus one ENGINEERED numeric column with a recognisable
    double-underscore name (poly / cross convention) that is NOT in raw X."""
    rng = np.random.default_rng(seed)
    raw = {"a": rng.normal(size=n), "b": rng.normal(size=n)}
    eng_name = "a__x__b"
    aug = pd.DataFrame({**raw, eng_name: rng.normal(size=n)})
    y = (aug[eng_name] > 0).astype(int).to_numpy()
    raw_cols = list(raw.keys())
    return aug, y, raw_cols, eng_name


def _replay_all(recipes, appended, X_raw):
    """Replay every appended recipe against raw X; return the first raised exc."""
    for r in recipes:
        if r.name in appended:
            try:
                apply_recipe(r, X_raw)
            except Exception as exc:
                return r, exc
    return None, None


# --------------------------------------------------------------------------
# GENERATOR layer: pins the bug class (engineered-source recipe -> KeyError).
# --------------------------------------------------------------------------


def test_conditional_residual_recipe_on_engineered_source_keyerrors_on_raw():
    """Conditional residual recipe on engineered source keyerrors on raw."""
    aug, y, raw_cols, eng = _augmented_num_frame(seed=1)
    _, appended, recipes, _ = hybrid_conditional_residual_fe(
        aug,
        y,
        num_cols=None,
        n_bins=8,
        top_k=20,
        max_pair_cols=6,
        mi_gate=False,
    )
    assert any(eng in r.src_names for r in recipes if r.name in appended), (
        "fixture must produce at least one engineered-source recipe to be a meaningful sensor for the bug class"
    )
    X_raw = aug[raw_cols]
    _bad_recipe, exc = _replay_all(recipes, appended, X_raw)
    assert isinstance(exc, KeyError), f"expected KeyError replaying engineered-source recipe on raw X; got {exc!r}"
    assert eng in str(exc)


def test_conditional_dispersion_recipe_on_engineered_source_keyerrors_on_raw():
    """Conditional dispersion recipe on engineered source keyerrors on raw."""
    aug, y, raw_cols, eng = _augmented_num_frame(seed=2)
    _, appended, recipes, _ = hybrid_conditional_dispersion_fe(
        aug,
        y,
        num_cols=None,
        n_bins=8,
        top_k=20,
        max_pair_cols=6,
        mi_gate=False,
    )
    assert any(eng in r.src_names for r in recipes if r.name in appended)
    X_raw = aug[raw_cols]
    _bad_recipe, exc = _replay_all(recipes, appended, X_raw)
    assert isinstance(exc, KeyError)
    assert eng in str(exc)


def test_grouped_quantile_recipe_on_engineered_source_keyerrors_on_raw():
    """Grouped quantile recipe on engineered source keyerrors on raw."""
    rng = np.random.default_rng(5)
    n = 900
    g = rng.integers(0, 6, size=n).astype("int64")
    eng = "x0__T2"
    aug = pd.DataFrame({"g": g, "val": rng.normal(size=n), eng: rng.normal(size=n) ** 2})
    y = ((aug["val"] + (g == 2)) > 0.5).astype(int).to_numpy()
    _, appended, recipes, _ = hybrid_grouped_quantile_fe(
        aug,
        y,
        group_cols=None,
        num_cols=None,
        top_k=20,
        min_mi=0.0,
        min_uplift=-1.0,
    )
    assert any(eng in r.src_names for r in recipes if r.name in appended)
    X_raw = aug[["g", "val"]]
    _bad_recipe, exc = _replay_all(recipes, appended, X_raw)
    assert isinstance(exc, KeyError)
    assert eng in str(exc)


# --------------------------------------------------------------------------
# FIT layer: pins the fix -- MRMR.fit must scope each family to RAW columns.
# --------------------------------------------------------------------------


@pytest.fixture
def _mrmr_cls():
    """Mrmr cls."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR


def _capture_family_cols(monkeypatch, target_module, func_name, captured):
    """Wrap a hybrid_*_fe so we record the num_cols / group_cols it is called
    with, then return an empty (no-op) augmentation so the fit stays cheap."""
    import importlib

    mod = importlib.import_module(target_module)
    orig = getattr(mod, func_name)

    def _wrapper(X, y, *args, **kwargs):
        """Stand-in for the wrapped FE function: record its num_cols/group_cols kwargs then return an empty no-op augmentation."""
        captured.append(
            {
                "num_cols": kwargs.get("num_cols"),
                "group_cols": kwargs.get("group_cols"),
            }
        )
        return X.copy(), [], [], pd.DataFrame()

    monkeypatch.setattr(mod, func_name, _wrapper)
    return orig


def _fit_with_families(MRMR, X, y, **flags):
    """Fit with families."""
    m = MRMR(
        fe_pairwise_ratio_enable=True,
        fe_grouped_agg_enable=True,
        fe_univariate_basis_enable=True,
        fe_local_mi_gate=False,
        **flags,
    )
    m.fit(X, y)
    return m


def test_fit_conditional_residual_only_passes_raw_columns(_mrmr_cls, monkeypatch):
    """Fit conditional residual only passes raw columns."""
    rng = np.random.default_rng(7)
    n = 700
    X = pd.DataFrame({c: rng.normal(size=n) for c in ("a", "b", "c", "d")})
    X["g"] = rng.integers(0, 5, size=n)
    y = ((X["a"] * X["b"]) > 0).astype(int).to_numpy()
    raw_cols = set(X.columns)

    captured: list[dict] = []
    _capture_family_cols(
        monkeypatch,
        "mlframe.feature_selection.filters._extra_fe_families",
        "hybrid_conditional_residual_fe",
        captured,
    )
    _fit_with_families(_mrmr_cls, X, y, fe_conditional_residual_enable=True)

    assert captured, "conditional_residual stage was not exercised"
    for call in captured:
        nc = call["num_cols"]
        assert nc is not None, "fix must hand an explicit RAW num_cols, never None"
        assert set(nc) <= raw_cols, f"engineered source leaked into num_cols: {nc}"


def test_fit_conditional_dispersion_only_passes_raw_columns(_mrmr_cls, monkeypatch):
    """Fit conditional dispersion only passes raw columns."""
    rng = np.random.default_rng(8)
    n = 700
    X = pd.DataFrame({c: rng.normal(size=n) for c in ("a", "b", "c", "d")})
    X["g"] = rng.integers(0, 5, size=n)
    y = ((X["a"] * X["b"]) > 0).astype(int).to_numpy()
    raw_cols = set(X.columns)

    captured: list[dict] = []
    _capture_family_cols(
        monkeypatch,
        "mlframe.feature_selection.filters._extra_fe_families",
        "hybrid_conditional_dispersion_fe",
        captured,
    )
    _fit_with_families(_mrmr_cls, X, y, fe_conditional_dispersion_enable=True)

    assert captured, "conditional_dispersion stage was not exercised"
    for call in captured:
        nc = call["num_cols"]
        assert nc is not None
        assert set(nc) <= raw_cols, f"engineered source leaked into num_cols: {nc}"


def test_fit_grouped_quantile_only_passes_raw_columns(_mrmr_cls, monkeypatch):
    """Fit grouped quantile only passes raw columns."""
    rng = np.random.default_rng(9)
    n = 700
    X = pd.DataFrame({c: rng.normal(size=n) for c in ("a", "b", "c", "d")})
    X["g"] = rng.integers(0, 5, size=n)
    y = ((X["a"] * X["b"]) > 0).astype(int).to_numpy()
    raw_cols = set(X.columns)

    captured: list[dict] = []
    _capture_family_cols(
        monkeypatch,
        "mlframe.feature_selection.filters._grouped_quantile_fe",
        "hybrid_grouped_quantile_fe",
        captured,
    )
    _fit_with_families(_mrmr_cls, X, y, fe_grouped_quantile_enable=True)

    assert captured, "grouped_quantile stage was not exercised"
    for call in captured:
        nc, gc = call["num_cols"], call["group_cols"]
        assert nc is not None and gc is not None, "fix must hand explicit RAW num_cols + group_cols, never None"
        assert set(nc) <= raw_cols, f"engineered source leaked into num_cols: {nc}"
        assert set(gc) <= raw_cols, f"engineered source leaked into group_cols: {gc}"
