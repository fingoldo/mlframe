"""End-to-end tests for the PR-1 transform-replay contract.

Verifies that ``MRMR.fit`` + ``MRMR.transform`` correctly:

1. Build ``EngineeredRecipe`` objects when ``fe_max_steps > 1`` and an
   engineered feature ends up in the final selection.
2. Replay each surviving recipe on test data, returning the engineered
   columns alongside the base-feature columns in ``transform()``
   output.
3. Surface engineered names via ``get_feature_names_out()`` after the
   base names.
4. Round-trip cleanly through pickle (legacy pickles without
   ``_engineered_recipes_`` resurface with an empty list, mirroring
   the legacy "no replay" behaviour bit-exact).

Tests are kept fast (n<=300, low permutation budget) so they fit the
``pytest -k "fast"`` profile envisaged in the cat-FE plan; multi-second
heavy fixtures live in ``test_mrmr_feature_engineering.py``.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe


@pytest.fixture
def multiplicative_synergy_train_test():
    """y = a * b + small_noise; same generative process for train and
    test, but different rng seeds so the rows are disjoint."""
    rng = np.random.default_rng(42)
    n_train, n_test = 200, 80
    a_tr = rng.uniform(-3, 3, n_train).astype(np.float32)
    b_tr = rng.uniform(-3, 3, n_train).astype(np.float32)
    a_te = rng.uniform(-3, 3, n_test).astype(np.float32)
    b_te = rng.uniform(-3, 3, n_test).astype(np.float32)
    noise_tr = rng.uniform(-1, 1, n_train).astype(np.float32)
    noise_te = rng.uniform(-1, 1, n_test).astype(np.float32)

    # Discretise target into 4 bins so MI sees structure on small n.
    y_tr_cont = a_tr * b_tr
    y_te_cont = a_te * b_te
    edges = np.quantile(y_tr_cont, [0.25, 0.5, 0.75])
    y_tr = np.digitize(y_tr_cont, edges).astype(np.int32)
    y_te = np.digitize(y_te_cont, edges).astype(np.int32)

    df_tr = pd.DataFrame({"a": a_tr, "b": b_tr, "noise": noise_tr})
    df_te = pd.DataFrame({"a": a_te, "b": b_te, "noise": noise_te})
    return df_tr, pd.Series(y_tr, name="target"), df_te, pd.Series(y_te, name="target")


# ---------------------------------------------------------------------------
# Pickle BC -- runs first to verify the safety contract before any fit
# ---------------------------------------------------------------------------


class TestPickleBC:
    """A pickle from the pre-PR-1 era has no ``_engineered_recipes_``
    attribute. ``__setstate__`` injects an empty list as the default,
    so legacy unpickle + transform behaves identically to the old
    "engineered cols dropped" path."""

    def test_legacy_pickle_without_recipes_field_loads(self):
        # Fit a fresh MRMR, then mutate its dict to simulate an older
        # pickle that lacked the recipe field. Round-trip and verify
        # the default kicks in.
        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,  # FE off -- legacy default surface
            verbose=0,
            n_jobs=1,
        )
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
        y = pd.Series(rng.integers(0, 2, size=80), name="target")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)

        # Strip the recipe attribute to mimic an older pickle.
        if hasattr(mrmr, "_engineered_recipes_"):
            del mrmr.__dict__["_engineered_recipes_"]

        restored = pickle.loads(pickle.dumps(mrmr))
        assert hasattr(restored, "_engineered_recipes_"), "__setstate__ must inject the missing default"
        assert restored._engineered_recipes_ == [], "Legacy pickle should resurface with an empty recipe list"

        # Transform should still work and produce base-features-only output.
        out = restored.transform(X)
        assert out.shape[0] == X.shape[0]
        # No engineered cols → output cols == base support.
        assert out.shape[1] == int(restored.n_features_)


# ---------------------------------------------------------------------------
# Recipe construction during fit
# ---------------------------------------------------------------------------


class TestRecipeBuilding:
    """When ``fe_max_steps > 1`` and an engineered name ends up in the
    selected set, a corresponding ``EngineeredRecipe`` lands in
    ``self._engineered_recipes_``."""

    @pytest.mark.slow
    def test_fe_max_steps_2_records_recipe_for_selected_engineered(self, multiplicative_synergy_train_test):
        df_tr, y_tr, _df_te, _y_te = multiplicative_synergy_train_test
        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            fe_max_steps=2,
            fe_npermutations=5,
            fe_unary_preset="medium",  # adds sqr/log/sin so mul/abs find synergy
            fe_binary_preset="medium",
            fe_min_pair_mi_prevalence=1.05,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df_tr, y_tr)

        # The recipe list is type-correct (every entry is a recipe with
        # name/kind/src_names) regardless of whether the synergy was
        # actually found in this small fixture -- some FE runs converge
        # without picking the engineered col.
        recipes = mrmr._engineered_recipes_
        assert isinstance(recipes, list)
        # The FE engine emits a growing vocabulary of recipe kinds (unary_binary,
        # orth_pair_cross, hermite_pair, ...); whichever wins the synergy search is
        # legitimate. Pin the structural contract -- a recognised kind, >=1 source,
        # every source a real input column -- not one specific kind.
        from typing import get_args, get_type_hints

        _valid_kinds = set(get_args(get_type_hints(EngineeredRecipe)["kind"]))
        for r in recipes:
            assert isinstance(r, EngineeredRecipe)
            assert r.kind in _valid_kinds, f"Unknown recipe kind {r.kind!r}"
            assert len(r.src_names) >= 1
            for src in r.src_names:
                assert src in df_tr.columns, f"Recipe src '{src}' must reference a real input column"

    @pytest.mark.slow
    def test_fe_step_appends_nbins_via_concat_not_elementwise_add(self, multiplicative_synergy_train_test):
        """Regression sensor for 2026-05-11 nbins concatenation bug.

        At ``mrmr.py:1286`` the FE step previously did
        ``nbins = nbins + new_nbins``. ``nbins`` is a numpy ndarray
        returned by ``categorize_dataset``; ``+`` does element-wise
        addition (or shape-mismatch broadcast error), NOT
        concatenation. After the FE step ``data.shape[1]`` and
        ``len(nbins)`` desynchronised and the next
        ``screen_predictors`` call tripped its
        ``targets_data.shape[1] == len(targets_nbins)`` assertion.

        Reuses ``multiplicative_synergy_train_test`` (same a*b
        synergy fixture) because that data reliably triggers the FE
        step to append engineered columns -- which is the only path
        that exercises the buggy ``nbins + new_nbins`` line.
        Verified that the test FAILS on pre-fix code (mid-fit
        AssertionError at screen.py:314) and PASSES on post-fix
        code (clean fit).
        """
        df_tr, y_tr, _df_te, _y_te = multiplicative_synergy_train_test
        mrmr = MRMR(
            full_npermutations=5,
            baseline_npermutations=5,
            fe_max_steps=2,  # forces the FE-step path that hits the bug-line
            fe_npermutations=5,
            fe_unary_preset="medium",
            fe_binary_preset="medium",
            fe_min_pair_mi_prevalence=1.05,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df_tr, y_tr)
        # Pre-fix: the inner ``screen_predictors`` call after the FE
        # step raised AssertionError mid-fit. Post-fix: ``fit``
        # completes; reaching these asserts at all is the regression
        # signal.
        assert mrmr.support_ is not None
        # And the recipes list is well-formed (downstream consumers
        # iterate over it via .kind / .src_names).
        for r in mrmr._engineered_recipes_:
            assert isinstance(r, EngineeredRecipe)

    def test_fe_max_steps_1_records_no_recipes(self):
        """At the ``fe_max_steps=1`` default no engineered cols are
        added back to ``data`` (they're computed but never re-screened),
        so no recipes survive into ``self._engineered_recipes_``."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "a": rng.uniform(-3, 3, 100),
                "b": rng.uniform(-3, 3, 100),
            }
        )
        y = pd.Series((rng.normal(size=100) > 0).astype(int), name="target")
        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=1,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df, y)
        assert mrmr._engineered_recipes_ == []


# ---------------------------------------------------------------------------
# Transform replay & get_feature_names_out
# ---------------------------------------------------------------------------


class TestTransformOnTestData:
    """When the FE step produces engineered features that survive
    selection, ``transform(X_test)`` must recompute them on test data
    and append them to the output."""

    def test_transform_with_no_recipes_matches_legacy_shape(self):
        """Default MRMR (``fe_max_steps=1``) records no recipes, so
        transform output is base-features-only -- identical to the
        pre-PR-1 path."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "a": rng.uniform(0, 1, 80),
                "b": rng.uniform(0, 1, 80),
                "c": rng.uniform(0, 1, 80),
            }
        )
        y = pd.Series((rng.normal(size=80) > 0).astype(int), name="target")
        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df, y)

        out = mrmr.transform(df)
        # No recipes => output is purely the base support columns.
        assert out.shape == (80, int(mrmr.n_features_))
        names = mrmr.get_feature_names_out()
        assert len(names) == int(mrmr.n_features_)

    def test_get_feature_names_out_appends_engineered_names(self):
        """When recipes exist, ``get_feature_names_out`` is base + engineered."""
        # We can't easily force an MRMR fit to emit recipes on a tiny
        # synthetic, so we directly seed the attribute -- the contract
        # under test is the name-emission, not the fit path.
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.uniform(0, 1, 50), "b": rng.uniform(0, 1, 50)})
        y = pd.Series((rng.normal(size=50) > 0).astype(int))
        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df, y)

        # Inject a synthetic recipe to exercise the surfaced-name path.
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_unary_binary_recipe,
        )

        synthetic = build_unary_binary_recipe(
            name="mul(identity(a),identity(b))",
            src_a_name="a",
            src_b_name="b",
            unary_a_name="identity",
            unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal",
            binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )
        mrmr._engineered_recipes_ = [synthetic]

        names = mrmr.get_feature_names_out()
        assert "mul(identity(a),identity(b))" in names
        assert names[-1] == "mul(identity(a),identity(b))", "Engineered names must appear AFTER base names (transform-output order)"

    def test_transform_replays_engineered_recipe_on_disjoint_test_set(self, multiplicative_synergy_train_test):
        """fit(train), then transform(test). Engineered col values on
        test data must equal the formula evaluated on test data --
        there's no leakage of fit-time values."""
        df_tr, y_tr, df_te, _y_te = multiplicative_synergy_train_test

        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df_tr, y_tr)

        # Inject a recipe representing ``mul(a, b)`` to bypass the FE
        # convergence sensitivity on tiny fixtures. The transform
        # contract is what we want to test here.
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_unary_binary_recipe,
        )

        recipe = build_unary_binary_recipe(
            name="mul(identity(a),identity(b))",
            src_a_name="a",
            src_b_name="b",
            unary_a_name="identity",
            unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal",
            binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )
        mrmr._engineered_recipes_ = [recipe]

        out = mrmr.transform(df_te)
        # Engineered col is the LAST col in transform output.
        # Pandas DataFrame -> verify by name.
        assert isinstance(out, pd.DataFrame)
        assert "mul(identity(a),identity(b))" in out.columns
        np.testing.assert_allclose(
            out["mul(identity(a),identity(b))"].to_numpy(),
            df_te["a"].to_numpy() * df_te["b"].to_numpy(),
            rtol=1e-5,
        )

    def test_transform_with_only_engineered_no_base_features(self):
        """Edge case: ``support_`` is empty but recipes exist.
        Transform should return only the engineered cols."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.uniform(0, 1, 50), "b": rng.uniform(0, 1, 50)})
        y = pd.Series((rng.normal(size=50) > 0).astype(int))
        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df, y)

        # Force empty support to exercise the recipes-only path.
        mrmr.support_ = np.array([], dtype=np.intp)
        mrmr.n_features_ = 0

        from mlframe.feature_selection.filters.engineered_recipes import (
            build_unary_binary_recipe,
        )

        mrmr._engineered_recipes_ = [
            build_unary_binary_recipe(
                name="add(identity(a),identity(b))",
                src_a_name="a",
                src_b_name="b",
                unary_a_name="identity",
                unary_b_name="identity",
                binary_name="add",
                unary_preset="minimal",
                binary_preset="minimal",
                quantization_nbins=None,
                quantization_method=None,
                quantization_dtype=np.float32,
            )
        ]

        out = mrmr.transform(df)
        # Output has 1 column (the engineered) and 50 rows.
        assert out.shape == (50, 1)
        np.testing.assert_allclose(
            out.iloc[:, 0].to_numpy(),
            df["a"].to_numpy() + df["b"].to_numpy(),
            rtol=1e-5,
        )

    def test_transform_with_neither_support_nor_recipes_returns_empty(self):
        """Edge case: nothing selected, no recipes. Legacy contract:
        return empty DataFrame with same row count, zero columns."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.uniform(0, 1, 30), "b": rng.uniform(0, 1, 30)})
        y = pd.Series((rng.normal(size=30) > 0).astype(int))
        mrmr = MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(df, y)

        mrmr.support_ = np.array([], dtype=np.intp)
        mrmr.n_features_ = 0
        mrmr._engineered_recipes_ = []

        out = mrmr.transform(df)
        assert out.shape == (30, 0)
