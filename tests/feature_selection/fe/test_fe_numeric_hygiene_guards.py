"""Regression tests for silent-NaN / unguarded-arithmetic fixes across the FE candidate-generation and
recipe-replay code paths.

A production MRMR fit with FE enabled logged repeated ``RuntimeWarning: invalid value encountered in
multiply`` traced to unary reciprocal/power transforms (``1/x``, ``1/x**2``, ``1/x**3``, ...) producing
+-inf at ``x=0`` with no epsilon guard, unlike the codebase's own hardened ``_safe_div`` for the binary
case; the +-inf then multiplies with another operand's zero to produce ``0*inf=nan``. Several sibling
materialisation sites also lacked the ``np.errstate`` suppression + ``nan_to_num`` scrub that ~20 other
call sites in this codebase already apply. Each test class below pins one fixed site.
"""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Finding 1: reciprocal-power unary transforms get an epsilon floor at x=0
# (feature_engineering.py: _safe_pow + create_unary_transformations).
# ---------------------------------------------------------------------------


class TestSafePowEpsilonFloor:
    """Groups tests covering TestSafePowEpsilonFloor."""
    @pytest.mark.parametrize(
        "preset,name",
        [
            ("minimal", "reciproc"),
            ("medium", "reciproc"),
            ("medium", "invsquared"),
            ("medium", "invqubed"),
            ("medium", "invcbrt"),
            ("medium", "invsqrt"),
        ],
    )
    def test_zero_input_is_finite(self, preset, name):
        """Zero input is finite."""
        from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

        ut = create_unary_transformations(preset)
        out = np.asarray(ut[name](np.array([0.0])))
        assert np.all(np.isfinite(out)), f"{name} at x=0 must be finite post-fix, got {out}"
        # Magnitude bound: the substituted value must be comparable to the codebase's normal ~1e9
        # zero-substitution scale (_safe_div's own x/eps), not exponent-dependent. A first cut of this fix
        # used a fixed eps=1e-9 FLOOR (then raised to the power), which gives eps**-2=1e18 / eps**-3=1e27
        # for invsquared/invqubed -- large enough to dominate an unscaled downstream model by many orders
        # of magnitude and traced to a real biz-value AUC regression (0.86 -> 0.54) via a compound
        # engineered feature. 1e10 leaves headroom above the 1e9 ceiling while still catching that class.
        assert abs(out[0]) <= 1e10, f"{name} at x=0 must floor to a bounded magnitude, got {out}"

    def test_zero_substitution_magnitude_independent_of_exponent(self):
        """Regression pin for the traced biz-value failure: reciproc/invsquared/invqubed/invcbrt/invsqrt
        must ALL floor a genuine x=0 to the SAME order of magnitude, not one that explodes with the
        exponent (a fixed eps floor gives eps**-1=1e9 but eps**-2=1e18 / eps**-3=1e27)."""
        from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

        ut = create_unary_transformations("medium")
        x = np.array([0.0])
        values = {name: float(np.asarray(ut[name](x))[0]) for name in ("reciproc", "invsquared", "invqubed", "invcbrt", "invsqrt")}
        lo, hi = min(values.values()), max(values.values())
        assert hi / max(lo, 1e-300) <= 10.0, f"zero-substitution magnitudes must be within one order of magnitude of each other, got {values}"

    @pytest.mark.parametrize(
        "preset,name,exponent,x",
        [
            ("minimal", "reciproc", -1, np.array([2.0, 3.5, 10.0, -2.0, -5.5])),
            ("medium", "reciproc", -1, np.array([2.0, 3.5, 10.0, -2.0, -5.5])),
            ("medium", "invsquared", -2, np.array([2.0, 3.5, 10.0, -2.0, -5.5])),
            ("medium", "invqubed", -3, np.array([2.0, 3.5, 10.0, -2.0, -5.5])),
            # invcbrt / invsqrt: positive-only reference values -- a negative base with a fractional
            # exponent is an unrelated, pre-existing nan domain restriction (np.power itself warns/nans
            # on e.g. (-2.0)**-0.5), out of scope for this eps-floor-at-zero fix.
            ("medium", "invcbrt", -1 / 3, np.array([2.0, 3.5, 10.0, 100.0])),
            ("medium", "invsqrt", -1 / 2, np.array([2.0, 3.5, 10.0, 100.0])),
        ],
    )
    def test_nonzero_input_unchanged_vs_pre_fix_formula(self, preset, name, exponent, x):
        """The eps floor must only perturb the exact-zero position; every nonzero value must match the
        pre-fix ``np.power(x, exponent)`` formula exactly, mirroring ``_safe_div``'s own no-perturbation
        guarantee for nonzero denominators."""
        from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

        ut = create_unary_transformations(preset)
        out = np.asarray(ut[name](x))
        expected = np.power(x, exponent)
        np.testing.assert_array_equal(out, expected)

    def test_reciproc_scalar_exact(self):
        """Reciproc scalar exact."""
        from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

        ut = create_unary_transformations("minimal")
        out = float(np.asarray(ut["reciproc"](np.array([2.0])))[0])
        assert out == pytest.approx(0.5)

    def test_mixed_zero_and_nonzero_no_warning(self):
        """A single array holding both an exact zero and ordinary values must produce a finite output at
        the zero position with no RuntimeWarning, and leave the nonzero positions untouched."""
        from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

        ut = create_unary_transformations("medium")
        x = np.array([0.0, 4.0, 9.0, 0.0, 16.0])
        for name, exponent in (("reciproc", -1), ("invsquared", -2), ("invqubed", -3)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out = np.asarray(ut[name](x))
            runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
            assert not runtime_warnings, f"{name}: {[str(x.message) for x in runtime_warnings]}"
            assert np.all(np.isfinite(out))
            nonzero_mask = x != 0.0
            np.testing.assert_array_equal(out[nonzero_mask], np.power(x[nonzero_mask], exponent))


# ---------------------------------------------------------------------------
# Finding 2: _rebuild_full_survivor_col wraps its unary/binary dispatch in np.errstate
# and reassigns nan_to_num's return value (feature_engineering.py).
# ---------------------------------------------------------------------------


class TestRebuildFullSurvivorColErrstate:
    """Groups tests covering TestRebuildFullSurvivorColErrstate."""
    def test_overflow_binary_no_runtime_warning_and_finite_output(self):
        """Overflow binary no runtime warning and finite output."""
        from mlframe.feature_selection.filters.feature_engineering import (
            _rebuild_full_survivor_col,
            create_binary_transformations,
            create_unary_transformations,
        )

        # Negative control: the SAME arithmetic, unwrapped, DOES warn -- proves the scenario is
        # warning-triggering in principle (plain np.multiply overflow on huge float64 operands is not
        # njit-compiled, so it goes through numpy's C-level warning machinery).
        huge = np.full(8, 1e200)
        with warnings.catch_warnings(record=True) as w_ctrl:
            warnings.simplefilter("always")
            np.multiply(huge, huge)
        assert any("overflow" in str(x.message) for x in w_ctrl), "control did not reproduce the overflow warning"

        unary = create_unary_transformations("minimal")
        binary = create_binary_transformations("minimal")
        n = 8
        X_full = pd.DataFrame({"a": np.full(n, 1e200), "b": np.full(n, 1e200)})
        original_cols = {0: 0, 1: 1}
        config = (((0, "identity"), (1, "identity")), "mul", 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _rebuild_full_survivor_col(config, X_full, original_cols, unary, binary, cols=["a", "b"])
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, f"expected no RuntimeWarning post-fix, got {[str(x.message) for x in runtime_warnings]}"
        assert np.all(np.isfinite(out)), f"survivor column must be finite (scrubbed), got {out}"
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Finding 3: _apply_unary_binary / _apply_side extends the errstate wrap to the general
# unary call + binary dispatch (engineered_recipes/_recipe_unary_binary.py).
# ---------------------------------------------------------------------------


class TestApplyUnaryBinaryErrstate:
    """Groups tests covering TestApplyUnaryBinaryErrstate."""
    def test_general_unary_and_binary_path_no_runtime_warning(self, monkeypatch):
        """General unary and binary path no runtime warning."""
        monkeypatch.delenv("MLFRAME_FE_GPU_STRICT_RESIDENT", raising=False)
        from mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary import (
            _apply_unary_binary,
            build_unary_binary_recipe,
        )

        n = 8
        X = pd.DataFrame({"a": np.full(n, 1e200), "b": np.full(n, 1e200)})
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
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _apply_unary_binary(recipe, X)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------------
# Finding 4: generate_pair_cross_basis_features wraps h_a * h_b in np.errstate + nan_to_num
# (_orthogonal_univariate_fe/_orth_pair_cross_fe.py).
# ---------------------------------------------------------------------------


class TestPairCrossBasisScrub:
    """Groups tests covering TestPairCrossBasisScrub."""
    def test_overflowing_basis_product_scrubbed_no_warning(self, monkeypatch):
        """Overflowing basis product scrubbed no warning."""
        import mlframe.feature_selection.filters._orthogonal_univariate_fe as _orth_pkg
        from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import (
            generate_pair_cross_basis_features,
        )

        def _fake_eval(x, basis, deg, **kwargs):
            # Both legs return huge finite values; h_a * h_b overflows via a plain numpy multiply
            # (the same "overflow encountered in multiply" class proven in TestRebuildFullSurvivorColErrstate).
            """Fake eval."""
            return np.full_like(np.asarray(x, dtype=np.float64), 1e200)

        monkeypatch.setattr(_orth_pkg, "_evaluate_basis_column", _fake_eval)

        n = 30
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = generate_pair_cross_basis_features(X, [("a", "b")], max_degree=1, min_degree=1, basis="hermite")
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert not out.empty
        for col in out.columns:
            vals = out[col].to_numpy()
            assert np.all(np.isfinite(vals)), f"{col} not scrubbed: {vals}"

    def test_normal_range_input_produces_finite_columns(self):
        """Non-mocked sanity check: the real basis evaluators on ordinary data never hit the guard,
        so the fix must not perturb the common case."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import (
            generate_pair_cross_basis_features,
        )

        n = 200
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
        out = generate_pair_cross_basis_features(X, [("a", "b")], max_degree=2, min_degree=1)
        assert not out.empty
        for col in out.columns:
            assert np.all(np.isfinite(out[col].to_numpy()))


# ---------------------------------------------------------------------------
# Finding 5: _apply_orth_pair_cross scrubs h_a * h_b before returning
# (engineered_recipes/_orth_basis_recipes.py).
# ---------------------------------------------------------------------------


class TestApplyOrthPairCrossScrub:
    """Groups tests covering TestApplyOrthPairCrossScrub."""
    def test_overflowing_product_scrubbed_in_final_output(self, monkeypatch):
        """Overflowing product scrubbed in final output."""
        import mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes as _obr
        from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe

        def _fake_eval(vals, basis, degree, *, pre_transform="raw", preprocess_params=None):
            """Fake eval."""
            return np.full(len(np.asarray(vals)), 1e200)

        monkeypatch.setattr(_obr, "_eval_orth_basis_column", _fake_eval)

        n = 10
        recipe = EngineeredRecipe(
            name="a*b__He1_He1",
            kind="orth_pair_cross",
            src_names=("a", "b"),
            extra={"basis_i": "hermite", "basis_j": "hermite", "deg_a": 1, "deg_b": 1},
        )
        X = pd.DataFrame({"a": np.linspace(-1, 1, n), "b": np.linspace(-1, 1, n)})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _obr._apply_orth_pair_cross(recipe, X)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert np.all(np.isfinite(np.asarray(out))), f"final output must be scrubbed, got {out}"


# ---------------------------------------------------------------------------
# Finding 6: _finalize_survivor_column drops the redundant nan_to_num + float64 upcast
# (_feature_engineering_pairs/_pairs_emit.py).
# ---------------------------------------------------------------------------


class TestFinalizeSurvivorColumn:
    """Groups tests covering TestFinalizeSurvivorColumn."""
    def test_identical_values_float32_source(self):
        """Identical values float32 source."""
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit import (
            _finalize_survivor_column,
        )

        src = np.array([1.0, 2.5, -3.0, 0.0], dtype=np.float32)
        pre_fix = np.nan_to_num(np.asarray(src, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        post_fix = _finalize_survivor_column(src)
        np.testing.assert_array_equal(post_fix.astype(np.float64), pre_fix)
        # dtype is now preserved (not force-upcast to float64) -- the common producer dtype.
        assert post_fix.dtype == np.float32

    def test_identical_values_float64_source(self):
        """The recompute-fallback path's ``_safe_div`` yields float64 (not float32) -- the fix must NOT
        truncate that precision, since forcing float32 here would be a real (if tiny) behavior change,
        not a pure redundancy removal."""
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit import (
            _finalize_survivor_column,
        )

        src = np.array([1.0, 2.5, -3.0, 0.0], dtype=np.float64)
        pre_fix = np.nan_to_num(np.asarray(src, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        post_fix = _finalize_survivor_column(src)
        np.testing.assert_array_equal(post_fix, pre_fix)
        assert post_fix.dtype == np.float64

    def test_no_second_scrub_needed_already_finite(self):
        """Sanity: an already-scrubbed (finite) array passes through byte-identical -- the function
        performs no numeric transformation at all, only a dtype-preserving asarray."""
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit import (
            _finalize_survivor_column,
        )

        src = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
        # NOTE: _finalize_survivor_column intentionally does NOT scrub -- every real caller has already
        # scrubbed upstream. This test documents that contract, not a claim that raw nan/inf never reaches it.
        out = _finalize_survivor_column(src)
        assert np.isnan(out[1])
        assert np.isposinf(out[2])
        assert np.isneginf(out[3])


# ---------------------------------------------------------------------------
# Bonus finding A: _apply_hermite_pair wraps eval_dispatch + bin_func in np.errstate and
# scrubs the result (engineered_recipes/_recipe_poly_cluster.py). Found by a wider-tree
# sweep beyond the original 7 findings: this replay path had NO guard at all.
# ---------------------------------------------------------------------------


class TestApplyHermitePairScrub:
    """Groups tests covering TestApplyHermitePairScrub."""
    def test_overflowing_eval_dispatch_scrubbed_no_warning(self, monkeypatch):
        """Overflowing eval dispatch scrubbed no warning."""
        import mlframe.feature_selection.filters.hermite_fe as _hf
        from mlframe.feature_selection.filters.engineered_recipes._recipe_poly_cluster import (
            _apply_hermite_pair,
            build_hermite_pair_recipe,
        )

        def _fake_eval_dispatch(z, coef):
            """Fake eval dispatch."""
            return np.full_like(np.asarray(z, dtype=np.float64), 1e200)

        monkeypatch.setitem(_hf._POLY_BASES["hermite"], "eval_dispatch", _fake_eval_dispatch)

        n = 10
        hermite_result = SimpleNamespace(
            coef_a=np.array([0.0, 1.0]),
            coef_b=np.array([0.0, 1.0]),
            basis="hermite",
            bin_func_name="mul",
            preprocess_a={"mean": 0.0, "std": 1.0},
            preprocess_b={"mean": 0.0, "std": 1.0},
            degree_a=1,
            degree_b=1,
        )
        recipe = build_hermite_pair_recipe(name="hermite_test", src_names=("a", "b"), hermite_result=hermite_result)
        X = pd.DataFrame({"a": np.linspace(-1, 1, n), "b": np.linspace(-1, 1, n)})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _apply_hermite_pair(recipe, X)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------------
# Bonus finding B: _apply_orth_fourier wraps the power-warp + sin/cos in np.errstate and
# scrubs the result (engineered_recipes/_orth_basis_recipes.py). ``fourier_powers=(1,2)`` is
# the FE default, and this replay path had no guard at all.
# ---------------------------------------------------------------------------


class TestApplyOrthFourierScrub:
    """Groups tests covering TestApplyOrthFourierScrub."""
    def test_power_overflow_scrubbed_no_warning(self):
        """Power overflow scrubbed no warning."""
        from mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes import (
            _apply_orth_fourier,
        )
        from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe

        recipe = EngineeredRecipe(
            name="a__Fsin1",
            kind="orth_fourier",
            src_names=("a",),
            extra={"kind": "sin", "freq": 1.0, "lo": 0.0, "span": 1.0, "power": 2},
        )
        X = pd.DataFrame({"a": np.array([1e200, 2.0, 3.0, -1e200, 0.0])})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _apply_orth_fourier(recipe, X)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert np.all(np.isfinite(np.asarray(out)))

    def test_normal_range_input_unaffected(self):
        """Ordinary data never hits the guard; the fix must not perturb the common case."""
        from mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes import (
            _apply_orth_fourier,
        )
        from mlframe.feature_selection.filters.engineered_recipes._recipe_core import EngineeredRecipe

        recipe = EngineeredRecipe(
            name="a__Fsin1",
            kind="orth_fourier",
            src_names=("a",),
            extra={"kind": "sin", "freq": 1.0, "lo": -1.0, "span": 2.0, "power": 1},
        )
        X = pd.DataFrame({"a": np.linspace(-1, 1, 20)})
        out = _apply_orth_fourier(recipe, X)
        z = (X["a"].to_numpy() - (-1.0)) / 2.0
        expected = np.sin(2.0 * np.pi * 1.0 * z)
        np.testing.assert_allclose(out, expected)


# ---------------------------------------------------------------------------
# Bonus findings C/D: 3-way / 4-way orth basis products (fit-time generator + replay) wrap
# the product in np.errstate + nan_to_num, mirroring the already-fixed pair-cross site.
# Both families are constructor-default-enabled (fe_hybrid_orth_triplet_enable /
# fe_hybrid_orth_quadruplet_enable).
# ---------------------------------------------------------------------------


class TestTripletQuadrupletBasisProductScrub:
    """Groups tests covering TestTripletQuadrupletBasisProductScrub."""
    def test_triplet_generator_overflow_scrubbed_no_warning(self, monkeypatch):
        """Triplet generator overflow scrubbed no warning."""
        import mlframe.feature_selection.filters._orthogonal_triplet_fe as _tri

        def _fake_eval(x, basis, deg, **kwargs):
            """Fake eval."""
            return np.full_like(np.asarray(x, dtype=np.float64), 1e200)

        monkeypatch.setattr(_tri, "_evaluate_basis_column", _fake_eval)

        n = 20
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n), "c": rng.standard_normal(n)})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _tri.generate_triplet_cross_basis_features(X, [("a", "b", "c")], max_degree=1, min_degree=1, basis="hermite")
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert not out.empty
        for col in out.columns:
            assert np.all(np.isfinite(out[col].to_numpy())), f"{col} not scrubbed"

    def test_triplet_replay_overflow_scrubbed_in_final_output(self, monkeypatch):
        """Triplet replay overflow scrubbed in final output."""
        import mlframe.feature_selection.filters.engineered_recipes as _er
        from mlframe.feature_selection.filters._orthogonal_triplet_fe_recipes import (
            _apply_orth_triplet_cross,
        )

        def _fake_eval(vals, basis, degree, *, pre_transform="raw", preprocess_params=None):
            """Fake eval."""
            return np.full(len(np.asarray(vals)), 1e200)

        monkeypatch.setattr(_er, "_eval_orth_basis_column", _fake_eval)

        n = 10
        recipe = SimpleNamespace(
            src_names=("a", "b", "c"),
            extra={"basis_i": "hermite", "basis_j": "hermite", "basis_k": "hermite", "deg_a": 1, "deg_b": 1, "deg_c": 1},
        )
        X = pd.DataFrame({"a": np.linspace(-1, 1, n), "b": np.linspace(-1, 1, n), "c": np.linspace(-1, 1, n)})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _apply_orth_triplet_cross(recipe, X)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert np.all(np.isfinite(np.asarray(out)))

    def test_quadruplet_generator_overflow_scrubbed_no_warning(self, monkeypatch):
        """Quadruplet generator overflow scrubbed no warning."""
        import mlframe.feature_selection.filters._orthogonal_quadruplet_fe as _quad

        def _fake_eval(x, basis, deg, **kwargs):
            """Fake eval."""
            return np.full_like(np.asarray(x, dtype=np.float64), 1e200)

        monkeypatch.setattr(_quad, "_evaluate_basis_column", _fake_eval)

        n = 20
        rng = np.random.default_rng(3)
        X = pd.DataFrame(
            {
                "a": rng.standard_normal(n),
                "b": rng.standard_normal(n),
                "c": rng.standard_normal(n),
                "d": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _quad.generate_quadruplet_cross_basis_features(X, [("a", "b", "c", "d")], max_degree=1, min_degree=1, basis="hermite")
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert not out.empty
        for col in out.columns:
            assert np.all(np.isfinite(out[col].to_numpy())), f"{col} not scrubbed"

    def test_quadruplet_replay_overflow_scrubbed_in_final_output(self, monkeypatch):
        """Quadruplet replay overflow scrubbed in final output."""
        import mlframe.feature_selection.filters.engineered_recipes as _er
        from mlframe.feature_selection.filters._orthogonal_quadruplet_fe_recipes import (
            _apply_orth_quadruplet_cross,
        )

        def _fake_eval(vals, basis, degree, *, pre_transform="raw", preprocess_params=None):
            """Fake eval."""
            return np.full(len(np.asarray(vals)), 1e200)

        monkeypatch.setattr(_er, "_eval_orth_basis_column", _fake_eval)

        n = 10
        recipe = SimpleNamespace(
            src_names=("a", "b", "c", "d"),
            extra={
                "basis_i": "hermite",
                "basis_j": "hermite",
                "basis_k": "hermite",
                "basis_l": "hermite",
                "deg_a": 1,
                "deg_b": 1,
                "deg_c": 1,
                "deg_d": 1,
            },
        )
        X = pd.DataFrame(
            {
                "a": np.linspace(-1, 1, n),
                "b": np.linspace(-1, 1, n),
                "c": np.linspace(-1, 1, n),
                "d": np.linspace(-1, 1, n),
            }
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = _apply_orth_quadruplet_cross(recipe, X)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert np.all(np.isfinite(np.asarray(out)))


# ---------------------------------------------------------------------------
# Bonus finding E: the ``poly_`` hermval branch in _build_operand_table wraps its call in
# np.errstate, matching its "prewarp" / plain-unary sibling branches
# (_feature_engineering_pairs/_pairs_setup.py). Cosmetic (the value ultimately stored is
# always rebuilt through the already-guarded _rebuild_full_survivor_col), still a genuine
# warning-hygiene gap during the MI search itself.
# ---------------------------------------------------------------------------


class TestBuildOperandTablePolyErrstate:
    """Groups tests covering TestBuildOperandTablePolyErrstate."""
    def test_poly_branch_overflow_no_runtime_warning(self):
        """Poly branch overflow no runtime warning."""
        from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_setup import (
            _build_operand_table,
        )

        n = 2
        raw_vars_pair = ("a",)
        prospective_pairs = {(raw_vars_pair, 1.0): None}
        transformed_vars = np.empty((n, 1), dtype=np.float32)
        unary_transformations = {"poly_test": np.array([0.0, 1e200])}
        vals = np.array([1e200, 2.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vars_transformations = _build_operand_table(
                prospective_pairs=prospective_pairs,
                transformed_vars=transformed_vars,
                _unary_names_eff=["poly_test"],
                unary_transformations=unary_transformations,
                _extval_raw_col=lambda var: vals if var == "a" else None,
                _prewarp_active=False,
                _prewarp_spec_by_var={},
                _gate_med_median_by_var={},
                cols=["a"],
                verbose=0,
                logger=logging.getLogger(__name__),
                gpu_compatible_unary_names=lambda: set(),
            )
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert not runtime_warnings, [str(x.message) for x in runtime_warnings]
        assert ("a", "poly_test") in vars_transformations


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
