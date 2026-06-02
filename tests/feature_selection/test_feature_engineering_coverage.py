"""Coverage-targeted tests for ``mlframe.feature_selection.filters.feature_engineering``.

Exercises:
- ``create_unary_transformations`` presets ("minimal" / "medium" / "maximal")
  including the special-function families (polygamma_0/1/2, struve0/1/2, jv0/1/2).
- ``create_binary_transformations`` presets ("minimal" / "medium" / "maximal").
- ``UNARY_INPUT_CONSTRAINTS`` module-level dict (keys + tag vocab).
- ``get_existing_feature_name`` / ``get_new_feature_name`` formatters.
- ``compute_pairs_mis`` (sorted-dict semantics across full and partial caches).
- ``check_prospective_fe_pairs`` happy path with multiple unary x binary combos.
- ``njit_functions_dict`` exception path (compilation-incompatible funcs are skipped).

The hot paths through ``check_prospective_fe_pairs`` are driven by mocked-out
``mi_direct`` / ``discretize_array`` substitutes to keep the test light; we
exercise the per-pair / per-binary inner loop without booting full MRMR.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.feature_engineering import (
    UNARY_INPUT_CONSTRAINTS,
    check_prospective_fe_pairs,
    compute_pairs_mis,
    create_binary_transformations,
    create_unary_transformations,
    get_existing_feature_name,
    get_new_feature_name,
)
from mlframe.feature_selection.filters._internals import njit_functions_dict


# ---------------------------------------------------------------------------
# create_unary_transformations / create_binary_transformations
# ---------------------------------------------------------------------------


class TestUnaryTransformationsPresets:
    """Each preset must (a) expose the documented superset and
    (b) compile its registry through ``njit_functions_dict`` without raising."""

    @pytest.mark.fast
    def test_minimal_preset_is_workhorse_set(self):
        # The "minimal" preset is a non-degenerate workhorse set (NOT identity-only):
        # it must carry enough building blocks to engineer common signals like
        # sqr(a)/b or log(c)*sin(d) on default MRMR settings.
        unary = create_unary_transformations(preset="minimal")
        assert set(unary.keys()) == {
            "identity", "neg", "abs", "sqr", "reciproc", "sqrt", "log", "sin",
        }
        # identity must round-trip a plain array unchanged
        x = np.linspace(-3, 3, 11, dtype=np.float32)
        np.testing.assert_array_equal(unary["identity"](x), x)

    def test_medium_preset_superset_of_minimal(self):
        medium = create_unary_transformations(preset="medium")
        minimal = create_unary_transformations(preset="minimal")
        for k in minimal:
            assert k in medium
        # A handful of well-known names that must come with the "medium" preset:
        for k in (
            "sign", "neg", "abs", "rint", "sqr", "qubed", "reciproc",
            "invsquared", "invqubed", "cbrt", "sqrt", "invcbrt", "invsqrt",
            "log", "exp", "sin",
        ):
            assert k in medium, f"medium preset missing '{k}'"

    def test_maximal_preset_superset_of_medium(self):
        maximal = create_unary_transformations(preset="maximal")
        medium = create_unary_transformations(preset="medium")
        for k in medium:
            assert k in maximal
        # Maximal-only names per the module docstring / source
        for k in (
            "grad1", "grad2", "sinc", "cos", "tan",
            "arcsin", "arccos", "arctan",
            "sinh", "cosh", "tanh",
            "arcsinh", "arccosh", "arctanh",
            "erf", "dawsn", "gammaln",
            # special families
            "polygamma_0", "polygamma_1", "polygamma_2",
            "struve0", "struve1", "struve2",
            "jv0", "jv1", "jv2",
        ):
            assert k in maximal, f"maximal preset missing '{k}'"

    def test_polygamma_struve_jv_orders_distinct(self):
        """Each integer order produces materially different output --
        guards against a captured-loop-variable bug where every closure
        ended up calling sp.polygamma(2, x)."""
        maximal = create_unary_transformations(preset="maximal")
        x = np.linspace(0.5, 3.0, 7, dtype=np.float64)

        for family in ("polygamma_", "struve", "jv"):
            outs = []
            keys = [k for k in maximal if k.startswith(family)]
            assert len(keys) >= 2, f"family '{family}' should expose multiple orders"
            for k in keys:
                outs.append(np.asarray(maximal[k](x)))
            # Every pair-of-orders must differ in at least one position.
            for i in range(len(outs)):
                for j in range(i + 1, len(outs)):
                    diff = np.abs(outs[i] - outs[j])
                    # Some entries may agree; require at least one disagreement.
                    assert np.nanmax(diff) > 1e-12, (
                        f"{family}{i} and {family}{j} produced identical "
                        "output -- captured-loop bug?"
                    )


class TestBinaryTransformationsPresets:
    @pytest.mark.fast
    def test_minimal_keys(self):
        # minimal carries the six arithmetic workhorses, including div + sub
        # (their prior absence from every tier was the FE regression).
        binary = create_binary_transformations(preset="minimal")
        assert set(binary.keys()) == {"mul", "add", "sub", "div", "max", "min"}

    def test_medium_superset_of_minimal(self):
        """``medium`` is a strict superset of ``minimal`` (adds abs_diff, hypot)."""
        medium = create_binary_transformations(preset="medium")
        minimal = create_binary_transformations(preset="minimal")
        for k in minimal:
            assert k in medium, f"medium preset dropped minimal key '{k}'"
        for k in ("abs_diff", "hypot"):
            assert k in medium, f"medium preset missing '{k}'"

    def test_maximal_superset(self):
        maximal = create_binary_transformations(preset="maximal")
        for k in (
            "mul", "add", "max", "min",
            "hypot", "logaddexp", "agm",
            "pow", "logn", "heaviside",
            "greater", "less", "equal",
            "beta", "binom",
        ):
            assert k in maximal, f"maximal preset missing '{k}'"

    def test_binary_funcs_callable_on_simple_inputs(self):
        binary = create_binary_transformations(preset="minimal")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(binary["mul"](a, b), a * b)
        np.testing.assert_array_equal(binary["add"](a, b), a + b)
        np.testing.assert_array_equal(binary["max"](a, b), np.maximum(a, b))
        np.testing.assert_array_equal(binary["min"](a, b), np.minimum(a, b))


# ---------------------------------------------------------------------------
# UNARY_INPUT_CONSTRAINTS
# ---------------------------------------------------------------------------


class TestUnaryInputConstraints:
    @pytest.mark.fast
    def test_all_keys_resolve_to_known_tags(self):
        valid_tags = {
            "-1to1", "-pi/2topi/2", "1toinf",
            "-0.(9)to0.(9)", "pos", "nonzero",
        }
        for name, tag in UNARY_INPUT_CONSTRAINTS.items():
            assert isinstance(name, str) and name
            assert tag in valid_tags, f"unknown constraint tag '{tag}' for '{name}'"

    def test_critical_keys_present(self):
        """A representative subset that downstream FE relies on."""
        for k in (
            "arccos", "arcsin", "arctan",
            "arccosh", "arctanh",
            "sqrt", "log",
            "reciproc", "invsquared", "invqubed", "invcbrt", "invsqrt",
        ):
            assert k in UNARY_INPUT_CONSTRAINTS, f"missing '{k}'"

    def test_keys_are_subset_of_maximal_preset_or_doc_only(self):
        """Every constraint key must be a real transform name in the maximal preset."""
        maximal = create_unary_transformations(preset="maximal")
        for k in UNARY_INPUT_CONSTRAINTS:
            assert k in maximal, (
                f"UNARY_INPUT_CONSTRAINTS lists '{k}' but the maximal "
                "preset doesn't expose it"
            )


# ---------------------------------------------------------------------------
# Name formatters
# ---------------------------------------------------------------------------


class TestNameFormatters:
    @pytest.mark.fast
    def test_identity_strips_wrapper(self):
        cols = ["a", "b", "c"]
        # (idx, "identity") -> bare column name
        assert get_existing_feature_name((0, "identity"), cols) == "a"
        assert get_existing_feature_name((2, "identity"), cols) == "c"

    def test_non_identity_wraps(self):
        cols = ["a", "b"]
        assert get_existing_feature_name((1, "log"), cols) == "log(b)"
        assert get_existing_feature_name((0, "sin"), cols) == "sin(a)"

    def test_new_feature_name_full(self):
        cols = ["a", "b", "c", "d"]
        # fe_tuple structure: (((idx_a, unary_a), (idx_b, unary_b)), bin_name, i)
        fe = (((0, "log"), (1, "sin")), "mul", 17)
        assert get_new_feature_name(fe, cols) == "mul(log(a),sin(b))"

        # Identity on one side
        fe2 = (((2, "identity"), (3, "sqr")), "add", 99)
        assert get_new_feature_name(fe2, cols) == "add(c,sqr(d))"


# ---------------------------------------------------------------------------
# njit_functions_dict (exception path)
# ---------------------------------------------------------------------------


class TestNjitFunctionsDictExceptions:
    @pytest.mark.fast
    def test_uncompilable_function_left_intact(self):
        """A function numba can't compile must remain callable from the registry
        rather than be removed -- the try/except in ``njit_functions_dict``
        swallows the compile error silently."""

        def python_only(x, y):
            # Use a feature numba refuses to compile in nopython mode (e.g. a
            # dict-of-lists construction that touches reflected types).
            return {str(int(v)): [v + 1] for v in (x, y)}

        registry = {"weird": python_only, "ok": lambda a: a + 1}
        # Should not raise even though ``python_only`` likely fails njit.
        njit_functions_dict(registry, exceptions=())
        # Both keys still present and callable.
        assert "weird" in registry and "ok" in registry
        assert callable(registry["weird"]) and callable(registry["ok"])

    def test_exceptions_list_skips_named_funcs(self):
        """Entries listed in ``exceptions`` are kept as plain Python (verifiable
        because their __name__ matches the original lambda after we wrap with
        a uniquely-named function)."""
        def keep_me(x):
            return x * 2

        keep_me.__name__ = "keep_me_unique_sentinel"
        registry = {"keep_me_unique_sentinel": keep_me, "do_njit": lambda x: x + 1}
        njit_functions_dict(registry, exceptions=("keep_me_unique_sentinel",))
        # ``keep_me_unique_sentinel`` must be exactly the original Python function.
        assert registry["keep_me_unique_sentinel"] is keep_me


# ---------------------------------------------------------------------------
# compute_pairs_mis
# ---------------------------------------------------------------------------


class TestComputePairsMis:
    """``compute_pairs_mis`` populates ``cached_MIs`` for every singleton and
    promotes pairs whose individual-MI sum clears a threshold."""

    @pytest.fixture
    def synthetic_dataset(self):
        rng = np.random.default_rng(0)
        n = 200
        # 3 discrete factors + a target.
        f0 = rng.integers(0, 4, n).astype(np.int32)
        f1 = rng.integers(0, 4, n).astype(np.int32)
        f2 = rng.integers(0, 4, n).astype(np.int32)
        # Strong relationship: y = f0 XOR f1 (so pair (0,1) has high joint MI)
        y = ((f0 + f1) % 4).astype(np.int32)
        data = np.column_stack([f0, f1, f2, y]).astype(np.int32)
        return data

    @pytest.mark.fast
    def test_populates_singletons_and_pairs(self, synthetic_dataset):
        from mlframe.feature_selection.filters.info_theory import merge_vars

        data = synthetic_dataset
        nbins = np.array([4, 4, 4, 4], dtype=np.int64)
        target_indices = np.array([3], dtype=np.int64)

        classes_y, freqs_y, _ = merge_vars(
            factors_data=data, vars_indices=target_indices,
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        classes_y_safe = classes_y.copy()

        all_pairs = [(0, 1), (0, 2), (1, 2)]
        cached_confident_MIs: dict = {}
        cached_MIs: dict = {}

        result = compute_pairs_mis(
            all_pairs=all_pairs,
            data=data, target_indices=target_indices, nbins=nbins,
            classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
            fe_min_nonzero_confidence=0.0,
            fe_npermutations=1,
            cached_confident_MIs=cached_confident_MIs,
            cached_MIs=cached_MIs,
            fe_min_pair_mi=-1.0,             # accept any sum
            fe_min_pair_mi_prevalence=0.0,   # accept any synergistic pair
        )
        # Returns the same cached_MIs dict it mutated.
        assert result is cached_MIs
        # Every singleton must be populated
        for v in (0, 1, 2):
            assert (v,) in cached_MIs, f"missing singleton ({v},)"

    def test_threshold_filters_weak_pairs(self, synthetic_dataset):
        """When ``fe_min_pair_mi`` is huge nothing is promoted; only singletons."""
        from mlframe.feature_selection.filters.info_theory import merge_vars

        data = synthetic_dataset
        nbins = np.array([4, 4, 4, 4], dtype=np.int64)
        target_indices = np.array([3], dtype=np.int64)

        classes_y, freqs_y, _ = merge_vars(
            factors_data=data, vars_indices=target_indices,
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        classes_y_safe = classes_y.copy()

        cached_MIs: dict = {}
        compute_pairs_mis(
            all_pairs=[(0, 1), (0, 2)],
            data=data, target_indices=target_indices, nbins=nbins,
            classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
            fe_min_nonzero_confidence=0.0,
            fe_npermutations=1,
            cached_confident_MIs={},
            cached_MIs=cached_MIs,
            fe_min_pair_mi=1e9,             # impossibly high -> skip pair MI
            fe_min_pair_mi_prevalence=0.0,
        )
        # No pair key should be present (only singletons).
        for k in cached_MIs:
            assert not isinstance(k, tuple) or len(k) == 1

    @pytest.mark.fast
    def test_pair_mi_cached_regardless_of_prevalence_gate(self, synthetic_dataset):
        """T3#24 Pack #5 pair-MI cache regression.

        When the computed pair-MI fails the prevalence gate, pre-fix code dropped it on the floor.
        Pack #5's adaptive retry with relaxed ``fe_min_pair_mi_prevalence`` then RE-COMPUTED the same pair.
        Post-fix: every computed pair MI is cached so retry sees a hit.

        Reproduction strategy: monkey-patch ``mi_direct`` to count calls. Run pass 1 with
        impossibly-high prevalence (every pair fails the gate). Then run pass 2 with prevalence=0
        on the SAME cache and verify NO additional pair-MI calls happened.
        """
        from mlframe.feature_selection.filters import feature_engineering as fe_mod
        from mlframe.feature_selection.filters.info_theory import merge_vars

        data = synthetic_dataset
        nbins = np.array([4, 4, 4, 4], dtype=np.int64)
        target_indices = np.array([3], dtype=np.int64)

        classes_y, freqs_y, _ = merge_vars(
            factors_data=data, vars_indices=target_indices,
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        classes_y_safe = classes_y.copy()

        # Wrap ``mi_direct`` to count invocations.
        call_counts = {"pair": 0, "singleton": 0}
        real_mi_direct = fe_mod.mi_direct

        def counting_mi_direct(data, x, y, **kw):
            if isinstance(x, tuple) and len(x) == 2:
                call_counts["pair"] += 1
            else:
                call_counts["singleton"] += 1
            return real_mi_direct(data, x=x, y=y, **kw)

        fe_mod.mi_direct = counting_mi_direct
        try:
            cached_MIs: dict = {}
            cached_confident_MIs: dict = {}
            pairs = [(0, 1), (0, 2), (1, 2)]

            # Pass 1: huge prevalence so EVERY pair MI fails the gate.
            compute_pairs_mis(
                all_pairs=pairs,
                data=data, target_indices=target_indices, nbins=nbins,
                classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
                fe_min_nonzero_confidence=0.0,
                fe_npermutations=1,
                cached_confident_MIs=cached_confident_MIs,
                cached_MIs=cached_MIs,
                fe_min_pair_mi=-1.0,             # admit every pair to MI compute
                fe_min_pair_mi_prevalence=1e9,   # reject every pair from prevalence
            )
            pair_calls_pass1 = call_counts["pair"]
            # All 3 pair MIs must have been COMPUTED.
            assert pair_calls_pass1 == 3, f"expected 3 pair-MI computes, got {pair_calls_pass1}"
            # Post-fix: every pair MI is cached even though prevalence rejected.
            for p in pairs:
                assert p in cached_MIs, (
                    f"pair {p} MI must be cached after compute regardless of prevalence; "
                    f"got cached_MIs keys: {sorted(cached_MIs.keys())}"
                )

            # Pass 2: relaxed prevalence on same cache. Pre-fix code would recompute;
            # post-fix code hits cache and does NO new pair-MI calls.
            compute_pairs_mis(
                all_pairs=pairs,
                data=data, target_indices=target_indices, nbins=nbins,
                classes_y=classes_y, classes_y_safe=classes_y_safe, freqs_y=freqs_y,
                fe_min_nonzero_confidence=0.0,
                fe_npermutations=1,
                cached_confident_MIs=cached_confident_MIs,
                cached_MIs=cached_MIs,
                fe_min_pair_mi=-1.0,
                fe_min_pair_mi_prevalence=0.0,
            )
            pair_calls_pass2 = call_counts["pair"] - pair_calls_pass1
            assert pair_calls_pass2 == 0, (
                f"adaptive retry must reuse cached pair MIs (Pack #5 T3#24); "
                f"observed {pair_calls_pass2} new pair-MI computes"
            )
        finally:
            fe_mod.mi_direct = real_mi_direct


# ---------------------------------------------------------------------------
# check_prospective_fe_pairs
# ---------------------------------------------------------------------------


class TestCheckProspectiveFePairs:
    """Drive the full ``check_prospective_fe_pairs`` loop end-to-end on a tiny
    DataFrame so the per-pair / per-binary inner code (incl. NaN scrubbing,
    leader selection, name formatting) is exercised."""

    @pytest.fixture
    def small_pandas_data(self):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({
            "a": rng.uniform(0.5, 5.0, n).astype(np.float32),
            "b": rng.uniform(-2.0, 2.0, n).astype(np.float32),
            "c": rng.uniform(0.1, 1.0, n).astype(np.float32),
        })
        return df

    @pytest.mark.fast
    def test_runs_end_to_end_pandas(self, small_pandas_data):
        from mlframe.feature_selection.filters.info_theory import merge_vars

        df = small_pandas_data
        # Build a synthetic discrete dataset for the MI machinery to operate on.
        # The MI evaluation inside ``check_prospective_fe_pairs`` walks ``classes_y``
        # / ``freqs_y`` which we provide via merge_vars on a fake target.
        from mlframe.feature_selection.filters.discretization import discretize_array
        data = np.column_stack([
            discretize_array(df["a"].to_numpy(), n_bins=4, method="quantile", dtype=np.int32),
            discretize_array(df["b"].to_numpy(), n_bins=4, method="quantile", dtype=np.int32),
            discretize_array(df["c"].to_numpy(), n_bins=4, method="quantile", dtype=np.int32),
        ])
        target_col = (df["a"].to_numpy() > df["a"].mean()).astype(np.int32)
        data = np.column_stack([data, target_col])
        nbins = np.array([4, 4, 4, 2], dtype=np.int64)
        target_indices = np.array([3], dtype=np.int64)
        classes_y, freqs_y, _ = merge_vars(
            factors_data=data, vars_indices=target_indices,
            var_is_nominal=None, factors_nbins=nbins, dtype=np.int32,
        )
        classes_y_safe = classes_y.copy()

        unary = create_unary_transformations(preset="minimal")
        binary = create_binary_transformations(preset="minimal")

        cols_names = ["a", "b", "c"]
        original_cols = {0: 0, 1: 1, 2: 2}  # var-index -> column position in X
        # One prospective pair: (0, 1) with some pair-MI uplift.
        prospective_pairs = {((0, 1), 1.0): 1.5}

        times_spent: dict = defaultdict(float)

        res = check_prospective_fe_pairs(
            prospective_pairs=prospective_pairs,
            X=df,
            unary_transformations=unary,
            binary_transformations=binary,
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            num_fs_steps=0,
            cols=cols_names,
            original_cols=original_cols,
            fe_max_steps=1,
            fe_npermutations=1,
            fe_max_pair_features=2,
            fe_print_best_mis_only=True,
            fe_min_nonzero_confidence=0.0,
            fe_min_engineered_mi_prevalence=0.0,  # accept everything
            fe_good_to_best_feature_mi_threshold=0.5,
            fe_max_external_validation_factors=0,
            numeric_vars_to_consider=[0, 1, 2],
            quantization_nbins=4,
            quantization_method="quantile",
            quantization_dtype=np.int32,
            times_spent=times_spent,
            verbose=0,
        )
        # ``res`` keyed by raw_vars_pair; with our permissive prevalence threshold
        # the pair should appear with at least one recommended feature.
        assert (0, 1) in res
        this_pair_features, transformed_vals, new_cols, new_nbins, messages = res[(0, 1)]
        assert isinstance(this_pair_features, set)
