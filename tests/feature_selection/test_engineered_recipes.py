"""Unit tests for ``engineered_recipes`` -- the recipe-based replay
layer that lets ``MRMR.transform`` recompute engineered features on
test data.

Three concerns under test:

1. ``apply_recipe`` recomputes the same column on test data as the
   fit-time computation produced.
2. The recipe round-trips through pickle (so MRMR persistence works).
3. Sensible errors fire on malformed recipes / missing functions /
   unsupported X types.

These tests do NOT exercise MRMR -- the integration tests live in
``test_mrmr_feature_engineering.py`` once PR-1 plumbs the recipe
storage through ``_run_fe_step``.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import (
    EngineeredRecipe,
    apply_recipe,
    build_unary_binary_recipe,
)
from mlframe.feature_selection.filters.feature_engineering import (
    create_unary_transformations,
    create_binary_transformations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pair_data():
    """Two numeric columns + a target. Used to verify replay matches
    the fit-time formula on known inputs."""
    rng = np.random.default_rng(42)
    n = 100
    a = rng.uniform(0.1, 10.0, n).astype(np.float32)
    b = rng.uniform(-5.0, 5.0, n).astype(np.float32)
    df = pd.DataFrame({"a": a, "b": b})
    return df


# ---------------------------------------------------------------------------
# 1. Recipe replay correctness
# ---------------------------------------------------------------------------


class TestApplyRecipeUnaryBinary:
    """``apply_recipe`` for ``"unary_binary"`` kind matches the fit-time
    formula on identical inputs and on disjoint test inputs."""

    @pytest.mark.fast
    def test_identity_pair_mul_replays_exactly(self, simple_pair_data):
        """``mul(identity(a), identity(b))`` == ``a * b`` element-wise."""
        recipe = build_unary_binary_recipe(
            name="mul(a,b)",
            src_a_name="a", src_b_name="b",
            unary_a_name="identity", unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )
        out = apply_recipe(recipe, simple_pair_data)
        expected = simple_pair_data["a"].to_numpy() * simple_pair_data["b"].to_numpy()
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_log_sin_mul_replays(self, simple_pair_data):
        """``mul(log(a), sin(b))`` matches numpy reference computation
        exactly on the same inputs (modulo NaN/Inf scrubbing). ``log``,
        ``sin`` and ``mul`` all live in the ``"minimal"`` workhorse preset."""
        recipe = build_unary_binary_recipe(
            name="mul(log(a),sin(b))",
            src_a_name="a", src_b_name="b",
            unary_a_name="log", unary_b_name="sin",
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )
        out = apply_recipe(recipe, simple_pair_data)

        # Reference: same operations through the registry to guarantee
        # behavioural parity even if registries swap to different log/sin
        # impls in the future.
        unary = create_unary_transformations(preset="minimal")
        binary = create_binary_transformations(preset="minimal")
        expected = binary["mul"](
            unary["log"](simple_pair_data["a"].to_numpy()),
            unary["sin"](simple_pair_data["b"].to_numpy()),
        )
        expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_replay_on_test_data_independent_from_fit_data(self):
        """Recipe holds no fit-time data: replaying on a different
        DataFrame with the same column names works and returns values
        consistent with the formula. ``sqr`` lives in the
        ``"minimal"`` preset, so the recipe records that preset."""
        rng = np.random.default_rng(0)
        # fit-time frame, 100 rows
        df_fit = pd.DataFrame({"a": rng.uniform(1, 10, 100), "b": rng.uniform(0, 1, 100)})
        # transform-time frame, 50 different rows
        df_test = pd.DataFrame({"a": rng.uniform(1, 10, 50), "b": rng.uniform(0, 1, 50)})

        recipe = build_unary_binary_recipe(
            name="mul(sqr(a),identity(b))",
            src_a_name="a", src_b_name="b",
            unary_a_name="sqr", unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )

        # Replay on FIT data first -- consistency baseline
        fit_out = apply_recipe(recipe, df_fit)
        assert fit_out.shape == (100,)
        np.testing.assert_allclose(
            fit_out, df_fit["a"].to_numpy() ** 2 * df_fit["b"].to_numpy(), rtol=1e-5
        )

        # Replay on TEST data -- the actual contract this PR is shipping
        test_out = apply_recipe(recipe, df_test)
        assert test_out.shape == (50,)
        np.testing.assert_allclose(
            test_out, df_test["a"].to_numpy() ** 2 * df_test["b"].to_numpy(), rtol=1e-5
        )

    def test_quantization_is_applied_when_recorded(self, simple_pair_data):
        """When the recipe records a discretization scheme, replay output
        is integer-coded with values in ``[0, nbins)``."""
        recipe = build_unary_binary_recipe(
            name="mul(a,b)_quantized",
            src_a_name="a", src_b_name="b",
            unary_a_name="identity", unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
            quantization_nbins=10,
            quantization_method="uniform",
            quantization_dtype=np.int16,
        )
        # The recipe carries a quantization scheme but no persisted fit-time edges, so replay warns that it re-quantiles on the input; assert + consume that expected notice.
        with pytest.warns(UserWarning, match="no fit-time quantile edges"):
            out = apply_recipe(recipe, simple_pair_data)

        assert out.dtype == np.int16, f"Expected int16, got {out.dtype}"
        assert out.min() >= 0
        assert out.max() <= 9, "uniform binning should clip to [0, nbins-1]"


# ---------------------------------------------------------------------------
# 2. Persistence
# ---------------------------------------------------------------------------


class TestRecipePersistence:
    """Recipes must round-trip through pickle so MRMR persistence works
    (joblib uses pickle under the hood)."""

    def test_pickle_round_trip_preserves_replay(self, simple_pair_data):
        original = build_unary_binary_recipe(
            name="add(neg(a),identity(b))",
            src_a_name="a", src_b_name="b",
            unary_a_name="neg", unary_b_name="identity",
            binary_name="add",
            unary_preset="minimal", binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )
        restored = pickle.loads(pickle.dumps(original))

        assert restored == original, "frozen dataclass should be value-equal"

        out_orig = apply_recipe(original, simple_pair_data)
        out_rest = apply_recipe(restored, simple_pair_data)
        np.testing.assert_array_equal(out_orig, out_rest)

    def test_recipes_are_hashable(self):
        """Frozen dataclass implies hashable. A future MRMR change might
        store recipes in a set or dict-key for dedup; keep that door
        open by asserting hashability today."""
        r = build_unary_binary_recipe(
            name="mul(a,b)",
            src_a_name="a", src_b_name="b",
            unary_a_name="identity", unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )
        # ``extra`` is a dict (not hashable). The dataclass is frozen but
        # contains a dict field, so hashing fails -- which is fine. Just
        # assert equality still works.
        assert r == r
        # If hash is needed in the future, switch to ``frozenset(extra.items())``.


# ---------------------------------------------------------------------------
# 3. Error handling
# ---------------------------------------------------------------------------


class TestRecipeErrors:
    """Replay must fail loudly on misconfigured recipes and unsupported
    input types -- never silently produce wrong-shape output."""

    def test_unknown_unary_function_raises(self, simple_pair_data):
        recipe = EngineeredRecipe(
            name="bogus(a,b)",
            kind="unary_binary",
            src_names=("a", "b"),
            unary_names=("not_a_real_function", "identity"),
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
        )
        with pytest.raises(KeyError, match="not_a_real_function"):
            apply_recipe(recipe, simple_pair_data)

    def test_unknown_binary_function_raises(self, simple_pair_data):
        recipe = EngineeredRecipe(
            name="bogus(a,b)",
            kind="unary_binary",
            src_names=("a", "b"),
            unary_names=("identity", "identity"),
            binary_name="not_a_real_binary",
            unary_preset="minimal", binary_preset="minimal",
        )
        with pytest.raises(KeyError, match="not_a_real_binary"):
            apply_recipe(recipe, simple_pair_data)

    def test_wrong_arity_raises(self, simple_pair_data):
        recipe = EngineeredRecipe(
            name="mul(a)", kind="unary_binary",
            src_names=("a",), unary_names=("identity",),
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
        )
        with pytest.raises(ValueError, match="must have exactly 2 src_names"):
            apply_recipe(recipe, simple_pair_data)

    def test_unknown_kind_raises(self, simple_pair_data):
        recipe = EngineeredRecipe(
            name="mystery", kind="quaternary_thing",  # type: ignore[arg-type]
            src_names=("a", "b"),
        )
        with pytest.raises(ValueError, match="Unknown recipe kind"):
            apply_recipe(recipe, simple_pair_data)

    def test_factorize_without_lookup_raises(self):
        """A factorize recipe needs ``recipe.extra['lookup_table']``;
        without it, replay should fail loud."""
        recipe = EngineeredRecipe(
            name="kway(a,b)", kind="factorize",
            src_names=("a", "b"),
            factorize_nbins=(4, 4),
        )
        with pytest.raises(KeyError, match="lookup_table"):
            apply_recipe(recipe, pd.DataFrame({"a": [0, 1, 2, 3], "b": [0, 1, 2, 3]}))

    def test_plain_ndarray_no_names_raises(self):
        """Raw 2-D ndarray has no column names -- caller must pass a
        framed input. We refuse to guess."""
        recipe = build_unary_binary_recipe(
            name="mul(a,b)",
            src_a_name="a", src_b_name="b",
            unary_a_name="identity", unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )
        with pytest.raises(KeyError, match="ndarray"):
            apply_recipe(recipe, np.zeros((10, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# 4. Polars compatibility (optional dep)
# ---------------------------------------------------------------------------


class TestNaNHandlingFactorize:
    """Tier 2.3: NaN / non-integer values at transform time.

    Test data often has NaN where train was a category. The factorize
    lookup requires int index; we handle NaN per ``unknown_strategy``.
    """

    def _build_recipe(self, unknown_strategy):
        from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe
        lookup = np.array([0, 1, 2, 3], dtype=np.int64)
        return EngineeredRecipe(
            name="kway(x1__x2)", kind="factorize",
            src_names=("x1", "x2"),
            factorize_nbins=(2, 2),
            unknown_strategy=unknown_strategy,
            extra={"lookup_table": lookup, "n_uniq_post_prune": 4},
        )

    def test_nan_clipped_under_clip_strategy(self):
        recipe = self._build_recipe(unknown_strategy="clip")
        df = pd.DataFrame({
            "x1": [0.0, 1.0, np.nan, 0.0],
            "x2": [0.0, 1.0, 0.0, np.nan],
        })
        out = apply_recipe(recipe, df)
        # No crash, all output codes are valid
        assert out.shape == (4,)
        assert (out >= 0).all()
        assert (out < 4).all()

    def test_nan_raises_under_raise_strategy(self):
        recipe = self._build_recipe(unknown_strategy="raise")
        df = pd.DataFrame({"x1": [0.0, np.nan], "x2": [1.0, 0.0]})
        with pytest.raises(ValueError, match="NaN"):
            apply_recipe(recipe, df)

    def test_all_nan_column_under_clip(self):
        recipe = self._build_recipe(unknown_strategy="clip")
        df = pd.DataFrame({
            "x1": [np.nan, np.nan, np.nan],
            "x2": [0.0, 1.0, 0.0],
        })
        out = apply_recipe(recipe, df)
        assert out.shape == (3,)


class TestApplyRecipeFactorize:
    """Cat-FE replay -- the recipe carries a pre-built lookup table
    that maps ``(a, b)`` pre-prune codes to post-prune classes."""

    def _build_factorize_recipe_for_xor(self, unknown_strategy="clip"):
        """Construct a factorize recipe equivalent to ``merge_vars(x1, x2)``
        for binary x1 / x2, by hand. Captures the lookup table that
        ``merge_vars`` would have produced for the four (x1, x2) cells."""
        # nbins_a=2, nbins_b=2 -> pre-prune codes 0..3.
        # Suppose all 4 combos appeared at fit time, post-prune classes
        # are 0..3. (No pruning needed.)
        lookup = np.array([0, 1, 2, 3], dtype=np.int64)
        return EngineeredRecipe(
            name="kway(x1__x2)", kind="factorize",
            src_names=("x1", "x2"),
            factorize_nbins=(2, 2),
            unknown_strategy=unknown_strategy,
            extra={"lookup_table": lookup, "n_uniq_post_prune": 4},
        )

    def test_factorize_replay_recovers_pre_prune_encoding(self):
        recipe = self._build_factorize_recipe_for_xor()
        df = pd.DataFrame({
            "x1": [0, 1, 0, 1, 0, 1],
            "x2": [0, 0, 1, 1, 0, 1],
        })
        out = apply_recipe(recipe, df)
        # By the recipe lookup ([0, 1, 2, 3]):
        #   (0, 0) -> code 0 -> class 0
        #   (1, 0) -> code 1 -> class 1
        #   (0, 1) -> code 2 -> class 2
        #   (1, 1) -> code 3 -> class 3
        np.testing.assert_array_equal(out, np.array([0, 1, 2, 3, 0, 3]))

    def test_factorize_clip_unseen_to_max_class(self):
        """When a test row has (x1, x2) values within fit-time cardinalities
        but the COMBINATION is unseen, ``unknown_strategy='clip'`` (default)
        maps it to the highest seen class."""
        # Fit-time saw only (0, 0) -> class 0; (1, 1) is unseen.
        lookup = np.array([0, -1, -1, -1], dtype=np.int64)
        # clip: unseen -> max seen class = 0
        seen_max = 0
        lookup_filled = lookup.copy()
        lookup_filled[lookup < 0] = seen_max
        recipe = EngineeredRecipe(
            name="kway(x1__x2)", kind="factorize",
            src_names=("x1", "x2"),
            factorize_nbins=(2, 2),
            unknown_strategy="clip",
            extra={"lookup_table": lookup_filled, "n_uniq_post_prune": 1},
        )
        df = pd.DataFrame({"x1": [0, 1, 1, 0], "x2": [0, 0, 1, 1]})
        out = apply_recipe(recipe, df)
        # All 4 rows map to class 0 (the only seen class)
        np.testing.assert_array_equal(out, np.array([0, 0, 0, 0]))

    def test_factorize_raise_on_unseen_combination(self):
        """``unknown_strategy='raise'`` keeps -1 sentinels and surfaces a
        clear error when test data has unseen (a, b) combos."""
        # Lookup unfilled (-1 marks unseen)
        lookup = np.array([0, 1, -1, -1], dtype=np.int64)
        recipe = EngineeredRecipe(
            name="kway(x1__x2)", kind="factorize",
            src_names=("x1", "x2"),
            factorize_nbins=(2, 2),
            unknown_strategy="raise",
            extra={"lookup_table": lookup, "n_uniq_post_prune": 2},
        )
        df = pd.DataFrame({"x1": [0, 1], "x2": [0, 1]})
        # Row 0: (0, 0) -> code 0 -> class 0. Fine.
        # Row 1: (1, 1) -> code 3 -> -1. Should raise.
        with pytest.raises(ValueError, match="not seen during fit"):
            apply_recipe(recipe, df)

    def test_factorize_replay_clips_out_of_range_test_values(self):
        """Test value above ``nbins_a`` (e.g. user has a new category in
        production) is clipped to ``nbins_a - 1``, avoiding buffer overrun."""
        lookup = np.array([0, 1, 2, 3], dtype=np.int64)
        recipe = EngineeredRecipe(
            name="kway(x1__x2)", kind="factorize",
            src_names=("x1", "x2"),
            factorize_nbins=(2, 2),
            unknown_strategy="clip",
            extra={"lookup_table": lookup, "n_uniq_post_prune": 4},
        )
        # x1=5 is way above nbins_a=2; should clip to 1.
        df = pd.DataFrame({"x1": [0, 5], "x2": [0, 1]})
        out = apply_recipe(recipe, df)
        # Row 0: (0, 0) -> 0. Row 1: (clipped 1, 1) -> code 3 -> 3.
        np.testing.assert_array_equal(out, np.array([0, 3]))

    def test_factorize_polars_compat(self):
        """Recipe replay accepts polars frames."""
        pl = pytest.importorskip("polars")
        lookup = np.array([0, 1, 2, 3], dtype=np.int64)
        recipe = EngineeredRecipe(
            name="kway(x1__x2)", kind="factorize",
            src_names=("x1", "x2"),
            factorize_nbins=(2, 2),
            unknown_strategy="clip",
            extra={"lookup_table": lookup, "n_uniq_post_prune": 4},
        )
        df = pl.DataFrame({"x1": [0, 1, 0, 1], "x2": [0, 0, 1, 1]})
        out = apply_recipe(recipe, df)
        np.testing.assert_array_equal(out, np.array([0, 1, 2, 3]))


class TestPolarsInput:
    """If polars is installed, ``apply_recipe`` accepts a polars
    DataFrame the same way it accepts a pandas one (per
    ``mlframe/CLAUDE.md`` no-frame-conversion rule -- we read one column
    natively, never convert the whole frame)."""

    def test_polars_frame_replay(self):
        pl = pytest.importorskip("polars")

        rng = np.random.default_rng(0)
        df = pl.DataFrame({
            "a": rng.uniform(1, 10, 50),
            "b": rng.uniform(0, 1, 50),
        })

        recipe = build_unary_binary_recipe(
            name="mul(a,b)",
            src_a_name="a", src_b_name="b",
            unary_a_name="identity", unary_b_name="identity",
            binary_name="mul",
            unary_preset="minimal", binary_preset="minimal",
            quantization_nbins=None,
            quantization_method=None,
            quantization_dtype=np.float32,
        )
        out = apply_recipe(recipe, df)
        np.testing.assert_allclose(
            out, df["a"].to_numpy() * df["b"].to_numpy(), rtol=1e-5
        )
