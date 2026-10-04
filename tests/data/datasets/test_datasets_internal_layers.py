"""Direct behavioural tests for the latent, derived-column and label-corruption layers of the synthetic dataset generator."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.data.datasets._derived import derived_order, realize_derived
from mlframe.data.datasets._latent import delta_weights, realize_latent, realize_latents
from mlframe.data.datasets._noise import apply_corruption, binning_pushforward, feature_dependent_flip, uniform_flip
from mlframe.data.datasets._scm import CausalGraph
from mlframe.data.datasets.ground_truth import Edge
from mlframe.data.datasets.spec import GateSpec, LatentSpec, NoiseSpec

N = 5000


def _latent(**kwargs) -> LatentSpec:
    """Build a latent with three reflections unless overridden."""
    base = dict(name="z", reflections=("r1", "r2", "r3"))
    base.update(kwargs)
    return LatentSpec(**base)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two vectors."""
    return float(np.corrcoef(a, b)[0, 1])


def test_delta_weights_alternate_in_sign_and_decay_geometrically():
    """Weights start at 1, flip sign each step and shrink by 1.6x, so no equal-weight mean can reproduce them."""
    w = delta_weights(4)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(-1.0 / 1.6)
    assert w[2] == pytest.approx(1.0 / 1.6**2)
    assert w[3] == pytest.approx(-1.0 / 1.6**3)
    assert delta_weights(0) == ()


def test_realize_latent_without_private_part_gives_exact_standardised_copies():
    """With distinct_sd=0 and noise_sd=0 every reflection is the latent itself, standardised, and the group is rank-1 exact."""
    real = realize_latent(_latent(), N, root_seed=3, spec_name="ds")
    assert real.deltas == {}
    assert real.reflections, "the latent spec must produce at least one reflection"
    for name, column in real.reflections.items():
        assert column.shape == (N,)
        assert float(column.mean()) == pytest.approx(0.0, abs=1e-9)
        assert float(column.std()) == pytest.approx(1.0, abs=1e-9)
        assert _corr(column, real.values) == pytest.approx(1.0, abs=1e-9)
        assert real.scales[name] == pytest.approx(float(real.values.std()))
    group = real.redundancy_group()
    assert group.exact is True and group.rank == 1
    assert group.members == ("r1", "r2", "r3") and group.source == "z"


def test_realize_latent_private_part_makes_reflections_distinct_and_group_inexact():
    """distinct_sd=1 gives each reflection a private delta: pairwise correlation ~0.5 and rank 1 + number of deltas."""
    real = realize_latent(_latent(distinct_sd=1.0), N, root_seed=3, spec_name="ds")
    assert set(real.deltas) == {"r1", "r2", "r3"}
    assert _corr(real.reflections["r1"], real.reflections["r2"]) == pytest.approx(0.5, abs=0.05)
    assert _corr(real.reflections["r1"], real.deltas["r1"]) == pytest.approx(np.sqrt(0.5), abs=0.05)
    group = real.redundancy_group()
    assert group.exact is False and group.rank == 4


def test_realize_latent_negative_loading_anticorrelates_and_noise_lowers_correlation():
    """A negative loading flips the sign against the latent; noise_sd=1 drops the correlation with it to ~0.71."""
    real = realize_latent(_latent(reflections=("a", "b"), loadings=(1.0, -1.0), noise_sd=1.0), N, root_seed=5, spec_name="ds")
    assert _corr(real.reflections["a"], real.values) == pytest.approx(np.sqrt(0.5), abs=0.05)
    assert _corr(real.reflections["b"], real.values) == pytest.approx(-np.sqrt(0.5), abs=0.05)


def test_realize_latent_is_deterministic_per_seed_and_namespaced_by_spec_name():
    """Same (seed, spec name) reproduces bit-identically; another seed or spec name changes the draw."""
    first = realize_latent(_latent(distinct_sd=0.5), 200, root_seed=9, spec_name="ds")
    again = realize_latent(_latent(distinct_sd=0.5), 200, root_seed=9, spec_name="ds")
    other_seed = realize_latent(_latent(distinct_sd=0.5), 200, root_seed=10, spec_name="ds")
    other_name = realize_latent(_latent(distinct_sd=0.5), 200, root_seed=9, spec_name="other")
    np.testing.assert_array_equal(first.values, again.values)
    np.testing.assert_array_equal(first.reflections["r2"], again.reflections["r2"])
    assert not np.array_equal(first.values, other_seed.values)
    assert not np.array_equal(first.values, other_name.values)


def test_realize_latent_t_family_has_heavier_tails_than_normal():
    """family='t' draws Student-t(4) factors whose excess kurtosis is well above the normal's ~0."""
    normal = realize_latent(_latent(), 50000, root_seed=1, spec_name="ds").values
    heavy = realize_latent(_latent(family="t"), 50000, root_seed=1, spec_name="ds").values

    def kurt(v):
        """Excess kurtosis."""
        z = (v - v.mean()) / v.std()
        return float((z**4).mean() - 3.0)

    assert abs(kurt(normal)) < 0.3
    assert kurt(heavy) > 1.5


def test_realize_latents_collects_columns_factors_scales_and_only_multi_reflection_groups():
    """Columns/scales are merged, delta factors are exposed under '<latent>::delta::<col>', and a single-reflection latent forms no group."""
    latents = (_latent(distinct_sd=0.5), LatentSpec(name="w", reflections=("solo",)))
    columns, factors, scales, groups = realize_latents(latents, 300, root_seed=2, spec_name="ds")
    assert set(columns) == {"r1", "r2", "r3", "solo"}
    assert set(scales) == {"r1", "r2", "r3", "solo"}
    assert set(factors) == {"z", "w", "z::delta::r1", "z::delta::r2", "z::delta::r3"}
    assert len(groups) == 1 and groups[0].source == "z"
    single = realize_latent(latents[1], 300, root_seed=2, spec_name="ds")
    np.testing.assert_array_equal(columns["solo"], single.reflections["solo"])


def _graph() -> CausalGraph:
    """a, b -> c -> (parents of y);  y -> d;  target y has parent c."""
    edges = (
        Edge(source="a", target="c", kind="direct", weight=2.0),
        Edge(source="b", target="c", kind="direct", weight=1.0),
        Edge(source="c", target="y", kind="direct", weight=1.0),
        Edge(source="y", target="d", kind="direct", weight=1.0),
    )
    return CausalGraph(edges, target="y", observed=("a", "b", "c", "d"))


def test_derived_order_splits_columns_into_before_and_after_target():
    """c feeds the target so it is built before it; d is a child of the target so it is built after it; roots are not derived."""
    before, after = derived_order(_graph(), ["a", "b", "c", "d"], "y")
    assert before == ("c",)
    assert after == ("d",)


def test_derived_order_skips_columns_built_by_the_latent_layer():
    """A column listed as a latent reflection is never rebuilt, even though the graph gives it parents."""
    before, after = derived_order(_graph(), ["a", "b", "c", "d"], "y", reflections=("c",))
    assert before == ()
    assert after == ("d",)


def test_realize_derived_builds_column_from_weighted_parents_in_place():
    """c becomes a standardised 2a+b plus residual: ~0.966 correlated with 2a+b, parents untouched, scale recorded."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(N), rng.standard_normal(N)
    columns = {"a": a.copy(), "b": b.copy(), "c": np.zeros(N)}
    scales: dict = {}
    realize_derived(["c"], _graph(), columns, scales, "y", {}, root_seed=4, spec_name="ds")
    np.testing.assert_array_equal(columns["a"], a)
    assert _corr(columns["c"], 2 * a + b) == pytest.approx(np.sqrt(5.0 / 5.36), abs=0.01)
    assert float(columns["c"].std()) == pytest.approx(1.0, abs=1e-9)
    assert scales["c"] == pytest.approx(np.sqrt(5.36), rel=0.05)


def test_realize_derived_child_of_target_uses_realised_label_and_requires_it():
    """d reflects the centred realised label; asking for it before the target exists raises KeyError."""
    rng = np.random.default_rng(1)
    y = (rng.random(N) < 0.3).astype(float)
    columns = {"d": np.zeros(N)}
    realize_derived(["d"], _graph(), columns, {}, "y", {"y": y}, root_seed=4, spec_name="ds")
    assert _corr(columns["d"], y) == pytest.approx(1.0 / np.sqrt(1.0 + 0.36 / y.var()), abs=0.02)
    with pytest.raises(KeyError, match="not realised yet"):
        realize_derived(["d"], _graph(), {"d": np.zeros(N)}, {}, "y", {}, root_seed=4, spec_name="ds")


def test_realize_derived_is_deterministic():
    """The residual stream is addressed by (seed, spec, column), so repeating the call reproduces the column exactly."""
    a = np.random.default_rng(0).standard_normal(100)
    first = {"a": a, "b": a[::-1].copy(), "c": np.zeros(100)}
    second = {"a": a, "b": a[::-1].copy(), "c": np.zeros(100)}
    realize_derived(["c"], _graph(), first, {}, "y", {}, root_seed=1, spec_name="ds")
    realize_derived(["c"], _graph(), second, {}, "y", {}, root_seed=1, spec_name="ds")
    np.testing.assert_array_equal(first["c"], second["c"])


def test_flip_probability_formulas_are_exact():
    """uniform_flip and feature_dependent_flip implement p(1-f)+(1-p)f and reject rates outside [0, 1]."""
    p = np.array([0.0, 0.25, 1.0])
    np.testing.assert_allclose(uniform_flip(p, 0.2), [0.2, 0.25 * 0.8 + 0.75 * 0.2, 0.8])
    np.testing.assert_allclose(feature_dependent_flip(p, np.array([0.5, 0.0, 0.1])), [0.5, 0.25, 0.9])
    with pytest.raises(ValueError, match="flip rate"):
        uniform_flip(p, 1.2)
    with pytest.raises(ValueError, match="per-row"):
        feature_dependent_flip(p, np.array([0.1, -0.1, 0.1]))


def test_binning_pushforward_maps_each_probability_to_its_equal_width_bin_centre():
    """Four equal-width bins have centres 0.125, 0.375, 0.625, 0.875; p=1.0 lands in the last bin."""
    p = np.array([0.0, 0.05, 0.3, 0.55, 0.99, 1.0])
    np.testing.assert_allclose(binning_pushforward(p, 4), [0.125, 0.125, 0.375, 0.625, 0.875, 0.875])
    with pytest.raises(ValueError, match="at least two bins"):
        binning_pushforward(p, 1)


def test_apply_corruption_none_returns_input_unchanged_without_caveat():
    """kind='none' is the identity and has nothing to warn about."""
    p = np.array([0.1, 0.9])
    out, caveat = apply_corruption(p, NoiseSpec(), {}, np.random.default_rng(0))
    assert out is p and caveat is None


def test_apply_corruption_uniform_flip_updates_true_prob_exactly():
    """A 20% uniform flip sends p to 0.6p + 0.2 with no caveat."""
    p = np.array([0.0, 0.5, 1.0])
    out, caveat = apply_corruption(p, NoiseSpec(kind="uniform_flip", rate=0.2, true_prob_update="uniform_flip"), {}, np.random.default_rng(0))
    np.testing.assert_allclose(out, [0.2, 0.5, 0.8])
    assert caveat is None


def test_apply_corruption_feature_dependent_flip_only_touches_rows_inside_the_gate():
    """Rows with g >= 0 flip at rate 0.3, rows outside keep their probability, and the caveat names the gating column."""
    g = np.array([-1.0, 1.0, -2.0, 2.0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    noise = NoiseSpec(kind="feature_dependent_flip", rate=0.3, true_prob_update="feature_dependent_flip", gate=GateSpec(column="g", low=0.0))
    out, caveat = apply_corruption(p, noise, {"g": g}, np.random.default_rng(0))
    np.testing.assert_allclose(out, [0.9, 0.9 * 0.7 + 0.1 * 0.3, 0.1, 0.1 * 0.7 + 0.9 * 0.3])
    assert caveat is not None and "'g'" in caveat


def test_apply_corruption_binning_derives_bin_count_from_rate_and_defaults_to_ten():
    """rate=0.25 gives 4 bins, rate=0 falls back to 10 bins, and both carry a quantisation caveat."""
    p = np.array([0.05, 0.32, 0.63])
    noise4 = NoiseSpec(kind="binning", rate=0.25, true_prob_update="binning_pushforward")
    out4, caveat4 = apply_corruption(p, noise4, {}, np.random.default_rng(0))
    np.testing.assert_allclose(out4, [0.125, 0.375, 0.625])
    assert caveat4 is not None and "4 equal-width bins" in caveat4
    noise10 = NoiseSpec(kind="binning", rate=0.0, true_prob_update="binning_pushforward")
    out10, caveat10 = apply_corruption(p, noise10, {}, np.random.default_rng(0))
    np.testing.assert_allclose(out10, [0.05, 0.35, 0.65])
    assert caveat10 is not None and "10 equal-width bins" in caveat10
