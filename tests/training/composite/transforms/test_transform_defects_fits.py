"""Regression tests for fit-side transform defects: degenerate and edge regimes (small n, ties, constant inputs, missing group labels), unit-dependent
floors, skipped identity candidates, dropped sample weights and in-sample target encoding.

Each test reproduces the wrong output the pre-fix code produced on a concrete input.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.composite.transforms import get_transform
from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.mark.parametrize("n", [300, 500])
def test_monotonic_residual_small_n_does_not_flatten_the_spline(n: int) -> None:
    """Under-populated edge slabs took the GLOBAL median and the cumulative max dragged every lower knot up to it (var_explained 0.57 at n=300)."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 10, n)
    y = base + rng.normal(0, 0.3, n)
    p = get_transform("monotonic_residual").fit(y, base)
    knots_y = np.asarray(p["knots_y"])
    assert p["var_explained"] > 0.95
    assert np.mean(np.diff(knots_y) > 0) >= 0.8


def test_monotonic_residual_grouped_small_groups_keep_their_slope() -> None:
    """Per-group fits run on a few hundred rows, which is exactly where the edge-knot flattening struck."""
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(4), 150)
    base = rng.uniform(0, 10, 600)
    y = base + rng.normal(0, 0.3, 600)
    tr = get_transform("monotonic_residual_grouped")
    p = tr.fit(y, base, groups=groups)
    t = tr.forward(y, base, p, groups=groups)
    assert float(np.var(t)) < 0.05 * float(np.var(y))


def test_quantile_normal_round_trip_exact_on_tail_rows() -> None:
    """The ECDF clip used 1/(2*n_knots), inside the knot range once n > 1000, so the extreme train rows collapsed onto T = +/-3.29 (error 0.8 sigma)."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(100_000)
    tr = get_transform("quantile_normal_y")
    p = tr.fit(y, None)
    back = tr.inverse(tr.forward(y, None, p), None, p)
    assert float(np.max(np.abs(back - y))) < 1e-6


@pytest.mark.parametrize("levels", [2, 5])
def test_gaussian_copula_recovers_every_level_of_a_discrete_target(levels: int) -> None:
    """The copula's eps came from the number of UNIQUE values, so a binary target's top knot was clipped to u = 0.75 and every y=1 row inverted to 0.52."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(1000)
    y = rng.integers(0, levels, 1000).astype(np.float64)
    tr = get_transform("gaussian_copula_residual")
    p = tr.fit(y, base)
    back = tr.inverse(tr.forward(y, base, p), base, p)
    assert float(np.max(np.abs(back - y))) < 1e-6


def test_linear_residual_constant_base_does_not_extrapolate() -> None:
    """lstsq on a rank-1 design split the level between alpha and beta (alpha 19.2 on base 5), so base 10 inverted to ~196 instead of mean(y) ~100."""
    rng = np.random.default_rng(0)
    y = rng.normal(100.0, 1.0, 200)
    base = np.full(200, 5.0)
    tr = get_transform("linear_residual")
    p = tr.fit(y, base)
    assert p["alpha"] == 0.0
    np.testing.assert_allclose(tr.inverse(np.zeros(1), np.array([10.0]), p), [np.mean(y)])
    pw = tr.fit(y, base, sample_weight=np.ones(200))
    assert pw["alpha"] == 0.0


def test_linear_residual_grouped_constant_base_group_keeps_global_slope() -> None:
    """A base constant within one group identifies only that group's level; its slope must be the global one, not a minimum-norm artifact."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.array(["a", "b", "c", "d", "e"]), 60)
    base = rng.normal(0, 1, 300)
    base[groups == "e"] = 3.0
    y = 2.0 * base + rng.normal(0, 0.1, 300)
    p = get_transform("linear_residual_grouped").fit(y, base, groups=groups)
    assert p["per_group_alphas"]["e"] == pytest.approx(p["alpha_global"])
    assert p["per_group_betas"]["e"] == pytest.approx(float(np.mean(y[groups == "e"])) - p["alpha_global"] * 3.0)


@pytest.mark.parametrize("scale", [1e-3, 1e6])
def test_logratio_soft_cap_is_scale_invariant(scale: float) -> None:
    """The MAD floor was 1e-3 * std(y) in raw y units against a log-scale cap: at scale 1e6 the cap was ~1e3 log units and never bound."""
    rng = np.random.default_rng(0)
    base = np.exp(rng.normal(0, 0.3, 500))
    y = base * np.exp(rng.normal(0, 0.05, 500))
    tr = get_transform("logratio")
    p1 = tr.fit(y, base)
    ps = tr.fit(y * scale, base * scale)
    assert ps["mad_eff"] == pytest.approx(p1["mad_eff"], rel=1e-9)
    t_hat = np.array([5.0, -5.0])
    np.testing.assert_allclose(tr.inverse(t_hat, base[:2] * scale, ps), scale * tr.inverse(t_hat, base[:2], p1), rtol=1e-9)


@pytest.mark.parametrize("name", ["linear_residual_grouped", "quantile_residual_grouped", "target_encoding_residual", "monotonic_residual_grouped"])
def test_grouped_transforms_accept_missing_group_labels(name: str) -> None:
    """``np.unique`` on an object column mixing strings with None / NaN raised TypeError, so every grouped transform crashed on a nullable column."""
    rng = np.random.default_rng(0)
    labels = np.array(["a", None, "b", np.nan, "a"], dtype=object)
    groups = np.tile(labels, 80)
    base = rng.normal(0, 1, groups.size)
    y = base + rng.normal(0, 0.1, groups.size)
    tr = get_transform(name)
    p = tr.fit(y, base, groups=groups)
    t = tr.forward(y, base, p, groups=groups)
    assert np.all(np.isfinite(t))
    pred_groups = np.array(["a", None, "zz", np.nan], dtype=object)
    out = tr.inverse(np.zeros(4), base[:4], p, groups=pred_groups)
    assert np.all(np.isfinite(out))


def test_monotonic_residual_grouped_shrinks_toward_the_global_median() -> None:
    """Per-group MEDIANS were shrunk toward the global MEAN, so on a skewed target every group's level moved up by c * (mean - median_g)."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 200)
    base = rng.uniform(0, 1, groups.size)
    y = np.exp(rng.normal(0, 1.0, groups.size))
    p = get_transform("monotonic_residual_grouped").fit(y, base, groups=groups)
    assert p["global"]["y_train_median"] == pytest.approx(float(np.median(y)))
    if p["shrinkage_factor"] > 0:
        med_gap = [float(np.median(y[groups == g])) for g in range(8)]
        assert abs(float(np.mean(np.median(y) - np.asarray(med_gap)))) < 0.1 * float(np.std(y))


@pytest.mark.parametrize("value", [7.0, 1e-6, -3e9])
def test_rank_ecdf_constant_target_inverts_to_the_constant(value: float) -> None:
    """A constant column got a synthetic knot at value + 1.0 (raw units), so a 1e-6 T error inverted to value + 1."""
    tr = get_transform("rank_ecdf_residual")
    y = np.full(50, value)
    base = np.linspace(0, 1, 50)
    p = tr.fit(y, base)
    out = tr.inverse(tr.forward(y, base, p) + 1e-6, base, p)
    np.testing.assert_allclose(out, value, rtol=1e-9, atol=0)


def test_quantile_residual_bin_iqrs_scale_with_the_target() -> None:
    """The per-bin IQR floor was an absolute 1e-6, so a 1e-7-scale target had every bin replaced by the global IQR."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 10, 2000)
    y = (1.0 + base) * rng.normal(1.0, 0.3, 2000)
    tr = get_transform("quantile_residual")
    p1 = tr.fit(y, base)
    ps = tr.fit(y * 1e-8, base)
    np.testing.assert_allclose(np.asarray(ps["bin_iqrs"]), 1e-8 * np.asarray(p1["bin_iqrs"]), rtol=1e-9)


def test_signed_power_keeps_identity_on_a_symmetric_target() -> None:
    """The exponent grid never contained 1.0, so an already-symmetric target was always compressed (p <= 0.9)."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(5000)
    assert get_transform("signed_power_y").fit(y, None)["p"] == 1.0


def test_signed_power_still_compresses_a_skewed_target() -> None:
    """The identity preference must not stop a heavy right tail from being compressed."""
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(0, 1.0, 5000))
    assert get_transform("signed_power_y").fit(y, None)["p"] < 1.0


def test_chain_honours_sample_weight_like_its_bivariate_half() -> None:
    """Chain fits had no ``sample_weight`` parameter, so the signature gate dropped the weights silently and the chain fitted unweighted OLS."""
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 400)
    y = 3.0 * base + rng.normal(0, 0.1, 400)
    y[200:] = -5.0 * base[200:]
    w = np.r_[np.ones(200), np.zeros(200)]
    chain = get_transform("chain_linres_cbrt")
    p = chain.fit(y, base, sample_weight=w)
    ref = get_transform("linear_residual").fit(y[:200], base[:200])
    assert p["bivariate_params"]["alpha"] == pytest.approx(ref["alpha"], rel=1e-9)
    assert p["bivariate_params"]["beta"] == pytest.approx(ref["beta"], rel=1e-9, abs=1e-12)


def test_nadaraya_watson_improves_with_more_data() -> None:
    """Knots were 2000 single raw observations while h shrank with n, so 100x more data barely reduced the error (0.085 -> 0.076)."""
    rng = np.random.default_rng(0)
    tr = get_transform("nadaraya_watson_residual")
    grid = np.linspace(0.05, 0.95, 400)
    errs = []
    for n in (2000, 200_000):
        b = rng.uniform(0, 1, n)
        y = np.sin(6 * b) + rng.standard_normal(n)
        p = tr.fit(y, b)
        g = tr.inverse(np.zeros(grid.size), grid, p)
        errs.append(float(np.sqrt(np.mean((g - np.sin(6 * grid)) ** 2))))
    assert errs[1] < 0.5 * errs[0]


def test_interaction_bases_rejects_a_mis_shaped_train_mask() -> None:
    """A train mask of the wrong length was silently ignored, so the caller who meant to stop the test-scale leak still got it."""
    from mlframe.training.composite.transforms.interaction_bases import generate_interaction_bases

    cand = {"a": np.arange(1.0, 11.0), "b": np.arange(2.0, 12.0)}
    with pytest.raises(ValueError, match="train_mask"):
        generate_interaction_bases(cand, ops=("div",), top_k=2, train_mask=np.ones(5, dtype=bool))


def test_interaction_bases_eps_uses_train_rows_only() -> None:
    """With a correct mask the divisor eps is the median over the train rows only."""
    from mlframe.training.composite.transforms.interaction_bases import generate_interaction_bases

    b = np.r_[np.full(8, 2.0), np.full(2, 1e6)]
    cand = {"a": np.ones(10), "b": b}
    mask = np.r_[np.ones(8, dtype=bool), np.zeros(2, dtype=bool)]
    _, prov = generate_interaction_bases(cand, ops=("div",), top_k=2, eps_div_floor_factor=1e-3, train_mask=mask)
    assert prov["a__div__b"]["scale_eps_b"] == pytest.approx(2e-3)


def test_yeo_johnson_fit_failure_is_logged_as_yeo_johnson(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The Yeo-Johnson MLE failure was logged at DEBUG as a 'Box-Cox' failure."""
    import scipy.optimize

    def _boom(*args: object, **kwargs: object) -> None:
        """Stand-in for ``minimize_scalar`` that always fails."""
        raise RuntimeError("forced optimiser failure")

    monkeypatch.setattr(scipy.optimize, "minimize_scalar", _boom)
    reset_throttle_counts("yeo_johnson_lambda_mle_failed")
    with caplog.at_level(logging.WARNING):
        p = get_transform("yeo_johnson_y").fit(np.linspace(-3, 10, 100), None)
    assert p["lambda"] == 1.0
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Yeo-Johnson" in m for m in msgs)
    assert not any("Box-Cox" in m for m in msgs)


def test_box_cox_adapter_is_built_once() -> None:
    """``_registry_extended`` rebuilt all six unary adapters but registered only Box-Cox; the registered adapter must be the one ``registry`` re-exports."""
    from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY, registry, _registry_extended

    assert TRANSFORMS_REGISTRY["box_cox_y"].fit is registry._bc_fit_a is _registry_extended._bc_fit_a
    assert not hasattr(_registry_extended, "_cbrt_fit")


def test_target_encoding_train_t_is_out_of_fold() -> None:
    """Each train row's own y was folded into its category mean, so a singleton's train T was 20/21 of the deviation an unseen row would get."""
    rng = np.random.default_rng(0)
    groups = np.array([f"s{i}" for i in range(100)] + ["big"] * 400, dtype=object)
    y = rng.normal(0, 1, 500)
    tr = get_transform("target_encoding_residual")
    p = tr.fit(y, None, groups=groups)
    t = tr.forward(y, None, p, groups=groups)
    fold = np.arange(500) % 5
    for i in range(0, 100, 17):
        gm_out = float(np.mean(y[fold != fold[i]]))
        assert t[i] == pytest.approx(y[i] - gm_out, rel=1e-9, abs=1e-12)


def test_target_encoding_predict_side_uses_the_full_encoding() -> None:
    """Out-of-fold T applies only to the fit's own rows; any other batch (and every inverse) keeps the full-train encoding, so it round-trips."""
    rng = np.random.default_rng(1)
    groups = np.repeat(np.array(["a", "b", "c"], dtype=object), 100)
    y = rng.normal(0, 1, 300) + np.repeat([0.0, 3.0, -3.0], 100)
    tr = get_transform("target_encoding_residual")
    p = tr.fit(y, None, groups=groups)
    sub_y, sub_g = y[:50], groups[:50]
    np.testing.assert_allclose(tr.inverse(tr.forward(sub_y, None, p, groups=sub_g), None, p, groups=sub_g), sub_y, atol=1e-12)
    p_in = tr.fit(y, None, groups=groups, oof_folds=None)
    np.testing.assert_allclose(tr.inverse(tr.forward(y, None, p_in, groups=groups), None, p_in, groups=groups), y, atol=1e-12)
