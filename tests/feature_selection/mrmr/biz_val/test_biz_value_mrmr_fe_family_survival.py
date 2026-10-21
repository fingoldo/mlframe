"""End-to-end survival of every feature-engineering (FE) family of MRMR: the family's engineered column must be created, survive selection, replay and lift.

A gate that silently rejects a whole family is invisible to unit tests that call the low-level generator directly. Each case below builds an ``MRMR``
whose every default-True ``fe_*_enable`` flag (read from the constructor signature) is off, switches on only the family under test and fits it on a
scenario whose target only that family can express. The four assertions per family are

* CREATED and SURVIVED: a recipe of the family's ``kind`` exists in ``_engineered_recipes_`` and one of them is in ``get_feature_names_out()``;
* REPLAY: ``transform`` of a held-out frame has the selected shape and only finite values;
* LIFT: a held-out ridge / logistic model (or a small boosted tree where noted) scores clearly higher on the selected output than on the raw
  columns, over seeds 0, 1, 2, with ``mean(lift) - 2 sd(lift) > 0``;
* NEGATIVE CONTROL: with the target permuted (independent of every column) no column of the family is selected.

Family table (flag -> recipe kind -> engineered column name pattern):

=====================  ==================================================  ===========================  ============================
family                 flags                                               recipe kind                  scenario / downstream model
=====================  ==================================================  ===========================  ============================
hinge                  fe_hinge_enable                                     hinge_basis                  y = relu(x0 - 0.5), ridge
univariate_basis       fe_hybrid_orth_enable + fe_univariate_basis_enable  orth_univariate              y = x0 squared, ridge
univariate_fourier     fe_hybrid_orth_enable + fe_univariate_fourier_...   orth_fourier                 y = sin(2 pi 6 u), ridge
wavelet                fe_wavelet_enable                                   orth_wavelet                 y = localised dyadic step of u, ridge
pair_cross             fe_hybrid_orth_enable + fe_hybrid_orth_pair_enable  orth_pair_cross/hermite_pair  y = x0 * x1, ridge
modular                fe_modular_enable                                   modular                      y = cos(2 pi x / 7), ridge
pairwise_modular       discrete master + fe_pairwise_modular_enable        pairwise_modular             y = ((a + b) mod 7 < 2), boosted tree
integer_lattice        discrete master + fe_integer_lattice_enable         pairwise_integer_lattice     y = gcd(a, b) >= 4, boosted tree
row_argmax             discrete master + fe_row_argmax_enable              row_argmax                   y = argmax of 5 columns, ridge
conditional_gate       discrete master + fe_conditional_gate_enable        conditional_gate             y = c > 0 ? a : b, ridge
conditional_dispersion fe_conditional_dispersion_enable                    conditional_dispersion       y = |a| / sigma(g), ridge
offset_product         fe_offset_product_enable                            offset_product               y = ln(2c) sin(d / 3), ridge
oof_warp               fe_oof_warp_enable                                  oof_warp1d                   y = sin(9.4 a) + 0.5 b, ridge
row_stat               fe_row_stat_enable                                  row_stat                     y = range of 4 columns, ridge
oof_cell2d             fe_oof_cell2d_enable + usability_aware_lists        oof_cell2d                   Gaussian bump over two columns, ridge
binned_numeric_agg     fe_binned_numeric_agg_enable                        binned_numeric_agg           y = sigma(cell of g), ridge
kfold_target_encoding  fe_kfold_te_enable                                  kfold_target_encoded         y = mean of a 40-level category, ridge
=====================  ==================================================  ===========================  ============================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.mrmr.biz_val._fe_family_survival_common import (
    checked,
    Family,
    check_created_and_survived,
    check_false_positive_rate,
    check_lift,
    check_noise_control,
    check_power,
    check_replay,
)

N = 6000


def _normal_frame(seed: int, p: int = 6, n: int = N):
    """Standard-normal columns ``x0 .. x{p-1}`` and the generator that made them."""
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]), r


def _uniform_frame(seed: int, p: int = 6, n: int = N):
    """Uniform(0, 1) columns ``a, b, c ...`` and the generator that made them."""
    r = np.random.default_rng(seed)
    return pd.DataFrame({chr(97 + i): r.random(n) for i in range(p)}), r


def build_hinge(seed: int):
    """``y = relu(x0 - 0.5)`` plus small noise: a kink no linear model can express."""
    X, r = _normal_frame(seed)
    return X, np.maximum(X["x0"].to_numpy() - 0.5, 0.0) * 2.0 + 0.1 * r.standard_normal(len(X))


def build_square(seed: int):
    """``y = x0^2`` plus noise: a symmetric effect no linear model can use."""
    X, r = _normal_frame(seed)
    return X, X["x0"].to_numpy() ** 2 + 0.3 * r.standard_normal(len(X))


def build_sine(seed: int):
    """``y = sin(2 pi 6 u)`` for ``u`` uniform on (0, 1): six periods, finer than a 10-bin MI resolves."""
    X, r = _uniform_frame(seed, p=5)
    return X, np.sin(2 * np.pi * 6.0 * X["a"].to_numpy()) + 0.2 * r.standard_normal(len(X))


def build_bump(seed: int):
    """``y = 1`` on the dyadic cell ``0.25 < u < 0.375`` of a uniform column, plus noise: a localised step."""
    X, r = _uniform_frame(seed, p=5)
    u = X["a"].to_numpy()
    return X, ((u > 0.25) & (u < 0.375)) * 1.0 + 0.1 * r.standard_normal(len(X))


def build_product(seed: int):
    """``y = x0 * x1`` plus noise: two columns with flat marginals."""
    X, r = _normal_frame(seed, p=4)
    return X, X["x0"].to_numpy() * X["x1"].to_numpy() + 0.2 * r.standard_normal(len(X))


def build_cyclic(seed: int):
    """Integer ``t`` in 0..199, ``y = cos(2 pi t / 7)`` plus noise: a period-7 cycle in a monotone counter."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({"t": r.integers(0, 200, N), "x1": r.standard_normal(N), "x2": r.standard_normal(N)})
    return X, np.cos(2 * np.pi * X["t"].to_numpy() / 7.0) + 0.2 * r.standard_normal(N)


def build_pairwise_modular(seed: int):
    """``y = ((a + b) mod 7 < 2)`` on integer columns: a residue class of a sum."""
    r = np.random.default_rng(seed)
    a, b = r.integers(0, 50, N), r.integers(0, 50, N)
    X = pd.DataFrame({"a": a, "b": b, "z0": r.integers(0, 50, N), "z1": r.standard_normal(N)})
    return X, ((a + b) % 7 < 2).astype(int)


def build_gcd(seed: int):
    """``y = gcd(a, b) >= 4`` on even integers."""
    r = np.random.default_rng(seed)
    a, b = r.integers(1, 40, N) * 2, r.integers(1, 40, N) * 2
    X = pd.DataFrame({"a": a.astype(float), "b": b.astype(float), "z0": r.standard_normal(N), "z1": r.standard_normal(N)})
    return X, (np.gcd(a, b) >= 4).astype(int)


def build_argmax(seed: int):
    """``y = index of the largest of five columns`` plus noise."""
    X, r = _normal_frame(seed, p=7)
    idx = np.argmax(X[[f"x{i}" for i in range(5)]].to_numpy(), axis=1).astype(float)
    return X, idx + 0.2 * r.standard_normal(len(X))


def build_gate(seed: int):
    """``y = a if c > 0 else b`` plus noise: a data dependent regime switch."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({"a": r.standard_normal(N), "b": r.standard_normal(N), "c": r.standard_normal(N), "z0": r.standard_normal(N)})
    return X, np.where(X["c"] > 0, X["a"], X["b"]) + 0.1 * r.standard_normal(N)


def build_dispersion(seed: int):
    """``aux ~ N(0, sigma(g))`` and ``y = |aux| / sigma(g)``: the anomaly score is a function of a conditional scale."""
    r = np.random.default_rng(seed)
    g = r.uniform(0, 1, N)
    sigma = 0.3 + 2.5 * g
    aux = r.normal(0, sigma)
    X = pd.DataFrame({"g": g, "aux": aux, "z0": r.standard_normal(N), "z1": r.standard_normal(N)})
    return X, np.abs(aux) / sigma + 0.05 * r.standard_normal(N)


def build_offset_product(seed: int):
    """``y = ln(2c) sin(d / 3)``: a product whose sign changes inside the range of ``c``."""
    X, r = _uniform_frame(seed, p=5)
    return X, np.log(2.0 * X["c"].to_numpy()) * np.sin(X["d"].to_numpy() / 3.0) + 0.02 * r.standard_normal(len(X))


def build_oof_warp(seed: int):
    """``y`` is a fixed random 12-step function of ``a`` plus ``0.5 b`` and noise: a lookup curve no closed-form unary operator reproduces."""
    X, r = _uniform_frame(seed, p=4)
    levels = np.random.default_rng(999).normal(0, 1, 12)
    step = levels[np.minimum((X["a"].to_numpy() * 12).astype(int), 11)]
    return X, step + 0.5 * X["b"].to_numpy() + 0.3 * r.standard_normal(len(X))


def build_row_stat(seed: int):
    """``y = range of x1, x3, x5, x6`` plus noise: flat marginals and invisible to every pair form."""
    X, r = _normal_frame(seed, p=10)
    A = X[["x1", "x3", "x5", "x6"]].to_numpy()
    return X, (A.max(1) - A.min(1)) + 0.3 * r.standard_normal(len(X))


def build_cell2d(seed: int):
    """``y`` is a fixed random 4 x 4 table indexed by the quartile cells of ``(x3, x4)`` plus a linear term in ``x1``."""
    X, r = _normal_frame(seed, p=6)
    table = np.random.default_rng(998).normal(0, 1.5, (4, 4))
    edges = [-0.67, 0.0, 0.67]
    i, j = np.digitize(X["x3"].to_numpy(), edges), np.digitize(X["x4"].to_numpy(), edges)
    return X, table[i, j] + X["x1"].to_numpy() + 0.3 * r.standard_normal(len(X))


def build_binagg(seed: int):
    """``aux ~ N(0, 0.5 + 2 |g - 0.5|)`` and ``y`` equal to that spread plus noise."""
    r = np.random.default_rng(seed)
    g = r.uniform(0, 1, N)
    sigma = 0.5 + 2.0 * np.abs(g - 0.5)
    X = pd.DataFrame({"g": g, "aux": r.normal(0, sigma), "noise": r.standard_normal(N)})
    return X, sigma + 0.1 * r.standard_normal(N)


def build_kfold_te(seed: int):
    """A 40-level string category whose level mean is the target, next to three noise columns."""
    r = np.random.default_rng(seed)
    levels = np.array([f"L{i:02d}" for i in range(40)])
    effect = r.normal(0, 1, 40)
    codes = r.integers(0, 40, N)
    X = pd.DataFrame({"cat": pd.Categorical(levels[codes]), "z0": r.standard_normal(N), "z1": r.standard_normal(N), "z2": r.standard_normal(N)})
    return X, effect[codes] + 0.3 * r.standard_normal(N)


_DISCRETE = {"fe_discrete_structural_operators_enable": True}
_ORTH = {"fe_hybrid_orth_enable": True}

FAMILIES = [
    Family("hinge", {"fe_hinge_enable": True}, ("hinge_basis",), build_hinge, panel=14),
    Family("univariate_basis", {**_ORTH, "fe_univariate_basis_enable": True}, ("orth_univariate",), build_square, panel=14),
    Family("univariate_fourier", {**_ORTH, "fe_univariate_fourier_enable": True}, ("orth_fourier", "orth_univariate", "orth_extra_basis"), build_sine, pattern=r"__(sin|cos)"),
    Family("wavelet", {**_ORTH, "fe_wavelet_enable": True}, ("orth_wavelet",), build_bump),
    Family("pair_cross", {**_ORTH, "fe_hybrid_orth_pair_enable": True}, ("orth_pair_cross", "hermite_pair"), build_product, panel=14),
    Family("modular", {"fe_modular_enable": True}, ("modular",), build_cyclic),
    Family("pairwise_modular", {**_DISCRETE, "fe_pairwise_modular_enable": True}, ("pairwise_modular",), build_pairwise_modular, "classification", "tree", panel=14),
    Family("integer_lattice", {**_DISCRETE, "fe_integer_lattice_enable": True}, ("pairwise_integer_lattice",), build_gcd, "classification", "tree"),
    Family("row_argmax", {**_DISCRETE, "fe_row_argmax_enable": True}, ("row_argmax",), build_argmax, panel=14),
    Family("conditional_gate", {**_DISCRETE, "fe_conditional_gate_enable": True}, ("conditional_gate",), build_gate),
    Family("conditional_dispersion", {"fe_conditional_dispersion_enable": True}, ("conditional_dispersion",), build_dispersion),
    Family("offset_product", {"fe_offset_product_enable": True}, ("offset_product",), build_offset_product),
    Family("oof_warp", {"fe_oof_warp_enable": True}, ("oof_warp1d",), build_oof_warp, extra={"usability_aware_lists": True}, panel=14),
    Family("row_stat", {"fe_row_stat_enable": True}, ("row_stat",), build_row_stat),
    Family("oof_cell2d", {"fe_oof_cell2d_enable": True}, ("oof_cell2d",), build_cell2d, extra={"usability_aware_lists": True}, panel=14),
    Family("binned_numeric_agg", {"fe_binned_numeric_agg_enable": True}, ("binned_numeric_agg",), build_binagg, panel=14),
    Family("kfold_target_encoding", {"fe_kfold_te_enable": True}, ("kfold_target_encoded",), build_kfold_te),
]
_IDS = [f.name for f in FAMILIES]


@pytest.mark.parametrize("family", checked(FAMILIES, "created"))
def test_family_column_is_created_and_survives_selection(family):
    """A recipe of the family's kind exists and at least one of its columns is in the selected output."""
    check_created_and_survived(family)


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_family_replay_on_a_held_out_frame_is_finite(family):
    """``transform`` of the held-out rows has the selected shape and only finite engineered values."""
    check_replay(family)


@pytest.mark.parametrize("family", checked(FAMILIES, "lift"))
def test_family_lifts_a_held_out_model(family):
    """The selected output beats the raw columns on held-out rows for every seed, by more than twice the seed-to-seed spread of the lift."""
    check_lift(family)


@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_family_adds_nothing_on_a_permuted_target(family):
    """With the target independent of every column no column of the family is selected."""
    check_noise_control(family)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("family", checked(FAMILIES, "power"))
def test_family_detection_power_over_the_seed_panel(family):
    """The family's column is created and selected on enough seeds that the Clopper-Pearson lower bound of the detection rate reaches the policy floor."""
    check_power(family)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("family", FAMILIES, ids=_IDS)
def test_family_false_positive_rate_over_the_seed_panel(family):
    """On permuted targets the Clopper-Pearson upper bound of the rate of runs selecting a column of the family stays below the policy ceiling."""
    check_false_positive_rate(family)
