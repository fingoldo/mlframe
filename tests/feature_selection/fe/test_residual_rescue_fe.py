"""Residual rescue: a term that is weak in rank space but large in variance is admitted on the residual of the selected design, noise is not."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._residual_rescue_fe import residual_rescue
from mlframe.feature_selection.filters.mrmr import MRMR


def _two_term(n=20000, seed=0, outlier_share=0.03):
    """y = a**2/b + 3*log(c)*sin(d) + 0.3*e: the ratio holds most of the variance through a few outlier rows, the (c, d) term dominates every rank statistic."""
    r = np.random.default_rng(seed)
    a, e, d = (r.uniform(-3, 3, n) for _ in range(3))
    b = r.uniform(0.25, 15, n)
    c = r.uniform(0.001, 15, n)
    out = r.random(n) < outlier_share
    a = np.where(out, a * 2.5, a)
    b = np.where(out & (r.random(n) < 0.3), 0.25, b)
    y = a**2 / b + 3 * np.log(c) * np.sin(d) + 0.3 * e
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), y


def test_weak_in_rank_space_term_is_admitted_first():
    """The ratio, 83% of the variance but a rank-MI of 0.07, is the first column admitted on top of the (c, d) design."""
    X, y = _two_term()
    design = {"d": X["d"].to_numpy(), "cd": np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())}
    pool = {"a": X["a"].to_numpy(), "b": X["b"].to_numpy(), "a2/b": (X["a"] ** 2 / X["b"]).to_numpy()}
    got = residual_rescue(design, pool, y)
    assert got and got[0][0] == "a2/b"


def test_pure_noise_pool_is_family_wise_controlled():
    """Thirty pure-noise columns are admitted in at most 3 of 20 calls (family-wise 5% per call)."""
    X, y = _two_term(n=6000)
    design = {"d": X["d"].to_numpy(), "cd": np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy()), "a2/b": (X["a"] ** 2 / X["b"]).to_numpy(), "e": X["e"].to_numpy()}
    hits = 0
    for seed in range(20):
        pool = {f"z{j}": np.random.default_rng(1000 * seed + j).standard_normal(len(y)) for j in range(30)}
        hits += bool(residual_rescue(design, pool, y))
    assert hits <= 3  # family-wise 5% per call: 20 calls, a binomial tail far below this bar


def test_copy_of_a_design_column_scores_nothing():
    """An affine copy and a monotone transform of a design column leave nothing in the residual."""
    X, y = _two_term()
    cd = np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())
    design = {"cd": cd, "a2/b": (X["a"] ** 2 / X["b"]).to_numpy(), "e": X["e"].to_numpy(), "d": X["d"].to_numpy()}
    assert residual_rescue(design, {"cd_copy": cd * 2.0 + 1.0, "cd_monotone": np.arctan(cd)}, y) == []


def test_practical_effect_floor_is_a_knob():
    """min_relative_gain and max_admit gate the admission."""
    X, y = _two_term()
    design = {"d": X["d"].to_numpy(), "cd": np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy())}
    pool = {"a2/b": (X["a"] ** 2 / X["b"]).to_numpy()}
    assert residual_rescue(design, pool, y, min_relative_gain=0.0)
    assert residual_rescue(design, pool, y, min_relative_gain=0.99) == []
    assert residual_rescue(design, pool, y, max_admit=0) == []


def _operands(names):
    """Single-letter source columns named by the selected features (wavelet / hinge suffixes after ``__`` ignored)."""
    out = set()
    for n in names:
        word = ""
        for ch in str(n).split("__")[0] + " ":
            if ch.isalnum() or ch == "_":
                word += ch
            else:
                if len(word) == 1:
                    out.add(word)
                word = ""
    return out


@pytest.mark.timeout(900)
def test_estimator_recovers_the_outlier_masked_ratio_only_with_the_stage():
    """two_pairs_strong under 3% outliers: ``a**2/b`` holds 83% of the variance in a few rows, its rank MI is 0.07 against 1.85 for the (c, d) term, so the greedy drops both operands."""
    from tests.feature_selection import _synthetic_distributions as sd

    domains = {"a": sd.DOMAIN_ANY, "b": sd.DOMAIN_DIVISOR, "c": sd.DOMAIN_POSITIVE, "d": sd.DOMAIN_ANY, "e": sd.DOMAIN_ANY}
    data = sd.sample_operands(seed=42, n=20000, domains=domains, profile="with_outliers")
    y = data["a"] ** 2 / data["b"] + 3.0 * np.log(data["c"]) * np.sin(data["d"]) + 0.3 * data["e"]
    X = pd.DataFrame({k: data[k] for k in domains})
    on = MRMR(verbose=0, random_seed=42).fit(X, pd.Series(y, name="y"))
    off = MRMR(verbose=0, random_seed=42, fe_residual_rescue_enable=False).fit(X, pd.Series(y, name="y"))
    assert "a" in _operands(on.get_feature_names_out())
    assert "a" not in _operands(off.get_feature_names_out())


def test_constructor_default_and_legacy_state():
    """Constructor default is on; a pre-feature pickle reloads with it off."""
    assert MRMR().fe_residual_rescue_enable is True
    assert MRMR(fe_residual_rescue_enable=False).fe_residual_rescue_enable is False
    from mlframe.feature_selection.filters.mrmr._mrmr_setstate_defaults import _SETSTATE_LEGACY_DEFAULTS as D

    assert D["fe_residual_rescue_enable"] is False
