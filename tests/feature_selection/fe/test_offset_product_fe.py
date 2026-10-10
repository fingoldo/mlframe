"""Offset-product FE ``(u + s) * (v + t)``: shift recovery, held-out acceptance, replay and the MRMR wiring."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._offset_product_fe import (
    OFFSET_UNARIES,
    apply_offset_product_recipe,
    hybrid_offset_product_fe,
)
from mlframe.feature_selection.filters._offset_product_kernels import ols2_shift

N_BINS = 10


def _binned_mi(feature: np.ndarray, y: np.ndarray) -> float:
    """Plug-in MI (nats) of ``feature`` and ``y``, both cut into ``N_BINS`` quantile bins."""
    edges_f = np.quantile(feature, np.linspace(0, 1, N_BINS + 1)[1:-1])
    edges_y = np.quantile(y, np.linspace(0, 1, N_BINS + 1)[1:-1])
    joint = np.zeros((N_BINS, N_BINS))
    np.add.at(joint, (np.searchsorted(edges_f, feature, side="right"), np.searchsorted(edges_y, y, side="right")), 1)
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def _case2(n: int, seed: int = 0):
    """The sign-crossing interaction: ``ln(2c)`` changes sign at c = 0.5, next to a ratio term and a free additive term."""
    r = np.random.default_rng(seed)
    a, b, c, d, e, f = (r.random(n) for _ in range(6))
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), y


def _rank01(y: np.ndarray) -> np.ndarray:
    """Average ranks of ``y`` scaled into ``(0, 1]``."""
    from scipy.stats import rankdata

    return rankdata(y) / len(y)


def test_ols2_shift_recovers_known_shifts():
    """Unit: on ``y = (u + 0.3) * (v - 0.7)`` the regression returns the shifts (0.3, -0.7)."""
    r = np.random.default_rng(1)
    u, v = r.random(20000), r.random(20000)
    y = (u + 0.3) * (v - 0.7)
    sh = np.empty(4)
    ols2_shift(u, v, y, 0, 1, u.min(), u.max(), v.min(), v.max(), sh)
    assert sh[0] == pytest.approx(0.3, abs=1e-6)
    assert sh[1] == pytest.approx(-0.7, abs=1e-6)


def test_ols2_shift_nan_on_constant_input():
    """Unit: a constant factor makes the normal equations singular; the shifts are NaN, not garbage."""
    u = np.ones(100)
    v = np.random.default_rng(0).random(100)
    sh = np.empty(4)
    ols2_shift(u, v, v, 0, 1, 1.0, 1.0, v.min(), v.max(), sh)
    assert np.isnan(sh[:2]).all()


@pytest.mark.parametrize("n", [20000, 60000])
def test_business_value_sign_crossing_pair_beats_best_preset_form(n):
    """Business value: the accepted (c, d) column carries more MI about y than the best fixed-zero preset form ``mul(log(c), sin(d))``."""
    X, y = _case2(n)
    _, appended, _recipes, enc = hybrid_offset_product_fe(X, y, top_k=10)
    cd = [nm for nm in appended if "c" in nm and "d" in nm and "a" not in nm.replace("sqrt", "").replace("abs", "").replace("sqr", "")]
    assert cd, f"no (c,d) offset product among {appended}"
    best = max(_binned_mi(enc[nm].to_numpy(), y) for nm in cd)
    preset = _binned_mi(np.log(2 * X["c"].to_numpy()) * 0 + np.log(X["c"].to_numpy()) * np.sin(X["d"].to_numpy()), y)
    assert best > preset + 0.03, f"offset product MI {best:.4f} vs preset {preset:.4f}"


@pytest.mark.parametrize("seed", range(5))
def test_pure_noise_accepts_nothing(seed):
    """Noise control: an unrelated target and independent columns produce no offset product."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({k: r.random(20000) for k in "abcde"})
    _, appended, _, _ = hybrid_offset_product_fe(X, r.random(20000))
    assert appended == []


def test_additive_target_accepts_nothing():
    """No-interaction control: ``y = x**2 + z`` has no shifted-product structure beyond the shift-free baselines."""
    r = np.random.default_rng(3)
    X = pd.DataFrame({k: r.random(30000) for k in "abcd"})
    y = X["a"].to_numpy() ** 2 + X["b"].to_numpy() + 0.3 * r.standard_normal(30000)
    _, appended, _, _ = hybrid_offset_product_fe(X, y)
    assert appended == []


def test_replay_matches_fit_columns_and_survives_pickle_and_nan():
    """Replay: the recipe reproduces the fit column exactly on the training frame, survives pickle, and fills non-finite unary outputs on new data."""
    X, y = _case2(20000)
    _, _appended, recipes, enc = hybrid_offset_product_fe(X, y, top_k=10)
    assert recipes
    for rc in recipes:
        np.testing.assert_array_equal(apply_offset_product_recipe(rc, X), enc[rc.name].to_numpy())
        rc2 = pickle.loads(pickle.dumps(rc))  # nosec B301 -- round-trip of a locally-created, trusted object
        assert rc2 == rc
    Xn = X.copy()
    Xn.iloc[:50, :] = np.nan
    for rc in recipes:
        assert np.isfinite(apply_offset_product_recipe(rc, Xn)).all()


def test_unary_registry_is_replayable():
    """Every scanned unary exists in the minimal preset, so replay under the stored preset cannot KeyError."""
    from mlframe.feature_selection.filters.feature_engineering import create_unary_transformations

    assert set(OFFSET_UNARIES) <= set(create_unary_transformations(preset="minimal"))


def test_mrmr_fit_transform_roundtrip_with_offset_product():
    """Wiring: MRMR fits with the family on, exposes the roster attribute and transform() replays the column from the raw frame."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _case2(20000)
    fs = MRMR(verbose=0, fe_max_steps=2)
    fs.fit(X, pd.Series(y, name="y"))
    assert isinstance(fs.offset_product_features_, list)
    Xt = np.asarray(fs.transform(X))
    assert Xt.shape[0] == len(X) and np.isfinite(Xt).all()
    clone = pickle.loads(pickle.dumps(fs))  # nosec B301 -- round-trip of a locally-created, trusted object
    np.testing.assert_array_equal(np.asarray(clone.transform(X)), Xt)


def test_opt_out_flag_disables_the_family():
    """Opt-out: fe_offset_product_enable=False leaves no offset-product column and an empty roster."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _case2(20000)
    fs = MRMR(verbose=0, fe_max_steps=2, fe_offset_product_enable=False)
    fs.fit(X, pd.Series(y, name="y"))
    assert fs.offset_product_features_ == []
    assert not any(str(nm).startswith("offmul(") for nm in fs.get_feature_names_out())
