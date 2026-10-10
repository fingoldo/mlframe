"""Out-of-fold warp operator: leak safety of the cross-fit, the acceptance on a held-out gain, replay and the MRMR wiring."""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters._oof_warp_fe import apply_oof_warp1d_recipe, hybrid_oof_warp_fe
from mlframe.feature_selection.filters._oof_warp_service import apply_warp1d, fit_oof_warp1d


def _sine_case(n: int, seed: int = 0):
    """``a`` matters through sin(9.4 a) (1.5 periods: a 10-bin MI cannot resolve it), ``b`` linearly, ``c`` .. ``e`` are noise."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({k: r.random(n) for k in "abcde"})
    y = np.sin(9.4 * X["a"].to_numpy()) + 0.5 * X["b"].to_numpy() + 0.3 * r.standard_normal(n)
    return X, y


def test_oof_value_is_not_fitted_to_its_own_target():
    """On a pure-noise target the cross-fitted warp is uncorrelated with the target, while the same table applied to its own fitting rows shows the usual in-sample correlation."""
    oof_corr, in_corr = [], []
    for seed in range(8):
        r = np.random.default_rng(seed)
        x, y = r.random(4000), r.random(4000)
        fit = fit_oof_warp1d(x, y, seed=seed)
        oof_corr.append(np.corrcoef(fit["oof"], y)[0, 1])
        in_corr.append(np.corrcoef(apply_warp1d(x, fit["cx"], fit["cy"], fit["fill"]), y)[0, 1])
    assert abs(float(np.mean(oof_corr))) < 0.03
    assert float(np.mean(in_corr)) > 0.05 > abs(float(np.mean(oof_corr)))


def test_apply_equals_numpy_interp_and_fills_non_finite_values():
    """The njit table lookup is np.interp (clamped at both ends); a non-finite value gets the fill."""
    cx, cy = np.array([0.1, 0.4, 0.9]), np.array([0.2, 0.7, 0.3])
    x = np.array([-5.0, 0.1, 0.25, 0.4, 0.65, 2.0, np.nan, np.inf])
    got = apply_warp1d(x, cx, cy, 0.55)
    finite = np.isfinite(x)
    np.testing.assert_allclose(got[finite], np.interp(x[finite], cx, cy), rtol=1e-12)
    assert (got[~finite] == 0.55).all()


def test_business_value_sine_column_is_accepted_and_ridge_error_collapses():
    """The oscillating column is accepted (and only it); adding its warp cuts the held-out ridge MAE by more than half, and a linear model on the raw columns cannot use `a` at all."""
    X, y = _sine_case(20000)
    _, appended, recipes, _enc = hybrid_oof_warp_fe(X, y)
    assert appended == ["oofwarp(a)"]
    cut = 15000
    Xr = X.to_numpy()
    base = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xr[:cut], y[:cut])
    mae_raw = np.abs(y[cut:] - base.predict(Xr[cut:])).mean()
    aug = np.column_stack([Xr, apply_oof_warp1d_recipe(recipes[0], X)])
    with_warp = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(aug[:cut], y[:cut])
    mae_warp = np.abs(y[cut:] - with_warp.predict(aug[cut:])).mean()
    assert mae_warp < 0.5 * mae_raw, (mae_raw, mae_warp)


@pytest.mark.parametrize("seed", range(5))
def test_noise_and_linear_targets_accept_nothing(seed):
    """Noise control: an unrelated target, and a purely linear one (a monotone relation: the warp bins like the raw column), produce no warp."""
    r = np.random.default_rng(seed)
    X = pd.DataFrame({k: r.random(20000) for k in "abcde"})
    assert hybrid_oof_warp_fe(X, r.standard_normal(20000))[1] == []
    assert hybrid_oof_warp_fe(X, 2 * X["a"].to_numpy() + 0.3 * r.standard_normal(20000))[1] == []


def test_nominal_like_columns_are_skipped():
    """A column with at most FEW_CLASSES_MAX distinct values is a label: its warp would be a target encoding, so no warp is built even when the effect is strong."""
    r = np.random.default_rng(1)
    X = pd.DataFrame({"g": r.integers(0, 8, 20000).astype(float), "u": r.random(20000)})
    y = np.sin(2.0 * X["g"].to_numpy()) + 0.2 * r.standard_normal(20000)
    assert hybrid_oof_warp_fe(X, y)[1] == []


def test_replay_matches_the_training_column_survives_pickle_and_handles_nan():
    """Replay reproduces the fit column (cross-fit aside: correlation > 0.99), survives pickle and gives a finite fill for NaN input."""
    X, y = _sine_case(20000)
    _, _appended, recipes, enc = hybrid_oof_warp_fe(X, y)
    rec = recipes[0]
    assert np.corrcoef(apply_oof_warp1d_recipe(rec, X), enc[rec.name].to_numpy())[0, 1] > 0.99
    rec2 = pickle.loads(pickle.dumps(rec))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert rec2 == rec
    Xn = X.copy()
    Xn.iloc[:100, 0] = np.nan
    out = apply_oof_warp1d_recipe(rec, Xn)
    assert np.isfinite(out).all() and np.allclose(out[:100], rec.extra["fill"])


def test_performance_sanity_at_100k_rows():
    """Five columns at 100k rows take well under two seconds once compiled (about 0.2 s measured)."""
    X, y = _sine_case(100000)
    hybrid_oof_warp_fe(X.iloc[:2000], y[:2000])
    t0 = time.perf_counter()
    hybrid_oof_warp_fe(X, y)
    assert time.perf_counter() - t0 < 2.0


def test_mrmr_fit_exposes_the_roster_and_transform_replays():
    """Wiring: an MRMR fit with the family on lists the warp in `oof_warp_features_`, transform replays it, and the opt-out flag leaves no warp."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _sine_case(15000)
    fs = MRMR(verbose=0, fe_max_steps=2).fit(X, pd.Series(y, name="y"))
    assert isinstance(fs.oof_warp_features_, list)
    Xt = np.asarray(fs.transform(X))
    assert Xt.shape[0] == len(X) and np.isfinite(Xt).all()
    off = MRMR(verbose=0, fe_max_steps=2, fe_oof_warp_enable=False).fit(X, pd.Series(y, name="y"))
    assert off.oof_warp_features_ == []
    assert not any(str(n).startswith("oofwarp(") for n in off.get_feature_names_out())


def test_usability_pool_offers_the_warp_and_the_linear_greedy_takes_it():
    """The linear-downstream pool gets replayable warp candidates; on the sine target the usability greedy selects the warp of `a`."""
    from mlframe.feature_selection.filters._usability_aware_selection import build_usability_candidate_pool, usability_greedy
    from mlframe.feature_selection.filters._usability_warp_pool import warp_pool_candidates

    X, y = _sine_case(8000)
    names = list(X.columns)
    extra = warp_pool_candidates(X, y, names, np.float32, 10)
    assert extra and all(c.recipe is not None for c in extra)
    for c in extra:
        np.testing.assert_allclose(apply_oof_warp1d_recipe(c.recipe, X).astype(np.float32), c.values, atol=0.15)
    pool = build_usability_candidate_pool(X, y, names, feature_dtype=np.float32, quantization_nbins=10)
    picked = [c.name for c in usability_greedy(list(pool) + extra, y, w=0.85, seed=0)]
    assert "oofwarp(a)" in picked, picked
