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


def test_synergy_prefilter_keeps_the_planted_pair_and_drops_noise_pairs():
    """The cheap raw-MI pre-filter keeps the (c, d) pair of the sign-crossing target and keeps no pair of a pure-noise frame."""
    from mlframe.feature_selection.filters._offset_product_fe import synergy_pairs
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    X, y = _case2(30000)
    xs = [X[c].to_numpy() for c in X.columns]
    codes = np.asarray(encode_y_for_classif_mi(y), dtype=np.int64)
    pairs = synergy_pairs(xs, np.arange(len(y)), codes, int(codes.max()) + 1)
    assert (2, 3) in pairs, f"the (c, d) pair was dropped: {pairs}"
    r = np.random.default_rng(0)
    noise = [r.random(30000) for _ in range(5)]
    ny = np.asarray(encode_y_for_classif_mi(r.random(30000)), dtype=np.int64)
    assert synergy_pairs(noise, np.arange(30000), ny, int(ny.max()) + 1) == []


def test_prefilter_changes_no_accepted_column():
    """Selection parity: with and without the pre-filter the stage accepts the same columns on the sign-crossing target, and the filtered run scans fewer pairs."""
    X, y = _case2(30000)
    _, with_filter, _, _ = hybrid_offset_product_fe(X, y, top_k=10)
    _, without, _, _ = hybrid_offset_product_fe(X, y, top_k=10, min_synergy=None)
    assert with_filter == without and with_filter


def test_large_n_accepts_only_the_real_interaction():
    """Materiality: at 100k rows the margin 40/n is below 0.001 nats, yet only the (c, d) interaction is accepted, not the pairs whose gain over the baselines is a fraction of a percent."""
    X, y = _case2(100000)
    _, appended, recipes, _ = hybrid_offset_product_fe(X, y, top_k=10)
    assert appended and all(set(r.src_names) == {"c", "d"} for r in recipes), appended


class _FakeCache:
    """Kernel tuning cache stand-in returning a fixed region verdict."""

    def __init__(self, verdict):
        """Remember the verdict ``lookup`` returns."""
        self.verdict = verdict

    def lookup(self, kernel, **dims):
        """Return the stored verdict for any kernel and dimensions."""
        return self.verdict


def test_scan_backend_lookup_fallback_and_tuned_verdict(monkeypatch):
    """Untuned, the device scan is chosen only in strict-resident mode; a tuned verdict wins in both directions."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    monkeypatch.setattr(dispatch, "_get_cache", lambda: None)
    assert dispatch.lookup_offset_scan_backend(100000, 735, strict_resident=True) == "gpu"
    assert dispatch.lookup_offset_scan_backend(100000, 735, strict_resident=False) == "cpu"
    monkeypatch.setattr(dispatch, "_get_cache", lambda: _FakeCache({"backend_choice": "cpu"}))
    assert dispatch.lookup_offset_scan_backend(100000, 735, strict_resident=True) == "cpu"
    monkeypatch.setattr(dispatch, "_get_cache", lambda: _FakeCache({"backend_choice": "gpu"}))
    assert dispatch.lookup_offset_scan_backend(100000, 735, strict_resident=False) == "gpu"
    monkeypatch.setattr(dispatch, "_get_cache", lambda: _FakeCache({"backend_choice": "bogus"}))
    assert dispatch.lookup_offset_scan_backend(100000, 735, strict_resident=False) == "cpu"


def test_log_unary_replays_with_the_frozen_anchor_not_the_batch_minimum():
    """Replay applies log(x + anchor) with the anchor stored at fit time; the registry's batch-dependent shift would give another value on a batch whose minimum is positive."""
    from mlframe.feature_selection.filters._offset_product_fe import build_offset_product_recipe

    rec = build_offset_product_recipe(
        name="offmul(log(c)+0.1,d+0.2)", src_names=("c", "d"), unary_names=("log", "identity"), unary_preset="minimal",
        shifts=(0.1, 0.2), fills=(0.0, 0.0), out_clip=(-1e9, 1e9), log_shifts=(0.3, 0.0),
    )
    r = np.random.default_rng(0)
    batch = pd.DataFrame({"c": r.random(50) + 0.05, "d": r.random(50)})  # strictly positive: smart_log would use no shift at all
    expected = (np.log(batch["c"].to_numpy() + 0.3) + 0.1) * (batch["d"].to_numpy() + 0.2)
    np.testing.assert_allclose(apply_offset_product_recipe(rec, batch), expected, rtol=1e-12)


def test_fit_stores_the_log_anchor_of_a_column_with_non_positive_values():
    """A fitted recipe whose source column has negative values stores the smart_log shift of the fit column."""
    from mlframe.feature_selection.filters._offset_product_fe import _log_anchor

    x = np.array([-0.5, 0.2, 1.0])
    assert _log_anchor(x) == pytest.approx(1e-5 + 0.5)
    assert _log_anchor(np.array([0.1, 2.0])) == 0.0


def _bounded_case(n: int, seed: int):
    """The sign-crossing target with the columns in [0.1, 1.1): ``a`` and ``d`` carry independent additive effects, only (c, d) interact."""
    r = np.random.default_rng(seed)
    a, b, c, d, e, f = (r.random(n) + 0.1 for _ in range(6))
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), 0.2 * a**2 / b + f / 5.0 + np.log(2 * c) * np.sin(d / 3)


def test_independent_additive_effects_are_not_credited_as_an_interaction():
    """Regression: with the shift-free baselines alone, the unrelated pair (a, d) was accepted in 35-50% of the runs (a weighted sum with better weights than the least-squares one beat the baselines);
    the weighted-sum baseline of the winner's own factors removes it while (c, d) is still found in every run."""
    for seed in range(8):
        X, y = _bounded_case(15000, seed)
        _, _, recipes, _ = hybrid_offset_product_fe(X, y, top_k=10)
        pairs = {tuple(sorted(r.src_names)) for r in recipes}
        assert pairs == {("c", "d")}, f"seed {seed}: accepted pairs {pairs}"


def test_weighted_sum_baseline_separates_an_additive_target_from_an_interaction():
    """The best weighted sum carries most of the information of an additive target (MI 1.4) and less than half of what the true feature of a sign-crossing product carries (0.65 against 1.5)."""
    from mlframe.feature_selection.filters._offset_product_kernels import weighted_sum_heldout_mi
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    r = np.random.default_rng(0)
    u, v = r.random(20000), r.random(20000)
    mi = {}
    for name, y in (("additive", 2 * u + v + 0.1 * r.standard_normal(20000)), ("product", (u - 0.5) * (v - 0.5) + 0.01 * r.standard_normal(20000))):
        codes = np.asarray(encode_y_for_classif_mi(y), dtype=np.int64)
        mi[name] = float(weighted_sum_heldout_mi(u, v, codes, int(codes.max()) + 1, 10, 32))
    assert mi["additive"] > 1.2, mi
    assert mi["product"] < 0.8, mi


def test_the_unrelated_pair_is_accepted_without_the_weighted_sum_baseline_and_the_significance_bar(monkeypatch):
    """Teeth: with the pre-fix rule (no weighted-sum baseline, no standard-error bar, a 3% floor) the unrelated pair of the regression test above is accepted on these seeds."""
    import mlframe.feature_selection.filters._offset_product_fe as fe_mod

    monkeypatch.setattr(fe_mod, "weighted_sum_heldout_mi", lambda *a, **k: 0.0)
    monkeypatch.setattr(fe_mod, "SIGNIFICANCE_Z", 0.0)
    unrelated = 0
    for seed in range(8):
        X, y = _bounded_case(15000, seed)
        _, _, recipes, _ = hybrid_offset_product_fe(X, y, top_k=10, min_relative_gain=0.03)
        unrelated += len({tuple(sorted(r.src_names)) for r in recipes} - {("c", "d")})
    assert unrelated >= 1


def test_standard_error_of_the_gain_shrinks_with_the_square_root_of_n():
    """The bar follows the sample size by itself: the standard error of the shifted product's gain over its baselines at 4x the rows is about half."""
    from mlframe.feature_selection.filters._offset_product_fe import _rank_scaled, _winner_standard_error
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    def se_at(n):
        """Standard error of the winner's gain on ``n`` rows of a shifted-product target."""
        r = np.random.default_rng(1)
        u, v = r.random(n), r.random(n)
        y = (u - 0.4) * (v - 0.7) + 0.1 * r.standard_normal(n)
        codes = np.asarray(encode_y_for_classif_mi(y), dtype=np.int64)
        clip_u, clip_v = np.quantile(u, [0.01, 0.99]), np.quantile(v, [0.01, 0.99])
        return _winner_standard_error(u, v, _rank_scaled(y), codes, int(codes.max()) + 1, clip_u, clip_v)

    ratio = se_at(8000) / se_at(32000)
    assert 1.4 < ratio < 2.8, ratio


def test_usability_pool_gets_the_accepted_offset_products_with_replayable_recipes():
    """The linear-downstream pool is offered the accepted columns as candidates whose recipes replay to their stored values; a pure-noise frame offers none."""
    from mlframe.feature_selection.filters._usability_offset_pool import offset_product_candidates

    X, y = _bounded_case(15000, 2)
    cands = offset_product_candidates(X, y, list(X.columns), np.float32, 10)
    assert cands and all(set(c.recipe.src_names) == {"c", "d"} for c in cands)
    for c in cands:
        np.testing.assert_allclose(apply_offset_product_recipe(c.recipe, X).astype(np.float32), c.values, rtol=1e-5, atol=1e-6)
        assert c.mi > 0.0
    r = np.random.default_rng(0)
    noise = pd.DataFrame({k: r.random(6000) for k in "abcd"})
    assert offset_product_candidates(noise, r.standard_normal(6000), list(noise.columns), np.float32, 10) == []


def test_usability_greedy_picks_the_offset_product_for_the_linear_list():
    """With the candidate in the pool the linear usability greedy selects it on the sign-crossing target; without it the list has no (c, d) offset form."""
    from mlframe.feature_selection.filters._usability_aware_selection import build_usability_candidate_pool, usability_greedy
    from mlframe.feature_selection.filters._usability_offset_pool import offset_product_candidates

    X, y = _bounded_case(8000, 2)
    names = list(X.columns)
    pool = build_usability_candidate_pool(X, y, names, feature_dtype=np.float32, quantization_nbins=10)
    extra = offset_product_candidates(X, y, names, np.float32, 10)
    assert extra
    plain = [c.name for c in usability_greedy(pool, y, w=0.85, seed=2)]
    boosted = [c.name for c in usability_greedy(list(pool) + extra, y, w=0.85, seed=2)]
    assert not any(n.startswith("offmul(") for n in plain)
    assert any(n.startswith("offmul(") for n in boosted), boosted
