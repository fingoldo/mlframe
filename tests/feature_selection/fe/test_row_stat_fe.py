"""Row-statistic operator: kernel parity, the subset search, acceptance, replay and the MRMR wiring."""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters._row_stat_fe import apply_row_stat_recipe, hybrid_row_stat_fe
from mlframe.feature_selection.filters._row_stat_kernels import STAT_NAMES, eval_candidates, stat_column


def _frame(n: int, seed: int, p: int = 10):
    """Standard-normal columns ``x0 .. x{p-1}`` and the generator."""
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]), r


def _range_case(n: int, seed: int = 0):
    """``y`` is the range of four columns plus noise: flat marginals, invisible to every pair form."""
    X, r = _frame(n, seed)
    A = X[["x1", "x3", "x5", "x6"]].to_numpy()
    return X, (A.max(1) - A.min(1)) + 0.3 * r.standard_normal(n)


@pytest.mark.parametrize("stat", STAT_NAMES)
def test_kernel_statistic_matches_numpy(stat):
    """Each statistic of the njit column kernel equals the numpy reference on random data (soft max / min via log-mean-exp with sharpness +-4)."""
    r = np.random.default_rng(0)
    Z = r.standard_normal((5, 300))
    sub = np.array([4, 0, 2, 1], dtype=np.int64)
    got = stat_column(np.ascontiguousarray(Z), sub, len(sub), STAT_NAMES.index(stat))
    B = Z[sub]
    ref = {
        "min": B.min(0), "max": B.max(0), "med": np.median(B, 0), "rng": B.max(0) - B.min(0), "std": B.std(0),
    }.get(stat)
    if ref is None:
        beta = 4.0 if stat == "lse_pos" else -4.0
        t = beta * B
        mx = t.max(0)
        ref = (mx + np.log(np.exp(t - mx).mean(0))) / beta
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_candidate_scoring_prefers_the_true_subset():
    """eval_candidates gives the true four-column range a much higher MI than a subset with noise columns, on both the selection and the held-out rows."""
    from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

    X, y = _range_case(8000)
    Zt = np.ascontiguousarray(((X - X.mean()) / X.std()).to_numpy().T)
    codes = np.asarray(encode_y_for_classif_mi(y), dtype=np.int64)
    subsets = np.zeros((2, 8), dtype=np.int64)
    subsets[0, :4] = [1, 3, 5, 6]
    subsets[1, :4] = [0, 2, 4, 7]
    ev, od = np.empty(2), np.empty(2)
    eval_candidates(Zt, subsets, np.array([4, 4], dtype=np.int64), np.array([STAT_NAMES.index("rng")] * 2, dtype=np.int64), codes, int(codes.max()) + 1, 10, ev, od)
    assert ev[0] > 4 * ev[1] and od[0] > 4 * od[1]


def test_business_value_range_of_four_columns_is_found_and_helps_ridge_and_boosting():
    """The exact four columns are recovered; adding the statistic cuts the held-out ridge MAE by more than half and also lowers the gradient-boosting MAE."""
    X, y = _range_case(20000)
    _, appended, recipes, _enc = hybrid_row_stat_fe(X, y)
    assert appended and set(recipes[0].src_names) == {"x1", "x3", "x5", "x6"}
    cut = 15000
    base = X.to_numpy()
    aug = np.column_stack([base, apply_row_stat_recipe(recipes[0], X)])

    def mae(model, M):
        """Held-out MAE of ``model`` fitted on the first ``cut`` rows of ``M``."""
        return float(np.abs(y[cut:] - model.fit(M[:cut], y[:cut]).predict(M[cut:])).mean())

    def ridge():
        """A fresh ridge pipeline."""
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    assert mae(ridge(), aug) < 0.5 * mae(ridge(), base)
    def boost():
        """A fresh gradient-boosting regressor."""
        return HistGradientBoostingRegressor(max_iter=100, early_stopping=False, random_state=0)

    assert mae(boost(), aug) < mae(boost(), base)


@pytest.mark.parametrize("seed", range(4))
def test_noise_and_additive_and_pair_targets_accept_nothing(seed):
    """Noise control: a noise target, weighted additive and nonlinear additive targets, and a pair product produce no row statistic (two-column subsets belong to the pair forms)."""
    X, r = _frame(10000, seed)
    x = {c: X[c].to_numpy() for c in X.columns}
    targets = [
        r.standard_normal(10000),
        x["x1"] + 2 * x["x2"] - 1.5 * x["x3"] + 0.7 * r.standard_normal(10000),
        np.sin(2 * x["x1"]) + x["x2"] ** 2 + 0.5 * r.standard_normal(10000),
        x["x1"] * x["x2"] + 0.5 * r.standard_normal(10000),
    ]
    for y in targets:
        assert hybrid_row_stat_fe(X, y)[1] == []


def test_replay_equals_the_fit_column_survives_pickle_and_fills_nan():
    """Replay reproduces the stored column exactly, survives pickle, and a NaN source value is the training mean (0 after standardisation) rather than a NaN output."""
    X, y = _range_case(20000)
    _, _appended, recipes, enc = hybrid_row_stat_fe(X, y)
    rec = recipes[0]
    np.testing.assert_allclose(apply_row_stat_recipe(rec, X), enc[rec.name].to_numpy(), rtol=1e-12)
    assert pickle.loads(pickle.dumps(rec)) == rec  # nosec B301 -- round-trip of a locally-created, trusted object
    Xn = X.copy()
    Xn.loc[:50, "x1"] = np.nan
    assert np.isfinite(apply_row_stat_recipe(rec, Xn)).all()


def test_performance_sanity_at_100k_rows():
    """The search runs on a 20000-row sample, so 100k rows cost about as much as 20k (about 0.3 s measured): well under three seconds once compiled."""
    X, y = _range_case(100000)
    hybrid_row_stat_fe(X.iloc[:3000], y[:3000])
    t0 = time.perf_counter()
    hybrid_row_stat_fe(X, y)
    assert time.perf_counter() - t0 < 3.0


def test_mrmr_fit_exposes_the_roster_and_the_opt_out_works():
    """Wiring: an MRMR fit lists the statistic in `row_stat_features_`; transform replays it; the opt-out leaves none."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _range_case(15000)
    fs = MRMR(verbose=0, fe_max_steps=2).fit(X, pd.Series(y, name="y"))
    assert isinstance(fs.row_stat_features_, list)
    Xt = np.asarray(fs.transform(X))
    assert Xt.shape[0] == len(X) and np.isfinite(Xt).all()
    off = MRMR(verbose=0, fe_max_steps=2, fe_row_stat_enable=False).fit(X, pd.Series(y, name="y"))
    assert off.row_stat_features_ == [] and not any(str(n).startswith("rowstat_") for n in off.get_feature_names_out())


def test_usability_pool_offers_the_statistic_and_the_linear_greedy_takes_it():
    """The linear-downstream pool gets the replayable row statistic and the usability greedy selects it on the range target."""
    from mlframe.feature_selection.filters._usability_aware_selection import build_usability_candidate_pool, usability_greedy
    from mlframe.feature_selection.filters._usability_row_stat_pool import row_stat_pool_candidates

    X, y = _range_case(8000)
    names = list(X.columns)
    extra = row_stat_pool_candidates(X, y, names, np.float32, 10)
    assert extra and all(c.recipe is not None for c in extra)
    pool = build_usability_candidate_pool(X, y, names, feature_dtype=np.float32, quantization_nbins=10)
    picked = [c.name for c in usability_greedy(list(pool) + extra, y, w=0.85, seed=0)]
    assert any(n.startswith("rowstat_") for n in picked), picked
