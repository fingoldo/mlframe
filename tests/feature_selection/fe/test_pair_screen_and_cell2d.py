"""Residual pair screen (M) and the out-of-fold 2-D cell table (B): detection, null behaviour, leak safety, replay and the usability-pool wiring."""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters._oof_cell2d_fe import apply_oof_cell2d_recipe, cell2d_pool_candidates, fit_oof_cell2d
from mlframe.feature_selection.filters._pair_residual_screen import pair_interaction_screen


def _frame(n: int, seed: int, p: int = 20):
    """Standard-normal columns ``x0 .. x{p-1}`` and the generator."""
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)]), r


def _bump_case(n: int, seed: int = 0):
    """A Gaussian bump over (x4, x7) plus a linear term: no linear model can form the bump."""
    X, r = _frame(n, seed)
    y = 3 * np.exp(-(X["x4"].to_numpy() ** 2 + X["x7"].to_numpy() ** 2)) + X["x1"].to_numpy() + 0.3 * r.standard_normal(n)
    return X, y


@pytest.mark.parametrize(
    "name, target, expected",
    [
        ("bump", lambda X, r: 3 * np.exp(-(X["x4"].to_numpy() ** 2 + X["x7"].to_numpy() ** 2)) + X["x1"].to_numpy() + 0.3 * r.standard_normal(len(X)), ("x4", "x7")),
        ("product", lambda X, r: X["x5"].to_numpy() * X["x11"].to_numpy() + 2 * X["x1"].to_numpy() + 0.5 * r.standard_normal(len(X)), ("x11", "x5")),
        ("xor with flat marginals", lambda X, r: np.sign(X["x2"].to_numpy()) * np.sign(X["x9"].to_numpy()) + 0.3 * r.standard_normal(len(X)), ("x2", "x9")),
    ],
)
def test_screen_finds_exactly_the_interacting_pair(name, target, expected):
    """On every seed the only pair reported is the interacting one (a family-wise 5% level over the 190 pairs of 20 columns)."""
    for seed in range(4):
        X, r = _frame(10000, seed)
        res = pair_interaction_screen(X, target(X, r))
        assert [tuple(sorted(p[:2])) for p in res] == [expected], (name, seed, res)


@pytest.mark.parametrize("seed", range(4))
def test_screen_reports_nothing_on_noise_and_on_a_strongly_additive_target(seed):
    """Null behaviour: a noise target and a strong nonlinear additive one (a sine, a parabola, a linear term) give no pair."""
    X, r = _frame(10000, seed)
    assert pair_interaction_screen(X, r.standard_normal(10000)) == []
    additive = 2 * np.sin(X["x1"].to_numpy()) + X["x2"].to_numpy() ** 2 + X["x3"].to_numpy() + 0.5 * r.standard_normal(10000)
    assert pair_interaction_screen(X, additive) == []


def test_screen_does_not_treat_the_squashing_of_a_rank_target_as_interaction():
    """Regression: with a rank-scaled target the CDF squashing of an additive sum made (x1, x2) significant on every seed; the winsorised standardised target keeps additivity."""
    for seed in range(3):
        X, r = _frame(20000, seed)
        y = 2 * np.sin(X["x1"].to_numpy()) + X["x2"].to_numpy() ** 2 + X["x3"].to_numpy() + 0.5 * r.standard_normal(20000)
        assert pair_interaction_screen(X, y) == []


def test_cell_value_is_not_fitted_to_its_own_target():
    """On a noise target the cross-fitted cell value is uncorrelated with the target while the full table applied to its fitting rows shows the in-sample correlation."""
    oof_c, in_c = [], []
    for seed in range(8):
        r = np.random.default_rng(seed)
        a, b, y = r.random(4000), r.random(4000), r.random(4000)
        fit = fit_oof_cell2d(a, b, y, seed=seed)
        oof_c.append(np.corrcoef(fit["oof"], y)[0, 1])
        ia, ib = np.searchsorted(fit["ea"], a, side="right"), np.searchsorted(fit["eb"], b, side="right")
        in_c.append(np.corrcoef(fit["tab"][ia, ib], y)[0, 1])
    assert abs(float(np.mean(oof_c))) < 0.03 < float(np.mean(in_c))


def test_business_value_bump_table_collapses_the_ridge_error_and_replays():
    """The pool candidate for the bump pair cuts the held-out ridge MAE by more than 40%, replay agrees with the training column (correlation > 0.99) and the recipe pickles."""
    X, y = _bump_case(20000)
    cands = cell2d_pool_candidates(X, y, list(X.columns), np.float32, 10)
    assert cands and set(cands[0].recipe.src_names) == {"x4", "x7"}
    cut = 15000
    base = X.to_numpy()
    extra = np.asarray(apply_oof_cell2d_recipe(cands[0].recipe, X))
    assert np.corrcoef(extra, cands[0].values)[0, 1] > 0.99

    def mae(M):
        """Held-out ridge MAE on the first-``cut``-rows fit."""
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(M[:cut], y[:cut])
        return float(np.abs(y[cut:] - m.predict(M[cut:])).mean())

    assert mae(np.column_stack([base, extra])) < 0.6 * mae(base)
    rec = cands[0].recipe
    assert pickle.loads(pickle.dumps(rec)) == rec  # nosec B301 -- round-trip of a locally-created, trusted object
    Xn = X.copy()
    Xn.loc[:50, "x4"] = np.nan
    assert np.isfinite(apply_oof_cell2d_recipe(rec, Xn)).all()


def test_no_candidates_on_an_additive_target():
    """Without a significant interaction there is nothing to tabulate: the pool gets no table."""
    X, r = _frame(10000, 3)
    y = 2 * np.sin(X["x1"].to_numpy()) + X["x2"].to_numpy() ** 2 + 0.5 * r.standard_normal(10000)
    assert cell2d_pool_candidates(X, y, list(X.columns), np.float32, 10) == []


def test_performance_sanity_at_100k_rows():
    """Screening 20 columns and tabulating the significant pair at 100k rows takes about 0.5 s measured; the bound is generous."""
    X, y = _bump_case(100000)
    cell2d_pool_candidates(X.iloc[:3000], y[:3000], list(X.columns), np.float32, 10)
    t0 = time.perf_counter()
    cell2d_pool_candidates(X, y, list(X.columns), np.float32, 10)
    assert time.perf_counter() - t0 < 6.0


def test_usability_greedy_takes_the_table_and_the_mrmr_flag_is_wired():
    """On the bump target the linear usability greedy selects the table; the MRMR constructor carries the flag with its default on."""
    from mlframe.feature_selection.filters._usability_aware_selection import build_usability_candidate_pool, usability_greedy
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _bump_case(8000)
    names = list(X.columns)
    pool = build_usability_candidate_pool(X, y, names, feature_dtype=np.float32, quantization_nbins=10)
    extra = cell2d_pool_candidates(X, y, names, np.float32, 10)
    picked = [c.name for c in usability_greedy(list(pool) + extra, y, w=0.85, seed=0)]
    assert any(n.startswith("oofcell(") for n in picked), picked
    assert MRMR().fe_oof_cell2d_enable is True and MRMR(fe_oof_cell2d_enable=False).fe_oof_cell2d_enable is False
