"""Usability-aware feature selection (linear-downstream list) -- core-algorithm pin.

See `_usability_aware_selection.py` + `MRMR_USABILITY_AWARE_SELECTION_DESIGN.md`. On the
heavy-tailed F2 target a linear model needs the engineered interaction `mul(log(c),sin(d))`-
shaped feature to reach the irreducible `f/5` MAE floor (~0.05); MRMR's pure-MI selection
ranks a high-MI monotone warp over it. The usability-aware greedy (relevance blends MI with
the held-out |partial corr of the candidate with the RESIDUAL after the selected features|)
selects the genuine `(c,d)` interaction form and reaches the floor -- with REPLAYABLE recipes
so `transform()` reproduces the linear feature space on test data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tests.feature_selection.conftest import is_fast_mode


def test_usability_greedy_tiny_n_does_not_crash_on_empty_folds():
    """Regression (audit #1, 2026-06-13): the CV greedy used a random multinomial fold assignment
    that could leave a fold EMPTY at small n / large n_folds -> empty train fold crashed fit, empty
    test fold yielded NaN MAE. The balanced-partition fix must let tiny n run cleanly. Also covers
    n < n_folds and n < 2."""
    from mlframe.feature_selection.filters._usability_aware_selection import select_usability_aware_features

    rng = np.random.default_rng(0)
    for n in (3, 8, 15):
        df = pd.DataFrame({k: rng.random(n) for k in ("a", "b", "c", "d", "e")})
        y = df["a"].to_numpy() ** 2 / np.clip(df["b"].to_numpy(), 1e-3, None)
        # n_folds (8) deliberately >= n for the n=3/8 cases -> would crash/NaN pre-fix.
        sel = select_usability_aware_features(
            df, y, list(df.columns), w=0.7, K=3, seed=0, pool_kwargs=dict(max_pairs=4, max_per_pair=4), greedy_kwargs=dict(n_folds=8, shortlist=10)
        )
        assert isinstance(sel, list)  # returns cleanly (possibly empty), never raises / NaN-poisons
    # n < 2 short-circuits to []
    df1 = pd.DataFrame({k: rng.random(1) for k in ("a", "b", "c")})
    assert select_usability_aware_features(df1, np.array([1.0]), list(df1.columns)) == []


def _case2(n: int, seed: int = 0):
    """Helper that case2."""
    rng = np.random.default_rng(seed)
    a, b, c, d, e, f = (rng.random(n) for _ in range(6))
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2.0) * np.sin(d / 3.0)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, y.astype("float64")


@pytest.mark.slow
@pytest.mark.timeout(180)  # CV-MAE forward selection refits per (candidate, fold); see PERF TODO in _usability_aware_selection.py
def test_usability_aware_reaches_linear_floor_with_replayable_recipes():
    """w->1 usability selection on F2: a linear model on the selected (replayed) features reaches
    ~the f/5 floor (<= 0.07 MAE), selects a genuine (c,d) interaction form, and every pair recipe
    replays byte-consistently on a held-out slice. Pure-MI (w=0) does NOT select a (c,d) form."""
    from mlframe.feature_selection.filters._usability_aware_selection import (
        select_usability_aware_features,
    )
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_absolute_error

    n = 30_000 if is_fast_mode() else 50_000
    df, y = _case2(n=n, seed=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.9 * n) :]
    Xtr = df.iloc[tr].reset_index(drop=True)
    Xte = df.iloc[te].reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    ytr, yte = y[tr], y[te]

    pool_kwargs = dict(max_pairs=10, max_per_pair=8)  # p=5 -> all pairs; bounded for CI speed
    greedy_kwargs = dict(shortlist=14, n_folds=3)  # lighter CV than the defaults; still reaches the floor

    def _matrix(selected, Xframe):
        """Helper that matrix."""
        cols = []
        for s in selected:
            if s.recipe is None:
                cols.append(np.asarray(Xframe[s.name], dtype=float))
            else:
                rep = np.nan_to_num(np.asarray(apply_recipe(s.recipe, Xframe), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
                cols.append(rep)
        return np.column_stack(cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel_lin = select_usability_aware_features(Xtr, ytr, list(df.columns), w=0.8, K=6, seed=0, pool_kwargs=pool_kwargs, greedy_kwargs=greedy_kwargs)

    # (1) a genuine (c,d) interaction form is selected by the usability list.
    has_cd = lambda sel: any(("c" in s.src and "d" in s.src) for s in sel)
    assert has_cd(sel_lin), f"usability (w=0.8) selected no (c,d) form: {[s.name for s in sel_lin]}"

    # (2) every pair recipe in the usability list replays byte-consistently on a held-out slice.
    for s in sel_lin:
        if s.recipe is None:
            continue
        full = np.nan_to_num(np.asarray(apply_recipe(s.recipe, Xte), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        sl = Xte.iloc[1000:1500].reset_index(drop=True)
        part = np.nan_to_num(np.asarray(apply_recipe(s.recipe, sl), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        assert np.allclose(full[1000:1500], part, atol=1e-3), f"recipe {s.name!r} replay not slice-consistent"

    # (3) a linear model on the usability-selected (replayed) features reaches ~the f/5 floor.
    Mtr, Mte = _matrix(sel_lin, Xtr), _matrix(sel_lin, Xte)
    mdl = make_pipeline(StandardScaler(), LinearRegression()).fit(Mtr, ytr)
    mae_lin = mean_absolute_error(yte, mdl.predict(Mte))
    assert mae_lin <= 0.07, f"usability-aware linear MAE {mae_lin:.4f} did not reach the f/5 floor (~0.05); selected={[s.name for s in sel_lin]}"
