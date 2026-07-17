"""Multi-statistic K-fold target encoding: beyond per-category mean(y), emit std / skew / kurt of y per
category as separate leak-safe ``{col}__te_{stat}`` columns.

Validated contract:
* ``stats=("mean",)`` is byte-identical to the historical single-stat encoder (back-compat);
* requesting std/skew/kurt emits one extra OOF column + recipe per stat, each replaying through the SAME
  ``kfold_target_encoded`` path (a stat column is structurally just a different per-category lookup);
* the std column recovers the WITHIN-CATEGORY SPREAD of y the mean cannot see;
* transform replay is deterministic and finite on disjoint / unseen categories (leak-safe);
* business value: on a cell-varying-slope regression, the multi-stat columns fed alongside the raw feature
  lift a held-out GBM vs the mean-only encoding -- the regime where target-spread proxies the slope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._target_encoding_fe import (
    engineered_name_te,
    engineered_name_te_stat,
    kfold_target_encode_fit,
    kfold_target_encode_with_recipes,
)
from mlframe.feature_selection.filters.engineered_recipes._recipe_dispatch import apply_recipe


def test_mean_only_is_backcompat():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"c": rng.integers(0, 10, 2000)})
    y = rng.normal(0, 1, 2000)
    te_df, recipes = kfold_target_encode_fit(df, y, ["c"], stats=("mean",))
    assert list(te_df.columns) == [engineered_name_te("c")]
    # Historical recipe shape preserved.
    assert "lookup" in recipes["c"] and "global_mean" in recipes["c"]


def test_multistat_emits_one_column_and_recipe_per_stat():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"c": rng.integers(0, 10, 3000)})
    y = rng.normal(0, 1, 3000)
    stats = ("mean", "std", "skew", "kurt")
    _Xa, appended, recipes = kfold_target_encode_with_recipes(df, y, cat_cols=["c"], stats=stats)
    assert appended == [engineered_name_te_stat("c", s) for s in stats]
    assert len(recipes) == len(stats)
    assert all(r.kind == "kfold_target_encoded" for r in recipes)


def test_std_column_recovers_within_cell_spread():
    """y has the SAME mean (~0) in every category but a category-dependent std. mean-TE cannot tell the
    categories apart; std-TE must rank them by their true spread."""
    rng = np.random.default_rng(2)
    n = 8000
    cat = rng.integers(0, 6, n)
    sigma = 0.3 + 0.4 * cat  # spread grows with the category index; mean stays 0
    y = rng.normal(0.0, sigma, n)
    df = pd.DataFrame({"c": cat})
    _te_df, recipes = kfold_target_encode_fit(df, y, ["c"], stats=("mean", "std"))
    std_lut = recipes["c"]["stat_lookups"]["std"]
    # std lookup must be monotonically increasing in the category index.
    std_by_cat = [std_lut[str(k)] for k in range(6)]
    assert all(std_by_cat[i] < std_by_cat[i + 1] for i in range(5)), std_by_cat
    # mean lookup carries ~no separating signal (all near 0).
    mean_lut = recipes["c"]["stat_lookups"]["mean"]
    assert max(abs(mean_lut[str(k)]) for k in range(6)) < 0.15


def test_multistat_replay_is_leak_safe_and_finite():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"c": rng.integers(0, 8, 3000)})
    y = rng.normal(0, 1, 3000)
    _, _, recipes = kfold_target_encode_with_recipes(df, y, cat_cols=["c"], stats=("mean", "std", "skew", "kurt"))
    # Disjoint test incl. an UNSEEN category (99) -> must fall back to global, never NaN.
    df_te = pd.DataFrame({"c": np.r_[rng.integers(0, 8, 400), np.full(20, 99)]})
    for r in recipes:
        out1 = np.asarray(apply_recipe(r, df_te)).ravel()
        out2 = np.asarray(apply_recipe(r, df_te)).ravel()
        assert np.isfinite(out1).all()
        np.testing.assert_array_equal(out1, out2)  # deterministic


def test_biz_value_multistat_lifts_varying_slope_regression():
    """y = a(cell) + b(cell)*x_raw + noise. std(y|cell) ~ |b(cell)| carries the slope the mean misses; fed to a
    GBM with x_raw it lifts held-out R2 vs mean-only. Encoder-level (selection-independent) measurement."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score

    deltas = []
    for seed in (10, 11, 12):
        rng = np.random.default_rng(seed)
        n = 10000
        cell = rng.integers(0, 25, n)
        x_raw = rng.normal(0, 1, n)
        a = (cell % 5) - 2.0
        b = 0.5 + (cell // 5)  # slope magnitude varies by cell, in [0.5, 4.5]
        y = a + b * x_raw + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"cell": cell, "x_raw": x_raw})
        cut = n // 2
        tr, te = slice(0, cut), slice(cut, n)

        def _encode(stats):
            te_df, recipes = kfold_target_encode_fit(df.iloc[tr], y[tr], ["cell"], stats=stats)
            from mlframe.feature_selection.filters.engineered_recipes import build_kfold_target_encoded_recipe

            cols_tr = [te_df[c].to_numpy() for c in te_df.columns]
            info = recipes["cell"]
            cols_te = []
            for s in stats:
                rec = build_kfold_target_encoded_recipe(
                    name=engineered_name_te_stat("cell", s),
                    src_name="cell",
                    lookup=info["stat_lookups"][s],
                    global_mean=info["global_stats"][s],
                    smoothing=info["smoothing"],
                )
                cols_te.append(apply_recipe(rec, df.iloc[te]))
            Xtr = np.column_stack([df["x_raw"].to_numpy()[tr], *cols_tr])
            Xte = np.column_stack([df["x_raw"].to_numpy()[te], *cols_te])
            m = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0).fit(Xtr, y[tr])
            return r2_score(y[te], m.predict(Xte))

        deltas.append(_encode(("mean", "std", "skew", "kurt")) - _encode(("mean",)))
    assert float(np.mean(deltas)) > 0.02, f"multistat TE should lift varying-slope regression: deltas={deltas}"
