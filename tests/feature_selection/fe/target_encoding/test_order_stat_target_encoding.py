"""Robust / order-statistic K-fold target encodings: per-category median, symmetric-trimmed mean, target quantiles
(q10/q90), IQR, min, max of y, emitted as leak-safe ``{col}__te_{stat}`` columns alongside the moment stats.

Validated contract:
* each order stat equals the numpy/scipy reference per category (exact on a hand-checkable frame);
* the fit-time OOF column is the OUT-OF-FOLD estimate, never the full-cell stat (leak-safety);
* transform replay on held-out rows reproduces the stored full-data per-category stat, unseen -> global;
* a category below the stat's stability floor falls back to the GLOBAL stat value (rare-cell shrinkage discipline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import trim_mean

from mlframe.feature_selection.filters._target_encoding_fe import (
    TE_SUPPORTED_STATS,
    apply_target_encoding,
    engineered_name_te_stat,
    kfold_target_encode_fit,
    kfold_target_encode_with_recipes,
)
from mlframe.feature_selection.filters._target_encoding_order_stats import ORDER_STAT_N_MIN, ORDER_STATS
from mlframe.feature_selection.filters.engineered_recipes._recipe_dispatch import apply_recipe

_ORDER = ("median", "trimmed_mean", "q10", "q90", "iqr", "min", "max")


def test_order_stats_are_supported():
    """Order stats are supported."""
    for s in _ORDER:
        assert s in TE_SUPPORTED_STATS
        assert s in ORDER_STATS


def test_per_category_order_stats_match_numpy_reference():
    """Two well-populated categories (n=60 each, above every floor); each stat must equal the numpy/scipy reference."""
    rng = np.random.default_rng(0)
    ya = rng.normal(10.0, 2.0, 60)
    yb = rng.exponential(3.0, 60)  # skewed -> median != mean, nonzero IQR
    y = np.concatenate([ya, yb])
    df = pd.DataFrame({"c": ["A"] * 60 + ["B"] * 60})
    _, rec = kfold_target_encode_fit(df, y, ["c"], stats=_ORDER, n_folds=5)
    sl = rec["c"]["stat_lookups"]
    for name, seg in (("A", ya), ("B", yb)):
        assert sl["median"][name] == pytest.approx(float(np.median(seg)))
        assert sl["q10"][name] == pytest.approx(float(np.quantile(seg, 0.10)))
        assert sl["q90"][name] == pytest.approx(float(np.quantile(seg, 0.90)))
        assert sl["iqr"][name] == pytest.approx(float(np.quantile(seg, 0.90) - np.quantile(seg, 0.10)))
        assert sl["min"][name] == pytest.approx(float(seg.min()))
        assert sl["max"][name] == pytest.approx(float(seg.max()))
        assert sl["trimmed_mean"][name] == pytest.approx(float(trim_mean(seg, 0.10)))


def test_median_robust_to_outlier_vs_mean():
    """A single huge outlier in a category drags the mean far but leaves the median near the clean center."""
    y = np.concatenate([np.full(40, 5.0), np.full(40, -5.0)])
    y[0] = 1e6  # outlier in category A
    df = pd.DataFrame({"c": ["A"] * 40 + ["B"] * 40})
    _, rec = kfold_target_encode_fit(df, y, ["c"], stats=("mean", "median"), n_folds=5)
    sl = rec["c"]["stat_lookups"]
    assert sl["median"]["A"] == pytest.approx(5.0)
    assert sl["mean"]["A"] > 100.0  # mean is wrecked by the outlier


def test_oof_column_is_out_of_fold_not_full_cell():
    """Leak-safety: the fit-time OOF median column must NOT equal the full-data per-category median for a category whose
    per-fold medians differ. Construct a category whose y depends on fold membership so OOF != full-cell."""
    rng = np.random.default_rng(3)
    n = 500
    cat = np.array(["A"] * n)
    y = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"c": cat})
    te_df, rec = kfold_target_encode_fit(df, y, ["c"], stats=("median",), n_folds=5, random_state=0)
    full_med = rec["c"]["stat_lookups"]["median"]["A"]
    oof = te_df[engineered_name_te_stat("c", "median")].to_numpy()
    # Single category -> every OOF value is that fold's held-out median; at least one fold must differ from the full-cell
    # median (otherwise the OOF estimate would leak the full-cell value).
    assert np.any(np.abs(oof - full_med) > 1e-9)
    assert np.all(np.isfinite(oof))


def test_transform_replay_reproduces_stored_stat():
    """Transform replay reproduces stored stat."""
    rng = np.random.default_rng(4)
    n = 1200
    cat = rng.integers(0, 6, n).astype(str)
    y = rng.exponential(2.0, n)
    df = pd.DataFrame({"c": cat})
    _Xa, appended, recipes = kfold_target_encode_with_recipes(df, y, cat_cols=["c"], stats=("median", "q90", "iqr"))
    assert appended == [engineered_name_te_stat("c", s) for s in ("median", "q90", "iqr")]
    # Replay each recipe on held-out rows: one row per known category + one unseen category.
    known = sorted(set(cat))
    X_test = pd.DataFrame({"c": [*known, "ZZZ_unseen"]})
    for rec in recipes:  # recipe-dispatch replay path must be finite on known + unseen categories
        assert np.all(np.isfinite(apply_recipe(rec, X_test)))
    # Stored full-data lookups reproduce exactly at transform time; unseen -> global.
    _, raw = kfold_target_encode_fit(df, y, ["c"], stats=("median", "q90", "iqr"))
    for stat in ("median", "q90", "iqr"):
        sl = raw["c"]["stat_lookups"][stat]
        gm = raw["c"]["global_stats"][stat]
        enc = apply_target_encoding(X_test, "c", {"lookup": sl, "global_mean": gm})
        for i, k in enumerate(known):
            assert enc[i] == pytest.approx(sl[k])
        assert enc[-1] == pytest.approx(gm)  # unseen -> global


def test_rare_cell_falls_back_to_global_below_floor():
    """A category with fewer rows than the q10 floor (20) must emit the GLOBAL q10, not its own noisy estimate."""
    rng = np.random.default_rng(5)
    big = rng.normal(0.0, 1.0, 400)  # category BIG, well above every floor
    rare = np.array([100.0, 101.0, 102.0])  # category RARE, n=3 < all floors, wildly off-distribution
    y = np.concatenate([big, rare])
    df = pd.DataFrame({"c": ["BIG"] * 400 + ["RARE"] * 3})
    _, rec = kfold_target_encode_fit(df, y, ["c"], stats=("q10", "median", "min"), n_folds=5)
    for stat in ("q10", "median", "min"):
        assert ORDER_STAT_N_MIN[stat] > 3
        sl = rec["c"]["stat_lookups"][stat]
        g = rec["c"]["global_stats"][stat]
        # RARE (n=3) below floor -> global fallback, NOT the rare cell's own ~100 values.
        assert sl["RARE"] == pytest.approx(g)
        assert sl["RARE"] < 50.0  # global is dominated by the 400 near-zero rows


def test_moment_and_order_stats_coexist():
    """Moment and order stats coexist."""
    rng = np.random.default_rng(6)
    n = 3000
    df = pd.DataFrame({"c": rng.integers(0, 8, n)})
    y = rng.normal(0, 1, n)
    stats = ("mean", "std", "median", "trimmed_mean", "q10", "q90", "iqr", "min", "max")
    te_df, rec = kfold_target_encode_fit(df, y, ["c"], stats=stats)
    assert list(te_df.columns) == [engineered_name_te_stat("c", s) for s in stats]
    assert np.all(np.isfinite(te_df.to_numpy()))
    for s in stats:
        assert s in rec["c"]["stat_lookups"]
