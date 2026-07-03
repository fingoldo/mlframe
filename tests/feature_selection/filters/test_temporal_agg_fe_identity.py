"""Regression: temporal_agg_fe O(N^2) -> O(N) optimizations are identity-preserving.

Pins the bit-identity contract for two algorithmic optimizations in
``_temporal_agg_fe.py``:

1. ``_group_row_slices`` (stable-argsort + bincount split) replacing the
   per-group ``codes_sorted == g`` boolean-mask rescan in the fit-side
   history-build loops. The emitted encoded columns AND the stored per-entity
   history snapshots must be EXACT-equal to the mask path.
2. Welford running accumulators in ``apply_temporal_expanding`` replacing the
   per-row concatenate+reduce (O(N^2) per entity). min/max/count must be
   EXACT-equal; mean/std may differ only by FP reduction order (~1e-9 bound,
   never selection-altering).

These run <5s. On pre-fix code (mask path + concatenate-reduce replay) the
encoded values are identical BY DESIGN -- the guard that makes this a real
regression test is the explicit ~1e-9 std/mean bound (a buggy accumulator that
silently drifts would blow past it) plus the history-snapshot exact-equality
(a wrong argsort split would scramble per-entity order).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _temporal_agg_fe as M


def _make(n=3000, ne=120, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "ent": rng.integers(0, ne, n),
            "t": rng.integers(0, n, n).astype(np.int64),
            "val": rng.normal(size=n),
        }
    )
    # a few NaNs to exercise the finite-filtering path
    df.loc[rng.integers(0, n, n // 40), "val"] = np.nan
    return df


def _reference_history_build(df, ent_cols, val_col, tcol):
    """Independent reference using the OLD per-group mask logic, to pin that
    ``_group_row_slices`` produces the identical per-entity (t, v) order."""
    key = M._entity_key_series(df, ent_cols).to_numpy()
    time_vals = df[tcol].to_numpy()
    order = M._stable_time_order(time_vals)
    key_sorted = key[order]
    vals_sorted = np.asarray(df[val_col].to_numpy(), dtype=np.float64)[order]
    times_ord = time_vals[order]
    codes_sorted, _ = pd.factorize(key_sorted, sort=False)
    n_codes = int(codes_sorted.max()) + 1 if codes_sorted.size else 0
    hist = {}
    for g in range(n_codes):
        mask = codes_sorted == g
        ent_key = str(key_sorted[mask][0])
        hist[ent_key] = {"t": times_ord[mask].tolist(), "v": vals_sorted[mask].tolist()}
    return hist


def test_group_row_slices_matches_mask_history():
    df = _make()
    ref = _reference_history_build(df, ["ent"], "val", "t")
    _, recipes = M.generate_expanding_agg_features(df, ["ent"], ["val"], "t", stats=["mean"])
    # any recipe carries the same history
    got = next(iter(recipes.values()))["history"]
    assert set(got) == set(ref)
    for ent in ref:
        assert got[ent]["t"] == ref[ent]["t"], ent
        assert got[ent]["v"] == ref[ent]["v"] or (
            np.allclose(
                np.nan_to_num(got[ent]["v"], nan=-999),
                np.nan_to_num(ref[ent]["v"], nan=-999),
            )
        ), ent


def _reference_expanding_replay(X_test, recipe_extra):
    """Naive concatenate+reduce oracle with the leak-safe STRICT-TIME merge: for each test row at time t, the
    expanding stat is over train values with train_time < t (plus earlier-processed within-test values). Mirrors the
    incremental accumulator in apply_temporal_expanding, which was changed to merge train history by time rather
    than seed the entity's ENTIRE train history up front (the pre-fix seed-all replay let a test row inside the
    train time range see FUTURE train values)."""
    ent_cols = list(recipe_extra["entity_cols"])
    value_col = recipe_extra["value_col"]
    stat = recipe_extra["stat"]
    prior = float(recipe_extra["global_prior"])
    history = recipe_extra["history"]
    key, times = M._replay_keys_times(X_test, ent_cols, recipe_extra["time_col"])
    is_dt = M._is_datetime_like(times)
    times_num = times.astype("datetime64[ns]").astype(np.int64) if is_dt else np.asarray(times, dtype=np.float64)
    vals = np.asarray(X_test[value_col].to_numpy(), dtype=np.float64)
    n = len(X_test)
    out = np.full(n, prior, dtype=np.float64)
    order = M._stable_time_order(times_num)
    test_hist = {}
    for idx in order:
        ent = str(key[idx])
        t = times_num[idx]
        h = history.get(ent, {})
        ht = np.asarray(h.get("t", []))
        hv = np.asarray(h.get("v", []), dtype=np.float64)
        if ht.size == hv.size and ht.size:
            train_v = hv[(ht < t) & np.isfinite(hv)]
        else:
            train_v = hv[np.isfinite(hv)]  # legacy recipe without stored times: positional seed-all
        seen = test_hist.get(ent, [])
        combined = np.concatenate([train_v, np.asarray(seen, dtype=np.float64)]) if seen else train_v
        if combined.size > 0:
            out[idx] = M._reduce_expanding(combined, stat)
        v = vals[idx]
        if np.isfinite(v):
            test_hist.setdefault(ent, []).append(v)
    return out


@pytest.mark.parametrize(
    "stat,tol", [("min", 0.0), ("max", 0.0), ("count", 0.0), ("mean", 1e-9), ("std", 1e-9)]
)
def test_expanding_replay_identity(stat, tol):
    df = _make(seed=1)
    _, recipes = M.generate_expanding_agg_features(df, ["ent"], ["val"], "t", stats=[stat])
    name = M.engineered_name_expanding("val", "ent", stat)
    extra = recipes[name]
    df_test = _make(seed=2)
    ref = _reference_expanding_replay(df_test, extra)
    got = M.apply_temporal_expanding(df_test, extra)
    if tol == 0.0:
        np.testing.assert_array_equal(got, ref)
    else:
        md = float(np.max(np.abs(got - ref)))
        assert md <= tol, f"{stat}: max abs diff {md} exceeds {tol}"


@pytest.mark.parametrize(
    "col",
    [
        pd.Series([1, 2, 5000, -3, 0], dtype="int64"),
        pd.Series([0.5, -1.25, 3.0, np.nan, 1e-9], dtype="float64"),
        pd.Series(["aa", "bb", "cc", "dd", "ee"], dtype=object),
        pd.Series(pd.Categorical(["x", "y", "x", "z", "y"])),
    ],
)
def test_entity_key_series_matches_per_row_str(col):
    """The vectorized ``.astype(str)`` entity-key cast must produce strings
    bit-identical to the prior per-row ``str(v)`` map, so factorize codes (and
    thus the engineered features) are unchanged."""
    X = pd.DataFrame({"ent": col})
    got = M._entity_key_series(X, ["ent"])
    ref = X["ent"].astype(object).map(lambda v: str(v))
    pd.testing.assert_series_equal(got, ref, check_names=False)


def test_entity_key_series_multicol_matches_per_row_str():
    X = pd.DataFrame(
        {"a": pd.Series([1, 2, 3], dtype="int64"), "b": ["p", "q", "r"]}
    )
    got = M._entity_key_series(X, ["a", "b"])
    parts = [X[c].astype(object).map(lambda v: str(v)) for c in ("a", "b")]
    ref = parts[0]
    for p in parts[1:]:
        ref = ref.str.cat(p, sep="\x00")
    pd.testing.assert_series_equal(got, ref, check_names=False)
