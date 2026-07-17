"""Bit-identity guard for the vectorized count/frequency/cat-pair-cross replay.

The transform-time replay functions used a per-row Python ``dict.get`` loop over
the full test column; they were vectorized to resolve the lookup once per UNIQUE
category/pair and broadcast via the inverse index (mirroring the already-
vectorized fit side + _grouped_agg_fe._broadcast_lookup). These tests assert the
vectorized output is BIT-IDENTICAL to an independent per-row reference across
seen / unseen / NaN categories and both cat-pair encodings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._count_freq_interaction_fe import (
    apply_count_encoding,
    apply_frequency_encoding,
)
from mlframe.feature_selection.filters._cat_pair_fe import apply_cat_pair_cross
from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str


def _ref_count(cats, lookup, default):
    """Ref count."""
    return np.array([int(lookup.get(c, default)) for c in cats], dtype=np.int64)


def _ref_freq(cats, lookup, default):
    """Ref freq."""
    return np.array([float(lookup.get(c, default)) for c in cats], dtype=np.float64)


def test_count_encoding_vectorized_bit_identical():
    """Count encoding vectorized bit identical."""
    lookup = {"a": 5, "b": 3, "__nan__": 1}
    X = pd.DataFrame({"c": ["a", "b", "zzz", "a", np.nan, "b", "a"]})
    recipe = {"lookup": lookup, "default": 0}
    got = apply_count_encoding(X, "c", recipe)
    cats = np.asarray(_column_to_str(X["c"]))
    np.testing.assert_array_equal(got, _ref_count(cats, lookup, 0))
    assert got.dtype == np.int64


def test_frequency_encoding_vectorized_bit_identical():
    """Frequency encoding vectorized bit identical."""
    lookup = {"a": 0.4, "b": 0.25, "__nan__": 0.05}
    X = pd.DataFrame({"c": ["b", "a", "unseen", "a", np.nan, "b"]})
    recipe = {"lookup": lookup, "default": 0.0}
    got = apply_frequency_encoding(X, "c", recipe)
    cats = np.asarray(_column_to_str(X["c"]))
    np.testing.assert_array_equal(got, _ref_freq(cats, lookup, 0.0))
    assert got.dtype == np.float64


def test_count_freq_empty_and_all_unseen():
    """Count freq empty and all unseen."""
    rec_c = {"lookup": {"a": 2}, "default": 7}
    rec_f = {"lookup": {"a": 0.9}, "default": -1.0}
    empty = pd.DataFrame({"c": pd.Series([], dtype=object)})
    assert apply_count_encoding(empty, "c", rec_c).shape == (0,)
    assert apply_frequency_encoding(empty, "c", rec_f).shape == (0,)
    allun = pd.DataFrame({"c": ["x", "y", "z"]})
    np.testing.assert_array_equal(apply_count_encoding(allun, "c", rec_c), np.array([7, 7, 7], dtype=np.int64))
    np.testing.assert_array_equal(apply_frequency_encoding(allun, "c", rec_f), np.array([-1.0, -1.0, -1.0]))


def _ref_cat_pair(cats_i, cats_j, mapping, encoding, te_lookup, global_mean, sentinel):
    """Ref cat pair."""
    out = np.empty(len(cats_i), dtype=np.float64)
    lk = te_lookup or {}
    for r in range(len(cats_i)):
        if encoding == "target":
            code = mapping.get((cats_i[r], cats_j[r]))
            out[r] = global_mean if code is None else float(lk.get(code, global_mean))
        else:
            out[r] = float(mapping.get((cats_i[r], cats_j[r]), sentinel))
    return out


def test_cat_pair_cross_raw_vectorized_bit_identical():
    """Cat pair cross raw vectorized bit identical."""
    mapping = {("a", "x"): 0, ("a", "y"): 1, ("b", "x"): 2}
    X = pd.DataFrame(
        {
            "ci": ["a", "a", "b", "b", "a", "c"],  # (b,y) and (c,*) unseen
            "cj": ["x", "y", "x", "y", "x", "z"],
        }
    )
    got = apply_cat_pair_cross(X, "ci", "cj", mapping, encoding="raw")
    ci = np.asarray(_column_to_str(X["ci"]))
    cj = np.asarray(_column_to_str(X["cj"]))
    exp = _ref_cat_pair(ci, cj, mapping, "raw", None, 0.0, len(mapping))
    np.testing.assert_array_equal(got, exp)


def test_cat_pair_cross_target_vectorized_bit_identical():
    """Cat pair cross target vectorized bit identical."""
    mapping = {("a", "x"): 0, ("a", "y"): 1, ("b", "x"): 2}
    te_lookup = {0: 0.8, 1: 0.2}  # code 2 absent -> global_mean
    gm = 0.5
    X = pd.DataFrame(
        {
            "ci": ["a", "a", "b", "b", "a"],  # (b,x)->code2 (absent from te), (b,y) unseen pair
            "cj": ["x", "y", "x", "y", "x"],
        }
    )
    got = apply_cat_pair_cross(X, "ci", "cj", mapping, encoding="target", te_lookup=te_lookup, global_mean=gm)
    ci = np.asarray(_column_to_str(X["ci"]))
    cj = np.asarray(_column_to_str(X["cj"]))
    exp = _ref_cat_pair(ci, cj, mapping, "target", te_lookup, gm, len(mapping))
    np.testing.assert_array_equal(got, exp)


def test_cat_pair_cross_empty():
    """Cat pair cross empty."""
    mapping = {("a", "x"): 0}
    empty = pd.DataFrame({"ci": pd.Series([], dtype=object), "cj": pd.Series([], dtype=object)})
    assert apply_cat_pair_cross(empty, "ci", "cj", mapping, encoding="raw").shape == (0,)
    assert apply_cat_pair_cross(empty, "ci", "cj", mapping, encoding="target", global_mean=0.3).shape == (0,)


def _ref_column_to_str(arr):
    """Pre-fix per-row reference: canonicalise every row independently."""
    from mlframe.feature_selection.filters._internals import canonical_group_token

    out = np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        if v is None or (isinstance(v, float) and v != v):
            out[i] = "__nan__"
        else:
            out[i] = canonical_group_token(v)
    return out


def test_column_to_str_object_per_unique_canonicalises_not_per_row(monkeypatch):
    """The object-column fast path canonicalises once per UNIQUE value, not per
    row. Spy on ``canonical_group_token``: pre-fix (per-row loop) it fires len(arr)
    times; post-fix it fires <= n_unique << n. Sensor fails on the per-row code."""
    import mlframe.feature_selection.filters._internals as _internals
    import mlframe.feature_selection.filters._target_encoding_fe as _te

    calls = {"n": 0}
    real = _internals.canonical_group_token

    def spy(v):
        """Helper that spy."""
        calls["n"] += 1
        return real(v)

    monkeypatch.setattr(_internals, "canonical_group_token", spy)

    n, n_unique = 5000, 25
    vals = np.array([f"c{k}" for k in range(n_unique)], dtype=object)
    rng = np.random.default_rng(3)
    col = pd.Series(vals[rng.integers(0, n_unique, n)])
    out = _te._column_to_str(col)
    n_calls = calls["n"]  # capture BEFORE the reference (which also calls the spy)

    assert n_calls <= n_unique, f"per-unique path expected <= {n_unique} canonical_group_token calls; got {n_calls} (per-row regression)"
    monkeypatch.setattr(_internals, "canonical_group_token", real)
    np.testing.assert_array_equal(out.astype(str), _ref_column_to_str(col.to_numpy()).astype(str))


def test_column_to_str_bit_identical_across_dtypes_and_nan():
    """Per-unique fast path is bit-identical to the per-row reference on
    string / mixed-None-NaN / int / float / bool-in-object / all-NaN columns.
    The bool-in-object case must take the gated-out per-row branch."""
    rng = np.random.default_rng(1)
    from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str

    pool_mixed = np.array(["a", "b", None, float("nan"), 1, 1.0, 2, 2.5, "1"], dtype=object)
    pool_bool = np.array([True, False, 1, 0, "x"], dtype=object)
    cases = [
        pd.Series(np.array([f"c{k}" for k in rng.integers(0, 50, 800)], dtype=object)),
        pd.Series(pool_mixed[rng.integers(0, len(pool_mixed), 600)]),
        pd.Series(rng.integers(0, 40, 700)),
        pd.Series(rng.integers(0, 20, 700).astype(float)),
        pd.Series(pool_bool[rng.integers(0, len(pool_bool), 500)]),  # gated-out
        pd.Series(np.array([None] * 300, dtype=object)),
    ]
    for i, col in enumerate(cases):
        got = _column_to_str(col)
        arr = col.to_numpy()
        if arr.dtype.kind in ("i", "u", "b"):
            from mlframe.feature_selection.filters._internals import canonical_group_token

            ref = np.array([canonical_group_token(v) for v in arr], dtype=object)
        else:
            ref = _ref_column_to_str(arr)
        np.testing.assert_array_equal(got.astype(str), ref.astype(str), err_msg=f"case {i}")
