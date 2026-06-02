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
    return np.array([int(lookup.get(c, default)) for c in cats], dtype=np.int64)


def _ref_freq(cats, lookup, default):
    return np.array([float(lookup.get(c, default)) for c in cats], dtype=np.float64)


def test_count_encoding_vectorized_bit_identical():
    lookup = {"a": 5, "b": 3, "__nan__": 1}
    X = pd.DataFrame({"c": ["a", "b", "zzz", "a", np.nan, "b", "a"]})
    recipe = {"lookup": lookup, "default": 0}
    got = apply_count_encoding(X, "c", recipe)
    cats = np.asarray(_column_to_str(X["c"]))
    np.testing.assert_array_equal(got, _ref_count(cats, lookup, 0))
    assert got.dtype == np.int64


def test_frequency_encoding_vectorized_bit_identical():
    lookup = {"a": 0.4, "b": 0.25, "__nan__": 0.05}
    X = pd.DataFrame({"c": ["b", "a", "unseen", "a", np.nan, "b"]})
    recipe = {"lookup": lookup, "default": 0.0}
    got = apply_frequency_encoding(X, "c", recipe)
    cats = np.asarray(_column_to_str(X["c"]))
    np.testing.assert_array_equal(got, _ref_freq(cats, lookup, 0.0))
    assert got.dtype == np.float64


def test_count_freq_empty_and_all_unseen():
    rec_c = {"lookup": {"a": 2}, "default": 7}
    rec_f = {"lookup": {"a": 0.9}, "default": -1.0}
    empty = pd.DataFrame({"c": pd.Series([], dtype=object)})
    assert apply_count_encoding(empty, "c", rec_c).shape == (0,)
    assert apply_frequency_encoding(empty, "c", rec_f).shape == (0,)
    allun = pd.DataFrame({"c": ["x", "y", "z"]})
    np.testing.assert_array_equal(apply_count_encoding(allun, "c", rec_c), np.array([7, 7, 7], dtype=np.int64))
    np.testing.assert_array_equal(apply_frequency_encoding(allun, "c", rec_f), np.array([-1.0, -1.0, -1.0]))


def _ref_cat_pair(cats_i, cats_j, mapping, encoding, te_lookup, global_mean, sentinel):
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
    mapping = {("a", "x"): 0, ("a", "y"): 1, ("b", "x"): 2}
    X = pd.DataFrame({
        "ci": ["a", "a", "b", "b", "a", "c"],   # (b,y) and (c,*) unseen
        "cj": ["x", "y", "x", "y", "x", "z"],
    })
    got = apply_cat_pair_cross(X, "ci", "cj", mapping, encoding="raw")
    ci = np.asarray(_column_to_str(X["ci"])); cj = np.asarray(_column_to_str(X["cj"]))
    exp = _ref_cat_pair(ci, cj, mapping, "raw", None, 0.0, len(mapping))
    np.testing.assert_array_equal(got, exp)


def test_cat_pair_cross_target_vectorized_bit_identical():
    mapping = {("a", "x"): 0, ("a", "y"): 1, ("b", "x"): 2}
    te_lookup = {0: 0.8, 1: 0.2}  # code 2 absent -> global_mean
    gm = 0.5
    X = pd.DataFrame({
        "ci": ["a", "a", "b", "b", "a"],          # (b,x)->code2 (absent from te), (b,y) unseen pair
        "cj": ["x", "y", "x", "y", "x"],
    })
    got = apply_cat_pair_cross(X, "ci", "cj", mapping, encoding="target", te_lookup=te_lookup, global_mean=gm)
    ci = np.asarray(_column_to_str(X["ci"])); cj = np.asarray(_column_to_str(X["cj"]))
    exp = _ref_cat_pair(ci, cj, mapping, "target", te_lookup, gm, len(mapping))
    np.testing.assert_array_equal(got, exp)


def test_cat_pair_cross_empty():
    mapping = {("a", "x"): 0}
    empty = pd.DataFrame({"ci": pd.Series([], dtype=object), "cj": pd.Series([], dtype=object)})
    assert apply_cat_pair_cross(empty, "ci", "cj", mapping, encoding="raw").shape == (0,)
    assert apply_cat_pair_cross(empty, "ci", "cj", mapping, encoding="target", global_mean=0.3).shape == (0,)
