"""Regression sensor for the per-unique factorize-gather rewrite of
``_extra_fe_families._column_to_str``.

The Family A/B/C copy of ``_column_to_str`` (rare-category / conditional-residual
/ rankgauss replay) used a per-ROW ``pandas.Series.map(canonical_group_token)``,
running the Python-level token callback once per row -- 8s (int) / 20s (float+NaN)
at 10M rows. It now delegates to the canonical per-UNIQUE implementation in
``_target_encoding_fe`` (one token call per distinct value), which carries the
bool/0/1 collision gate that falls back to the exact per-row loop.

Two pins:

* ``test_..._calls_token_per_unique_not_per_row`` -- ``canonical_group_token`` is
  invoked at most once per DISTINCT value, never per row (FAILS on the pre-fix
  per-row map, which called it ~n times).
* ``test_..._bit_identical_to_per_row_reference`` -- the per-unique output is
  byte-for-byte identical to the per-row map across int / float+NaN / str /
  bool / mixed-bool-int / mixed-int-float columns (the bool-collision cases
  exercise the exact-loop fallback).
"""

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._internals as INT
from mlframe.feature_selection.filters._extra_fe_families import _column_to_str
from mlframe.feature_selection.filters._internals import canonical_group_token


def _per_row_reference(col) -> np.ndarray:
    s = pd.Series(col)
    if s.isna().any():
        return s.astype(object).map(
            lambda v: "__nan__" if (v is None or (isinstance(v, float) and v != v))
            else canonical_group_token(v)
        ).to_numpy()
    return s.astype(object).map(canonical_group_token).to_numpy()


def _cases():
    rng = np.random.default_rng(7)
    xf = rng.integers(0, 50, 1000).astype(float)
    xf[rng.random(1000) < 0.1] = np.nan
    return {
        "int": rng.integers(0, 50, 1000),
        "float+nan": xf,
        "str": np.array(["a", "b", "cc", None], dtype=object)[rng.integers(0, 4, 1000)],
        "bool": rng.integers(0, 2, 1000).astype(bool),
        "mixed_bool_int": np.array([True, False, 1, 0, 2, "x", None, 2.5], dtype=object)[rng.integers(0, 8, 1000)],
        "mixed_int_float": np.array([1, 1.0, 2, 2.5, None], dtype=object)[rng.integers(0, 5, 1000)],
    }


def test_extra_fe_column_to_str_calls_token_per_unique_not_per_row(monkeypatch):
    calls = {"n": 0}
    orig = INT.canonical_group_token

    def spy(v):
        calls["n"] += 1
        return orig(v)

    monkeypatch.setattr(INT, "canonical_group_token", spy)

    n = 20_000
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 100, n)  # 100 distinct values, no bool/0/1 fast-path block-out beyond value 0/1
    _column_to_str(pd.Series(arr))

    # Per-unique path calls the token at most once per distinct value (<=100),
    # never per row (the pre-fix map called it ~n=20000 times).
    n_unique = int(np.unique(arr).size)
    assert calls["n"] <= n_unique, f"expected <= {n_unique} token calls (per-unique); got {calls['n']} (per-row regression)"
    assert calls["n"] < n // 10, "token must not be called once per row"


def test_extra_fe_column_to_str_bit_identical_to_per_row_reference():
    for name, arr in _cases().items():
        got = _column_to_str(pd.Series(arr))
        ref = _per_row_reference(pd.Series(arr))
        assert np.array_equal(got.astype(str), ref.astype(str)), f"mismatch for case {name!r}"
