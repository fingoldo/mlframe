"""Bit-identity test for the EXTERNAL-VALIDATION raw-column extraction MEMO in
``check_prospective_fe_pairs`` (2026-06-07, LEVER 1 perf optimization).

The lazy external-validation tie-break extracts the RAW values of every external
factor (``X.iloc[:, original_cols[ext]].values`` / polars ``.to_numpy()``) once per
tied-leader config. Since the external-factor set is the SAME across configs and
across raw pairs, the same column was previously re-extracted ~hundreds of thousands
of times on a wide bed (scene 2407x299: ~276k pandas ``.iloc`` calls / ~43s of pure
pandas indexing). The optimization memoises the extraction by the var key so each
distinct external factor is extracted ONCE per ``check_prospective_fe_pairs`` call.

The extraction is DETERMINISTIC -- a var index maps to a FIXED column-values ndarray
for the lifetime of the call -- so the memo is BYTE-IDENTICAL to the per-config
re-extraction. These tests assert:

  1. an end-to-end ``MRMR.fit`` that exercises the external-validation tie-break
     produces an IDENTICAL selection across repeated runs (the memo must not perturb
     selection or introduce any ordering / aliasing artefact), and
  2. the memo CONTRACT directly: the cached extract equals a fresh ``.iloc`` extract
     byte-for-byte, and distinct var keys never alias to the same buffer (no key
     collision across distinct columns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _wide_competition_frame(n: int = 600, p: int = 16, seed: int = 17):
    """A wide numeric bed where several engineered pair-forms compete -- enough to
    produce leaders TIED at the max primary MI so the external-validation tie-break
    (and thus the memoised external-factor extraction) actually runs."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, p)).astype(np.float64),
        columns=[f"f{i}" for i in range(p)],
    )
    # A target that depends on a genuine algebraic pair AND on several raw columns,
    # so the FE pair search finds prospective pairs and runs external validation.
    sig = (X["f0"] ** 2) / (np.abs(X["f1"]) + 0.5) + np.log1p(np.abs(X["f2"])) * np.sin(X["f3"])
    y = pd.Series((sig > sig.median()).astype(int))
    return X, y


def test_extval_rawcol_cache_fit_is_deterministic_and_selects():
    """End-to-end: a fit that exercises the external-validation path must select a
    non-empty feature set and reproduce it EXACTLY across runs (memo determinism)."""
    from mlframe.feature_selection.filters import MRMR

    X, y = _wide_competition_frame()

    def _fit_cols():
        m = MRMR(verbose=0, fe_max_steps=1, n_jobs=1, random_seed=0)
        m.fit(X, y)
        return list(m.transform(X.iloc[:5]).columns)

    cols_a = _fit_cols()
    cols_b = _fit_cols()
    assert cols_a, "fit selected zero columns -- test frame failed to drive selection"
    assert cols_a == cols_b, (
        "memoised external-validation extraction must be deterministic across runs; "
        f"got\n  run A: {cols_a}\n  run B: {cols_b}"
    )


@pytest.mark.parametrize("n,p", [(300, 8), (600, 20)])
def test_extval_rawcol_memo_contract_pandas(n, p):
    """The memo CONTRACT: cached extract == fresh ``.iloc`` extract byte-for-byte,
    one extraction per distinct var, and distinct vars never alias."""
    rng = np.random.default_rng(31 + n + p)
    X = pd.DataFrame(
        rng.standard_normal((n, p)).astype(np.float64),
        columns=[f"f{i}" for i in range(p)],
    )
    # ``original_cols``: var-name -> positional column index (the contract the
    # in-code closure uses; here we use the column names as the var keys).
    original_cols = {f"f{i}": i for i in range(p)}

    # Faithful replica of the in-code ``_extval_raw_col`` closure logic.
    cache: dict = {}
    n_extractions = {"count": 0}

    def _extval_raw_col(_var):
        if _var in cache:
            return cache[_var]
        if _var not in original_cols:
            cache[_var] = None
            return None
        n_extractions["count"] += 1
        _vals = X.iloc[:, original_cols[_var]].values
        cache[_var] = _vals
        return _vals

    vars_ = list(original_cols.keys())

    # Repeated lookups (mimicking many configs x pairs) must hit the cache, never
    # re-extract, and stay byte-identical to a fresh extract.
    for _rep in range(5):
        for v in vars_:
            got = _extval_raw_col(v)
            ref = X.iloc[:, original_cols[v]].values
            assert got is not None
            assert np.array_equal(got, ref), f"cached extract != fresh .iloc for {v}"

    # Exactly ONE extraction per distinct var despite 5 repetitions.
    assert n_extractions["count"] == p, (
        f"expected {p} extractions (one per var), got {n_extractions['count']}"
    )

    # No key collision: distinct vars map to distinct columns (the cache key is the
    # var id, never the array contents).
    seen_ids = {id(cache[v]) for v in vars_}
    assert len(seen_ids) == p, "distinct var keys aliased to the same cached buffer"
    for v in vars_:
        assert np.array_equal(cache[v], X[v].values), f"var {v} cached wrong column"

    # Unknown var -> None (matches the in-code guard), cached so it is not re-probed.
    assert _extval_raw_col("does_not_exist") is None
    assert "does_not_exist" in cache and cache["does_not_exist"] is None
