"""MRMR critique batch C: FE leak fixes.

- FE-F2: temporal entity key used raw .astype(str), so an int-at-fit / float-at-serve entity id ('1' vs '1.0')
  missed every history entry and routed all test rows to the global prior. Now canonicalised (int/float collapse).
- FE-F4: _compute_target_encoding naive path (n_oof_folds<=0) emitted the row's own y as a training feature; now it
  falls back to OOF unless allow_naive_leak=True.
"""
import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._temporal_agg_fe import (
    generate_expanding_agg_features,
    apply_temporal_expanding,
)


def test_temporal_entity_key_survives_int_to_float_dtype_drift():
    # fit with INT entity ids
    Xtr = pd.DataFrame({"entity": np.array([1, 1, 1, 2, 2], dtype=np.int64),
                        "tcol": [1, 2, 3, 1, 2], "x0": [10.0, 20.0, 30.0, 5.0, 7.0]})
    _enc, rec = generate_expanding_agg_features(Xtr, ["entity"], ["x0"], "tcol", stats=("count",))
    extra = next(iter(rec.values()))
    # serve with the SAME entity ids but as FLOAT dtype (a NaN elsewhere / parquet round-trip promotes the column)
    Xte = pd.DataFrame({"entity": np.array([1.0, 2.0], dtype=np.float64),
                        "tcol": [4, 3], "x0": [99.0, 99.0]})
    out = apply_temporal_expanding(Xte, extra)
    # entity 1 has 3 earlier train rows, entity 2 has 2 -> counts must reflect history, NOT collapse to the prior 0
    assert out[0] == 3.0, f"int->float entity drift lost history for entity 1: {out[0]} (expected 3)"
    assert out[1] == 2.0, f"int->float entity drift lost history for entity 2: {out[1]} (expected 2)"


def _te_inputs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 6, n)
    b = rng.integers(0, 6, n)
    factors = np.column_stack([a, b]).astype(np.int32)
    y = (rng.random(n) < 0.5).astype(np.int64)
    nbins = np.array([6, 6], dtype=np.int64)
    return factors, y, nbins


def test_naive_target_encoding_falls_back_to_oof_without_optin(caplog):
    from mlframe.feature_selection.filters._cat_target_encoding_and_weighted import _compute_target_encoding

    factors, y, nbins = _te_inputs()
    with caplog.at_level(logging.WARNING):
        te_safe, _ = _compute_target_encoding(
            factors, (0, 1), np.array([2], dtype=np.int64), y, nbins,
            n_oof_folds=0, smoothing=1.0, dtype=np.int32,
        )
    assert any("target-LEAKING" in r.message for r in caplog.records), "expected the naive-leak fallback warning"
    # opt-in keeps the naive path -> different values than the OOF fallback
    te_naive, _ = _compute_target_encoding(
        factors, (0, 1), np.array([2], dtype=np.int64), y, nbins,
        n_oof_folds=0, smoothing=1.0, dtype=np.int32, allow_naive_leak=True,
    )
    assert not np.allclose(te_safe, te_naive), "safe OOF fallback must differ from the naive leaky encoding"
