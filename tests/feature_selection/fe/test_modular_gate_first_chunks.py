"""The gate-first modular family gives the same augmented frame and scores whatever the source-column chunk (the codes are filled into one buffer)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._periodic_fe import _hybrid_modular_fe_gate_first


def test_chunk_size_does_not_change_the_result():
    """A weekly-periodic signal among noise columns: 1-column chunks and one chunk agree."""
    rng = np.random.default_rng(2)
    n = 6000
    t = rng.integers(0, 700, n).astype(float)
    X = pd.DataFrame({"t": t, "a": rng.standard_normal(n), "b": rng.uniform(0, 50, n), "c": rng.integers(0, 30, n).astype(float)})
    y = ((t % 7) < 2).astype(int) ^ (rng.random(n) < 0.1)
    args = dict(periods=(7.0, 12.5, 30.0), top_k=5, n_boot=6, sample_fraction=0.8, seed=0, nbins=10, min_uplift_lcb=1.0)
    cols = list(X.columns)
    one_X, one_s = _hybrid_modular_fe_gate_first(X, y, cols, len(cols), **args)
    many_X, many_s = _hybrid_modular_fe_gate_first(X, y, cols, 1, **args)
    pd.testing.assert_frame_equal(many_X, one_X)
    pd.testing.assert_frame_equal(many_s.reset_index(drop=True), one_s.reset_index(drop=True))
    assert any(c.startswith("t__") for c in one_X.columns)


def test_family_flags_same_on_int8_codes(monkeypatch):
    """The family test gives the same pass flags on int8 codes as on int16, host and device paths."""
    from mlframe.feature_selection.filters._family_maxt_gate import family_codes_of_rows, family_maxt_pass_codes

    rng = np.random.default_rng(0)
    n = 8000
    vals = rng.standard_normal((120, n)).astype(np.float32)
    vals[10] = np.round(vals[10])
    y = ((vals[3] > 0.5) ^ (rng.random(n) < 0.2)).astype(int)
    cols, nx = family_codes_of_rows(vals, 10)
    for strict in ("0", "1"):
        monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", strict)
        want = family_maxt_pass_codes(cols, nx, y, seed=0)
        got = family_maxt_pass_codes(np.ascontiguousarray(cols.astype(np.int8)), nx, y, seed=0)
        assert np.array_equal(got, want) and want.any()
