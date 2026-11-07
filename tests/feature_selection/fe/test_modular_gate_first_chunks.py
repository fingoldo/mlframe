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


def test_streamed_family_test_gives_the_same_frame_and_scores(monkeypatch):
    """A family above the stream threshold is tested group by group with its codes regenerated; the augmented frame, the scores and the flags equal the single-buffer test."""
    from mlframe.feature_selection.filters import _periodic_fe as pf

    rng = np.random.default_rng(2)
    n = 6000
    t = rng.integers(0, 700, n).astype(float)
    X = pd.DataFrame({"t": t, "a": rng.standard_normal(n), "b": rng.uniform(0, 50, n), "c": rng.integers(0, 30, n).astype(float)})
    y = ((t % 7) < 2).astype(int) ^ (rng.random(n) < 0.1)
    args = dict(periods=(7.0, 12.5, 30.0), top_k=5, n_boot=6, sample_fraction=0.8, seed=0, nbins=10, min_uplift_lcb=1.0)
    cols = list(X.columns)
    want_X, want_s = pf._hybrid_modular_fe_gate_first(X, y, cols, 2, **args)
    monkeypatch.setattr(pf, "_STREAM_FAMILY_BYTES", 1)
    monkeypatch.setattr(pf, "_STREAM_GROUP_BYTES", 40 * n)
    calls = []
    real = pf._ModularChunks.codes
    monkeypatch.setattr(pf._ModularChunks, "codes", lambda self, k: (calls.append(k), real(self, k))[1])
    got_X, got_s = pf._hybrid_modular_fe_gate_first(X, y, cols, 2, **args)
    assert calls, "the streamed path regenerated no chunk"
    pd.testing.assert_frame_equal(got_X, want_X)
    pd.testing.assert_frame_equal(got_s.reset_index(drop=True), want_s.reset_index(drop=True))
    assert any(c.startswith("t__") for c in got_X.columns)
