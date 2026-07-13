"""RESIDENT UPLOAD (wave 10, 2026-07-13): ``_gpu_build_and_score_univariate``'s ``used_x`` fallback (taken
whenever ``_Mr`` is None -- i.e. ``basis != "auto"``, ``y is None``, GPU routing is disabled, or the
routing target is degenerate) previously uploaded the WHOLE (n, n_used) candidate matrix as ONE raw
``cp.asarray`` blob every call, never deduping across repeated calls on the SAME raw columns. It now
routes through ``assemble_resident_matrix`` keyed by column NAME (mirroring ``raw_mat``/``_Mr``'s own
per-column resident construction), so each distinct raw column uploads ONCE per fit regardless of which
caller (univariate-decide / pair-cross / triplet / ...) asks for it.

Skips when cupy is unavailable (CI without a GPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_gpu_resident import _gpu_build_and_score_univariate
from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _make_frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2.0, 2.0, n)
    b = rng.normal(0.0, 1.0, n)
    c = rng.uniform(0.1, 5.0, n)
    y = (a**2) + 0.4 * b - 0.2 * np.log(c) + rng.normal(0, 0.05, n)
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    return X, y


def test_used_m_fallback_dedups_raw_column_upload_across_calls(monkeypatch):
    """Two calls to _gpu_build_and_score_univariate with basis='legendre' (forces _Mr is None, taking
    the used_x fallback) on the SAME raw columns -- mirroring the same builder re-run by
    univariate-decide then pair-cross over the same X/y -- must upload each raw column's (n,) float64
    content only ONCE across both calls."""
    monkeypatch.setenv("MLFRAME_FE_GPU_ROUTING", "0")  # force _Mr None regardless of basis
    X, y = _make_frame()
    # Match the production pipeline's precision path EXACTLY: the host candidate column is cast to
    # _crit_np_dtype() (float32 under the default MLFRAME_CRIT_DTYPE_RELAXED) BEFORE assemble_resident_
    # matrix upcasts it to float64 for the device eval -- so the array actually uploaded is this float32
    # round-trip, not a direct float64 cast of the raw pandas column.
    from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
    a64 = np.ascontiguousarray(X["a"].to_numpy(), dtype=_crit_np_dtype()).astype(np.float64)

    upload_calls = {"n": 0}
    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        if isinstance(arr, np.ndarray) and arr.shape == a64.shape and arr.dtype == np.float64 and np.array_equal(arr, a64):
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    monkeypatch.setattr(cp, "asarray", _counting_asarray)

    eng1, _names1, _scores1 = _gpu_build_and_score_univariate(X, ["a", "b", "c"], (2, 3), "legendre", y, nbins=10)
    eng2, _names2, _scores2 = _gpu_build_and_score_univariate(X, ["a", "b", "c"], (2, 3), "legendre", y, nbins=10)

    assert eng1 is not None and eng2 is not None
    assert upload_calls["n"] == 1, f"column 'a' upload called {upload_calls['n']} times across 2 calls (expected 1)"
    np.testing.assert_allclose(cp.asnumpy(eng1), cp.asnumpy(eng2), atol=1e-9, rtol=1e-9)


def test_used_m_fallback_bit_identical_vs_resident_cache_disabled(monkeypatch):
    """MLFRAME_FE_RESIDENT_OPERANDS=0 reproduces the exact pre-fix raw-upload path (assemble_resident_matrix's
    per-column resident_operand always uploads fresh, never touching the cache). Output must match the
    cached path within the FE MI selection-equivalence tolerance (~1e-9)."""
    monkeypatch.setenv("MLFRAME_FE_GPU_ROUTING", "0")
    X, y = _make_frame(seed=5)

    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "0")
    eng_raw, names_raw, scores_raw = _gpu_build_and_score_univariate(X, ["a", "b", "c"], (2, 3), "legendre", y, nbins=10)
    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "1")
    clear_fe_resident_operands()
    eng_cached, names_cached, scores_cached = _gpu_build_and_score_univariate(X, ["a", "b", "c"], (2, 3), "legendre", y, nbins=10)

    assert eng_raw is not None and eng_cached is not None
    assert names_raw == names_cached
    np.testing.assert_allclose(cp.asnumpy(eng_raw), cp.asnumpy(eng_cached), atol=1e-9, rtol=1e-9)
    pd.testing.assert_frame_equal(scores_raw, scores_cached, check_exact=False, atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
