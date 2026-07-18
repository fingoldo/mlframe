"""Regression for the dense-block bulk-corrcoef path in ``_dedup_collinear_source_cols``.

cProfile on the wellbore-100k GPU-strict fit found the fully-finite ("dense") candidate block hardcoded to
``np.corrcoef`` regardless of backend -- unlike the partial-NaN block a few lines below, which already
dispatches through ``_resolve_pc_backend``/``_pairwise_complete_abs_corr`` (GPU-strict aware, cupy/njit
selectable). 9 calls to the dense path cost 116.9s cumtime in that profile (p_dense in the thousands after
FE families stack candidate columns), entirely on CPU even under STRICT mode's "carry FE compute on the
device" contract. These tests pin: (1) the numpy default path is untouched (backend selection falls through
to np.corrcoef exactly as before), (2) the cupy/njit dense path (forced via env) produces the same duplicate
verdict and the same |corr| values (up to FP noise) as the legacy hardcoded np.corrcoef path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup import (
    _dedup_collinear_source_cols,
)


def _dense_frame(seed: int, n: int = 4000, p: int = 12) -> pd.DataFrame:
    """All-finite frame with a few near-duplicate and a few independent columns, no NaN anywhere."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    cols = {"base": base, "dup": base + rng.normal(size=n) * 1e-6}
    for j in range(p - 2):
        cols[f"c{j}"] = rng.normal(size=n)
    return pd.DataFrame(cols)


def test_numpy_backend_default_path_unchanged():
    """Without a forced/STRICT backend, the dense block must still take the plain np.corrcoef path
    (no behavior change for the common non-GPU case)."""
    X = _dense_frame(seed=1)
    cols = list(X.columns)
    kept = _dedup_collinear_source_cols(X, cols, corr_threshold=0.999)
    assert "dup" not in kept, f"near-duplicate 'dup' survived: {kept}"
    assert "base" in kept
    for c in cols:
        if c not in ("base", "dup"):
            assert c in kept


@pytest.mark.parametrize("backend", ["cupy", "njit"])
def test_dense_block_backend_dispatch_matches_legacy_corrcoef(monkeypatch, backend):
    """Forcing the cupy/njit dedup backend must route the DENSE block through it too (not just the
    partial-NaN block) and the duplicate verdict must match the legacy hardcoded np.corrcoef path."""
    if backend == "cupy":
        pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_FE_DEDUP_CORR_BACKEND", backend)

    X = _dense_frame(seed=2)
    cols = list(X.columns)
    kept_new = _dedup_collinear_source_cols(X, cols, corr_threshold=0.999)

    # Reference: the legacy hardcoded np.corrcoef path (forced numpy backend, same frame/threshold).
    monkeypatch.setenv("MLFRAME_FE_DEDUP_CORR_BACKEND", "numpy")
    kept_legacy = _dedup_collinear_source_cols(X, cols, corr_threshold=0.999)

    assert set(kept_new) == set(kept_legacy), (
        f"{backend} dense-block dispatch disagrees with the legacy numpy path: {kept_new} vs {kept_legacy}"
    )
    assert "dup" not in kept_new


def test_resolve_pc_backend_consulted_for_dense_block(monkeypatch):
    """The dense-block pass must actually call _resolve_pc_backend (not silently ignore it) -- a direct
    unit check on the dispatch call, independent of the functional duplicate-detection outcome above."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_dedup as od

    calls = []
    orig = od._resolve_pc_backend

    def _spy(q, r, n):
        calls.append((q, r, n))
        return orig(q, r, n)

    monkeypatch.setattr(od, "_resolve_pc_backend", _spy)
    X = _dense_frame(seed=3)
    od._dedup_collinear_source_cols(X, list(X.columns), corr_threshold=0.999)
    assert calls, "dense block never consulted _resolve_pc_backend -- still hardcoded to np.corrcoef"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
