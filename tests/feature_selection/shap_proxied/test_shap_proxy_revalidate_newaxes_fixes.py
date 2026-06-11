"""Regression tests for freshly-landed fixes in
``mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate``.

Covers:
  - [13] proxy_trust_guard reuses the single contiguous ``_phi_T`` it already
         builds for the coalition margins instead of letting
         subset_redundancy_many re-transpose ``phi`` a second time. The new
         ``subset_redundancy_many(phi, idx_list, phi_T=...)`` keyword must:
           * return a result bit-identical to the no-``phi_T`` (re-transpose) path
             (the optimisation may not change any numbers), and
           * skip the internal ``np.ascontiguousarray(phi.T)`` copy when a
             precomputed transpose is supplied (the actual perf win).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_calibrate import subset_redundancy_many


def test_phi_t_kwarg_is_bit_identical_to_retranspose():
    """Passing a precomputed phi_T must yield exactly the same redundancies as letting the helper
    transpose phi itself (perf-only change, no numeric drift). Includes singleton + empty-subset edges."""
    rng = np.random.default_rng(7)
    phi = rng.normal(size=(600, 16))
    idx_list = [[0, 1, 2], [5], [], [3, 7, 11, 15], list(range(16)), [4, 5]]

    baseline = subset_redundancy_many(phi, idx_list)  # internal transpose
    phi_T = np.ascontiguousarray(phi.T)
    reused = subset_redundancy_many(phi, idx_list, phi_T=phi_T)

    np.testing.assert_array_equal(baseline, reused)


def test_phi_t_kwarg_skips_internal_transpose(monkeypatch):
    """With phi_T supplied the helper must NOT build its own contiguous transpose (the redundant
    O(n_samples*n_units) copy the fix removes); without it, the helper must still transpose once."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_calibrate as calib_mod

    calls = {"n": 0}
    real_ascontig = np.ascontiguousarray

    def _counting_ascontig(a, *args, **kwargs):
        calls["n"] += 1
        return real_ascontig(a, *args, **kwargs)

    monkeypatch.setattr(calib_mod.np, "ascontiguousarray", _counting_ascontig)

    rng = np.random.default_rng(11)
    phi = rng.normal(size=(400, 12))
    idx_list = [[0, 1], [2, 3, 4]]
    phi_T = real_ascontig(phi.T)

    calls["n"] = 0
    subset_redundancy_many(phi, idx_list, phi_T=phi_T)
    assert calls["n"] == 0  # precomputed transpose reused, no extra copy

    calls["n"] = 0
    subset_redundancy_many(phi, idx_list)
    assert calls["n"] == 1  # legacy path still transposes exactly once


def test_proxy_trust_guard_passes_precomputed_phi_t(monkeypatch):
    """End-to-end: proxy_trust_guard builds _phi_T for coalition margins and must hand it to
    subset_redundancy_many (phi_T kwarg present, same array object), so the transpose happens once."""
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate as reval_mod

    seen = {}

    # subset_redundancy_many is imported lazily inside proxy_trust_guard from the calibrate module;
    # patch it there so the call inside the guard is observed.
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_calibrate as calib_mod
    real_calib_fn = calib_mod.subset_redundancy_many

    def _spy(phi, idx_list, *, phi_T=None):
        seen["phi_T_is_array"] = isinstance(phi_T, np.ndarray)
        seen["phi_T_shape"] = None if phi_T is None else phi_T.shape
        return real_calib_fn(phi, idx_list, phi_T=phi_T)

    monkeypatch.setattr(calib_mod, "subset_redundancy_many", _spy)

    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    import pandas as pd

    n_units = 6
    n_samples = 120
    # phi: row-major (n_samples, n_units) SHAP-attribution proxy block.
    phi = rng.normal(size=(n_samples, n_units))
    base = rng.normal(size=n_samples)
    # honest-retrain data lives in feature (== unit, no clustering) space.
    X_search = pd.DataFrame(rng.normal(size=(n_samples, n_units)), columns=[f"u{i}" for i in range(n_units)])
    y_search = (X_search.iloc[:, 0] + 0.2 * rng.normal(size=n_samples) > 0).astype(int).to_numpy()
    X_hold = pd.DataFrame(rng.normal(size=(80, n_units)), columns=X_search.columns)
    y_hold = (X_hold.iloc[:, 0] + 0.2 * rng.normal(size=80) > 0).astype(int).to_numpy()

    reval_mod.proxy_trust_guard(
        phi, base, y_search, LogisticRegression(max_iter=200), X_search, X_hold, y_hold,
        classification=True, n_anchors=8, rng=np.random.default_rng(1), n_jobs=1,
    )

    assert seen.get("phi_T_is_array") is True
    # _phi_T is the contiguous transpose of phi: shape (n_units, n_samples).
    assert seen.get("phi_T_shape") == (n_units, n_samples)
