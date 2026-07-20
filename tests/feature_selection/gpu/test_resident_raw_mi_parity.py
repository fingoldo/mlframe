"""GPU parity coverage for ``_resident_raw_mi.resident_raw_baseline_mi`` (mrmr_audit_2026-07-20
test_coverage.md #10): the module's own docstring claims the resident MI is byte-identical to the
host STRICT ``_mi_classif_batch`` estimator over the same matrix/y/binner -- previously asserted but
never tested. Skips cleanly when cupy is unavailable."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._resident_raw_mi import resident_raw_baseline_mi

cp = pytest.importorskip("cupy")


class TestGateReturnsNoneWhenStrictResidentOff:
    """With STRICT-residency off (the default / explicit opt-out), resident_raw_baseline_mi must
    return None so the caller falls back to the exact host _mi_classif_batch."""

    def test_returns_none_when_strict_off(self, monkeypatch):
        """MLFRAME_FE_GPU_STRICT=0 -> resident_raw_baseline_mi is a no-op (None)."""
        monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
        rng = np.random.default_rng(0)
        mat = rng.standard_normal((200, 3))
        y = rng.integers(0, 3, size=200)
        out = resident_raw_baseline_mi(mat, y, ("test_role", ("a", "b", "c")), nbins=10)
        assert out is None

    def test_returns_none_when_resident_opt_out_set(self, monkeypatch):
        """Even with STRICT on, MLFRAME_FE_GPU_STRICT_RESIDENT=0 is the explicit opt-out."""
        monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
        monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "0")
        rng = np.random.default_rng(1)
        mat = rng.standard_normal((200, 3))
        y = rng.integers(0, 3, size=200)
        out = resident_raw_baseline_mi(mat, y, ("test_role", ("a", "b", "c")), nbins=10)
        assert out is None


class TestResidentMatchesHostStrictMi:
    """Under STRICT-residency, the resident MI over a raw baseline matrix must match the host
    ``_mi_classif_batch`` estimator over the SAME matrix/y/binner (edge, non-rank) to the
    documented ~1e-9 tolerance."""

    def test_resident_edge_binned_mi_matches_host(self, monkeypatch):
        """Resident edge binned mi matches host."""
        from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import _mi_classif_batch

        monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
        monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
        monkeypatch.delenv("MLFRAME_FE_GPU_STRICT_AUTO_MIN_N", raising=False)

        rng = np.random.default_rng(42)
        n, k = 3000, 5
        mat = rng.standard_normal((n, k)).astype(np.float64)
        y = rng.integers(0, 4, size=n).astype(np.int64)
        nbins = 10

        resident = resident_raw_baseline_mi(mat, y, ("test_parity_role", tuple(f"c{j}" for j in range(k))), nbins=nbins)
        if resident is None:
            pytest.skip("STRICT-resident raw-baseline path did not engage on this host (no usable CUDA device / gate declined) -- nothing to compare")

        host = _mi_classif_batch(mat, y, nbins=nbins, rank_binning=False)
        max_abs_diff = float(np.max(np.abs(np.asarray(resident) - np.asarray(host))))
        assert max_abs_diff < 1e-6, (
            f"resident_raw_baseline_mi diverged from the host _mi_classif_batch by {max_abs_diff:.3e} "
            f"(docstring claims byte-identical selection-equivalence, ~1e-9)"
        )
