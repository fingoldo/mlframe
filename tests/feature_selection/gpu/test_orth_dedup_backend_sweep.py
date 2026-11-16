"""The pairwise-complete correlation of the dedup: the three backends agree, the fused cupy tail matches numpy, and the choice goes through the tuner entry."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_dedup as od


def _blocks(p: int = 24, n: int = 3000):
    """Sweep-shaped inputs: row-centred criterion-dtype blocks with missing values and planted near-duplicates."""
    return od._sweep_inputs({"p": p, "n": n})


def test_njit_matches_numpy_on_the_sweep_shape():
    """The parallel numba backend equals the BLAS backend to float32 summation rounding, nan entries in the same places."""
    q, r = _blocks()
    ref = od._pc_corr_numpy(q, r)
    got = od._pc_corr_njit(np.ascontiguousarray(q, dtype=np.float64), np.ascontiguousarray(r, dtype=np.float64))
    assert np.array_equal(np.isnan(ref), np.isnan(got))
    np.testing.assert_allclose(od._nan_to_flag(got), od._nan_to_flag(ref), atol=od._SWEEP_TOL)


def test_planted_duplicates_are_found_by_every_backend():
    """The first tenth of the rows of R repeat Q's rows with small noise: |r| is near 1 on that diagonal for each backend."""
    q, r = _blocks()
    k = max(1, q.shape[0] // 10)
    for corr in (od._pc_corr_numpy(q, r), od._pc_corr_njit(np.ascontiguousarray(q, dtype=np.float64), np.ascontiguousarray(r, dtype=np.float64))):
        assert (np.diag(corr)[:k] > 0.99).all()


def test_fused_cupy_tail_matches_numpy():
    """The device form (explicit GEMMs plus the fused ``pc_corr_final`` pass) equals numpy, including the nan mask of the fewer-than-8-common-rows and near-constant rules."""
    pytest.importorskip("cupy")
    q, r = _blocks()
    q[3, :] = 1.0  # a constant row: zero variance -> nan
    q[5, 8:] = np.nan  # fewer than 8 common rows -> nan
    ref = od._pc_corr_numpy(q, r)
    got = od._pc_corr_cupy(q, r)
    assert np.array_equal(np.isnan(ref), np.isnan(got))
    np.testing.assert_allclose(od._nan_to_flag(got), od._nan_to_flag(ref), atol=od._SWEEP_TOL)


def test_the_choice_goes_through_the_tuner_entry(monkeypatch):
    """Without a force or STRICT the resolver asks the registered entry and accepts only the three known backend names."""
    monkeypatch.delenv("MLFRAME_FE_DEDUP_CORR_BACKEND", raising=False)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")

    class _Spec:
        """A stand-in entry that answers with a fixed choice and records the dims it was asked with."""

        asked: dict = {}

        def choose(self, **dims):
            """Record the dims and answer 'njit'."""
            self.asked = dims
            return "njit"

    spec = _Spec()
    monkeypatch.setattr(od, "_SPEC", spec)
    assert od._resolve_pc_backend(40, 10, 5000) == "njit"
    assert spec.asked == {"p": 40, "n": 5000}
    monkeypatch.setattr(od, "_SPEC", type("Bad", (), {"choose": lambda self, **d: "tpu"})())
    assert od._resolve_pc_backend(40, 10, 5000) == "numpy"
