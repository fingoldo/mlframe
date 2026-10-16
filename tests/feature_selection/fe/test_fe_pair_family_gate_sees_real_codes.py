"""The pair family gate reads the pair's real discretised codes, not an unfilled deferred GPU buffer (which reads as all zeros and rejects every winner)."""

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._family_maxt_gate as fg


def _cuda_ok():
    """Cuda ok."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _frame(n=3000, seed=0):
    """Frame."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(8)})
    y = ((X["f0"] * X["f1"] + X["f2"] / (1 + X["f3"].abs()) + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    return X, y


@pytest.mark.skipif(not _cuda_ok(), reason="the deferred host-codes buffer exists only on the GPU path")
@pytest.mark.parametrize("strict", ["1", ""])
def test_family_test_never_runs_on_all_zero_codes(monkeypatch, strict):
    """Family test never runs on all zero codes."""
    if strict:
        monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", strict)
    seen = []
    real = fg.family_pass_batches

    def spy(batches, y_codes, alpha=0.01, n_perm=fg.FAMILY_MAXT_PERMUTATIONS, seed=0):
        """Spy."""
        seen.append([bool(np.asarray(b).any()) for b in batches])
        return real(batches, y_codes, alpha, n_perm, seed)

    monkeypatch.setattr(fg, "family_pass_batches", spy)
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame()
    MRMR(verbose=0, random_state=0, n_jobs=1, fit_cache_max=0).fit(X, y)
    assert seen, "the frame must reach the exact family test"
    assert all(any(flags) for flags in seen), f"family test ran on a family of all-zero codes: {[f for f in seen if not any(f)][:3]}"
