"""The batched device probe of the accuracy gate keeps the same columns as the per-candidate host probe."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_accuracy_gate as G
from mlframe.feature_selection.filters import _fe_accuracy_gate_batched as B


def _device_ok() -> bool:
    """Whether the batched probe can run here at all (a usable CUDA device and cupy)."""
    try:
        import cupy  # noqa: F401

        from mlframe.feature_selection.filters._gpu_policy import cuda_available_for_run

        return bool(cuda_available_for_run())
    except Exception:
        return False


needs_device = pytest.mark.skipif(not _device_ok(), reason="no usable CUDA device for the batched gate probe")


def _frame(n=2000, k=40, seed=0):
    """Sources, engineered columns and a binary target: some engineered columns add signal, some are noise, one is constant."""
    rng = np.random.default_rng(seed)
    srcs = [rng.standard_normal(n) for _ in range(k)]
    y = ((srcs[0] + 0.6 * srcs[1] ** 2 + 0.5 * rng.standard_normal(n)) > 0).astype(np.int64)
    engs = []
    for i, s in enumerate(srcs):
        if i % 4 == 0:
            engs.append(s**2)  # a real second-order lift for the columns that drive y
        elif i % 4 == 1:
            engs.append(rng.standard_normal(n))  # pure noise
        elif i % 4 == 2:
            engs.append(np.zeros(n))  # constant: zero spread
        else:
            engs.append(s * 2.0 + 1.0)  # an affine copy of the source: no uplift
    return srcs, engs, y


@needs_device
def test_batched_uplifts_keep_the_same_columns_as_the_host_probe():
    """Every covered candidate's keep/drop verdict matches the host probe's, and the uplifts are close."""
    srcs, engs, y = _frame()
    got = B.batched_uplifts_for_pairs(srcs, engs, [f"s{i}" for i in range(len(srcs))], y, seed=0)
    assert got is not None and np.isfinite(got).any(), "the batch must cover the finite candidates"
    host = np.array([G.measure_feature_uplift(s[:, None], e[:, None], y, classification=True) if G.measure_feature_uplift(s[:, None], e[:, None], y, classification=True) is not None else np.nan for s, e in zip(srcs, engs)])
    covered = np.isfinite(got) & np.isfinite(host)
    assert covered.sum() >= len(srcs) // 2, f"too few covered: {covered.sum()} of {len(srcs)}"
    verdict_batched = got[covered] >= G._FE_UPLIFT_MIN
    verdict_host = host[covered] >= G._FE_UPLIFT_MIN
    assert verdict_batched.any() and not verdict_batched.all(), "the fixture must produce both keeps and drops, or the verdict check has no teeth"
    same = verdict_batched == verdict_host
    assert same.all(), f"verdicts differ at {np.flatnonzero(covered)[~same]}: batched {got[covered][~same]} host {host[covered][~same]}"
    # the batch reduces over designs in one batched BLAS call, so an uplift can differ in the 4th decimal (a single design matches the host exactly); the contract is the verdict above,
    # and this bound only catches a gross error
    assert np.allclose(got[covered], host[covered], atol=5e-3), np.abs(got[covered] - host[covered]).max()


@needs_device
def test_pairs_the_batch_does_not_cover_come_back_nan():
    """A pair with a non-finite value keeps its own rows in the host probe, so the batch must not decide it."""
    srcs, engs, y = _frame(n=1500, k=8)
    engs[2] = engs[2].copy()
    engs[2][5] = np.nan
    srcs[3] = srcs[3].copy()
    srcs[3][7] = np.inf
    got = B.batched_uplifts_for_pairs(srcs, engs, [f"s{i}" for i in range(len(srcs))], y, seed=0)
    assert got is not None
    assert not np.isfinite(got[2]) and not np.isfinite(got[3]), "non-finite pairs must be left to the host probe"
    assert np.isfinite(got[0]), "the finite pairs are still covered"


def test_a_regression_or_multiclass_target_is_not_batched():
    """The batch reproduces the binary-logistic probe only; other targets fall through to the host path."""
    rng = np.random.default_rng(1)
    n = 800
    srcs = [rng.standard_normal(n) for _ in range(6)]
    assert B.batched_uplifts_for_pairs(srcs, [s**2 for s in srcs], [f"s{i}" for i in range(6)], rng.standard_normal(n)) is None
    assert B.batched_uplifts_for_pairs(srcs, [s**2 for s in srcs], [f"s{i}" for i in range(6)], rng.integers(0, 4, n)) is None
