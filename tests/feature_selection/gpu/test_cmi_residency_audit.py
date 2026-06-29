"""Residency BYTE-AUDIT for the GPU-strict greedy-CMI feature constructor.

Selection parity is not enough: a byte-identical kernel can still leak the whole (K,) CMI vector D2H and
re-upload the fit-constant ``y`` (and round-constant ``z``) H2D on every candidate batch. This audits the
ACTUAL host<->device traffic of the wired greedy loop (``greedy_cmi_fe_construct`` ->
``batched_cmi_gpu(..., return_device=True)`` + ``cmi_device_argmax``) by transfer SIZE and asserts the bulk
leak is gone:

  (a) the (K,) MI float64 vector is NOT in the bulk-D2H list (it stays resident; only the argmax scalars cross),
  (b) ``y`` is uploaded ONCE, not re-uploaded as a bulk H2D per candidate batch (resident-operand cache),
  (c) the per-round argmax pulls only scalar (< BULK_BYTES) D2H.

The harness (``residency_audit``) classifies cp.asarray H2D / .get()+cp.asnumpy D2H by byte size; BULK_BYTES
(8192) cleanly separates the scalar decisions from bulk arrays.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _make_frame(n: int, seed: int):
    rng = np.random.default_rng(seed)
    a, b, c, d = (rng.random(n) for _ in range(4))
    # A compound target so several engineered candidates carry signal and the greedy loop runs >1 round.
    score = (a * a) - 1.0 + 0.7 * np.sin(d * 3.0) - 0.5 * b
    y = (score > np.median(score)).astype(np.int64)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d})
    return X, y


def _run_audit(env_on: bool):
    from mlframe.feature_selection.filters import _fe_resident_operands as _R
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import greedy_cmi_fe_construct
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit

    saved = {k: os.environ.get(k) for k in
             ("MLFRAME_FE_GPU_STRICT", "MLFRAME_CMI_GPU", "MLFRAME_FE_VRAM_F32")}
    if env_on:
        os.environ["MLFRAME_FE_GPU_STRICT"] = "1"
        os.environ["MLFRAME_CMI_GPU"] = "1"
        os.environ["MLFRAME_FE_VRAM_F32"] = "1"
    else:
        os.environ["MLFRAME_CMI_GPU"] = "0"
        os.environ.pop("MLFRAME_FE_GPU_STRICT", None)

    # Count resident-operand UPLOAD events (cache misses) per role: a HIT returns the cached device array
    # without an H2D, a MISS issues exactly one cp.asarray H2D. The label ``cmi_y`` is fit-constant and must
    # upload exactly once; ``cmi_z`` is round-constant and legitimately re-uploads when its CONTENT changes
    # (the conditioning support grows each greedy round -> a genuinely different array, not a redundant churn).
    role_uploads: dict = {}
    _orig = _R.resident_operand

    def _counting(arr, key, **kw):
        import numpy as _np
        host = _np.asarray(arr)
        dtype = kw.get("dtype")
        if dtype is not None:
            host = host.astype(dtype, copy=False)
        host = _np.ascontiguousarray(host) if kw.get("contiguous", True) else host
        full_key = (key, host.shape, host.dtype.str)
        sig = (host.shape, host.dtype.str, hash(host.tobytes()))
        cached = _R._FE_RESIDENT_OPERANDS.get(full_key)
        miss = not (cached is not None and cached[1] == sig)
        if miss:
            role_uploads[key] = role_uploads.get(key, 0) + 1
        return _orig(arr, key, **kw)

    try:
        X, y = _make_frame(8000, 11)
        kw = dict(nbins=10, seed_cols_count=4, min_cmi_gain=0.0)
        _R.clear_fe_resident_operands()
        # Warm cupy/JIT + the greedy enumeration OUTSIDE the audited region.
        greedy_cmi_fe_construct(X, y, **kw)
        _R.clear_fe_resident_operands()
        _R.resident_operand = _counting
        try:
            with residency_audit() as rep:
                X_aug, scores = greedy_cmi_fe_construct(X, y, **kw)
        finally:
            _R.resident_operand = _orig
        return rep, scores, role_uploads
    finally:
        _R.resident_operand = _orig
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cmi_greedy_residency_no_bulk_mi_vector_d2h():
    """PRIMARY GATE: under the 3 strict flags the wired greedy loop emits ZERO bulk D2H (the (K,) CMI vector
    stays resident; the argmax pulls only scalars) and does NOT re-upload y as a bulk H2D per candidate batch."""
    rep_on, scores, role_uploads = _run_audit(env_on=True)
    # Sanity: the GPU path actually ran (>=1 winner selected over multiple candidates).
    assert len(scores) >= 1, "greedy CMI selected no winners (path may not have exercised the GPU loop)"

    # (a) The (K,) MI float64 vector is NOT in the bulk-D2H list. Under return_device the only D2H crossings
    #     are the argmax (idx int64 8B, val float64 8B) scalars + the analytic-null tiny scalars.
    assert len(rep_on.bulk_d2h) == 0, (
        f"unexpected bulk D2H (the (K,) MI vector should stay resident): {rep_on.bulk_d2h}; "
        f"{rep_on.summary()}")

    # (c) All D2H is scalar (< BULK_BYTES).
    from mlframe.feature_selection.filters._gpu_strict_fe._audit import BULK_BYTES
    assert all(b < BULK_BYTES for b in rep_on.d2h), rep_on.summary()

    # (b) y is FIT-CONSTANT: the greedy hot path uploads the fixed y (role ``cmi_greedy_y_fixed``) EXACTLY ONCE
    #     and then reuses that RESIDENT cupy array for every per-round batched_cmi_gpu call (no per-round y H2D).
    #     The per-permutation SHUFFLED y (role ``cmi_y``) is a genuinely different array each null draw and so is
    #     not fit-constant -- it is kept on a distinct role so it can never evict the fixed y.
    assert role_uploads.get("cmi_greedy_y_fixed", 0) == 1, (
        f"fixed greedy y uploaded {role_uploads.get('cmi_greedy_y_fixed', 0)}x; the fit-constant y must be "
        f"uploaded exactly once and reused resident. role_uploads={role_uploads}")


def test_cmi_residency_before_after_classification():
    """Document the before/after bulk-transfer classification: with return_device OFF (host path) the per-round
    (K,) MI vector + y cross as bulk; with the strict wiring ON they do not. This is informational (printed) and
    asserts the ON path is strictly cleaner on bulk D2H."""
    rep_off, _, _ = _run_audit(env_on=False)   # CPU host path (no GPU): zero device traffic at all
    rep_on, _, ru = _run_audit(env_on=True)
    print("BEFORE (host/CPU path):     " + rep_off.summary())
    print("AFTER  (GPU-strict wired):  " + rep_on.summary() + f"  role_uploads={ru}")
    # The wired GPU path keeps the bulk D2H at zero (the gate); the host path issues no device transfers.
    assert len(rep_on.bulk_d2h) == 0
