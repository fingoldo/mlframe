"""Phase 0 scaffold guard for the separate KTC-free GPU-resident FE path (_gpu_strict_fe).

Pins the residency-path INFRASTRUCTURE before the pipeline is wired:
  * the resident flag gates correctly (default OFF; needs both STRICT + RESIDENT);
  * the entry is an inert stub (raises NotImplementedError -> the FE step falls back to the existing path with
    ZERO behavior change) -- verified end-to-end on the F2 golden compound;
  * ResidentFEState uploads operands/y per device with a hw-spec-derived launch config + VRAM chunk, and drops
    its device handles on pickle;
  * the byte-size residency audit harness classifies bulk vs scalar transfers (the contract is audited by size,
    not by .get() count).
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def test_resident_flag_default_off():
    from mlframe.feature_selection.filters._gpu_strict_fe import fe_gpu_strict_resident_enabled

    # RESIDENT unset -> OFF regardless of STRICT (env not set here in the default test env).
    if os.environ.get("MLFRAME_FE_GPU_STRICT_RESIDENT", "") == "":
        assert fe_gpu_strict_resident_enabled() is False


def test_entry_stub_is_inert():
    from mlframe.feature_selection.filters._gpu_strict_fe import run_fe_step_gpu_strict

    with pytest.raises(NotImplementedError):
        run_fe_step_gpu_strict(None)


def test_resident_state_build_and_pickle():
    from mlframe.feature_selection.filters._gpu_strict_fe import ResidentFEState

    rng = np.random.default_rng(0)
    ops = rng.random((2000, 4))
    yc = rng.integers(0, 3, 2000)
    st = ResidentFEState.build(ops, list("abcd"), yc, y_cont_host=rng.random(2000), f32=True)
    d0 = st.device_ids()[0]
    assert st.operands(d0).shape == (2000, 4)
    assert st.y_codes(d0).shape == (2000,)
    assert st.y_cont(d0) is not None
    assert st.n_classes == int(yc.max()) - int(yc.min()) + 1
    cfg = st.launch_config(d0, ky=3)
    assert cfg["threads"] >= 1 and cfg["shared_per_block"] > 0 and isinstance(cfg["use_fused"], bool)
    assert st.k_chunk(d0, 50) >= 1
    # device handles must NOT survive pickle (module-resident-cache convention)
    red = pickle.loads(pickle.dumps(st))
    assert red._operands == {} and red._y_codes == {} and red.op_names == list("abcd")
    st.free()
    assert st._operands == {}  # freed


def test_residency_audit_classifies_bulk_vs_scalar():
    from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit, BULK_BYTES

    with residency_audit() as rep:
        a = cp.asarray(np.random.default_rng(1).random(20000))  # bulk H2D
        _ = a.get()  # bulk D2H
        _ = cp.asarray(np.array([1.0]))  # scalar H2D
    assert len(rep.bulk_h2d) == 1
    assert len(rep.bulk_d2h) == 1
    # the lone scalar H2D is below the bulk threshold
    assert any(b < BULK_BYTES for b in rep.h2d)
