"""The device operand-table build (rebuild of the plain unary columns from the raw operands, in column blocks, columns grouped by op code) gives the table of the cupy unary chain exactly; the public builder takes the route the kernel-tuning entry chooses (host upload or device rebuild) with the same table."""

from __future__ import annotations

import numpy as np
import pytest
from tests._cuda_require import require_cuda_device

from mlframe.feature_selection.filters import _gpu_resident_operand_build as ob
from mlframe.feature_selection.filters._gpu_resident_shared import _unary_apply


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    require_cuda_device()
    return cp


def _inputs(n=3000):
    """The sweep pool: host table and column specs."""
    return ob._sweep_inputs({"n_samples": n})


def test_rebuild_equals_the_cupy_unary_chain_exactly(monkeypatch):
    """Every batched column equals ``_unary_apply(cp, ...)`` cast to float32, value for value (NaN in the same cells), whatever the op-code order in the specs."""
    cp = _cp()
    table, specs = _inputs()
    monkeypatch.setattr(ob, "operand_table_device_chosen", lambda n, p=0: True)
    dev, n_gpu, _n_cpu = ob.build_resident_operand_table(table, specs)
    assert n_gpu == len(specs)
    got = cp.asnumpy(dev)
    for col, raw, name in specs:
        with np.errstate(all="ignore"):
            want = cp.asnumpy(_unary_apply(cp, name, cp.asarray(raw)).astype(cp.float32))
        np.testing.assert_array_equal(np.isnan(got[:, col]), np.isnan(want), err_msg=name)
        np.testing.assert_array_equal(got[:, col][~np.isnan(want)], want[~np.isnan(want)], err_msg=name)


def test_host_route_uploads_the_host_table(monkeypatch):
    """With the host chosen the builder returns the uploaded host table (no column built on the device)."""
    cp = _cp()
    table, specs = _inputs()
    monkeypatch.setattr(ob, "operand_table_device_chosen", lambda n, p=0: False)
    dev, n_gpu, n_cpu = ob.build_resident_operand_table(table, specs)
    assert (n_gpu, n_cpu) == (0, table.shape[1])
    np.testing.assert_array_equal(cp.asnumpy(dev), table)


def test_spec_blocks_give_the_one_block_table(monkeypatch):
    """A budget of one raw operand per block cuts the specs into several blocks; the table equals the single-block table."""
    cp = _cp()
    table, specs = _inputs()
    monkeypatch.setattr(ob, "operand_table_device_chosen", lambda n, p=0: True)
    whole, _, _ = ob.build_resident_operand_table(table, specs)
    blocks = ob._spec_blocks(specs, set(), table.shape[0], table.shape[0] * 8)
    assert len(blocks) > 1 and sum(len(b) for b in blocks) == len(specs)
    from mlframe.feature_selection.filters import _device_sort_safe as dss

    monkeypatch.setattr(dss, "device_room", lambda cp_: 4 * table.shape[0] * 8)  # a quarter of the room holds one raw operand
    blocked, n_gpu, _ = ob.build_resident_operand_table(table, specs)
    assert n_gpu == len(specs)
    np.testing.assert_array_equal(cp.asnumpy(whole), cp.asnumpy(blocked))
