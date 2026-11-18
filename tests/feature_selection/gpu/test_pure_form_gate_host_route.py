"""The host batch (numba ``pair_gate_scalars_njit``) returns the gate scalars of the device batch (to 1e-8 relative on a well-conditioned pool) and the same verdicts; the public entry takes the host route when the tuner chooses it and the device route otherwise; the form blocks of a tiny room give the one-block result."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_pure_form_retention_gpu_resident as m
from mlframe.feature_selection.filters import _pure_form_gate_njit as nj
from tests.feature_selection.gpu._gpu_support import has_device


def _cp():
    """cupy with a usable device, else skip."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp


def test_host_scalars_equal_the_device_scalars():
    """All four scalars, including exactly linear forms and a pair of the same operand twice."""
    cp = _cp()
    forms, pairs, names, cols, y = m._sweep_inputs({"n_samples": 6000})
    pairs[0], pairs[1] = (names[2], names[2]), (names[3], names[4])
    host = nj.host_gate_scalars(forms, cols, pairs, names, y)
    yc = cp.asarray(y - y.mean())
    basis = [m._basis_stack(cp, cp.asarray(c)) for c in cols]
    dev = m._batched_gate_scalars(cp, cp.asarray(np.stack(forms, 1)), basis, {nm: i for i, nm in enumerate(names)}, pairs, yc, len(pairs))
    # a form that is exactly linear in its operands leaves a residual at rounding level (ss 4e-6 of a total 1.2e4): the two solvers agree on every scalar to 1e-9 of the form's own sum of squares
    atol = 1e-9 * len(forms[0]) * host[:, :1] ** 2
    assert np.all(np.abs(host - dev) <= 1e-8 * np.abs(dev) + atol)


def test_unknown_operand_returns_none_on_the_host():
    """A pair naming an operand outside the base set sends the caller to the per-candidate CPU path."""
    forms, pairs, names, cols, y = m._sweep_inputs({"n_samples": 500})
    pairs[0] = (names[0], "unknown")
    assert nj.host_gate_scalars(forms, cols, pairs, names, y) is None


@pytest.mark.parametrize("device", [False, True])
def test_entry_follows_the_tuner_and_the_verdicts_agree(monkeypatch, device):
    """Host and device verdicts of the entry are identical on the sweep pool; the chosen route is the one that runs."""
    _cp()
    forms, pairs, names, cols, y = m._sweep_inputs({"n_samples": 6000})
    calls = {"device": 0, "host": 0}
    real_dev, real_host = m._device_verdicts, m._host_verdicts
    monkeypatch.setattr(m, "_device_verdicts", lambda *a, **k: (calls.__setitem__("device", calls["device"] + 1), real_dev(*a, **k))[1])
    monkeypatch.setattr(m, "_host_verdicts", lambda *a, **k: (calls.__setitem__("host", calls["host"] + 1), real_host(*a, **k))[1])
    monkeypatch.setattr(m, "pure_form_device_chosen", lambda n, p=0: device)
    got = m.adds_nonlinear_value_batch_gpu_resident(forms, pairs, names, cols, y, min_resid_frac=0.10, min_resid_corr=0.08)
    assert (calls["device"], calls["host"]) == ((1, 0) if device else (0, 1))
    monkeypatch.setattr(m, "pure_form_device_chosen", lambda n, p=0: not device)
    other = m.adds_nonlinear_value_batch_gpu_resident(forms, pairs, names, cols, y, min_resid_frac=0.10, min_resid_corr=0.08)
    assert list(got) == list(other)


def test_pair_blocks_give_the_single_block_verdicts(monkeypatch):
    """A room that holds only a few form columns forces several pair blocks; the verdicts equal the one-block run."""
    _cp()
    from mlframe.feature_selection.filters import _device_sort_safe as dss

    forms, pairs, names, cols, y = m._sweep_inputs({"n_samples": 4000})
    monkeypatch.setattr(m, "pure_form_device_chosen", lambda n, p=0: True)
    whole = m.adds_nonlinear_value_batch_gpu_resident(forms, pairs, names, cols, y, min_resid_frac=0.10, min_resid_corr=0.08)
    monkeypatch.setattr(dss, "device_room", lambda cp_: 4 * 4 * 4 * 4000 * 8)  # about four form columns per block
    blocked = m.adds_nonlinear_value_batch_gpu_resident(forms, pairs, names, cols, y, min_resid_frac=0.10, min_resid_corr=0.08)
    assert list(whole) == list(blocked)
