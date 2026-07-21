"""Edge case (mrmr_audit_2026-07-20 edge_cases.md): a GPU-resident-labeled FE family must never
silently no-op or produce wrong output when the GPU-resident path is unavailable/off -- it must
transparently run its exact CPU-equivalent sibling.

The GPU-resident sub-path for ``_fe_pure_form_retention.py``'s pair-interaction retention (the
``_fe_pure_form_retention_gpu_resident.adds_nonlinear_value_batch_gpu_resident`` batched
non-separability filter) is gated by ``fe_gpu_strict_resident_enabled()`` -- an ENV-VAR-driven
flag (``MLFRAME_FE_GPU_STRICT`` / ``_RESIDENT``), independent of MRMR's own ``use_gpu`` ctor
param, which mostly governs the direct MI-permutation dispatch (``mi_direct`` vs ``mi_direct_gpu``)
rather than these per-family GPU-resident sub-paths. The family's own code already wraps the
device call in a broad ``try/except`` that falls back to the exact per-candidate CPU
``_adds_nonlinear_value`` on ANY failure or when the flag is off (the default) -- this test pins
that contract as regression coverage: the default (no STRICT flags set) CPU path recovers the
trapped pair-interaction correctly, and it agrees with the GPU-resident path bit-for-bit in
selection when a CUDA device is available.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.mrmr import MRMR


def _trapped_product_fixture(seed: int = 0, n: int = 3000):
    """y = a*b (a pure pair-interaction) + a linear nuisance term -- the classic
    retain_usable_pure_forms target: the greedy MI loop can leave a*b trapped inside a cross-mix
    or represented only via its separate raw operands, and this family's job is to recover the
    PURE product form as its own linearly-usable engineered column."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    y = a * b + 0.3 * c + 0.1 * rng.standard_normal(n)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "noise1": rng.standard_normal(n), "noise2": rng.standard_normal(n)})
    return X, y


def _kw(**overrides):
    """Fast-fitting default MRMR constructor kwargs, overridable per test."""
    base = dict(random_state=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, fe_max_steps=1, skip_retraining_on_same_content=False)
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clean_strict_env(monkeypatch):
    """Ensure no STRICT env var leaks in from a prior test in the same process."""
    for k in ("MLFRAME_FE_GPU_STRICT", "MLFRAME_FE_GPU_STRICT_RESIDENT", "MLFRAME_CMI_GPU"):
        monkeypatch.delenv(k, raising=False)
    yield


def test_default_no_strict_env_recovers_trapped_pair_interaction():
    """With no STRICT env vars set (the default -- MRMR's own use_gpu ctor param does not gate this
    family), the exact CPU path must run and successfully recover the trapped a*b pair-interaction
    as its own engineered column, never silently producing an empty/wrong contribution."""
    X, y = _trapped_product_fixture()
    m = MRMR(**_kw())
    m.fit(X, y)
    names = [str(s) for s in m.get_feature_names_out()]
    assert any(
        "a" in n and "b" in n and n not in ("a", "b") for n in names
    ), f"expected a recovered a*b pair-interaction engineered column in support_, got {names}"


def test_gpu_resident_path_selection_equivalent_to_cpu_default(monkeypatch):
    """When a CUDA device IS available and STRICT-residency is explicitly forced on, the
    GPU-resident non-separability filter must select the SAME engineered columns as the exact CPU
    default path -- never a silent divergence or empty contribution from the device branch."""
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device visible")
    except Exception:
        pytest.skip("cupy unavailable")

    X, y = _trapped_product_fixture(seed=3)

    m_cpu = MRMR(**_kw())
    m_cpu.fit(X.copy(), y.copy())
    names_cpu = sorted(str(s) for s in m_cpu.get_feature_names_out())

    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")
    m_gpu = MRMR(**_kw())
    m_gpu.fit(X.copy(), y.copy())
    names_gpu = sorted(str(s) for s in m_gpu.get_feature_names_out())

    assert names_cpu == names_gpu, f"GPU-resident path diverged from the CPU default: cpu={names_cpu} vs gpu={names_gpu}"


def test_device_failure_falls_back_to_cpu_not_silent_empty(monkeypatch):
    """Forcing the device-resident helper to raise (simulating a cupy/device fault mid-fit) must
    fall back to the exact CPU path and still recover the trapped pair-interaction -- never
    silently swallow the failure into an empty/degenerate contribution."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")

    import mlframe.feature_selection.filters._fe_pure_form_retention_gpu_resident as _gr

    def _boom(*_a, **_k):
        """Stand-in for the device-resident helper that always raises, simulating a cupy/device fault."""
        raise RuntimeError("simulated device fault")

    monkeypatch.setattr(_gr, "adds_nonlinear_value_batch_gpu_resident", _boom, raising=True)

    X, y = _trapped_product_fixture(seed=7)
    m = MRMR(**_kw())
    m.fit(X, y)
    names = [str(s) for s in m.get_feature_names_out()]
    assert any(
        "a" in n and "b" in n and n not in ("a", "b") for n in names
    ), f"a simulated device fault must fall back to the CPU path and still recover the pair-interaction, got {names}"
