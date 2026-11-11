"""Register-pressure ratchet for fused RawKernels that expose a ``kernel(cp, real) -> CompiledFunction`` factory.

CUDA.md (section 3, `choose-launch-configuration`) says to take the register count from ``ptxas -v`` or the profiler, never
from a hand estimate, and that a local load or store (a spill) costs 30 to 800 cycles depending on which cache level
answers it. cupy's compiled ``RawKernel``/``RawModule`` functions report this directly: ``num_regs`` and
``local_size_bytes`` are read off the cubin, which is the ``ptxas -v`` number without needing the compiler's text
output. This test compiles every kernel in the registry below for every dtype it supports and asserts there is no
spill (``local_size_bytes == 0``), skipping the whole module honestly when no CUDA device is present rather than
asserting on host-only numbers.

The registry starts with the kernels this session wrote or touched; CUDA.md kernels are added here as they are
written or revisited, the same incremental-ratchet shape as ``_fe_gpu_contract.json``. A kernel enters
``_SPILL_EXEMPT`` only with the reason a spill is accepted for it (e.g. a kernel with too many live values to avoid
one even after trying), never silently.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

# name -> (module path, factory attribute name, dtypes to compile)
_REGISTRY: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "fe_accuracy_gate_kernel.gate_logistic_auc": (
        "mlframe.feature_selection.filters._fe_accuracy_gate_kernel",
        "kernel",
        ("float32", "float64"),
    ),
}

# kernel name -> reason a register spill is accepted for it. Empty is the default: fix the kernel instead of exempting
# it. Pack any thread-private triangular-matrix array (Hessian, Cholesky factor) before reaching for this dict - that
# cut this kernel's spill from 1512 to 936 bytes/thread for free (see the reason string for what remains).
_SPILL_EXEMPT: dict[str, str] = {
    "fe_accuracy_gate_kernel.gate_logistic_auc": (
        "936 bytes/thread remain (down from 1512 after packing the per-thread Hessian/Cholesky arrays into their "
        "lower triangle) in the per-row gradient accumulation loop's g[MAXP]/x[MAXP]/h[MAXTRI] work vectors. k is a "
        "runtime kernel argument (0..MAXK), so the compiler cannot unroll the loop that indexes them and keep them "
        "in registers. Removing the spill would need a second pass over the design columns (recompute z without "
        "storing x[]) at MAXK<=8, doubling the global reads this kernel does per Newton iteration, for a kernel "
        "that already had one silent-divergence incident from a tolerance mismatch (see "
        "audits/2026-10-22/profiles/_fe_accuracy_gate_kernel.md); not worth the risk for this occupancy-limited, "
        "already compute-bound kernel per CUDA.md section 3 (a spill is not worth a higher occupancy)."
    )
}


def _cupy_or_skip() -> Any:
    """Import cupy or skip the test; skip again if the device present cannot actually compile/launch a kernel."""
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover - environment without a usable CUDA device
        pytest.skip(f"no usable CUDA device: {exc}")
    return cp


@pytest.mark.parametrize("name", sorted(_REGISTRY))
def test_registered_kernel_has_no_register_spill(name: str):
    """Every dtype a registered kernel compiles for reports zero local_size_bytes, unless listed in _SPILL_EXEMPT."""
    cp = _cupy_or_skip()
    module_path, factory_name, dtypes = _REGISTRY[name]
    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name)
    for dtype in dtypes:
        real = getattr(cp, dtype)
        compiled = factory(cp, real)
        compiled.compile()
        local_bytes = int(compiled.local_size_bytes)
        if name in _SPILL_EXEMPT:
            continue
        assert local_bytes == 0, (
            f"{name} ({dtype}) spills {local_bytes} bytes/thread to local memory (num_regs={compiled.num_regs}); "
            f"either reduce live values per CUDA.md section 3 register-pressure-warp-occupancy, or add an entry to "
            f"_SPILL_EXEMPT with the reason."
        )


def test_every_exempted_kernel_states_its_reason():
    """Every entry in _SPILL_EXEMPT names a registered kernel and carries a non-empty reason."""
    for name, reason in _SPILL_EXEMPT.items():
        assert name in _REGISTRY, f"{name} is exempted but not registered"
        assert reason.strip(), f"{name} has an empty exemption reason"
