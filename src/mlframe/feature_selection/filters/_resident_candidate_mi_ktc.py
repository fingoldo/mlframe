"""MANDATE-1/2 KTC crossover for the resident-GPU FE-gate candidate-MI path.

Gates ``_resident_candidate_mi.best_existing_op_mi_resident`` (resident GPU candidate-gen + plug-in MI) against
the host ``_plugin_mi_classif_batch_njit`` it replaces. Per ``feedback_use_kernel_tuning_cache_for_gpu`` the
engage/skip decision is NOT a hardcoded k threshold: a ``kernel_tuner`` sweeps both paths over an (n, k) grid
and records, per region, which is faster. The resident path engages only where MEASURED faster; otherwise the
caller stays on the exact njit batch-MI.

HONEST EXPECTATION (GTX 1050 Ti, 6 SMs): the resident plug-in-MI crossover is ~k>=100 @ n=100k (the F2 gate
calls are k<=18-dominated), so on THIS card the sweep keeps small-k on CPU (the per-launch+sync overhead loses
sub-crossover) and selects the resident path for large-k / on stronger GPUs / larger p. The two paths are
selection-equivalent (percentile-edge vs rank binning, the approved FE-PAIR trade), so the sweep's equivalence
tol is loosened accordingly and it ranks by WALL.

CPU/no-cupy host: the sweep never runs, ``.choose()`` returns "njit", and the caller takes the exact host path.
"""
from __future__ import annotations

import logging

import numpy as np
from typing import Any, cast
from mlframe.utils.log_throttle import log_throttle

from ._ktc_fallback import selection_view

logger = logging.getLogger(__name__)

# (n, k) grid. k spans the F2 gate distribution (small-k k~16) up to the wide orth-univariate / large-screen
# regimes (k>=128) where the resident gen+MI amortises its launch. The default fallback keeps the un-tuned
# host path (njit) - the resident path is OPT-IN-BY-MEASUREMENT only, never the un-tuned default.
_RESCAND_SWEEP_N_SAMPLES = [50_000, 100_000, 300_000]
_RESCAND_SWEEP_K = [16, 64, 128, 256]
_RESCAND_SALT = 2  # 2: the sweep compares the selection view (best candidates) at the strict tolerance


def rescand_use_resident(n: int, k: int) -> bool:
    """Per-host engage decision for the resident-GPU candidate-MI path, from the kernel_tuning_cache.

    Returns ``True`` (use the resident GPU gen+MI) only on a measured-faster cache hit; ``False`` on a miss /
    no-cupy / lookup failure (caller stays on the exact host njit batch-MI). ``k`` snaps to the nearest swept
    bucket. STRICT GPU mode (``MLFRAME_FE_GPU_STRICT=1``, diagnostic, default OFF) forces the resident path:
    the two binning schemes are selection-equivalent (the approved FE-PAIR trade)."""
    # The global GPU opt-out outranks both the tuning cache and STRICT mode: a measured "resident is faster"
    # verdict says nothing about a run that asked for no GPU at all. Checked live rather than through the
    # memoised _cuda_usable() probe, which freezes whatever the environment said at its first call.
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        return False
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        if fe_gpu_strict_enabled(n=int(n), p=int(k)):
            return True
    except Exception as e:  # nosec B110 - optional dependency import guard
        log_throttle(logger, "_resident_candidate_mi_ktc.strict_check", logging.WARNING, "fe_gpu_strict_enabled() check failed, continuing to the crossover-based decision: %s", e)
    if _RESCAND_SPEC is None:
        return False
    k_bucket = min(_RESCAND_SWEEP_K, key=lambda b: abs(b - int(k)))
    try:
        choice = _RESCAND_SPEC.choose(n_samples=int(n), k=int(k_bucket))
    except Exception as e:
        log_throttle(logger, "_resident_candidate_mi_ktc.rescand_use_resident", logging.WARNING, "rescand_use_resident: KTC choose() failed, caller stays on the exact host njit batch-MI: %s", e)
        return False
    return bool(choice == "resident")


def _make_rescand_inputs(dims: dict):
    """An (n, k) host double candidate matrix + an int64 y, shaped like a gate MI call. The two probe
    variants score the SAME matrix (njit on host vs the resident GPU plug-in MI on its device copy), so the
    crossover measured is gen-agnostic launch+MI vs njit - the dominant cost the gate routes."""
    n = int(dims["n_samples"])
    k = int(dims["k"])
    rng = np.random.default_rng(0)
    a = rng.uniform(0.1, 1.1, (n, k)).astype(np.float64)  # Intentional float64: KTC probe fixture mirrors the gate's float64 host matrix (keeps tuned crossover)
    b = rng.uniform(0.1, 1.1, (n, k)).astype(np.float64)  # Intentional float64: KTC probe fixture mirrors the gate's float64 host matrix (keeps tuned crossover)
    mat = (a * a) / b  # heavy-tailed, the kind of engineered candidate column the gate scores
    y = rng.integers(0, 4, size=n).astype(np.int64)
    return (np.ascontiguousarray(mat), y)


def _rescand_njit(mat, y):
    """Sweep probe variant: score ``mat`` on the host exact njit batch plug-in MI (the path being gated against)."""
    from mlframe.feature_selection.filters.hermite_fe.shared import plugin_mi_classif_batch_njit as _plugin_mi_classif_batch_njit

    return selection_view(_plugin_mi_classif_batch_njit(mat, y, 20))


def _rescand_resident(mat, y):
    """Sweep probe variant: upload ``mat``/``y`` to the GPU and score via the resident plug-in MI kernel being gated."""
    import cupy as cp

    from . import hermite_fe as _hf  # noqa: F401 - full-init parent before sibling import
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    mat_gpu = cp.asarray(mat, dtype=cp.float64)  # Intentional float64: KTC probe device upload mirrors the gate's float64 host matrix
    y_gpu = cp.asarray(y, dtype=cp.int64)
    return selection_view(np.asarray(_plugin_mi_classif_batch_cuda_resident(mat_gpu, y_gpu, 20), dtype=np.float64))  # Intentional float64: comparison vector


def _run_rescand_sweep() -> list:
    """Time host njit MI vs the resident GPU plug-in MI across the (n, k) grid; faster EQUIVALENT wins per
    region. The two binning schemes (rank vs percentile-edge) differ only at ties -> selection-equivalent: the sweep
    compares what the gate reads from a scoring pass (the best candidates, ``selection_view``) at the strict tolerance, and ranks by WALL."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"njit": _rescand_njit, "resident": _rescand_resident}
    return cast(list, sweep_backend_grid(
        variants,
        {"n_samples": _RESCAND_SWEEP_N_SAMPLES, "k": _RESCAND_SWEEP_K},
        _make_rescand_inputs,
        reference="njit",
        repeats=3, equiv_rtol=1e-6, equiv_atol=1e-6,
    ))


def _rescand_fallback_choice(n_samples: int, k: int = 16) -> str:
    """Pre-sweep fallback: the host njit path (the resident path engages only when MEASURED faster)."""
    return "njit"


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _RESCAND_SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="fe_gate_resident_candidate_mi_crossover",
        variant_fns=(),  # GPU resident path covered by salt; njit is the reference
        tuner=_run_rescand_sweep,
        axes={"n_samples": list(_RESCAND_SWEEP_N_SAMPLES), "k": list(_RESCAND_SWEEP_K)},
        fallback=_rescand_fallback_choice,
        gpu_capable=True,
        salt=_RESCAND_SALT,
        cli_label="fe_gate_resident_candidate_mi_crossover",
    )
except Exception as e:
    logger.warning("fe_gate_resident_candidate_mi_crossover kernel_tuner registration failed: %s", e)
    _RESCAND_SPEC = None


# ----------------------------------------------------------------------------------------------------------------------
# Host or device-born tau grid of the conditional gate (kernel-tuning entry ``fe_gate_grid_backend``)
# ----------------------------------------------------------------------------------------------------------------------
_GRID_SWEEP_N_SAMPLES = [2_000, 10_000, 50_000, 200_000]
_GRID_SPECS = 12  # tau-grid specs of one flush (the gate flushes batches of this order: 27 specs x 17 taus at 100000 rows)
_GRID_TAUS = 17
_GRID_NBINS = 12
_GRID_SALT = 1


def _grid_inputs(dims: dict) -> tuple:
    """``(specs, y)``: ``_GRID_SPECS`` select-mode specs of ``_GRID_TAUS`` quantile thresholds each over three float32 operand columns of ``dims["n_samples"]`` rows, and a 3-class target the first operand drives."""
    n = int(dims["n_samples"])
    rng = np.random.default_rng(0)
    cols = [rng.normal(size=n).astype(np.float32) for _ in range(_GRID_SPECS + 2)]
    y = (np.digitize(cols[0], [-0.5, 0.5]) + (cols[1] > 0)) % 3
    specs = []
    for i in range(_GRID_SPECS):
        c, a, b = cols[(i + 2) % len(cols)], cols[i % len(cols)], cols[(i + 1) % len(cols)]
        specs.append(("select", (f"sweep_a{i}", f"sweep_b{i}", f"sweep_c{i}"), (c, a, b), np.quantile(c, np.linspace(0.1, 0.9, _GRID_TAUS)).astype(np.float32)))
    return (specs, y.astype(np.int64))


def _grid_host(specs: list, y: np.ndarray) -> np.ndarray:
    """The host path of the gate flush: the tau-grid blocks built on the host and scored by the batched rank-binned MI."""
    from ._conditional_gate_fe import _gate_build_feats, _gate_grid_mi

    big = np.ascontiguousarray(np.concatenate([_gate_build_feats(mode, operands, taus) for (mode, _ctup, operands, taus) in specs], axis=1))
    return np.asarray(_gate_grid_mi(big, y, _GRID_NBINS), dtype=np.float64)  # Intentional float64: comparison vector


def _grid_device(specs: list, y: np.ndarray) -> np.ndarray:
    """The device-born path: the grids built on the device from the resident operands (uploaded once, the cache keys are the spec's column names) and scored there, rank-binned like the host."""
    from ._resident_candidate_mi import gate_grid_mi_resident

    got = gate_grid_mi_resident(specs, y, _GRID_NBINS, rank_binning=True, y_min=int(y.min()), n_classes=int(y.max() - y.min()) + 1)
    if got is None:
        raise RuntimeError("device-born gate grid declined")
    return np.asarray(got, dtype=np.float64)  # Intentional float64: comparison vector


def _grid_sweep() -> list:
    """Time the host and the device-born tau grid over the row axis; they must agree to 1e-7 (measured 3e-9 to 7e-9 between the float32 host scan and the device grid on 10000 to 200000 rows, the two accumulate the plug-in sums in another order)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return cast(list, sweep_backend_grid({"host": _grid_host, "device": _grid_device}, {"n_samples": _GRID_SWEEP_N_SAMPLES}, _grid_inputs, reference="host", repeats=3, equiv_rtol=1e-7, equiv_atol=1e-7))


DEVICE_FALLBACK_ROWS = 10_000  # before any sweep: the device-born grid from the row count the first measurement put it ahead


def _grid_fallback_choice(n_samples: int) -> str:
    """Pre-sweep choice from the row mark."""
    return "device" if int(n_samples) >= DEVICE_FALLBACK_ROWS else "host"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner as _kernel_tuner

    _GRID_SPEC: Any = _kernel_tuner(
        kernel_name="fe_gate_grid_backend",
        variant_fns=(),  # both backends are timed by the sweep callable
        tuner=_grid_sweep,
        axes={"n_samples": list(_GRID_SWEEP_N_SAMPLES)},
        fallback=_grid_fallback_choice,
        gpu_capable=True,
        salt=_GRID_SALT,
        cli_label="fe_gate_grid_backend",
    )
except Exception as e:
    logger.warning("fe_gate_grid_backend kernel_tuner registration failed: %s", e)
    _GRID_SPEC = None


def gate_grid_use_device(n: int) -> bool:
    """Whether the tau grid of a conditional-gate flush over ``n`` rows is built and scored on the device (outside STRICT, which forces the device): the per-host tuning cache, else the row mark.
    ``MLFRAME_FE_GATE_GRID_DEVICE=0|1`` forces a side."""
    import os

    forced = os.environ.get("MLFRAME_FE_GATE_GRID_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _GRID_SPEC is None:
        return int(n) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_GRID_SPEC.choose(n_samples=int(n))) == "device"
    except Exception as exc:
        log_throttle(logger, "_resident_candidate_mi_ktc.gate_grid_use_device", logging.WARNING, "gate_grid_use_device: KTC choose() failed, using the row mark: %s", exc)
        return int(n) >= DEVICE_FALLBACK_ROWS
