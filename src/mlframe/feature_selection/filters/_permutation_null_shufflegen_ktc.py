"""KTC crossover for the maxT permutation-null SHUFFLE-GENERATION backend (large-n lever, 2026-06-24).

The order-1 maxT floor (``_permutation_null.pooled_permutation_null_gain_floor``) pre-generates ``nperm``
full-length target shuffles before its histogram+MI kernel runs. Profiling the F2 fit on a GTX 1050 Ti box
showed this GENERATION -- not the GPU/njit kernel -- is the dominant LARGE-N cost: the sequential numpy
``rng.shuffle`` loop scales O(nperm * n) and measured ~2.2s at n=600k / nperm=200 versus only ~0.29s for the
njit histogram kernel it feeds (~88% of the floor wall, the single biggest large-n-specific reducible cost).

The K shuffles are mutually independent, so ``_permutation_null._gen_target_shuffles_par_njit`` parallelises
the Fisher-Yates across permutations (per-row seed -> thread-count-independent, reproducible). Bench (8
threads): 2.69s -> 1.00s at n=600k (2.7x), beating even the best GPU argsort-keys gen (1.4s chunked) with no
GPU contention / OOM on a small card. The numba stream is a DIFFERENT (but valid uniform) draw sequence than
numpy's, so the resulting floor is statistically equivalent but NOT byte-identical -- per
``feedback_use_kernel_tuning_cache_for_gpu`` / ``feedback_fastest_default_with_dispatch`` the engage/skip
decision is NOT a hardcoded threshold: a ``kernel_tuner`` sweeps both backends over an (n, nperm) grid and
records, per region, which is faster. The numba path engages ONLY where MEASURED faster (large n); otherwise
the floor stays on the exact sequential numpy stream -- which keeps the small-n canonical/F2 suite (N<=100k)
byte-stable in selection (the crossover sits well above those sizes, so they never take the numba branch).

Equivalence: the two backends produce DIFFERENT permutations but each row is a true permutation of the same
target, so the column-wise value multiset (``np.sort`` of every row) is IDENTICAL between backends -- the
sweep ranks the two by WALL with equivalence checked on that permutation-invariant signature (loose tol).

CPU/no-numba host or sweep failure: ``.choose()`` returns "numpy" and the caller takes the exact legacy path.
"""
from __future__ import annotations

import numpy as np

# (n, nperm) grid. n spans the small canonical sizes (where numpy wins / the suite must stay byte-stable) up
# to the large-n regime where the parallel Fisher-Yates amortises its JIT + thread fan-out. nperm spans the
# legacy 25 and the default 200. Default fallback keeps the legacy numpy stream (numba is opt-in-by-measure).
_SHUFFLEGEN_SWEEP_N = [50_000, 100_000, 300_000, 600_000]
_SHUFFLEGEN_SWEEP_NPERM = [25, 200]
_SHUFFLEGEN_SALT = 1


def shufflegen_use_numba(n: int, nperm: int) -> bool:
    """Per-host engage decision for the parallel-njit shuffle generator, from the kernel_tuning_cache.

    Returns ``True`` (use the parallel Fisher-Yates) only on a measured-faster cache hit; ``False`` on a miss /
    no-numba / lookup failure (caller stays on the exact sequential numpy stream). Each axis snaps to the
    nearest swept bucket."""
    import os as _os
    _force = _os.environ.get("MLFRAME_FDR_SHUFFLEGEN", "")  # "numba" / "numpy" force (A/B + escape hatch)
    if _force == "numba":
        return True
    if _force == "numpy":
        return False
    if _SHUFFLEGEN_SPEC is None:
        return False
    pb = min(_SHUFFLEGEN_SWEEP_NPERM, key=lambda b: abs(b - int(nperm)))
    try:
        choice = _SHUFFLEGEN_SPEC.choose(n_samples=int(n), nperm=int(pb))
    except Exception:
        return False
    return choice == "numba"


def _make_shufflegen_inputs(dims: dict):
    """An (n, nperm) workload shaped like a maxT floor's gen step: the int32 target codes + the perm count."""
    n = int(dims["n_samples"])
    nperm = int(dims["nperm"])
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, size=n).astype(np.int32)
    return (y, nperm)


def _shufflegen_numpy(y, nperm):
    rng = np.random.default_rng(12345)
    yp = y.copy()
    out = np.empty((int(nperm), y.shape[0]), dtype=y.dtype)
    for k in range(int(nperm)):
        rng.shuffle(yp)
        out[k] = yp
    # permutation-invariant equivalence signature: per-row sorted multiset (identical across backends).
    return np.sort(out, axis=1)


def _shufflegen_numba(y, nperm):
    from ._permutation_null import _gen_target_shuffles_par_njit

    out = _gen_target_shuffles_par_njit(y, int(nperm), np.int64(777))
    return np.sort(out, axis=1)


def _shufflegen_gpu(y, nperm):
    """Device argsort-keys gen brought back to host for the sweep's permutation-invariant equivalence check.
    Raises (caught by the sweep -> marked unavailable/slow) on a card without the VRAM for the (nperm, n)
    key+order buffers, so a small-VRAM host never selects it."""
    import cupy as cp

    from ._permutation_null_resident import gen_target_shuffles_cupy

    out = gen_target_shuffles_cupy(y, int(nperm), y.dtype, 777)
    if out is None:
        raise RuntimeError("gpu shufflegen unavailable")
    return np.sort(cp.asnumpy(out), axis=1)


def shufflegen_use_gpu(n: int, nperm: int) -> bool:
    """Per-host engage decision for the DEVICE-resident argsort-keys shuffle generator, from the
    kernel_tuning_cache. ``True`` only on a measured-faster cache hit (a big-VRAM host where the device gen
    + the resident floor beat the host gen + H2D); ``False`` on a miss / no-cupy / lookup failure (caller
    uses the host gen). ``MLFRAME_FDR_SHUFFLEGEN=gpu`` forces it for an A/B."""
    import os as _os

    _force = _os.environ.get("MLFRAME_FDR_SHUFFLEGEN", "")
    if _force == "gpu":
        return True
    if _force in ("numba", "numpy"):
        return False
    # STRICT GPU-RESIDENT (no KTC on this path): the residency contract is 100% device data+kernels, wall-loss
    # on a weak card ACCEPTED. ``permnull_use_resident`` already forces the resident floor under STRICT; the
    # shuffle-gen must match or the (nperm,n) y_perms matrix is host-generated then uploaded (a bulk H2D that
    # violates residency). The device argsort-keys gen yields a statistically-equivalent uniform null (each row
    # a true permutation) -> selection-equivalent (not byte-identical); the resident gen self-guards to a host
    # fallback if VRAM is short, so this never crashes. Verified F2 selection unchanged.
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        if fe_gpu_strict_enabled():
            return True
    except Exception:
        pass
    if _SHUFFLEGEN_SPEC is None:
        return False
    pb = min(_SHUFFLEGEN_SWEEP_NPERM, key=lambda b: abs(b - int(nperm)))
    try:
        choice = _SHUFFLEGEN_SPEC.choose(n_samples=int(n), nperm=int(pb))
    except Exception:
        return False
    return choice == "gpu"


def _run_shufflegen_sweep() -> list:
    """Time the sequential numpy stream vs the parallel njit Fisher-Yates across the (n, nperm) grid; faster
    EQUIVALENT wins per region. Both yield true uniform permutations, so the per-row sorted multiset is
    IDENTICAL between backends -> equivalence holds exactly and the sweep ranks by WALL."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"numpy": _shufflegen_numpy, "numba": _shufflegen_numba, "gpu": _shufflegen_gpu}
    return sweep_backend_grid(
        variants,
        {"n_samples": _SHUFFLEGEN_SWEEP_N, "nperm": _SHUFFLEGEN_SWEEP_NPERM},
        _make_shufflegen_inputs,
        reference="numpy",
        repeats=3, equiv_rtol=0.0, equiv_atol=0.0,
    )


def _shufflegen_fallback_choice(n_samples: int, nperm: int = 200) -> str:
    """Pre-sweep fallback: the legacy sequential numpy stream (numba engages only when MEASURED faster)."""
    return "numpy"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SHUFFLEGEN_SPEC = kernel_tuner(
        kernel_name="fe_maxt_permnull_shufflegen_backend",
        variant_fns=(),
        tuner=_run_shufflegen_sweep,
        axes={"n_samples": list(_SHUFFLEGEN_SWEEP_N), "nperm": list(_SHUFFLEGEN_SWEEP_NPERM)},
        fallback=_shufflegen_fallback_choice,
        gpu_capable=True,
        salt=_SHUFFLEGEN_SALT,
        cli_label="fe_maxt_permnull_shufflegen_backend",
    )
except Exception:
    _SHUFFLEGEN_SPEC = None
