"""KTC crossover for the resident-GPU batched pair-combo MI TABLE (iter17, 2026-06-23).

Gates ``_usability_pool_resident.score_pair_combos_table_resident`` (ALL pairs' combo columns into ONE
resident matrix, ONE batched bin+MI pass) against the per-pair host njit kernel
``_usability_njit_pool.score_pair_combos`` -> ``_pair_combo_mi_njit_table_parallel`` it replaces. Per
``feedback_use_kernel_tuning_cache_for_gpu`` the engage/skip decision is NOT a hardcoded threshold: a
``kernel_tuner`` sweeps both paths over an (n_rows, npairs, n_combos) grid and records, per region, which is
faster. The resident path engages only where MEASURED faster; otherwise the caller stays on the exact njit
per-pair table kernel.

WHY BATCHED-ACROSS-PAIRS (not per-pair): iter13/iter16 proved per-PAIR ``_pair_combo_mi_cupy`` is a 3x LOSS
(F2 100k 34.8s -> 102.5s) -- each pair pays a fresh operand H2D + tiny launch grid that swamps the ~1.0s CPU
kernel. Batching every pair's nc combos into one (npairs*nc, n) device matrix amortises launch overhead over
the WHOLE sweep, so the GPU only competes at LARGE total work (many pairs x large nc and/or large n / a
stronger card). On the dev GTX 1050 Ti the F2 pool is narrow (joint-MI prune -> few pairs, modest nc) and the
sweep is expected to keep it on CPU -- correct (njit is already ~1.0s there).

BIT-FAITHFUL: the resident table reuses the SAME bit-faithful GPU primitives the per-pair cupy twin uses
(rank-based binning + Miller-Madow MI), so it matches the njit table to ~6e-15 -- the equivalence tol is
tight (not the looser percentile-edge trade). CPU/no-cupy host: the sweep never runs, ``.choose()`` returns
"njit", and the caller takes the exact host per-pair path -- byte-for-byte unchanged.
"""
from __future__ import annotations

import numpy as np

# (n_rows, npairs, n_combos) grid. npairs spans the narrow joint-MI-pruned F2 pool (few pairs) up to a wide
# sweep (max_pairs~60); n_combos ~ |unary|^2*|binary| (the medium preset -> ~1734) down to a minimal preset.
# n_rows axis: GPU H2D/launch overhead amortises with n. Default fallback keeps the un-tuned host njit path.
_POOLRES_SWEEP_N = [10_000, 50_000, 200_000]
_POOLRES_SWEEP_NPAIRS = [4, 16, 60]
_POOLRES_SWEEP_NCOMBOS = [96, 578, 1734]
_POOLRES_SALT = 1


def pool_table_use_resident(n_rows: int, npairs: int, n_combos: int) -> bool:
    """Per-host engage decision for the resident-GPU batched pair-combo MI table, from the
    kernel_tuning_cache. Returns ``True`` (use the resident path) only on a measured-faster cache hit;
    ``False`` on a miss / no-cupy / lookup failure (caller stays on the exact host per-pair njit kernel).
    Each axis snaps to the nearest swept bucket. STRICT GPU mode (``MLFRAME_FE_GPU_STRICT=1``, diagnostic,
    default OFF) forces the resident path: it is bit-faithful to the njit table (~6e-15) -> selection-equivalent."""
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        if fe_gpu_strict_enabled():
            return True
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    if _POOLRES_SPEC is None:
        return False
    pb = min(_POOLRES_SWEEP_NPAIRS, key=lambda b: abs(b - int(npairs)))
    cb = min(_POOLRES_SWEEP_NCOMBOS, key=lambda b: abs(b - int(n_combos)))
    try:
        choice = _POOLRES_SPEC.choose(n_rows=int(n_rows), npairs=int(pb), n_combos=int(cb))
    except Exception:
        return False
    return bool(choice == "resident")


def _make_pooltable_inputs(dims: dict):
    """A workload shaped like a full pool table call: ``npairs`` operand pairs, the shared y codes/terms, and
    the op-code arrays -- the SAME arguments both paths consume (the njit reference is invoked per-pair, the
    resident path once over all pairs), so the crossover measured is the real table-build work."""
    n = int(dims["n_rows"])
    npairs = int(dims["npairs"])
    ncombos = int(dims["n_combos"])
    rng = np.random.default_rng(0)
    # derive a (nu, nb) op grid whose nu*nu*nb is closest to the requested n_combos (binary preset 'minimal'
    # is 6 ops; pick nu so nu^2*6 ~ ncombos, capped at the 17-op 'medium' unary set).
    nb = 6
    nu = max(1, min(17, int(round((ncombos / nb) ** 0.5))))
    ua_codes = list(range(nu))
    ub_codes = list(range(nu))
    bn_codes = list(range(nb))
    operands = []
    for p in range(npairs):
        x1 = np.ascontiguousarray(rng.standard_normal(n))
        x2 = np.ascontiguousarray(rng.exponential(1.2, n))
        operands.append((x1, x2))
    y = operands[0][0] * operands[0][0] / (operands[0][1] + 0.5) + 0.1 * rng.standard_normal(n)
    edges = np.quantile(y, np.linspace(0.0, 1.0, 11))
    y_codes = np.searchsorted(np.unique(edges)[1:-1], y, side="right").astype(np.int64)
    k_y = int(y_codes.max()) + 1
    cy = np.bincount(y_codes).astype(np.float64) / n
    h_y = float(-(cy[cy > 0] * np.log(cy[cy > 0])).sum())
    y_terms = (y_codes, h_y, k_y)
    nbins = 10
    return (operands, y_codes, y_terms, nbins, ua_codes, ub_codes, bn_codes)


def _pooltable_njit(operands, y_codes, y_terms, nbins, ua_codes, ub_codes, bn_codes):
    """Reference: the host per-pair njit table kernel, looped over every pair (what the resident path replaces)."""
    from ._usability_njit_pool import score_pair_combos

    rows = [score_pair_combos(x1, x2, y_codes, y_terms, nbins, ua_codes, ub_codes, bn_codes) for (x1, x2) in operands]
    return np.stack(rows, axis=0)


def _pooltable_resident(operands, y_codes, y_terms, nbins, ua_codes, ub_codes, bn_codes):
    from ._usability_pool_resident import score_pair_combos_table_resident

    out = score_pair_combos_table_resident(operands, y_codes, y_terms, nbins, ua_codes, ub_codes, bn_codes)
    if out is None:
        # no-cupy / device error in the sweep: return the njit reference so the sweep does not crash.
        return _pooltable_njit(operands, y_codes, y_terms, nbins, ua_codes, ub_codes, bn_codes)
    return out


def _run_pooltable_sweep() -> list:
    """Time host per-pair njit vs the resident batched table across the (n_rows, npairs, n_combos) grid;
    faster EQUIVALENT wins per region. Both paths use rank-based binning + Miller-Madow MI -> bit-faithful
    (~6e-15), so the equivalence tol is tight; the sweep ranks by WALL."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"njit": _pooltable_njit, "resident": _pooltable_resident}
    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        variants,
        {"n_rows": _POOLRES_SWEEP_N, "npairs": _POOLRES_SWEEP_NPAIRS, "n_combos": _POOLRES_SWEEP_NCOMBOS},
        _make_pooltable_inputs,
        reference="njit",
        repeats=3, equiv_rtol=1e-9, equiv_atol=1e-9,
    )


def _pooltable_fallback_choice(n_rows: int, npairs: int = 16, n_combos: int = 578) -> str:
    """Pre-sweep fallback: the host per-pair njit path (resident engages only when MEASURED faster)."""
    return "njit"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _POOLRES_SPEC = kernel_tuner(
        kernel_name="fe_usability_pool_combo_mi_table_resident_crossover",
        variant_fns=(),  # GPU resident path covered by salt; njit is the reference
        tuner=_run_pooltable_sweep,
        axes={"n_rows": list(_POOLRES_SWEEP_N), "npairs": list(_POOLRES_SWEEP_NPAIRS), "n_combos": list(_POOLRES_SWEEP_NCOMBOS)},
        fallback=_pooltable_fallback_choice,
        gpu_capable=True,
        salt=_POOLRES_SALT,
        cli_label="fe_usability_pool_combo_mi_table_resident_crossover",
    )
except Exception:
    _POOLRES_SPEC = None
