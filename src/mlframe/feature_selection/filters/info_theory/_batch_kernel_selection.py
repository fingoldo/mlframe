"""Per-host choice between the two batched FE-candidate MI kernels (``batch_mi_with_noise_gate`` v1 and the fused v2).

Carved out of ``_batch_kernels.py`` (1k-LOC budget), which re-exports every name here.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, cast

import numpy as np

from ._batch_kernels import batch_mi_with_noise_gate, batch_mi_with_noise_gate_v2

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# ---- batched FE-candidate MI kernel selector (F2 fused observed-MI dispatch, 2026-06-22) -------------
# ``batch_mi_with_noise_gate_v2`` fuses the per-column dense-code write with the observed-MI joint
# accumulation (one n-row pass instead of two), measured 1.18-1.21x faster than the original kernel at the
# canonical 30k-subsample K~3888 chunk and BIT-IDENTICAL (maxdiff 0.0 on the observed MI AND the
# noise-gated output, both npermutations=0 and =25). It is structurally never slower (it removes a full
# strided column re-read), so it is the default. Per the repo "keep all kernel versions; dispatch by
# size via kernel_tuning_cache" rule the ORIGINAL kernel is retained and the choice is routed per-host
# through the KTC (axes n_rows x n_cols); the v2 fused path is the measurement-backed fallback for an
# un-tuned host (the safe faster default here). ``MLFRAME_BATCH_MI_KERNEL=v1|v2`` force-overrides.
_BATCH_MI_KERNEL_CODE_VERSION = "batch_mi_noise_gate_kernel-v2-fused-2026-06-22"


def _batch_mi_kernel_fallback_choice(n_rows: int, n_cols: int) -> str:
    """Pre-sweep default kernel for the batched FE-candidate MI: the fused v2 (measured uniformly faster
    + bit-identical). The KTC sweep can later refine per-host if a pathological size ever regresses."""
    return "v2"


def select_batch_mi_kernel(n_rows: int, n_cols: int) -> Callable:
    """Return the batched FE-candidate MI kernel (``batch_mi_with_noise_gate`` v1 or the fused v2) for this
    (n_rows, n_cols) on this host. Routed through ``pyutilz.system.kernel_tuning_cache`` (per-host cache,
    code-version checked, async background sweep, measurement-backed fallback) - NOT a hardcoded threshold.
    ``MLFRAME_BATCH_MI_KERNEL`` env (``v1``/``v2``) force-overrides for A/B + safety. Defaults to v2 (the
    fused, measured-faster, bit-identical kernel) on any miss/failure so an un-tuned host gets the win."""
    _env = os.environ.get("MLFRAME_BATCH_MI_KERNEL", "").strip().lower()
    if _env == "v1":
        return cast(Callable, batch_mi_with_noise_gate)
    if _env == "v2":
        return cast(Callable, batch_mi_with_noise_gate_v2)
    choice = "v2"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        res = KernelTuningCache.load_or_create().get_or_tune(
            "batch_mi_fe_kernel",
            dims={"n_rows": int(n_rows), "n_cols": int(n_cols)},
            tuner=_run_batch_mi_kernel_sweep,
            axes=["n_rows", "n_cols"],
            fallback={"kernel_choice": _batch_mi_kernel_fallback_choice(int(n_rows), int(n_cols))},
            code_version=_BATCH_MI_KERNEL_CODE_VERSION,
            async_sweep=True,
        )
        if isinstance(res, str):
            choice = res
        elif res:
            choice = str(res.get("kernel_choice", "v2"))
    except Exception as exc:
        # a genuine kernel-selection bug silently degrades
        # every fit to the hardcoded v2 default with no diagnostic trail.
        logger.debug("mrmr: batch-MI kernel selection failed; defaulting to v2: %r", exc, exc_info=True)
        choice = "v2"
    return cast(Callable, batch_mi_with_noise_gate if choice == "v1" else batch_mi_with_noise_gate_v2)


def _run_batch_mi_kernel_sweep():
    """Per-host v1-vs-v2 crossover sweep for the batched FE-candidate MI kernel -> kernel_choice regions
    keyed on (n_rows, n_cols). Both kernels are bit-identical so equivalence holds at a tight tol; the
    sweep ranks by wall only. Returns [] when the benchmarking helper is unavailable (-> fallback v2)."""
    try:
        from pyutilz.dev.benchmarking import sweep_backend_grid
        from ..discretization import discretize_2d_quantile_batch
    except Exception as exc:
        # Distinguish "benchmarking helper genuinely
        # unavailable" from a real import bug - both silently fell back to v2 with zero trace before.
        logger.debug("mrmr: batch-MI kernel sweep unavailable (falling back to v2 default): %r", exc, exc_info=True)
        return []

    def _make_inputs(dims):
        """Build a synthetic (discretized candidates, y-classes, y-frequencies) fixture at the requested (n_rows, n_cols) grid point, using a shared a**2/b-derived base signal so v1/v2 see realistic non-degenerate MI."""
        n = int(dims["n_rows"]); K = int(dims["n_cols"]); nbins = 10
        rng = np.random.default_rng(0)
        a = rng.uniform(0.1, 1.1, n); b = rng.uniform(0.1, 1.1, n); base = a ** 2 / b
        cand = np.empty((n, K), dtype=np.float32)
        for k in range(K):
            cand[:, k] = (base * (1.0 + 0.01 * rng.standard_normal(n)) + 0.001 * k).astype(np.float32)
        np.nan_to_num(cand, copy=False)
        disc = discretize_2d_quantile_batch(cand, n_bins=nbins, dtype=np.int8, assume_finite=True)
        y = a**2 / b
        edges = np.quantile(y, np.linspace(0, 1, nbins + 1)[1:-1])
        yc = np.searchsorted(edges, y).astype(np.int64)
        fy = np.bincount(yc, minlength=int(yc.max()) + 1).astype(np.float64) / n
        return (disc, np.full(K, nbins, dtype=np.int64), yc, fy)

    def _call(kernel):
        """Bind a fixed permutation-MI call signature to ``kernel`` (v1 or v2) so ``sweep_backend_grid`` can invoke both with identical positional data args."""
        def _f(disc, fn, yc, fy):
            """Invoke the bound kernel on one sweep-generated fixture with a fixed permutation/seed/threshold configuration."""
            return kernel(
                disc_2d=disc, factors_nbins=fn, classes_y=yc, classes_y_safe=yc, freqs_y=fy,
                npermutations=25, base_seed=np.uint64(0), min_nonzero_confidence=0.0, use_su=False,
                dtype=np.int32, classes_dtype=np.int8,
            )
        return _f

    return cast(list, sweep_backend_grid(
        {"v1": _call(batch_mi_with_noise_gate), "v2": _call(batch_mi_with_noise_gate_v2)},
        {"n_rows": [30_000, 100_000], "n_cols": [600, 3888]},
        _make_inputs,
        reference="v1", repeats=3, equiv_rtol=1e-12, equiv_atol=1e-15,
        decision_key="kernel_choice",
    ))
