"""Backend dispatcher for per-group sequential-recursion FE kernels.

``bocpd_features`` and ``online_bayesian_linear_regression`` run an
inherently sequential recursion WITHIN each group, but groups (wells /
panels / series) are independent. The parallel axis is therefore
``prange`` across groups. Which backend wins depends on the live host
and on the workload shape ``(n_samples, n_groups)``:

* few groups (1-3) -> ``prange`` cannot saturate cores; serial njit wins
  (no thread spawn overhead).
* many groups (tens to thousands) -> parallel njit wins, scaling with
  core count.

Hardcoding the crossover is wrong on any machine other than the one it
was measured on, so the choice routes through the per-host
``pyutilz.performance.kernel_tuning.cache`` (same infra as
``joint_hist_batched`` / ``plugin_mi_classif_dispatch``). Until an
auto-tune sweep has populated the cache for the live host, a
measurement-backed fallback heuristic is used.

Override with ``MLFRAME_FE_RECURSION_BACKEND={serial,parallel}``.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Measurement-backed fallback crossovers (Intel host, 2026-05-27). The
# right crossover is PER KERNEL because per-group work differs by orders
# of magnitude:
#   * fe_bocpd: heavy per-row recursion (growing run-length posterior).
#     prange wins from a handful of groups -- measured 3.4x at 8 wells /
#     3.2k rows, 5.1x at 100 wells / 100k rows.
#   * fe_oblr: trivial per-row work (k=2 Sherman-Morrison rank-1 update).
#     Thread-spawn overhead dominates at small scale (0.69x at 8 groups /
#     3.2k rows) but the crossover is low: parallel measured 1.35x at 100
#     groups / 100k rows and 1.29x at 400 groups / 800k rows. Pick parallel
#     from ~16 groups + 50k rows; stay serial below to dodge the penalty.
# Hosts with different core counts / memory bandwidth refine these via the
# kernel_tuning_cache auto-tune sweep; these are only the pre-sweep guess.
_FALLBACK_THRESHOLDS: dict[str, tuple[int, int]] = {
    # kernel_name: (min_groups, min_samples) to pick parallel
    "fe_bocpd": (4, 2_000),
    "fe_oblr": (16, 50_000),
}
_DEFAULT_THRESHOLD = (8, 20_000)


def _env_override() -> str | None:
    forced = os.environ.get("MLFRAME_FE_RECURSION_BACKEND", "").strip().lower()
    if forced in ("serial", "parallel"):
        return forced
    return None


def _fallback_backend(kernel_name: str, n_samples: int, n_groups: int) -> str:
    """Per-kernel measurement-backed default used before the cache is
    populated. Parallel only when there are enough groups AND enough total
    work to beat the thread-spawn overhead for THIS kernel."""
    min_groups, min_samples = _FALLBACK_THRESHOLDS.get(kernel_name, _DEFAULT_THRESHOLD)
    if n_groups >= min_groups and n_samples >= min_samples:
        return "parallel"
    return "serial"


def dispatch_recursion_backend(
    kernel_name: str, n_samples: int, n_groups: int,
    *, run_auto_tune: bool = False,
) -> str:
    """Return ``"serial"`` or ``"parallel"`` for a per-group recursion kernel.

    ``kernel_name`` is e.g. ``"fe_bocpd"`` / ``"fe_oblr"`` so each kernel
    can carry its own measured crossover in the cache. Env override wins;
    then the kernel_tuning_cache; then the measurement-backed fallback.
    """
    # env override + the n_groups<=1 short-circuit stay FIRST (env must win even
    # at n_groups<=1, and a single group can never use prange) -- so env_key is
    # NOT delegated to get_or_tune below; it is handled here to preserve order.
    forced = _env_override()
    if forced is not None:
        return forced
    if n_groups <= 1:
        return "serial"
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
        from ._recursion_autotune import _run_sweep, recursion_code_version

        # Shared orchestrator: per-host cache (code-version checked) -> on-miss
        # sweep (only when run_auto_tune, once/process, cross-process locked) ->
        # measurement-backed fallback. Replaces the hand-rolled
        # lookup/miss/sweep/re-lookup dance.
        result = KernelTuningCache.load_or_create().get_or_tune(
            kernel_name,
            dims={"n_samples": n_samples, "n_groups": n_groups},
            tuner=(lambda: _run_sweep(kernel_name)) if run_auto_tune else (lambda: None),
            axes=["n_samples", "n_groups"],
            fallback={"backend_choice": _fallback_backend(kernel_name, n_samples, n_groups)},
            code_version=recursion_code_version(kernel_name),
            async_sweep=True,  # FIT-TIME: measure off the hot path (when run_auto_tune drives a real sweep)
        )
        backend = str((result or {}).get("backend_choice", "")) if not isinstance(result, str) else result
        if backend in ("serial", "parallel"):
            return backend
    except Exception as e:  # pyutilz missing / cache error -> fallback
        logger.debug("recursion dispatch get_or_tune failed: %s", e)
    return _fallback_backend(kernel_name, n_samples, n_groups)


__all__ = ["dispatch_recursion_backend"]
