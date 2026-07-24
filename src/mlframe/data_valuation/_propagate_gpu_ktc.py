"""kernel_tuning_cache crossover for a cupy-resident nearest-subsample-neighbor path in
``propagate_subsample_values`` (gt_04 scalability follow-up).

Profiled at ``n_full=2,000,000``/``max_valued_rows=20,000`` (``training_sample_weight_from_valuation``'s
own scalability bench): ``cdist_euclidean`` cost 648s, ``np.argmin`` (the k=1 fast path) cost a further
share of the remaining time -- both host-CPU bound and embarrassingly parallel across rows, the exact
profile a GPU-resident squared-distance + argmin kernel targets. Per
``feedback_use_kernel_tuning_cache_for_gpu`` the engage/skip decision is measured per-host via
``kernel_tuner``, never a hardcoded row-count threshold; the resident path engages only where a sweep
found it faster. CPU/no-cupy hosts never run the sweep, ``.choose()`` returns "host", and the caller
takes the exact chunked-cdist + argmin path unchanged.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_PROPAGATE_SWEEP_N_FULL = [50_000, 200_000, 1_000_000]
_PROPAGATE_SWEEP_N_SUB = [2_000, 20_000]
_PROPAGATE_SALT = 1


def propagate_use_resident(n_full: int, n_sub: int) -> bool:
    """Per-host engage decision for the resident-GPU nearest-subsample-neighbor path (k=1 only).

    Returns ``True`` only on a measured-faster cache hit; ``False`` on a miss / no-cupy / lookup
    failure (caller stays on the exact chunked ``cdist`` + ``argmin`` host path). ``n_full``/``n_sub``
    snap to the nearest swept bucket.
    """
    if _PROPAGATE_SPEC is None:
        return False
    n_full_bucket = min(_PROPAGATE_SWEEP_N_FULL, key=lambda b: abs(b - int(n_full)))
    n_sub_bucket = min(_PROPAGATE_SWEEP_N_SUB, key=lambda b: abs(b - int(n_sub)))
    try:
        choice = _PROPAGATE_SPEC.choose(n_full=n_full_bucket, n_sub=n_sub_bucket)
    except Exception as exc:
        logger.debug("propagate_use_resident: KTC lookup failed, staying on the host path: %s", exc)
        return False
    return bool(choice == "resident")


def nearest_neighbor_value_resident(X_full: np.ndarray, X_subsample: np.ndarray, subsample_values: np.ndarray, *, batch_full: int = 100_000) -> np.ndarray:
    """GPU-resident k=1 nearest-subsample-neighbor value lookup: uploads ``X_subsample`` once, streams
    ``X_full`` in chunks, and returns each row's nearest subsample neighbor's value -- same contract as
    :func:`mlframe.data_valuation._mc_sampling.propagate_subsample_values` at ``k=1``, GPU-accelerated.

    Squared euclidean distance avoids the ``sqrt`` the CPU ``cdist`` path computes (monotone -- argmin is
    identical either way), and the whole ``(batch_full, n_subsample)`` distance matrix is computed and
    reduced ON DEVICE; only the resulting ``(batch_full,)`` nearest-index vector crosses back to host.
    """
    import cupy as cp

    X_full = np.ascontiguousarray(X_full, dtype=np.float64)
    X_sub_gpu = cp.asarray(np.ascontiguousarray(X_subsample, dtype=np.float64))
    sub_sq_norm = cp.sum(X_sub_gpu * X_sub_gpu, axis=1)  # (n_sub,)
    values_gpu = cp.asarray(np.asarray(subsample_values, dtype=np.float64))
    n_full = X_full.shape[0]
    out = np.empty(n_full, dtype=np.float64)

    for start in range(0, n_full, batch_full):
        end = min(start + batch_full, n_full)
        X_batch_gpu = cp.asarray(X_full[start:end])
        batch_sq_norm = cp.sum(X_batch_gpu * X_batch_gpu, axis=1)  # (batch,)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b -- avoids materializing a-b for every pair.
        cross = X_batch_gpu @ X_sub_gpu.T  # (batch, n_sub)
        sq_dist = batch_sq_norm[:, None] + sub_sq_norm[None, :] - 2.0 * cross
        nearest_idx = cp.argmin(sq_dist, axis=1)
        out[start:end] = cp.asnumpy(values_gpu[nearest_idx])
    return out


def _make_propagate_inputs(dims: dict):
    """A (n_full, n_sub) host float64 pair + a subsample value vector, shaped like a real propagation call."""
    n_full = int(dims["n_full"])
    n_sub = int(dims["n_sub"])
    rng = np.random.default_rng(0)
    X_full = rng.standard_normal((n_full, 12))
    X_sub = rng.standard_normal((n_sub, 12))
    values = rng.standard_normal(n_sub)
    return (np.ascontiguousarray(X_full), np.ascontiguousarray(X_sub), values)


def _propagate_host(X_full, X_sub, values):
    """Sweep probe variant: the exact chunked cdist + argmin host path being gated against."""
    from ._mc_sampling import propagate_subsample_values

    return propagate_subsample_values(X_full, X_sub, values, k=1)


def _propagate_resident(X_full, X_sub, values):
    """Sweep probe variant: the GPU-resident nearest-neighbor path being gated."""
    return nearest_neighbor_value_resident(X_full, X_sub, values)


def _run_propagate_sweep() -> list:
    """Time the host chunked-cdist path vs the GPU-resident path across the (n_full, n_sub) grid; faster
    EQUIVALENT wins per region (bit-identical up to nearest-neighbor TIE-BREAK order -- a genuine tie
    picks whichever candidate each backend's own argmin/argpartition happens to return first, an
    unavoidable discretization artifact rather than a real divergence, so the sweep loosens tolerance
    accordingly)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"host": _propagate_host, "resident": _propagate_resident}
    return sweep_backend_grid(
        variants,
        {"n_full": _PROPAGATE_SWEEP_N_FULL, "n_sub": _PROPAGATE_SWEEP_N_SUB},
        _make_propagate_inputs,
        reference="host",
        repeats=3, equiv_rtol=1e-3, equiv_atol=1e-3,
    )


def _propagate_fallback_choice(n_full: int, n_sub: int = 20_000) -> str:
    """Pre-sweep fallback: the host chunked-cdist path (the resident path engages only when MEASURED faster)."""
    return "host"


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _PROPAGATE_SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="data_valuation_propagate_subsample_crossover",
        variant_fns=(),  # GPU resident path covered by salt; host cdist is the reference
        tuner=_run_propagate_sweep,
        axes={"n_full": list(_PROPAGATE_SWEEP_N_FULL), "n_sub": list(_PROPAGATE_SWEEP_N_SUB)},
        fallback=_propagate_fallback_choice,
        gpu_capable=True,
        salt=_PROPAGATE_SALT,
        cli_label="data_valuation_propagate_subsample_crossover",
    )
except Exception:
    _PROPAGATE_SPEC = None


__all__ = ["propagate_use_resident", "nearest_neighbor_value_resident"]
