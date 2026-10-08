"""Per-host CPU-vs-GPU width gates for ShapProxiedFS clustering, from the kernel_tuning_cache.

Two dispatches used to carry a hardcoded feature-count threshold each (``GPU_MIN_FEATURES`` = 2000 for the dense Pearson edge scan, 500 for the pairwise
SU scan). The crossover depends on the host (GPU generation, CPU core count, cold cupy/NVRTC load), so each is now a measured per-host sweep through
``pyutilz.performance.kernel_tuning``: the CPU and GPU edge scans are timed over a (features, samples) grid and the faster one wins per region. The old
constants stay as the pre-sweep / tuner-failure fallback only.

The sweep times WARM steady-state calls: the one-time cupy/CUDA load and NVRTC compile of a cold process is not part of the measurement, so a host
where the GPU wins only warm can still lose on a process that pays the cold start for a single fit. Callers that must avoid it pass ``use_gpu=False`` or
an explicit ``gpu_min_features``.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

logger = logging.getLogger(__name__)

# Fallback widths, used until a sweep has run on this host (or when the tuner is unavailable). Dev-box calibration: the cold cupy/CUDA load + NVRTC compile
# (~17 s) dwarfs the dense CPU ``Z.T @ Z`` below f~2000 (~0.3 s at f=704/n=10000); the SU GPU launch + one-hot pack overhead dwarfs the parallel CPU kernel
# below f~500 (~0.14 s at f=500 / n_bins=10 / n=1500).
DENSE_FALLBACK_MIN_FEATURES = 2000
SU_FALLBACK_MIN_FEATURES = 500

_DENSE_SWEEP_F = [500, 2000, 4000]
_DENSE_SWEEP_N = [10_000]
_SU_SWEEP_F = [200, 500, 1500]
_SU_SWEEP_N = [2_000]
_SWEEP_SALT = 1
_SWEEP_THRESHOLD_DENSE = 0.05
_SWEEP_THRESHOLD_SU = 0.01
_SWEEP_N_BINS = 10


def _make_dense_inputs(dims: dict) -> tuple:
    """A standardized float32 ``(n, f)`` matrix with some correlated columns, so a real number of edges clears the sweep threshold."""
    n, f = int(dims["n"]), int(dims["f"])
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((n, f)).astype(np.float32)
    Z[:, 1::7] += 0.5 * Z[:, 0::7][:, : Z[:, 1::7].shape[1]]
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    return (np.ascontiguousarray(Z, dtype=np.float32),)


def _dense_edges_cpu_count(Z: np.ndarray) -> np.float64:
    """Edge count of the dense CPU path (the inline branch of ``cluster_correlated_features``)."""
    n, f = Z.shape
    C = (Z.T @ Z) / np.float32(n)
    iu = np.triu_indices(f, k=1)
    return np.float64(np.count_nonzero(np.abs(C[iu]) > np.float32(_SWEEP_THRESHOLD_DENSE)))


def _dense_edges_gpu_count(Z: np.ndarray) -> np.float64:
    """Edge count of the dense GPU path."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster import _edges_dense_gpu

    edges = _edges_dense_gpu(Z, _SWEEP_THRESHOLD_DENSE, 10**12)
    return np.float64(0 if edges is None else edges[0].shape[0])


def _run_dense_sweep() -> list:
    """CPU-vs-GPU dense Pearson edge scan over the (f, n) grid -> kernel_tuning_cache regions."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return cast(
        list,
        sweep_backend_grid(
            {"cpu": _dense_edges_cpu_count, "gpu": _dense_edges_gpu_count},
            {"f": _DENSE_SWEEP_F, "n": _DENSE_SWEEP_N},
            _make_dense_inputs,
            reference="cpu",
            repeats=3,
            equiv_rtol=0.0,
            equiv_atol=0.0,
        ),
    )


def _dense_fallback_choice(f: int, n: int = 0) -> str:
    """Pre-sweep fallback: GPU from the dev-box-calibrated width."""
    return "gpu" if int(f) >= DENSE_FALLBACK_MIN_FEATURES else "cpu"


def _make_su_inputs(dims: dict) -> tuple:
    """Packed SU kernel inputs for ``f`` binned columns over ``n`` samples, with some dependent column pairs."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import _setup_su_kernel_inputs

    n, f = int(dims["n"]), int(dims["f"])
    rng = np.random.default_rng(0)
    cols = [rng.integers(0, _SWEEP_N_BINS, size=n).astype(np.int32) for _ in range(f)]
    for j in range(1, f, 5):
        cols[j] = ((cols[j - 1] + rng.integers(0, 2, size=n)) % _SWEEP_N_BINS).astype(np.int32)
    packed = _setup_su_kernel_inputs(cols, None)
    assert packed is not None
    return packed


def _su_edges_cpu_count(bins_packed, nbins, freqs_packed, freqs_offsets, h_marginals, constant_mask) -> np.float64:
    """Edge count of the CPU pairwise SU scan (bitmap kernel when its gates pass, else the scalar prange kernel)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import _run_cpu_pairwise_su

    flags = _run_cpu_pairwise_su(
        bins_packed, nbins, freqs_packed, freqs_offsets, h_marginals, constant_mask, _SWEEP_THRESHOLD_SU,
        use_bitmap=True, bitmap_min_features=None, bitmap_max_n_bins=None,
    )
    return np.float64(np.count_nonzero(flags))


def _su_edges_gpu_count(bins_packed, nbins, freqs_packed, freqs_offsets, h_marginals, constant_mask) -> np.float64:
    """Edge count of the GPU pairwise SU scan."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import _pairwise_su_edges_gpu

    return np.float64(np.count_nonzero(_pairwise_su_edges_gpu(bins_packed, nbins, h_marginals, constant_mask, _SWEEP_THRESHOLD_SU)))


def _run_su_sweep() -> list:
    """CPU-vs-GPU pairwise SU scan over the (f, n) grid -> kernel_tuning_cache regions."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return cast(
        list,
        sweep_backend_grid(
            {"cpu": _su_edges_cpu_count, "gpu": _su_edges_gpu_count},
            {"f": _SU_SWEEP_F, "n": _SU_SWEEP_N},
            _make_su_inputs,
            reference="cpu",
            repeats=3,
            equiv_rtol=0.0,
            equiv_atol=0.0,
        ),
    )


def _su_fallback_choice(f: int, n: int = 0) -> str:
    """Pre-sweep fallback: GPU from the dev-box-calibrated width."""
    return "gpu" if int(f) >= SU_FALLBACK_MIN_FEATURES else "cpu"


_DENSE_SPEC = kernel_tuner(
    kernel_name="shap_proxy_cluster_dense_gpu",
    variant_fns=(_dense_edges_cpu_count, _dense_edges_gpu_count),
    tuner=_run_dense_sweep,
    axes={"f": _DENSE_SWEEP_F, "n": _DENSE_SWEEP_N},
    fallback=_dense_fallback_choice,
    gpu_capable=True,
    salt=_SWEEP_SALT,
    cli_label="shap_proxy_cluster_dense_gpu",
)

_SU_SPEC = kernel_tuner(
    kernel_name="shap_proxy_cluster_su_gpu",
    variant_fns=(_su_edges_cpu_count, _su_edges_gpu_count),
    tuner=_run_su_sweep,
    axes={"f": _SU_SWEEP_F, "n": _SU_SWEEP_N},
    fallback=_su_fallback_choice,
    gpu_capable=True,
    salt=_SWEEP_SALT,
    cli_label="shap_proxy_cluster_su_gpu",
)


def dense_gpu_pays_off(n_features: int, n_samples: int) -> bool:
    """Whether the dense Pearson edge scan should run on the GPU on this host at this size (measured; falls back to the width constant)."""
    try:
        choice = _DENSE_SPEC.choose(f=int(n_features), n=int(n_samples))
    except Exception as e:  # best-effort: the choice only routes a speed-equivalent CPU/GPU scan; both give the same edges
        logger.debug("shap_proxy_cluster_dense_gpu choose() failed, using the width fallback: %s", e)
        choice = _dense_fallback_choice(n_features)
    return bool(choice == "gpu")


def su_gpu_pays_off(n_features: int, n_samples: int) -> bool:
    """Whether the pairwise SU scan should run on the GPU on this host at this size (measured; falls back to the width constant)."""
    try:
        choice = _SU_SPEC.choose(f=int(n_features), n=int(n_samples))
    except Exception as e:  # best-effort: the choice only routes a speed-equivalent CPU/GPU scan; both give the same edges
        logger.debug("shap_proxy_cluster_su_gpu choose() failed, using the width fallback: %s", e)
        choice = _su_fallback_choice(n_features)
    return bool(choice == "gpu")
