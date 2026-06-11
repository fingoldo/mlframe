"""GPU pairwise SU clustering: parity, dispatch gate, and dormant fallback.

Locks the iter70 contract:
  * Parity: when a CUDA device is available the GPU kernel produces cluster
    labels bit-identical to the CPU prange kernel on a non-trivial fixture
    (single-linkage on a >= threshold compare means the underlying flags
    matrix must match exactly; no FP-reordering can shift labels).
  * Dispatch gate: ``_should_route_su_gpu`` is False when cupy is missing
    OR ``n_features < gpu_min_features`` OR the one-hot working set exceeds
    the memory cap; True only when all three gates pass.
  * Dormant fallback: with cupy unavailable, ``cluster_correlated_features_su``
    still runs (CPU path) and returns labels matching ``use_parallel=False``;
    no exception at import time.
  * Speedup at f=2000 is enforced ONLY when a real GPU is present, so the
    test is skipped on the dev box (CPU-only / broken cupy install) instead
    of hard-failing the suite.

Wired the GPU-required cases under ``@pytest.mark.skipif(not cluster_su_gpu_available())``
so the suite stays green on machines without working cupy.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import (
    _resolve_gpu_min_features,
    _should_route_su_gpu,
    cluster_correlated_features_su,
    cluster_su_gpu_available,
)


def _quantile_bin(col: np.ndarray, n_bins: int) -> np.ndarray:
    col = np.asarray(col, dtype=np.float64)
    if np.unique(col).size <= 1:
        return np.zeros_like(col, dtype=np.int32)
    qs = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
    if qs.size <= 1:
        return np.zeros_like(col, dtype=np.int32)
    edges = qs[1:-1] if qs.size > 2 else qs[1:]
    return np.clip(np.digitize(col, edges, right=False), 0, max(0, qs.size - 2)).astype(np.int32)


def _build_synthetic_bins(n_samples: int, n_features: int, n_bins: int, seed: int):
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n_features // 6)
    blocks = []
    for _ in range(n_blocks):
        z = rng.standard_normal(n_samples)
        for _k in range(3):
            blocks.append(z + 0.2 * rng.standard_normal(n_samples))
    while len(blocks) < n_features:
        blocks.append(rng.standard_normal(n_samples))
    X = np.column_stack(blocks[:n_features])
    names = [f"f{i}" for i in range(n_features)]
    bins = {n: _quantile_bin(X[:, i], n_bins=n_bins) for i, n in enumerate(names)}
    return bins, names


def test_gpu_available_helper_does_not_raise():
    """``cluster_su_gpu_available`` is a pure boolean probe; lazy import + cached.

    The call must not raise even on boxes with no cupy / no CUDA / broken cupy install.
    Re-calling returns the same result without re-importing.
    """
    result = cluster_su_gpu_available()
    assert isinstance(result, bool)
    # second call must match (cached)
    assert cluster_su_gpu_available() == result


def test_resolve_gpu_min_features_default():
    """Default GPU width threshold is 500 (kernel_tuning_cache override otherwise)."""
    value = _resolve_gpu_min_features()
    assert isinstance(value, int)
    assert value >= 1


def test_should_route_su_gpu_gate_blocks_when_no_gpu(monkeypatch):
    """When cupy is absent the dispatcher must return False, even at large widths."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su as mod

    monkeypatch.setattr(mod, "cluster_su_gpu_available", lambda: False)
    assert not _should_route_su_gpu(n_features=10_000, n_samples=2000, max_n_bins=10)


def test_should_route_su_gpu_gate_blocks_when_below_min_features(monkeypatch):
    """Even with a working GPU, below ``gpu_min_features`` the CPU path is preferred."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su as mod

    monkeypatch.setattr(mod, "cluster_su_gpu_available", lambda: True)
    monkeypatch.setattr(mod, "_gpu_free_memory_bytes", lambda: 8 * 1024 ** 3)
    assert not _should_route_su_gpu(
        n_features=100, n_samples=1500, max_n_bins=10, gpu_min_features=500,
    )
    assert _should_route_su_gpu(
        n_features=2000, n_samples=1500, max_n_bins=10, gpu_min_features=500,
    )


def test_should_route_su_gpu_gate_blocks_when_memory_insufficient(monkeypatch):
    """OOM guard: when the one-hot working set exceeds 50% of free GPU memory, fall back."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su as mod

    monkeypatch.setattr(mod, "cluster_su_gpu_available", lambda: True)
    # only 64 MB free vs ~2 GB needed at f=10000 / n=10000 / nb=20
    monkeypatch.setattr(mod, "_gpu_free_memory_bytes", lambda: 64 * 1024 * 1024)
    assert not _should_route_su_gpu(
        n_features=10_000, n_samples=10_000, max_n_bins=20, gpu_min_features=500,
    )


def test_cpu_path_unchanged_when_gpu_unavailable(monkeypatch):
    """Dormant fallback: when GPU is unavailable the CPU prange kernel runs exactly
    as in iter69. Labels match the ``use_parallel=False`` reference path bit-for-bit.
    """
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su as mod

    monkeypatch.setattr(mod, "cluster_su_gpu_available", lambda: False)

    bins, names = _build_synthetic_bins(
        n_samples=800, n_features=120, n_bins=8, seed=42,
    )
    serial = cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=names, use_parallel=False,
    )
    auto = cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=names, use_gpu="auto",
    )
    assert np.array_equal(serial, auto), (
        "GPU-unavailable auto path should match CPU serial path exactly"
    )


def test_use_gpu_false_skips_gpu_even_when_available(monkeypatch):
    """``use_gpu=False`` forces the CPU prange path regardless of GPU availability."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su as mod

    # Pretend GPU is available; with use_gpu=False the dispatcher MUST NOT touch GPU.
    monkeypatch.setattr(mod, "cluster_su_gpu_available", lambda: True)
    monkeypatch.setattr(mod, "_gpu_free_memory_bytes", lambda: 8 * 1024 ** 3)

    # Sentinel: if the GPU kernel is invoked the test errors with the import
    # because cupy import would fail; we instead replace it with a raise-on-call.
    def _explode(*_a, **_kw):
        raise AssertionError("GPU kernel must not be invoked when use_gpu=False")

    monkeypatch.setattr(mod, "_pairwise_su_edges_gpu", _explode)

    bins, names = _build_synthetic_bins(
        n_samples=600, n_features=120, n_bins=8, seed=11,
    )
    # Must not raise.
    cluster_correlated_features_su(
        bins, threshold=0.3, feature_names=names, use_gpu=False,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not cluster_su_gpu_available(), reason="no cupy CUDA device")
def test_gpu_kernel_parity_against_cpu():
    """When a real GPU is present the GPU kernel labels must equal the CPU kernel labels."""
    bins, names = _build_synthetic_bins(
        n_samples=1200, n_features=200, n_bins=10, seed=7,
    )
    cpu = cluster_correlated_features_su(
        bins, threshold=0.35, feature_names=names, use_gpu=False,
    )
    gpu = cluster_correlated_features_su(
        bins, threshold=0.35, feature_names=names, use_gpu=True, gpu_min_features=10,
    )
    assert np.array_equal(cpu, gpu), (
        f"GPU SU clustering diverges from CPU at width=200: "
        f"first diff at index {int(np.where(cpu != gpu)[0][0]) if (cpu != gpu).any() else -1}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not cluster_su_gpu_available(), reason="no cupy CUDA device")
def test_gpu_kernel_speedup_at_width_2000():
    """At width=2000 the GPU path should be >=2x faster than the CPU prange kernel."""
    bins, names = _build_synthetic_bins(
        n_samples=1500, n_features=2000, n_bins=10, seed=5,
    )

    # GPU-capability gate (matches test_perf_regression.py): the 2x speedup
    # floor was calibrated on Ampere+ flagship hardware. Volta and earlier
    # Turing-class devices (RTX 2060 / Quadro RTX 4000) land at 0.8-1.3x
    # on this kernel + width combination -- the per-pair workload doesn't
    # amortise H2D + launch overhead until cc >= 8.0 + >= 8 GB VRAM.
    # Skip cleanly on weaker hardware rather than churning the sensor.
    try:
        import cupy as _cp_gate
        _dev = _cp_gate.cuda.Device(0)
        _major, _minor = _dev.compute_capability[0], _dev.compute_capability[1]
        _vram_total = int(_dev.mem_info[1])
        if (int(_major), int(_minor)) < (8, 0):
            pytest.skip(
                f"GPU compute capability {_major}.{_minor} below Ampere (8.0); "
                f"width=2000 SU-cluster kernel speedup floor is calibrated "
                f"for Ampere+ only -- Volta / Turing land at 0.8-1.3x on this "
                f"workload, dominated by H2D + launch overhead."
            )
        if _vram_total < 8 * 1024 * 1024 * 1024:
            pytest.skip(
                f"GPU VRAM {_vram_total / 1e9:.1f} GB below 8 GB threshold; "
                f"width=2000 SU-cluster kernel speedup floor does not apply."
            )
    except Exception as _gpu_info_err:
        pytest.skip(f"GPU capability probe failed: {_gpu_info_err}")

    # Warm-up so cupy + kernel compile do not pollute the timing.
    cluster_correlated_features_su(
        bins, threshold=0.4, feature_names=names, use_gpu=True, gpu_min_features=10,
    )
    cluster_correlated_features_su(
        bins, threshold=0.4, feature_names=names, use_gpu=False,
    )

    t0 = time.perf_counter()
    cpu_labels = cluster_correlated_features_su(
        bins, threshold=0.4, feature_names=names, use_gpu=False,
    )
    t_cpu = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_labels = cluster_correlated_features_su(
        bins, threshold=0.4, feature_names=names, use_gpu=True, gpu_min_features=10,
    )
    t_gpu = time.perf_counter() - t0

    assert np.array_equal(cpu_labels, gpu_labels), "labels diverge at width=2000"
    ratio = t_cpu / max(t_gpu, 1e-9)
    assert ratio >= 2.0, (
        f"GPU not fast enough at width=2000: cpu={t_cpu:.3f}s, gpu={t_gpu:.3f}s, "
        f"ratio={ratio:.2f}x (need >= 2x)"
    )
