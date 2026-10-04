"""Wave-23 sensor: all 8 GPU-dispatcher sites consult kernel_tuning_cache.

Wave 23 audit (2026-05-20) found 6 P1 + 2 P2 sites where hardcoded
CUDA / threshold / block-size constants determined dispatcher
behaviour without consulting ``pyutilz.performance.kernel_tuning.cache``.
Per memory rule ``feedback_use_kernel_tuning_cache_for_gpu``:

> "never hardcode CUDA thresholds / block sizes / kernel variants;
> integrate with pyutilz.performance.kernel_tuning.cache (mirror
> joint_hist_batched / plugin_mi_classif_dispatch). Hardcoded
> defaults are wrong on any HW other than dev machine - 2026-05-20
> incident left 2-4x speedups on the table."

The 8 sites now all share the same pattern:
1. Try ``KernelTuningCache.load_or_create().lookup(kernel_name, dims)``
2. Read the relevant per-HW tuned parameter from the result dict
3. Fall back to the source-code default (which IS the pre-wave-23
   hardcoded value) when:
   - pyutilz.performance.kernel_tuning.cache is not importable, OR
   - lookup returns None (no entry for live HW yet), OR
   - the lookup raises (corrupt sidecar etc.)

This sensor pins the post-fix shape at each site so a future refactor
that re-introduces the hardcoded default without the cache lookup
gets caught.
"""

from __future__ import annotations

import pytest


class _SpyKernelTuningCache:
    """Stands in for ``KernelTuningCache``: records every consultation and answers with a fixed tuned result."""

    def __init__(self, lookup_result=None, tuned_choice=None):
        self.calls = []
        self._lookup_result = lookup_result
        self._tuned_choice = tuned_choice

    def lookup(self, kernel_name, **dims):
        """Record the region lookup and return the configured tuned entry."""
        self.calls.append(("lookup", kernel_name, dims))
        return self._lookup_result

    def get_or_tune(self, kernel_name, dims=None, **kwargs):
        """Record the orchestrated lookup and return the tuned choice, or the supplied fallback."""
        self.calls.append(("get_or_tune", kernel_name, dims))
        if self._tuned_choice is not None:
            return {"backend_choice": self._tuned_choice}
        return kwargs["fallback"]

    def has(self, kernel_name):
        """No kernel counts as tuned, so a spec never memoises the spy's answer."""
        return False

    def code_version_stale(self, kernel_name, code_version):
        """Never stale."""
        return False

    def names(self):
        """The kernel names consulted so far."""
        return [name for _, name, _ in self.calls]


def _install_spy_cache(monkeypatch, lookup_result=None, tuned_choice=None):
    """Route every kernel-tuning-cache access (class entry point and process singleton) to one spy and return it."""
    from mlframe.feature_selection.filters import _kernel_tuning
    from pyutilz.performance.kernel_tuning import cache as cache_module

    spy = _SpyKernelTuningCache(lookup_result=lookup_result, tuned_choice=tuned_choice)
    spy_class = type("KernelTuningCache", (), {"load_or_create": staticmethod(lambda *args, **kwargs: spy)})
    monkeypatch.setattr(cache_module, "KernelTuningCache", spy_class)
    monkeypatch.setattr(_kernel_tuning, "get_kernel_tuning_cache", lambda *args, **kwargs: spy)
    return spy


_REGISTERED_SITES = {
    "batch_pair_mi_dispatch": ("mlframe.feature_selection.filters.batch_pair_mi_gpu", "batch_pair_mi"),
    "batch_pair_mi_cache_key": ("mlframe.feature_selection.filters.batch_pair_mi_gpu", "batch_pair_mi"),
    "cat_fe_perm_kernel": ("mlframe.feature_selection.filters._cat_confirm_permutation_tuning", "cat_fe_perm_kernel"),
    "unary_elementwise": ("mlframe.feature_selection.filters._unary_elementwise_tuning", "unary_elementwise"),
    "rff_matmul_crossover": ("mlframe.feature_engineering.transformer.random_features", "rff_matmul"),
}


@pytest.mark.parametrize("site", sorted(_REGISTERED_SITES))
def test_gpu_dispatcher_consults_kernel_tuning_cache(monkeypatch, site):
    """Every wave-23 dispatch site asks the kernel tuning cache for its decision and uses the tuned answer; the source default is only the fallback."""
    import importlib

    from pyutilz.performance.kernel_tuning.registry import get_registry

    module_name, cli_label = _REGISTERED_SITES[site]
    importlib.import_module(module_name)
    spec = next(s for s in get_registry().values() if s.cli_label == cli_label)
    dims = {axis: values[0] for axis, values in spec.axes.items()}
    spec._choice_cache.clear()
    monkeypatch.delenv(spec.env_key or "MLFRAME_UNSET_ENV_KEY", raising=False)

    try:
        fallback_spy = _install_spy_cache(monkeypatch)
        fallback_choice = spec.choose(**dims)
        spec._choice_cache.clear()
        tuned_spy = _install_spy_cache(monkeypatch, tuned_choice="tuned_variant_sentinel")
        tuned_choice = spec.choose(**dims)
    finally:
        spec._choice_cache.clear()

    assert fallback_spy.names() == [spec.kernel_name]
    assert tuned_spy.names() == [spec.kernel_name]
    assert tuned_choice == "tuned_variant_sentinel"
    assert fallback_choice != "tuned_variant_sentinel"
    assert fallback_choice == spec._fallback_choice(dims)


def test_joint_hist_dispatch_consults_kernel_tuning_cache(monkeypatch):
    """The joint-histogram variant lookup reads the tuned (kernel_variant, block_size) from the cache and falls back to the size-based default on a miss."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import lookup_joint_hist

    tuned = {"kernel_variant": "global", "block_size": 128}
    spy = _install_spy_cache(monkeypatch, lookup_result=tuned)
    assert lookup_joint_hist(n_samples=100_000, joint_size=64) == tuned
    assert spy.names() == ["joint_hist_batched"]
    assert spy.calls[0][2] == {"n_samples": 100_000, "joint_size": 64}

    miss_spy = _install_spy_cache(monkeypatch, lookup_result=None)
    default = lookup_joint_hist(n_samples=100_000, joint_size=64)
    assert miss_spy.names() == ["joint_hist_batched"]
    assert set(default) == {"kernel_variant", "block_size"}
    assert default != tuned


def test_polyeval_thresholds_consult_kernel_tuning_cache(monkeypatch):
    """The polynomial-evaluation crossover thresholds come from the cache entry for (basis, n) and fall back to the module defaults on a miss."""
    from mlframe.feature_selection.filters.hermite_fe import _hermite_oracle

    spy = _install_spy_cache(monkeypatch, lookup_result={"par_threshold": 123, "cuda_threshold": 456})
    assert _hermite_oracle._lookup_polyeval_thresholds("legendre", 1000) == (123, 456)
    assert spy.calls == [("lookup", "polyeval", {"basis": "legendre", "n_samples": 1000})]

    _install_spy_cache(monkeypatch, lookup_result=None)
    assert _hermite_oracle._lookup_polyeval_thresholds("legendre", 1000) == (_hermite_oracle._PAR_THRESHOLD, _hermite_oracle._CUDA_THRESHOLD)


def test_rmse_block_size_consults_kernel_tuning_cache(monkeypatch):
    """The numba.cuda RMSE dispatcher asks the cache for its block size and still returns the exact RMSE."""
    import numpy as np

    from mlframe.metrics import _gpu_metrics

    if not _gpu_metrics._is_numba_cuda_available():
        pytest.skip("numba.cuda unavailable: the RMSE cache lookup is only reached on the numba.cuda path")
    import cupy as cp

    spy = _install_spy_cache(monkeypatch, lookup_result={"block_n": 64})
    rng = np.random.default_rng(0)
    actual = rng.normal(size=5000)
    predicted = actual[:, None] + rng.normal(scale=[0.1, 0.5, 1.0], size=(5000, 3))

    got = cp.asnumpy(_gpu_metrics.gpu_multiple_rmse_scores(actual, predicted))

    assert spy.names() == ["rmse_partial_sum"]
    assert spy.calls[0][2] == {"n_samples": 5000, "n_cols": 3}
    np.testing.assert_allclose(got, np.sqrt(np.mean((actual[:, None] - predicted) ** 2, axis=0)), rtol=1e-9)


def _has_cuda_device() -> bool:
    """True when CuPy imports and sees at least one CUDA device."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def test_multi_pair_shared_memory_budget_is_probed_on_the_live_device(monkeypatch):
    """The multi-pair GPU histogram sizes its shared-memory kernel from the live device's budget (compute capability and opt-in passed through) and returns the exact pair MIs."""
    import numpy as np

    if not _has_cuda_device():
        pytest.skip("no CUDA device: the multi-pair histogram kernel needs one")
    from pyutilz.system import gpu_dispatch

    from mlframe.feature_selection.filters._gpu_pairs import mi_direct_gpu_batched_pairs

    seen = []
    real_budget = gpu_dispatch.get_shared_mem_budget_per_block

    def _spy_budget(cc_major, cc_minor, *args, **kwargs):
        """Record the probe arguments, then delegate to the real probe."""
        seen.append((cc_major, cc_minor, kwargs.get("allow_opt_in")))
        return real_budget(cc_major, cc_minor, *args, **kwargs)

    monkeypatch.setattr(gpu_dispatch, "get_shared_mem_budget_per_block", _spy_budget)
    rng = np.random.default_rng(0)
    n = 4000
    factors = rng.integers(0, 4, size=(n, 3)).astype(np.int32)
    y = ((factors[:, 0] + factors[:, 1]) % 2).astype(np.int32)
    classes_y = y
    freqs_y = np.bincount(y).astype(np.float64) / n
    nbins = np.array([4, 4, 4], dtype=np.int32)
    pairs_a, pairs_b = np.array([0, 0, 1]), np.array([1, 2, 2])

    got = mi_direct_gpu_batched_pairs(factors, pairs_a, pairs_b, nbins, classes_y, freqs_y)

    assert len(seen) == 1
    cc_major, cc_minor, allow_opt_in = seen[0]
    assert isinstance(cc_major, int) and cc_major >= 1 and isinstance(cc_minor, int)
    assert allow_opt_in is True

    def _plugin_mi(a, b):
        """Plug-in mutual information of the merged pair (a, b) with y, in nats."""
        merged = factors[:, a] * 4 + factors[:, b]
        joint = np.zeros((16, 2))
        np.add.at(joint, (merged, y), 1.0)
        p = joint / n
        px, py = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
        nz = p > 0
        return float(np.sum(p[nz] * np.log(p[nz] / (px @ py)[nz])))

    expected = np.array([_plugin_mi(a, b) for a, b in zip(pairs_a, pairs_b)])
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)


_WAVE23_SITE_MODULES = (
    "mlframe.feature_selection.filters.gpu",
    "mlframe.feature_selection.filters.batch_pair_mi_gpu",
    "mlframe.metrics._gpu_metrics",
    "mlframe.feature_selection.filters._cat_confirm_permutation_tuning",
    "mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch",
    "mlframe.feature_selection.filters._unary_elementwise_tuning",
    "mlframe.feature_engineering.transformer.random_features",
)

_BROKEN_CACHE_SNIPPET = """
import importlib
from pyutilz.performance.kernel_tuning import cache as _cache


class _BrokenCache:
    def __init__(self, *a, **k):
        raise RuntimeError("kernel tuning cache unavailable")

    @classmethod
    def load_or_create(cls, *a, **k):
        raise RuntimeError("kernel tuning cache unavailable")

    @classmethod
    def get_or_tune(cls, *a, **k):
        raise RuntimeError("kernel tuning cache unavailable")


_cache.KernelTuningCache = _BrokenCache
for _name in {modules!r}:
    importlib.import_module(_name)
print("IMPORTED", len({modules!r}))
"""


class _BrokenKernelTuningCache:
    """Stands in for a KernelTuningCache whose every entry point raises (missing or corrupt sidecar)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("kernel tuning cache unavailable")

    @classmethod
    def load_or_create(cls, *args, **kwargs):
        """Raises like a corrupt sidecar would."""
        raise RuntimeError("kernel tuning cache unavailable")


def test_wave23_falls_back_to_source_default_when_cache_unavailable(monkeypatch):
    """With the kernel tuning cache raising on every use, each GPU-dispatching site still imports
    (its @kernel_tuner registration falls back) and the RMSE dispatcher still returns the exact
    RMSE on its source-default block size. A missing or corrupt cache must never break dispatch."""
    import subprocess
    import sys

    import numpy as np

    proc = subprocess.run(
        [sys.executable, "-c", _BROKEN_CACHE_SNIPPET.format(modules=_WAVE23_SITE_MODULES)],
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0 and f"IMPORTED {len(_WAVE23_SITE_MODULES)}" in proc.stdout, proc.stderr[-3000:]

    from mlframe.metrics import _gpu_metrics

    if not _gpu_metrics._is_numba_cuda_available():
        pytest.skip("numba.cuda unavailable: the RMSE cache lookup is only reached on the numba.cuda path")
    import cupy as cp
    from pyutilz.performance.kernel_tuning import cache as _cache

    monkeypatch.setattr(_cache, "KernelTuningCache", _BrokenKernelTuningCache)
    rng = np.random.default_rng(0)
    actual = rng.normal(size=5000)
    predicted = actual[:, None] + rng.normal(scale=[0.1, 0.5, 1.0], size=(5000, 3))
    got = cp.asnumpy(_gpu_metrics.gpu_multiple_rmse_scores(actual, predicted))
    np.testing.assert_allclose(got, np.sqrt(np.mean((actual[:, None] - predicted) ** 2, axis=0)), rtol=1e-9)


def test_wave23_smoke_imports():
    """All 7 wave-23 modules import cleanly under default conditions (no live GPU, no cache populated, no sweep done yet) and expose their dispatch entry points."""
    import importlib

    entry_points = {
        "mlframe.feature_selection.filters.gpu": "init_kernels",
        "mlframe.feature_selection.filters.batch_pair_mi_gpu": "batch_pair_mi_cupy",
        "mlframe.metrics.core": "fast_roc_auc",
        "mlframe.feature_selection.filters.cat_interactions": "resolve_max_combined_nbins",
        "mlframe.feature_selection.filters.feature_engineering": "check_prospective_fe_pairs",
        "mlframe.feature_selection.filters.hermite_fe": "polyeval_dispatch",
        "mlframe.feature_engineering.transformer.random_features": "compute_rff_features",
    }
    assert len(entry_points) == 7
    for module_name, attribute in entry_points.items():
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute)), (module_name, attribute)
