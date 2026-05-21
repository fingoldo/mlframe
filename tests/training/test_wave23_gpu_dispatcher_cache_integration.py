"""Wave-23 sensor: all 8 GPU-dispatcher sites consult kernel_tuning_cache.

Wave 23 audit (2026-05-20) found 6 P1 + 2 P2 sites where hardcoded
CUDA / threshold / block-size constants determined dispatcher
behaviour without consulting ``pyutilz.system.kernel_tuning_cache``.
Per memory rule ``feedback_use_kernel_tuning_cache_for_gpu``:

> "never hardcode CUDA thresholds / block sizes / kernel variants;
> integrate with pyutilz.system.kernel_tuning_cache (mirror
> joint_hist_batched / plugin_mi_classif_dispatch). Hardcoded
> defaults are wrong on any HW other than dev machine - 2026-05-20
> incident left 2-4x speedups on the table."

The 8 sites now all share the same pattern:
1. Try ``KernelTuningCache.load_or_create().lookup(kernel_name, dims)``
2. Read the relevant per-HW tuned parameter from the result dict
3. Fall back to the source-code default (which IS the pre-wave-23
   hardcoded value) when:
   - pyutilz.system.kernel_tuning_cache is not importable, OR
   - lookup returns None (no entry for live HW yet), OR
   - the lookup raises (corrupt sidecar etc.)

This sensor pins the post-fix shape at each site so a future refactor
that re-introduces the hardcoded default without the cache lookup
gets caught.
"""
from __future__ import annotations

import pathlib

import pytest


def _read(rel: str) -> str:
    import mlframe as _mlframe
    return (
        pathlib.Path(_mlframe.__file__).resolve().parent / rel
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("rel,marker,site", [
    # #1: gpu.py streamed variant lookup_joint_hist
    (
        "feature_selection/filters/gpu.py",
        "lookup_joint_hist(n_samples=n, joint_size=joint_size)",
        "streamed_joint_hist",
    ),
    # #2: batch_pair_mi_gpu cache-driven dispatcher
    (
        "feature_selection/filters/batch_pair_mi_gpu.py",
        "_lookup_batch_pair_mi_thresholds(n_samples=n_samples, n_pairs=n_pairs)",
        "batch_pair_mi_dispatch",
    ),
    (
        "feature_selection/filters/batch_pair_mi_gpu.py",
        'cache.lookup(\n            "batch_pair_mi"',
        "batch_pair_mi_cache_key",
    ),
    # #3: metrics RMSE BLOCK_N cache lookup (moved to ``_gpu_metrics.py`` when
    # ``metrics/core.py`` was split into siblings to drop the monolith below 1k LOC).
    (
        "metrics/_gpu_metrics.py",
        '_cache.lookup(\n                "rmse_partial_sum"',
        "rmse_block_n",
    ),
    # #4: _gpu_pairs.py multi-pair shared-mem device probe (moved out of
    # gpu.py during the multi-pair-MI split).
    (
        "feature_selection/filters/_gpu_pairs.py",
        "get_shared_mem_budget_per_block as _shared_budget",
        "multi_pair_shared_cap",
    ),
    # #5: cat_interactions perm-kernel cache lookup (moved to the
    # ``_cat_confirm_permutation.py`` sibling when ``cat_interactions.py`` was
    # split below 1k LOC).
    (
        "feature_selection/filters/_cat_confirm_permutation.py",
        '_cache.lookup(\n                "cat_fe_perm_kernel"',
        "cat_fe_perm_kernel",
    ),
    # #6: feature_engineering.py unary cache lookup
    (
        "feature_selection/filters/feature_engineering.py",
        '_cache.lookup(\n                                    "unary_elementwise"',
        "unary_elementwise",
    ),
    # P2: hermite_fe polyeval lookup (linter may use dict-args or kwargs form)
    (
        "feature_selection/filters/hermite_fe.py",
        '_cache.lookup("polyeval"',
        "polyeval_thresholds",
    ),
    # P2: random_features RFF matmul lookup
    (
        "feature_engineering/transformer/random_features.py",
        '_cache.lookup("rff_matmul"',
        "rff_matmul_crossover",
    ),
])
def test_gpu_dispatcher_consults_kernel_tuning_cache(rel, marker, site):
    """Source-level guard: each of the 8 (now 9 -- batch_pair_mi has 2
    related markers) wave-23 sites must contain a kernel_tuning_cache
    lookup. Falling back to source-code defaults is allowed (and
    expected on first-run / no-sweep HW), but the lookup MUST be
    attempted -- the bug was 'never try the cache'."""
    src = _read(rel)
    assert marker in src, (
        f"Wave 23 regression at site={site!r}: kernel_tuning_cache "
        f"lookup pattern {marker!r} missing from {rel}. Reverting to "
        f"hardcoded defaults silently lifts 2-4x speedups on non-dev HW."
    )


def test_wave23_falls_back_to_source_default_when_cache_unavailable():
    """The cache-lookup pattern at each site MUST be wrapped in a
    try/except that falls back to the pre-wave-23 default. Otherwise
    a missing pyutilz.system.kernel_tuning_cache module would break
    every GPU-dispatching call site."""
    import mlframe as _mlframe
    root = pathlib.Path(_mlframe.__file__).resolve().parent
    # Pin the wrapping shape at each of the 6 sites that introduced
    # the lookup-with-fallback pattern (the polyeval site uses a named
    # helper which itself has the try/except).
    sites = [
        "feature_selection/filters/gpu.py",
        "feature_selection/filters/batch_pair_mi_gpu.py",
        # ``metrics/core.py`` was split; the RMSE GPU dispatcher lives in
        # the ``_gpu_metrics`` sibling now.
        "metrics/_gpu_metrics.py",
        # ``cat_interactions.py`` was split; the perm-kernel cache lookup
        # lives in the ``_cat_confirm_permutation`` sibling now.
        "feature_selection/filters/_cat_confirm_permutation.py",
        "feature_selection/filters/feature_engineering.py",
        "feature_engineering/transformer/random_features.py",
    ]
    for rel in sites:
        src = (root / rel).read_text(encoding="utf-8")
        # Every lookup site MUST be inside a try/except so the cache
        # being missing doesn't break the dispatcher. Accept either the
        # direct pyutilz import OR the project-local _kernel_tuning shim
        # that wraps pyutilz's cache with named project semantics.
        has_direct = "from pyutilz.system.kernel_tuning_cache import KernelTuningCache" in src
        has_shim = "from ._kernel_tuning import get_kernel_tuning_cache" in src
        has_pkg_shim = "from .._kernel_tuning import get_kernel_tuning_cache" in src
        has_uplevel_shim = "from .._kernel_tuning_cache.dispatch import" in src
        has_bench_dispatch = "from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import" in src
        assert (has_direct or has_shim or has_pkg_shim or has_uplevel_shim or has_bench_dispatch), (
            f"{rel}: kernel_tuning_cache integration missing -- expected "
            f"either the direct pyutilz import OR the project-local "
            f"_kernel_tuning shim OR the _benchmarks.kernel_tuning_cache "
            f"dispatch helper. The lookup path was removed?"
        )
        assert "except Exception" in src, (
            f"{rel}: the cache lookup must have a try/except fallback; "
            f"missing pyutilz module would otherwise break dispatch."
        )


def test_wave23_smoke_imports():
    """All 7 wave-23 modules import cleanly under default conditions
    (no live GPU, no cache populated, no sweep done yet)."""
    import mlframe.feature_selection.filters.gpu  # noqa: F401
    import mlframe.feature_selection.filters.batch_pair_mi_gpu  # noqa: F401
    import mlframe.metrics.core  # noqa: F401
    import mlframe.feature_selection.filters.cat_interactions  # noqa: F401
    import mlframe.feature_selection.filters.feature_engineering  # noqa: F401
    import mlframe.feature_selection.filters.hermite_fe  # noqa: F401
    import mlframe.feature_engineering.transformer.random_features  # noqa: F401
