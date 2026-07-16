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

import pathlib

import pytest


def _read(rel: str) -> str:
    """Read a source file relative to the installed mlframe package root."""
    import mlframe as _mlframe
    _path = pathlib.Path(_mlframe.__file__).resolve().parent / rel
    if not _path.exists() and _path.suffix == ".py":
        # Monolith-split compat: the flat module became a subpackage
        # (``X.py`` -> ``X/__init__.py`` + submodules). Read __init__ + every submodule.
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    return _path.read_text(encoding="utf-8")


@pytest.mark.parametrize("rel,marker,site", [
    # #1: streamed variant lookup_joint_hist. Moved to ``_gpu_batched.py`` when ``gpu.py`` was split
    # into siblings (the batched joint-hist dispatch, including this cache lookup, now lives there).
    (
        "feature_selection/filters/_gpu_batched.py",
        "lookup_joint_hist(n_samples=n, joint_size=joint_size)",
        "streamed_joint_hist",
    ),
    # #2: batch_pair_mi_gpu cache-driven dispatcher. Migrated to the @kernel_tuner registry + get_or_tune
    # orchestrator API (the old ``_cache.lookup("batch_pair_mi")`` shape was replaced); the registration is the
    # forward-stable marker that the site is still wired to per-host tuning rather than hardcoded defaults.
    (
        "feature_selection/filters/batch_pair_mi_gpu.py",
        "kernel_tuner(",
        "batch_pair_mi_dispatch",
    ),
    (
        "feature_selection/filters/batch_pair_mi_gpu.py",
        'cli_label="batch_pair_mi"',
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
    # #5: cat_interactions perm-kernel cache lookup. The @kernel_tuner registration
    # moved to the ``_cat_confirm_permutation_tuning.py`` sibling when
    # ``_cat_confirm_permutation.py`` was split below 1k LOC.
    (
        "feature_selection/filters/_cat_confirm_permutation_tuning.py",
        'cli_label="cat_fe_perm_kernel"',
        "cat_fe_perm_kernel",
    ),
    # #6: feature_engineering.py unary cache lookup
    # 2026-05-22: ``check_prospective_fe_pairs`` (which contains the
    # unary-elementwise dispatch + cache lookup) moved to
    # ``_feature_engineering_pairs.py`` during the feature_engineering
    # monolith split. The marker now lives in the sibling.
    # unary-elementwise tuning moved out of _feature_engineering_pairs.py into the dedicated
    # _unary_elementwise_tuning.py sibling during the @kernel_tuner migration; the dispatch site calls
    # unary_elementwise_backend_choice() from there.
    (
        "feature_selection/filters/_unary_elementwise_tuning.py",
        'cli_label="unary_elementwise"',
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
        'cli_label="rff_matmul"',
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
    a missing pyutilz.performance.kernel_tuning.cache module would break
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
        # ``cat_interactions.py`` was split; the perm-kernel @kernel_tuner
        # registration lives in the ``_cat_confirm_permutation_tuning`` sibling now.
        "feature_selection/filters/_cat_confirm_permutation_tuning.py",
        # ``feature_engineering.py`` was split, then ``_feature_engineering_pairs`` itself was carved into a subpackage;
        # the batched-MI GPU dispatch + KernelTuningCache.get_or_tune lookup now lives in the ``_pairs_dispatch`` submodule.
        "feature_selection/filters/_feature_engineering_pairs/_pairs_dispatch.py",
        "feature_selection/filters/_unary_elementwise_tuning.py",
        "feature_engineering/transformer/random_features.py",
    ]
    for rel in sites:
        src = (root / rel).read_text(encoding="utf-8")
        # Every lookup site MUST be inside a try/except so the cache
        # being missing doesn't break the dispatcher. Accept either the
        # direct pyutilz import OR the project-local _kernel_tuning shim
        # that wraps pyutilz's cache with named project semantics OR the
        # new @kernel_tuner registry import (post-migration sites register
        # a spec and consult it via the get_or_tune orchestrator).
        has_direct = "from pyutilz.performance.kernel_tuning.cache import KernelTuningCache" in src
        has_shim = "from ._kernel_tuning import get_kernel_tuning_cache" in src
        has_pkg_shim = "from .._kernel_tuning import get_kernel_tuning_cache" in src
        has_uplevel_shim = "from .._kernel_tuning_cache.dispatch import" in src
        has_bench_dispatch = "from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import" in src
        has_registry = "from pyutilz.performance.kernel_tuning.registry import kernel_tuner" in src
        assert has_direct or has_shim or has_pkg_shim or has_uplevel_shim or has_bench_dispatch or has_registry, (
            f"{rel}: kernel_tuning_cache integration missing -- expected "
            f"either the direct pyutilz import OR the project-local "
            f"_kernel_tuning shim OR the _benchmarks.kernel_tuning_cache "
            f"dispatch helper. The lookup path was removed?"
        )
        assert "except Exception" in src, f"{rel}: the cache lookup must have a try/except fallback; " f"missing pyutilz module would otherwise break dispatch."


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
