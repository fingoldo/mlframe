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


@pytest.mark.parametrize(
    "rel,marker,site",
    [
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
        # gpu.py during the multi-pair-MI split). 2026-07-16: the probe call itself was found broken
        # (get_shared_mem_budget_per_block() called with 0 args -- always raised TypeError, silently
        # swallowed, so the dynamic probe never actually engaged); fixed to pass (cc_major, cc_minor,
        # allow_opt_in=True), which no longer imports the function under an alias.
        (
            "feature_selection/filters/_gpu_pairs.py",
            "get_shared_mem_budget_per_block(_summary[",
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
    ],
)
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
    """All 7 wave-23 modules import cleanly under default conditions
    (no live GPU, no cache populated, no sweep done yet)."""
    import mlframe.feature_selection.filters.gpu
    import mlframe.feature_selection.filters.batch_pair_mi_gpu
    import mlframe.metrics.core
    import mlframe.feature_selection.filters.cat_interactions
    import mlframe.feature_selection.filters.feature_engineering
    import mlframe.feature_selection.filters.hermite_fe
    import mlframe.feature_engineering.transformer.random_features  # noqa: F401
