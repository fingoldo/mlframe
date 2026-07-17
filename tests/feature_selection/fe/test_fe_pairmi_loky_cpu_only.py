"""Regression test: the FE pair-MI SCORING loky workers must be CPU-ONLY.

Root cause (2026-07-06 wellbore diag, 1M wellbore run): the parallel branch of
``compute_pair_mis_and_floor`` (``_step_pairmi.py``) fanned per-chunk
``compute_pairs_mis`` work out via ``parallel_run(..., backend='threading')``. The
per-chunk body is GIL-bound CPU (``mi_direct`` per pair: joint plug-in MI + the
analytic/permutation null over ``fe_npermutations`` shuffles), so ``backend="threading"``
serialised the whole chunk list onto ONE core -- py-spy showed the MainThread stuck
in joblib ``_retrieve`` sleep-poll at ~1.1 cores for ~42 min, GPU util 0%.

The fix routes the parallel branch to a loky PROCESS pool for real multi-core
parallelism, with every worker forced CPU-ONLY via a worker ``initializer`` that
sets ``CUDA_VISIBLE_DEVICES=""`` before any cupy import (so no per-worker CUDA
context fills a small card). These tests pin that mechanism; the first FAILS on the
pre-fix code (threading string backend, no initializer) and the second pins the
shared initializer contract.
"""

import os
from types import SimpleNamespace

import numpy as np

from joblib._parallel_backends import LokyBackend

from mlframe.feature_selection.filters import _joblib_safe
from mlframe.feature_selection.filters._mrmr_fe_step import _step_pairmi


def test_worker_initializer_disables_cuda():
    """The shared initializer must blank CUDA_VISIBLE_DEVICES so a fresh worker sees no GPU."""
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        _joblib_safe.disable_cuda_in_worker()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "", (
            "initializer must set CUDA_VISIBLE_DEVICES='' so the loky worker sees no CUDA device and creates no cupy context"
        )
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


def test_pairmi_parallel_uses_cpu_only_loky_backend(monkeypatch):
    """>1 job + enough pairs -> parallel branch must build a CPU-only LokyBackend.

    Pre-fix code called ``parallel_run(..., backend='threading')`` -> this test fails
    (no LokyBackend / no CUDA-disabling initializer is ever built).

    The spy returns an EMPTY result list (not an exception) so the run completes normally: raising here
    would be caught by ``compute_pair_mis_and_floor``'s own loky-pool-failure retry/fallback (2026-07-10
    fix) and silently re-dispatched to the batched-CPU-retry path, defeating the capture instead of
    signalling it.
    """
    # Disable the GPU/CPU batch pre-fill so the function reaches the per-chunk
    # parallel branch directly (the batch path is a separate dispatch we are not testing).
    monkeypatch.setenv("MLFRAME_MRMR_BATCH_PAIR_MI", "0")

    captured = {}

    def _spy_parallel(*args, **kwargs):
        captured["backend"] = kwargs.get("backend")

        def _run(_tasks):
            return []

        return _run

    monkeypatch.setattr(_step_pairmi, "Parallel", _spy_parallel)

    n, n_cols = 200, 8
    rng = np.random.default_rng(0)
    data = rng.integers(0, 8, size=(n, n_cols)).astype(np.int32)
    cols = [f"c{i}" for i in range(n_cols)]
    nbins = np.full(n_cols, 8, dtype=np.int64)
    classes_y = (rng.random(n) > 0.5).astype(np.int32)
    freqs_y = np.bincount(classes_y).astype(np.float64)

    # Treat every col as RAW (in feature_names_in_) so the engineered feed-forward
    # cap does not prune the pool below the parallel-branch pair threshold.
    self = SimpleNamespace(feature_names_in_=cols)

    numeric_vars_to_consider = set(range(n_cols))  # C(8,2)=28 pairs >= n_jobs

    _step_pairmi.compute_pair_mis_and_floor(
        self,
        data=data,
        cols=cols,
        nbins=nbins,
        X=None,
        classes_y=classes_y,
        classes_y_safe=classes_y,
        freqs_y=freqs_y,
        target_indices=(n_cols - 1,),
        cached_MIs={},
        cached_confident_MIs={},
        numeric_vars_to_consider=numeric_vars_to_consider,
        _prevalence_debias_auto=False,
        n_jobs=16,
        prefetch_factor=1,
        parallel_kwargs={"backend": "threading"},
        fe_min_nonzero_confidence=0.95,
        fe_npermutations=10,
        fe_min_pair_mi=0.0,
        fe_min_pair_mi_prevalence=0.0,
        verbose=0,
    )

    backend = captured["backend"]
    assert isinstance(backend, LokyBackend), (
        "FE pair-MI parallel scoring must pass a LokyBackend INSTANCE (not a "
        "backend='threading'/'loky' string), so the initializer + "
        "inner_max_num_threads are actually honoured in joblib 1.5.x"
    )
    assert backend.backend_kwargs.get("initializer") is _joblib_safe.disable_cuda_in_worker, (
        "loky workers must be started with the CUDA-disabling initializer so they create NO per-worker cupy CUDA context (small-card VRAM fill guard)"
    )
    assert backend.inner_max_num_threads == 1, "workers must cap inner thread pools to 1 to avoid CPU oversubscription"
