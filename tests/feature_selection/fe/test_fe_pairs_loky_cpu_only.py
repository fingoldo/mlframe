"""Regression test: the polynom-pair-FE loky workers must be CPU-ONLY.

Root cause (2026-07-05 wellbore diag, 4 GB GTX 1050 Ti): the >=16-pair parallel
branch of ``run_polynom_pair_fe`` fanned the per-pair search out to loky worker
PROCESSES. Each worker imported cupy and grabbed its own ~250 MB CUDA context;
16 contexts (~3.3 GB) nearly filled the card, the workers then blocked on GPU
allocations, the joblib parent sleep-polled at ~1 core, and the pair search
STALLED ~2h (seconds-to-minutes expected).

The fix forces the loky workers CPU-only via a worker ``initializer`` that sets
``CUDA_VISIBLE_DEVICES=""`` before any cupy import, so no per-worker CUDA context
is created. These tests pin that mechanism; they FAIL on the pre-fix code, which
passed ``backend="loky"`` (a bare string) with no initializer (workers touched
the GPU).
"""

import os

import numpy as np
import pandas as pd
import pytest

from joblib._parallel_backends import LokyBackend

from mlframe.feature_selection.filters import polynom_pair_fe as ppf


def test_worker_initializer_disables_cuda():
    """The initializer must blank CUDA_VISIBLE_DEVICES so a fresh worker sees no GPU."""
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        ppf._poly_worker_disable_cuda()
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "", (
            "initializer must set CUDA_VISIBLE_DEVICES='' so the loky worker "
            "sees no CUDA device and creates no cupy context"
        )
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


class _CapturedBackend(Exception):
    """Raised by the Parallel spy once the backend has been captured."""

    def __init__(self, backend):
        self.backend = backend


def test_pair_search_parallel_uses_cpu_only_loky_backend(monkeypatch):
    """>=16 pairs -> parallel branch must build a LokyBackend with the CUDA-disabling initializer.

    Pre-fix code passed ``backend="loky"`` (str) with no initializer -> this test
    fails (captured backend is a str, has no ``_poly_worker_disable_cuda``).
    """
    captured = {}

    def _spy_parallel(*args, **kwargs):
        captured["backend"] = kwargs.get("backend")

        def _run(_tasks):
            raise _CapturedBackend(captured["backend"])

        return _run

    monkeypatch.setattr(ppf, "Parallel", _spy_parallel)

    n, n_cols = 40, 8
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {f"c{i}": rng.standard_normal(n) for i in range(n_cols)}
    )
    cols = list(X.columns)
    classes_y = (rng.random(n) > 0.5).astype(np.int64)

    # >=16 prospective pairs so the parallel branch (threshold=16 inside
    # run_polynom_pair_fe) fires. C(8,2)=28 pairs.
    pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
    assert len(pairs) >= 16
    prospective_pairs = {((i, j), 0.5): None for (i, j) in pairs}

    with pytest.raises(_CapturedBackend) as ei:
        ppf.run_polynom_pair_fe(
            X=X,
            is_polars_input=False,
            prospective_pairs=prospective_pairs,
            classes_y=classes_y,
            cols=cols,
            nbins=np.full(n_cols, 8, dtype=np.int64),
            data=X.values.copy(),
            engineered_features=set(),
            engineered_recipes={},
            hermite_features_list=[],
            feature_names_in=cols,
            fe_smart_polynom_iters=1,
            fe_smart_polynom_optimization_steps=1,
            fe_min_polynom_degree=1,
            fe_max_polynom_degree=3,
            fe_min_polynom_coeff=-2.0,
            fe_max_polynom_coeff=2.0,
            fe_min_engineered_mi_prevalence=0.1,
            fe_hermite_l2_penalty=0.0,
            fe_polynomial_basis="hermite",
            fe_mi_estimator="plugin",
            fe_optimizer="cma",
            fe_warm_start=False,
            fe_multi_fidelity=False,
            quantization_nbins=8,
            quantization_method="quantile",
            quantization_dtype=np.int16,
            n_jobs=16,
            verbose=0,
        )

    backend = ei.value.backend
    assert isinstance(backend, LokyBackend), (
        "parallel pair-search must pass a LokyBackend instance (not the "
        "backend='loky' string), so the initializer + inner_max_num_threads "
        "are actually honoured in joblib 1.5.x"
    )
    assert backend.backend_kwargs.get("initializer") is ppf._poly_worker_disable_cuda, (
        "loky workers must be started with the CUDA-disabling initializer so "
        "they create NO per-worker cupy CUDA context"
    )
    assert backend.inner_max_num_threads == 1, (
        "workers must cap inner thread pools to 1 (sampler-side, not kernel-side "
        "parallelism) to avoid CPU oversubscription"
    )
