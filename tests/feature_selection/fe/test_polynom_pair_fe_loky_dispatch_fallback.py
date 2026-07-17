"""Regression test for MRMR audit follow-on fix (2026-07-10): a loky pool dispatch failure inside
``run_polynom_pair_fe`` must fall back to the serial path, not crash the whole training run.

Reproduced live at n=3M production scale (the wellbore_train.py validation run): a GPU OOM earlier in
the same MRMR fit left the main process's CUDA context poisoned; cloudpickle serializing the per-pair
task closure for the loky dispatch then raised ``_pickle.PicklingError`` (a ``CUDADriverError`` as its
``__cause__``), which propagated all the way up through ``_run_fe_step`` / ``_fit_impl`` /
``train_mlframe_models_suite`` and crashed the process -- even though this pool's WORKERS are CPU-only
and have nothing to do with the GPU state that actually failed.

This test forces the SAME failure mode (patch ``Parallel`` to raise) and asserts the function completes
via the serial fallback instead of propagating the exception.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import polynom_pair_fe as ppf


def test_loky_dispatch_failure_falls_back_to_serial(monkeypatch, caplog):
    """A Parallel(...) dispatch failure (simulating a poisoned-CUDA-context pickling error) must not
    propagate -- the function must complete via the serial per-pair path instead."""
    calls = {"parallel": 0}

    def _boom_parallel(*args, **kwargs):
        calls["parallel"] += 1

        def _run(_tasks):
            raise RuntimeError("simulated poisoned-CUDA-context pickling failure")

        return _run

    monkeypatch.setattr(ppf, "Parallel", _boom_parallel)

    n, n_cols = 40, 8
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"c{i}": rng.standard_normal(n) for i in range(n_cols)})
    cols = list(X.columns)
    classes_y = (rng.random(n) > 0.5).astype(np.int64)

    # >=16 prospective pairs so the parallel branch (threshold=16) fires and hits the patched Parallel.
    pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
    assert len(pairs) >= 16
    prospective_pairs = {((i, j), 0.5): None for (i, j) in pairs}

    import logging

    with caplog.at_level(logging.WARNING):
        # Must complete WITHOUT raising -- the whole point of the fix.
        result = ppf.run_polynom_pair_fe(
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

    assert calls["parallel"] == 1, "the patched Parallel must have been reached (parallel branch taken)"
    assert result is not None, "run_polynom_pair_fe must return a result via the serial fallback, not raise"
    assert any("falling back to the serial" in rec.message for rec in caplog.records), (
        "the fallback must be logged so a real production failure is diagnosable, not silent"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
