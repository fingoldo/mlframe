"""Biz-value test: PySR-augmented suite must beat PySR-off suite on a
ground-truth equation rediscovery setup.

Sensor for the PySR FE upgrade. Generates a synthetic dataset where the
ground truth is ``y = 3*sin(x1) + log(|x2|+1) - 0.5*x3**2 + noise`` and
runs ``train_mlframe_models_suite`` twice:

- baseline: ``pysr_enabled=False``
- treatment: ``pysr_enabled=True`` with the standard operator preset

The treatment must EITHER produce a measurably lower test RMSE OR
rediscover at least 2 of the 3 ground-truth equation forms
(``sin``, ``log``/``safe_log``, ``square``).

Marked ``@pytest.mark.slow`` because PySR + Julia first-run cost ~30-60s.

Marked ``@pytest.mark.no_xdist`` because PythonCall / Julia
periodically segfaults (Windows access violation) inside ``pysr.sr.fit``
under multi-worker xdist load — the native crash kills the worker, the
master execnet channel dies, and ``BrokenPipeError`` cascades through
the scheduler taking down the rest of the suite. Symptoms verified on
the user's R: machine 2026-05-23. The PySR side is genuine third-party
native instability (Julia threadpool corruption when multiple xdist
workers boot Julia in parallel); guard by skipping under multi-worker
runs and letting the test pass in single-worker / direct invocation.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

_PYTEST_XDIST_WORKER_COUNT = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0") or "0")


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _PYTEST_XDIST_WORKER_COUNT > 1,
        reason=(
            "PySR/Julia (PythonCall) segfaults intermittently under multi-worker "
            "xdist on Windows — the worker crash propagates as BrokenPipeError to "
            "the master and aborts the rest of the suite. Run this test with -n0 "
            "or via direct pytest invocation."
        ),
    ),
]


def _make_synth(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Same generator as ``_benchmarks/bench_pysr_fe.py`` but tabular."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 6)).astype(np.float32)
    y = (3.0 * np.sin(x[:, 1]) + np.log(np.abs(x[:, 2]) + 1.0) - 0.5 * x[:, 3] ** 2 + 0.3 * rng.standard_normal(n)).astype(np.float32)
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(6)])
    df["target"] = y
    return df


def _holdout_rmse_from_models(models: dict) -> float:
    """Best test-RMSE across all trained model entries.

    Mlframe's per-target loop attaches a ``.metrics`` dict to every trained model
    object with the shape ``{split: {metric_name: value}}``. Iterate every leaf
    (target_type -> target_name -> [list of model entries]) and pull the
    minimum test-split RMSE.
    """
    rmses: list[float] = []
    if not isinstance(models, dict):
        return float("inf")
    for tt_block in models.values():
        if not isinstance(tt_block, dict):
            continue
        for entries in tt_block.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                m = getattr(entry, "metrics", None) or (entry.get("metrics") if isinstance(entry, dict) else None)
                if not isinstance(m, dict):
                    continue
                test_block = m.get("test") or {}
                # `test_block` may be flat {metric: value} or {class_idx: {...}} (binary/multiclass).
                _candidates = [test_block] + [v for v in test_block.values() if isinstance(v, dict)]
                for cand in _candidates:
                    if not isinstance(cand, dict):
                        continue
                    for k, v in cand.items():
                        if "rmse" in str(k).lower() and isinstance(v, (int, float)):
                            rmses.append(float(v))
    return min(rmses) if rmses else float("inf")


def _equation_forms_rediscovered(metadata: dict) -> int:
    """Count how many of {sin, log/safe_log, square} appear across all
    ``metadata['pysr_equations']`` entries.
    """
    eqs = metadata.get("pysr_equations") or {}
    if not isinstance(eqs, dict):
        return 0
    blob = " ".join(str(v).lower() for v in eqs.values())
    hits = 0
    if "sin" in blob:
        hits += 1
    if "log" in blob or "safe_log" in blob:
        hits += 1
    # square: PySR emits "x3 * x3" or "x3 ^ 2" or "square(x3)".
    if "square" in blob or "x3 * x3" in blob or "x3*x3" in blob or "x3 ^ 2" in blob:
        hits += 1
    return hits


def test_pysr_upgrade_beats_pysr_off_on_synthetic_ground_truth(tmp_path):
    """End-to-end: PySR-on either lowers test RMSE OR rediscovers >= 2 of the
    3 ground-truth equation forms. Either outcome demonstrates the FE upgrade
    delivers value vs PySR-off.
    """
    pytest.importorskip("pysr")
    pytest.importorskip("juliacall")
    pytest.importorskip("sklearn")

    from mlframe.training.configs import (
        PreprocessingExtensionsConfig,
        TrainingSplitConfig,
        OutputConfig,
    )
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
    from mlframe.training.core.main import train_mlframe_models_suite

    df = _make_synth(n=800, seed=0)
    fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["target"])
    split = TrainingSplitConfig(test_size=0.2, val_size=0.1, shuffle_test=True, shuffle_val=True)
    output_off = OutputConfig(data_dir=str(tmp_path / "off"), models_dir="m")
    output_on = OutputConfig(data_dir=str(tmp_path / "on"), models_dir="m")

    # Baseline: PySR off
    models_off, _meta_off = train_mlframe_models_suite(
        df=df,
        target_name="target",
        model_name="bench_off",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        preprocessing_extensions=PreprocessingExtensionsConfig(pysr_enabled=False),
        split_config=split,
        output_config=output_off,
        use_mlframe_ensembles=False,
        verbose=0,
    )

    # Treatment: PySR on with new standard preset defaults
    models_on, meta_on = train_mlframe_models_suite(
        df=df,
        target_name="target",
        model_name="bench_on",
        features_and_targets_extractor=fte,
        mlframe_models=["lgb"],
        preprocessing_extensions=PreprocessingExtensionsConfig(
            pysr_enabled=True,
            pysr_operator_preset="standard",
            # Trim for test budget: niter=80 still gives the GA enough rope to
            # find sin/log/square on this clean signal, but caps wall-clock at
            # ~30-60s on a typical workstation (vs default niter=400 ~3 min).
            pysr_niterations=80,
            pysr_top_k=5,
        ),
        split_config=split,
        output_config=output_on,
        use_mlframe_ensembles=False,
        verbose=0,
    )

    rmse_off = _holdout_rmse_from_models(models_off)
    rmse_on = _holdout_rmse_from_models(models_on)
    forms_found = _equation_forms_rediscovered(meta_on)

    # Either improvement axis is acceptable:
    rmse_improved = rmse_on < rmse_off
    forms_rediscovered = forms_found >= 2

    assert rmse_improved or forms_rediscovered, (
        f"PySR upgrade delivered no measurable value on the synthetic ground-truth setup. "
        f"RMSE off={rmse_off:.4f}, on={rmse_on:.4f} (improved={rmse_improved}); "
        f"forms rediscovered (sin/log/square)={forms_found}/3 (>=2 required)."
    )
