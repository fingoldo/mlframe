"""Sensor: ``run_honest_diagnostics`` populates all 4 artefact buckets.

Builds a minimal synthetic ctx + model-entries map, exercises the aggregator,
and asserts ``metadata["honest_diagnostics"]`` carries the four top-level
keys (``bootstrap_ci``, ``drift_psi``, ``calibration``, ``provenance``) plus
a non-empty per-target bootstrap CI block.
"""

from __future__ import annotations

import os
import tempfile
import types

import numpy as np
import pandas as pd


def _make_binary_entry(n: int = 400, seed: int = 5):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, size=n)
    true_p = 1.0 / (1.0 + np.exp(-4.0 * (raw - 0.5)))
    y = (rng.uniform(0, 1, size=n) < true_p).astype(np.int64)
    probs = np.column_stack([1.0 - raw, raw])
    preds = (raw > 0.5).astype(np.int64)
    entry = types.SimpleNamespace(
        model_name="cb_dummy",
        model=types.SimpleNamespace(oof_probs=probs.copy()),
        test_target=y,
        test_probs=probs,
        test_preds=preds,
        oof_probs=probs.copy(),
        oof_target=y.copy(),
    )
    return entry, y


def _make_df(n: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n),
            "cat_b": rng.choice(["x", "y", "z"], size=n),
            "cat_c": rng.choice(["P", "Q"], size=n),
        }
    )


def test_run_honest_diagnostics_populates_all_four_blocks():
    from mlframe.training.honest_diagnostics import run_honest_diagnostics

    entry, _y = _make_binary_entry(n=500, seed=3)
    models = {"BINARY_CLASSIFICATION": {"y_dummy": [entry]}}

    ctx = types.SimpleNamespace(
        train_df=_make_df(300, seed=1),
        val_df=_make_df(100, seed=2),
        test_df=_make_df(100, seed=3),
        data_dir="",
        models_dir="",
        models=models,
        metadata={},
    )
    metadata = {"provenance": {"pre_screen": {"source": "train_only", "n_rows": 300, "ts": "2026-05-25T00:00:00+00:00"}}}

    out = run_honest_diagnostics(ctx, models, metadata)

    assert "honest_diagnostics" in metadata
    payload = metadata["honest_diagnostics"]
    assert payload is out
    for key in ("bootstrap_ci", "drift_psi", "calibration", "provenance"):
        assert key in payload, key

    assert len(payload["bootstrap_ci"]) >= 1
    _boot_key, boot_val = next(iter(payload["bootstrap_ci"].items()))
    # Either a metric block or an explicit skip reason — never empty.
    assert isinstance(boot_val, dict) and len(boot_val) >= 1, boot_val

    # drift_psi block: status must be ok with at least 2 categorical features detected.
    assert payload["drift_psi"].get("status") == "ok", payload["drift_psi"]

    # provenance block: status ok, n_steps reflects the seeded entry.
    assert payload["provenance"]["status"] == "ok"
    assert payload["provenance"]["n_steps"] >= 1
    assert "step" in payload["provenance"]["table"]


def test_run_honest_diagnostics_calibration_block_emits_plot_when_reports_dir_present():
    from mlframe.training.honest_diagnostics import run_honest_diagnostics

    entry, _y = _make_binary_entry(n=600, seed=9)
    models = {"BINARY_CLASSIFICATION": {"y_dummy": [entry]}}

    with tempfile.TemporaryDirectory() as td:
        sub = os.path.join(td, "data")
        os.makedirs(sub, exist_ok=True)
        ctx = types.SimpleNamespace(
            train_df=_make_df(200, seed=1),
            val_df=_make_df(80, seed=2),
            test_df=_make_df(80, seed=3),
            data_dir=td,
            models_dir="data",
            models=models,
            metadata={},
        )
        metadata: dict = {}
        out = run_honest_diagnostics(ctx, models, metadata)

        cal_block = out["calibration"]
        assert len(cal_block) == 1
        _, cal = next(iter(cal_block.items()))
        assert cal["status"] == "ok"
        assert cal["chosen"] in {"Sigmoid", "Isotonic", "Beta", "Spline"}
        assert cal["plot_path"] and os.path.exists(cal["plot_path"]), cal


def test_reporting_config_default_honest_diagnostics_on():
    from mlframe.training._reporting_configs import ReportingConfig

    rc = ReportingConfig()
    assert rc.honest_estimator_diagnostics is True, "honest_estimator_diagnostics should default True"


def test_run_honest_diagnostics_handles_missing_test_probs():
    from mlframe.training.honest_diagnostics import run_honest_diagnostics

    entry = types.SimpleNamespace(
        model_name="dummy",
        model=None,
        test_target=None,
        test_probs=None,
    )
    models = {"BINARY_CLASSIFICATION": {"y_x": [entry]}}
    ctx = types.SimpleNamespace(
        train_df=_make_df(50, seed=1),
        val_df=_make_df(20, seed=2),
        test_df=_make_df(20, seed=3),
        data_dir="",
        models_dir="",
        models=models,
        metadata={},
    )
    metadata: dict = {}
    out = run_honest_diagnostics(ctx, models, metadata)
    boot = out["bootstrap_ci"]
    assert len(boot) == 1
    _, block = next(iter(boot.items()))
    assert block.get("status") == "skipped"
    assert "test_target" in block.get("reason", "") or "test_probs" in block.get("reason", "")
