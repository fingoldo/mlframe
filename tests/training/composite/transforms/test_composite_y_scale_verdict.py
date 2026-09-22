"""Suite-end verdict must compare composite targets against raw y using the per-model hook's y-scale metrics.

When the end-of-target wrap pass runs with skip_wrap_pass_predict=True it records no y-scale metrics, so before the fix
every composite row in the verdict showed ``val_RMSE=-`` and the operator could not tell whether log/cbrt/residual beat raw.
"""

from __future__ import annotations

import logging
import types

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.training.core._phase_composite_post_summary import _run_suite_end_dummy_baselines_summary
from mlframe.training.core._phase_composite_wrapping import emit_per_model_composite_y_scale_test


def _fit_entry(name: str, seed: int):
    """A fitted model entry named ``name`` on a small synthetic regression."""
    rng = np.random.default_rng(seed)
    n = 90
    x = rng.normal(0, 1, n)
    y = np.exp(1.0 + 0.5 * x + rng.normal(0, 0.1, n))
    df = pd.DataFrame({"x": x})
    tr, va, te = np.arange(0, 50), np.arange(50, 70), np.arange(70, n)
    inner = Ridge(alpha=1.0).fit(df.iloc[tr], np.log1p(y[tr]))
    entry = types.SimpleNamespace(model=inner, model_name=name)
    return df, y, tr, va, te, entry


def test_hook_records_val_and_test_and_verdict_compares_with_raw(caplog) -> None:
    """The hook records y-scale val and test metrics and the summary verdict compares the composite with the raw model."""
    comp = "y-logY"
    spec = {"name": comp, "transform_name": "log_y", "base_column": None, "fitted_params": {"offset": 1.0}}
    metadata: dict = {"composite_target_specs": {"regression": {"y": [spec]}}}
    df, y, tr, va, te, entry = _fit_entry("ridge_a", 0)
    emit_per_model_composite_y_scale_test(
        entry=entry, composite_spec=spec, orig_target_name="y", composite_name=comp, target_name=comp,
        y_full=y, test_idx=te, test_df_pd=df.iloc[te], train_idx=tr, val_idx=va, val_df=df.iloc[va],
        metadata=metadata, target_type="regression",
    )
    rows = metadata["composite_target_y_scale_metrics"]["regression"][comp]
    assert len(rows) == 1 and rows[0]["model_name"] == "ridge_a"
    assert set(rows[0]["metrics"]) == {"val", "test"}
    comp_val = rows[0]["metrics"]["val"]["RMSE"]
    # Re-running the hook for the same model must not duplicate its row.
    emit_per_model_composite_y_scale_test(
        entry=entry, composite_spec=spec, orig_target_name="y", composite_name=comp, target_name=comp,
        y_full=y, test_idx=te, test_df_pd=df.iloc[te], train_idx=tr, val_idx=va, val_df=df.iloc[va],
        metadata=metadata, target_type="regression",
    )
    assert len(metadata["composite_target_y_scale_metrics"]["regression"][comp]) == 1

    comp_test = rows[0]["metrics"]["test"]["RMSE"]
    # The verdict is decided on TEST (the composite was selected on val), so the raw model carries both splits.
    raw_entry = types.SimpleNamespace(model=object(), model_name="cb_raw", metrics={"val": {"RMSE": comp_val * 2.0}, "test": {"RMSE": comp_test * 2.0}})
    ens_entry = types.SimpleNamespace(model=object(), model_name="EnsARITHM [ridge]")
    models = {"regression": {"y": [raw_entry], comp: [entry, ens_entry]}}
    rep = {"strongest": "mean", "primary_metric": "val_RMSE", "data": {"mean": {"val_RMSE": comp_val * 4.0}}}
    metadata["dummy_baselines"] = {"regression": {"y": dict(rep), comp: dict(rep)}}
    cfg = types.SimpleNamespace(best_model_min_lift=1.5)
    with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_composite_post"):
        _run_suite_end_dummy_baselines_summary(models=models, metadata=metadata, dummy_baselines_config=cfg)
    text = "\n".join(r.getMessage() for r in caplog.records)
    comp_rows = [ln for ln in text.splitlines() if ln.startswith(comp)]
    # Cross-target verdict row now carries the composite's y-scale val metric instead of "-".
    assert any(f"val_RMSE={comp_val:.4f}"[:22] in ln for ln in comp_rows), text
    # The composite-vs-raw block answers the question directly, and names the ensemble lacking y-scale metrics.
    vs = [ln for ln in comp_rows if "COMPOSITE_BEATS_RAW" in ln]
    assert vs, text
    assert "2.000x" in vs[0] and "4.000x" in vs[0] and "cb_raw" in vs[0]
    assert "no y-scale metric: EnsARITHM [ridge]" in vs[0]


def test_the_composite_vs_raw_verdict_is_decided_on_test():
    """A composite that wins on val (where discovery selected it) and loses on test is RAW_BEATS_COMPOSITE."""
    from mlframe.training.core._phase_composite_post_summary import format_composite_vs_raw_block

    comp = "y-linres-b"
    metadata = {
        "dummy_baselines": {"regression": {"y": {"strongest": "mean", "primary_metric": "val_RMSE", "data": {"mean": {"val_RMSE": 5.0}}}}},
        "composite_target_y_scale_metrics": {"regression": {comp: [{"model_name": "lgb", "metrics": {"val": {"RMSE": 1.0}, "test": {"RMSE": 3.0}}}]}},
    }
    best = {("regression", "y"): {"val_RMSE": 2.0, "model_name": "lgb_raw", "test_RMSE": 2.0}}
    text = format_composite_vs_raw_block(models={"regression": {comp: []}}, metadata=metadata, best_metrics=best, composite_to_raw={("regression", comp): "y"})
    row = next(ln for ln in text.splitlines() if ln.startswith(comp))
    assert "RAW_BEATS_COMPOSITE" in row, row
    best_no_test = {("regression", "y"): {"val_RMSE": 2.0, "model_name": "lgb_raw (test fallback)", "test_RMSE": None}}
    text = format_composite_vs_raw_block(models={"regression": {comp: []}}, metadata=metadata, best_metrics=best_no_test, composite_to_raw={("regression", comp): "y"})
    assert "NO_TEST_METRIC_TO_COMPARE" in text, "without a raw test metric the val numbers must not decide the verdict"
