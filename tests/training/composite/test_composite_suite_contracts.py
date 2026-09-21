"""Discriminating end-to-end contracts for a composite-target suite.

The older integration tests assert ranges the y-clip guarantees on its own (``0 < RMSE < 100``, predictions inside 0.5x /
1.5x the y range), so a composite almost twice as bad as raw, a T-scale leak into a predict entry point, an ensemble built
on a broken OOF surface, or a crashing value report all passed. These contracts would catch each of them. One suite run
per model family is shared by every contract (module scope), on a fixture where ``diff`` / ``linear_residual`` on the
lag base is the true generating law.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig, TargetTypes
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.core._predict_main_from_models import predict_from_models

from tests.training.composite.test_composite_integration import (
    _LEAN_OUTPUT_CONFIG_KWARGS,
    _LEAN_REPORTING_CONFIG_KWARGS,
    _build_minimal_fte,
    _tvt_dataset,
)

_ADDITIVE = ("diff", "linear_residual", "additive_residual")


class _Collect(logging.Handler):
    """Collect every log record's message during the suite run."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        """Keep the formatted message."""
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module", params=["linear", "lgb"])
def suite(request, tmp_path_factory):
    """Train one composite suite for the family and keep its models, metadata and logs."""
    handler = _Collect()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        tmp = tmp_path_factory.mktemp(f"suite_{request.param}")
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, base_candidates=["TVT_prev"], transforms=["diff", "linear_residual"], mi_sample_n=200,
            top_k_after_mi=2, eps_mi_gain=-1.0, cross_target_ensemble_strategy="oof_weighted", skip_wrap_pass_predict=False,
        )
        models, metadata = train_mlframe_models_suite(
            df=_tvt_dataset(n=800), target_name="target", model_name="contracts",
            features_and_targets_extractor=_build_minimal_fte(), mlframe_models=[request.param],
            output_config={"data_dir": str(tmp / "data"), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
            reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
        )
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    return models, metadata, handler.messages


def _val_rmse(entry) -> float:
    """The val RMSE recorded on a suite entry."""
    return float((getattr(entry, "metrics", {}) or {})["val"]["RMSE"])


def _composite_keys(models) -> list[str]:
    """Composite-target entry names under regression."""
    return [k for k in models[TargetTypes.REGRESSION] if k != "target" and not str(k).startswith("_CT_ENSEMBLE__")]


def test_additive_composites_record_the_same_rmse_on_both_scales(suite):
    """For an additive transform, y - y_hat == T - T_hat, so the recorded y-scale and T-scale RMSE must agree."""
    models, metadata, _ = suite
    ysm = metadata["composite_target_y_scale_metrics"]["regression"]
    checked = 0
    for key in _composite_keys(models):
        entry = models[TargetTypes.REGRESSION][key][0]
        if not any(f"-{t_abbr}-" in key for t_abbr in ("diff", "linres", "addres")):
            continue
        y_scale = ysm[key][0]["metrics"]
        splits = [s for s in ("train", "val", "test") if "RMSE" in (entry.metrics.get(s) or {}) and "RMSE" in (y_scale.get(s) or {})]
        assert {"val", "test"} <= set(splits), f"{key}: both scales must record val and test RMSE; got {splits}"
        for split in splits:
            t_rmse = float(entry.metrics[split]["RMSE"])
            assert abs(float(y_scale[split]["RMSE"]) - t_rmse) < 1e-6, f"{key} {split}: y-scale {y_scale[split]['RMSE']} vs T-scale {t_rmse}"
        checked += 1
    assert checked, "the fixture must yield at least one additive composite"


def test_each_composite_is_within_20_percent_of_raw_on_val(suite):
    """The generating law is additive in the lag base, so no composite may be materially worse than raw y."""
    models, metadata, _ = suite
    raw = _val_rmse(models[TargetTypes.REGRESSION]["target"][0])
    ysm = metadata["composite_target_y_scale_metrics"]["regression"]
    for key in _composite_keys(models):
        comp = float(ysm[key][0]["metrics"]["val"]["RMSE"])
        assert comp <= 1.2 * raw, f"{key}: y-scale val RMSE {comp:.4f} vs raw {raw:.4f}"


def test_predict_from_models_serves_composites_on_the_y_scale(suite):
    """The in-memory predict entry point returns finite y-scale predictions for every composite, tracking raw."""
    models, metadata, _ = suite
    df_new = _tvt_dataset(n=800, seed=5)
    out = predict_from_models(df_new, models, metadata, features_and_targets_extractor=_build_minimal_fte(), verbose=0)
    preds = out["predictions"]
    def _pred_key(name):
        """The prediction key for an entry: ``regression_<name>`` with or without a model-name suffix."""
        return next((k for k in preds if k == f"regression_{name}" or k.startswith(f"regression_{name}_")), None)

    raw_key = _pred_key("target")
    assert raw_key is not None, f"no raw prediction; got {list(preds)}"
    raw = np.asarray(preds[raw_key], dtype=np.float64)
    y_mean = float(df_new["target"].mean())
    for key in _composite_keys(models):
        pred_key = _pred_key(key)
        assert pred_key is not None, f"predict_from_models dropped composite {key}; got {list(preds)}"
        comp = np.asarray(preds[pred_key], dtype=np.float64)
        assert np.isfinite(comp).all()
        assert abs(comp.mean() - y_mean) < 0.2 * abs(y_mean), f"{key} is not on the y scale: mean {comp.mean():.3f} vs y {y_mean:.3f}"
        assert np.corrcoef(comp, raw)[0, 1] > 0.9


def test_the_deployed_ensemble_is_not_worse_than_raw(suite):
    """Whatever the ensemble chose (weights or single best), its recorded val RMSE stays within 20% of raw's.

    A broken OOF weighting surface - the audit measured component OOF RMSE 17x its direct holdout RMSE - picks bad
    weights or a bad fallback, which shows up here as a val RMSE far above raw.
    """
    models, metadata, _ = suite
    raw = _val_rmse(models[TargetTypes.REGRESSION]["target"][0])
    ens = metadata["cross_target_ensemble_metrics"]["regression"]["target"]
    assert float(ens["val_RMSE"]) <= 1.2 * raw, f"CT_ENSEMBLE val RMSE {ens['val_RMSE']:.4f} vs raw {raw:.4f}"


def test_the_value_report_builds(suite):
    """A composite suite must produce its value report, not log that the build failed."""
    _, metadata, messages = suite
    assert not [m for m in messages if "report build failed" in m], "the composite value report failed to build"
    assert metadata.get("composite_value_report"), "the composite value report is missing from metadata"
