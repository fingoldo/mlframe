"""End-to-end: with ``mix_offset_enabled`` the suite trains a ``<target>__mixoff`` companion and records its feature-base mix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")


def _frame(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """A frame whose target carries a strong linear term in a base column, plus a product of two noise-free columns."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 0.95 * base + 0.5 * x1 * x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"TVT_prev": base, "x1": x1, "x2": x2, "target": y})


def test_mix_offset_companion_is_trained_and_recorded(tmp_path):
    """The suite trains ``target__mixoff`` and writes its beta and source into metadata when the held-out gate accepts."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite
    from tests.training.composite.test_composite_integration import _LEAN_OUTPUT_CONFIG_KWARGS, _LEAN_REPORTING_CONFIG_KWARGS, _build_minimal_fte

    cfg = CompositeTargetDiscoveryConfig(enabled=False, mix_offset_enabled=True)
    models, metadata = train_mlframe_models_suite(
        df=_frame(),
        target_name="target",
        model_name="mix_offset_e2e",
        features_and_targets_extractor=_build_minimal_fte(),
        mlframe_models=["linear"],
        output_config={"data_dir": str(tmp_path / "data"), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
        reporting_config=_LEAN_REPORTING_CONFIG_KWARGS,
        verbose=0,
        composite_target_discovery_config=cfg,
    )
    records = metadata.get("mix_offset_targets", {})
    assert records, "the gate should accept the strong linear base; no mix_offset_targets recorded"
    rec = next(iter(next(iter(records.values())).values()))
    assert rec["source"] == "target"
    assert abs(rec["beta"] - 0.95) < 0.2
    assert "target__mixoff" in str(models)
