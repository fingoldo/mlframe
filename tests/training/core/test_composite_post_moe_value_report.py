"""Deploy-path integration for the composite VALUE report + MoE selection gate.

Exercises ``run_composite_moe_and_value_report`` on a synthetic where the composite ensemble wins some groups
and the lag failsafe wins others: the gated deploy must beat always-composite AND be never-worse-than-lag per
group, the value report must label the split, and the flag-off / no-lag / no-group paths must no-op cleanly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training import TargetTypes
from mlframe.training.core._phase_composite_post_moe import (
    _MoEGatedDeployableModel,
    run_composite_moe_and_value_report,
)

_TT = TargetTypes.REGRESSION
_TNAME = "y"


class _ColumnStub:
    """Predict returns a named column verbatim (a stand-in for a fitted raw / composite model)."""

    def __init__(self, column: str) -> None:
        self.column = column

    def predict(self, X):
        """Predict."""
        return np.asarray(X[self.column], dtype=np.float64)


def _build_synthetic():
    """Two groups: A -> composite perfect / lag bad; B -> lag perfect / composite bad; raw mediocre both."""
    rng = np.random.default_rng(0)
    n_per = 10
    yA = np.arange(10, 10 + n_per, dtype=np.float64)
    yB = np.arange(100, 100 + n_per, dtype=np.float64)
    y = np.concatenate([yA, yB])
    grp = np.array(["A"] * n_per + ["B"] * n_per)
    composite_pred = np.concatenate([yA, yB + 5.0])  # A perfect, B off by 5
    raw_pred = y + 3.0  # mediocre everywhere
    lagcol = np.concatenate([yA + 5.0, yB])  # A off by 5, B perfect
    df = pd.DataFrame(
        {
            "grp": grp,
            "lagcol": lagcol,
            "composite_pred": composite_pred,
            "raw_pred": raw_pred,
            "feat": rng.normal(size=len(y)),
        }
    )
    return df, y, grp


def _make_models_metadata(*, with_lag: bool = True):
    """Make models metadata."""
    composite = _ColumnStub("composite_pred")
    raw = _ColumnStub("raw_pred")
    models = {
        _TT: {
            _TNAME: [SimpleNamespace(model=raw, pre_pipeline=None)],
            f"_CT_ENSEMBLE__{_TNAME}": [SimpleNamespace(model=composite)],
        }
    }
    metadata: dict = {}
    if with_lag:
        metadata["dummy_baselines"] = {
            str(_TT): {_TNAME: {"extras": {"lag_predict": {"feature_used": "lagcol"}}}},
        }
    return models, metadata, composite


def _config(**over):
    """Config."""
    base = dict(
        group_column="grp",
        emit_composite_value_report=True,
        moe_gate_enabled=True,
        moe_gate_shrink_rtol=0.0,
        moe_gate_min_group_rows=1,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _rmse(a, b):
    """Rmse."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_moe_gate_and_value_report_end_to_end():
    """Moe gate and value report end to end."""
    df, y, grp = _build_synthetic()
    idx = np.arange(len(y))
    models, metadata, composite = _make_models_metadata()
    ctx = SimpleNamespace(group_ids=grp, sample_weights=None, timestamps=None)

    run_composite_moe_and_value_report(
        models=models,
        metadata=metadata,
        target_by_type={_TT: {_TNAME: y}},
        composite_target_discovery_config=_config(),
        filtered_train_df=df,
        filtered_val_df=df,
        filtered_train_idx=idx,
        filtered_val_idx=idx,
        ctx=ctx,
    )

    # 1. Value report emitted + labels the split (composite worse than lag in >=1 group == group B).
    report = metadata["composite_value_report"][str(_TT)][_TNAME]
    assert report["n_groups"] == 2
    assert report["has_lag"] is True
    assert report["aggregate"]["n_worse_than_lag"] >= 1
    assert "B" in report["aggregate"]["worse_than_lag_groups"]

    # 2. Deployed model wrapped in the gate; guarantee pinned.
    deployed = models[_TT][f"_CT_ENSEMBLE__{_TNAME}"][0].model
    assert isinstance(deployed, _MoEGatedDeployableModel)
    guarantee = metadata["composite_moe_gate"][str(_TT)][_TNAME]["guarantee"]
    assert guarantee["not_worse_than_lag"] is True

    # 3. End-to-end guarantee: gated deploy beats always-composite AND is never worse than lag per group.
    gated = deployed.predict(df)
    always_composite = composite.predict(df)
    lag = df["lagcol"].to_numpy()
    assert _rmse(gated, y) <= _rmse(lag, y) + 1e-9
    assert _rmse(gated, y) < _rmse(always_composite, y)
    for g in ("A", "B"):
        m = grp == g
        assert _rmse(gated[m], y[m]) <= _rmse(lag[m], y[m]) + 1e-9


def test_flags_off_leaves_deploy_unchanged():
    """Flags off leaves deploy unchanged."""
    df, y, grp = _build_synthetic()
    idx = np.arange(len(y))
    models, metadata, composite = _make_models_metadata()
    ctx = SimpleNamespace(group_ids=grp, sample_weights=None)

    run_composite_moe_and_value_report(
        models=models,
        metadata=metadata,
        target_by_type={_TT: {_TNAME: y}},
        composite_target_discovery_config=_config(
            emit_composite_value_report=False,
            moe_gate_enabled=False,
        ),
        filtered_train_df=df,
        filtered_val_df=df,
        filtered_train_idx=idx,
        filtered_val_idx=idx,
        ctx=ctx,
    )
    assert models[_TT][f"_CT_ENSEMBLE__{_TNAME}"][0].model is composite
    assert "composite_value_report" not in metadata
    assert "composite_moe_gate" not in metadata


def test_no_lag_and_no_groups_no_op_gate():
    """No lag and no groups no op gate."""
    df, y, grp = _build_synthetic()
    idx = np.arange(len(y))

    # No lag expert -> value report still builds (lag=None), but NO gate wrap.
    models, metadata, composite = _make_models_metadata(with_lag=False)
    ctx = SimpleNamespace(group_ids=grp, sample_weights=None)
    run_composite_moe_and_value_report(
        models=models,
        metadata=metadata,
        target_by_type={_TT: {_TNAME: y}},
        composite_target_discovery_config=_config(),
        filtered_train_df=df,
        filtered_val_df=df,
        filtered_train_idx=idx,
        filtered_val_idx=idx,
        ctx=ctx,
    )
    assert models[_TT][f"_CT_ENSEMBLE__{_TNAME}"][0].model is composite
    assert metadata["composite_value_report"][str(_TT)][_TNAME]["has_lag"] is False
    assert "composite_moe_gate" not in metadata

    # Lag present but NO group ids -> gate no-ops (report has_lag True, no wrap).
    models2, metadata2, composite2 = _make_models_metadata(with_lag=True)
    ctx2 = SimpleNamespace(group_ids=None, sample_weights=None)
    run_composite_moe_and_value_report(
        models=models2,
        metadata=metadata2,
        target_by_type={_TT: {_TNAME: y}},
        composite_target_discovery_config=_config(),
        filtered_train_df=df,
        filtered_val_df=df,
        filtered_train_idx=idx,
        filtered_val_idx=idx,
        ctx=ctx2,
    )
    assert models2[_TT][f"_CT_ENSEMBLE__{_TNAME}"][0].model is composite2
    assert "composite_moe_gate" not in metadata2


def test_missing_group_column_in_frame_no_ops_gate():
    # Group ids exist at fit, but the predict frame lacks the group column -> gate would route globally to lag
    # (worse than the ensemble where composite wins), so the wrap is skipped and the ensemble ships unchanged.
    """Missing group column in frame no ops gate."""
    df, y, grp = _build_synthetic()
    df = df.drop(columns=["grp"])
    idx = np.arange(len(y))
    models, metadata, composite = _make_models_metadata()
    ctx = SimpleNamespace(group_ids=grp, sample_weights=None)
    run_composite_moe_and_value_report(
        models=models,
        metadata=metadata,
        target_by_type={_TT: {_TNAME: y}},
        composite_target_discovery_config=_config(),
        filtered_train_df=df,
        filtered_val_df=df,
        filtered_train_idx=idx,
        filtered_val_idx=idx,
        ctx=ctx,
    )
    assert models[_TT][f"_CT_ENSEMBLE__{_TNAME}"][0].model is composite
    assert "composite_moe_gate" not in metadata
