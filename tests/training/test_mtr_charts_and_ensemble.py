"""F-34 final pieces: E1 (per-target K-grid charts) + E2 (per-column
CT_ENSEMBLE)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlframe.training import TargetTypes
from mlframe.training.core._phase_composite_post_xt_ensemble import (
    MTRPerColumnEqualMeanEnsemble,
    _build_cross_target_ensemble_for_target,
    _build_mtr_per_column_ensemble,
)


# --- E2: MTRPerColumnEqualMeanEnsemble ---------------------------------------


class _StubComponent:
    """Tiny test double: predict() returns a fixed (N, K) array."""

    def __init__(self, preds):
        self._preds = np.asarray(preds)

    def predict(self, X):
        return self._preds


def test_mtr_ensemble_equal_mean_2_components():
    """Two components with predictions [(0, 0)] and [(2, 4)] -> mean (1, 2)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    c1 = _StubComponent(np.zeros((5, 2)))
    c2 = _StubComponent(np.tile([2.0, 4.0], (5, 1)))
    ens = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2], component_names=["c1", "c2"], n_targets=2,
    )
    preds = ens.predict(X)
    assert preds.shape == (5, 2)
    np.testing.assert_allclose(preds, np.tile([1.0, 2.0], (5, 1)))


def test_mtr_ensemble_promotes_1d_components_to_2d():
    """Per the contract: if a component returns (N,) it's promoted to
    (N, 1). Mixed (N,) and (N, 1) components must produce consistent
    shape; this test only verifies the (N,) -> (N, 1) promotion path."""
    c = _StubComponent(np.array([1.0, 2.0, 3.0]))  # (N=3,)
    ens = MTRPerColumnEqualMeanEnsemble(
        components=[c], component_names=["c1"], n_targets=1,
    )
    preds = ens.predict(np.zeros((3, 2)))
    assert preds.shape == (3, 1)


def test_mtr_ensemble_empty_components_raises():
    with pytest.raises(ValueError, match=r"at least 1 component"):
        MTRPerColumnEqualMeanEnsemble(
            components=[], component_names=[], n_targets=2,
        )


def test_mtr_ensemble_repr_includes_counts():
    c1 = _StubComponent(np.zeros((4, 2)))
    c2 = _StubComponent(np.ones((4, 2)))
    ens = MTRPerColumnEqualMeanEnsemble(
        components=[c1, c2], component_names=["alpha", "beta"], n_targets=2,
    )
    r = repr(ens)
    assert "n_components=2" in r
    assert "n_targets=2" in r
    assert "alpha" in r and "beta" in r


# --- E2 build helper ----------------------------------------------------------


def _make_models_dict_with_components(n_components=3, n_targets=2, n=20):
    """Build a synthetic ``models`` dict that looks like the suite-side
    structure: models[target_type][target_name] = [SimpleNamespace(
        model=<predict-capable>, pre_pipeline=None)]."""
    target_type = TargetTypes.MULTI_TARGET_REGRESSION
    target_name = "mtr_target"
    rng = np.random.default_rng(0)
    entries = []
    for i in range(n_components):
        comp_preds = rng.normal(loc=float(i), size=(n, n_targets))
        entries.append(SimpleNamespace(
            model=_StubComponent(comp_preds),
            pre_pipeline=None,
        ))
    return {target_type: {target_name: entries}}, target_type, target_name


def test_build_mtr_per_column_ensemble_appends_entry():
    """When 2+ components are present, the helper appends a CT_ENSEMBLE
    entry to models[target_type][target_name] and stamps metadata."""
    models, tt, tn = _make_models_dict_with_components(n_components=3, n_targets=2)
    metadata = {}
    target_by_type = {tt: {tn: np.zeros((20, 2))}}

    _build_mtr_per_column_ensemble(
        _tt_e=tt, _orig_tname=tn,
        models=models, metadata=metadata, target_by_type=target_by_type,
    )

    # Original 3 + 1 new ensemble entry.
    assert len(models[tt][tn]) == 4
    ens_entry = models[tt][tn][-1]
    assert getattr(ens_entry, "ct_ensemble", False) is True
    assert getattr(ens_entry, "mtr_ensemble", False) is True
    assert getattr(ens_entry, "ensemble_strategy", "") == "per_column_equal_mean"
    assert getattr(ens_entry, "n_components", 0) == 3
    # Metadata recorded.
    assert "mtr_ct_ensemble" in metadata
    assert tn in metadata["mtr_ct_ensemble"][str(tt)]


def test_build_mtr_per_column_ensemble_skips_when_single_component(caplog):
    """A single-component pool is NOT an ensemble (no averaging
    possible); helper logs INFO and returns without mutating."""
    models, tt, tn = _make_models_dict_with_components(n_components=1, n_targets=2)
    metadata = {}
    target_by_type = {tt: {tn: np.zeros((20, 2))}}
    with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_composite_post"):
        _build_mtr_per_column_ensemble(
            _tt_e=tt, _orig_tname=tn,
            models=models, metadata=metadata, target_by_type=target_by_type,
        )
    # Models length unchanged (only the 1 original component).
    assert len(models[tt][tn]) == 1
    assert metadata == {}
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("only 1 component" in m for m in msgs)


def test_build_mtr_per_column_ensemble_predict_averages_components():
    """End-to-end: build the ensemble, call its predict, verify it
    averages across the 3 stub components."""
    models, tt, tn = _make_models_dict_with_components(n_components=3, n_targets=2)
    metadata = {}
    target_by_type = {tt: {tn: np.zeros((20, 2))}}
    _build_mtr_per_column_ensemble(
        _tt_e=tt, _orig_tname=tn,
        models=models, metadata=metadata, target_by_type=target_by_type,
    )
    ens = models[tt][tn][-1].model
    # The stub components emit predictions with loc=0, 1, 2 across the
    # n=20 rows. The mean across (loc=0,1,2) per element is 1.0.
    pred = ens.predict(np.zeros((20, 5)))
    assert pred.shape == (20, 2)
    # Per-row mean of the three component locs (0+1+2)/3 = 1; with
    # gaussian noise across rows, the COLUMN mean should be near 1.
    assert abs(float(pred.mean()) - 1.0) < 0.3


# --- E2 integration with the CT_ENSEMBLE dispatcher ---------------------------


def test_ct_ensemble_dispatcher_routes_mtr_to_per_column_path():
    """When _build_cross_target_ensemble_for_target is called for MTR,
    the dispatch enters the per-column ensemble path and appends an
    ensemble entry (instead of just warning + skipping)."""
    models, tt, tn = _make_models_dict_with_components(n_components=3, n_targets=2)
    metadata = {}
    cache = {}

    _build_cross_target_ensemble_for_target(
        _tt_e=tt,
        _orig_tname=tn,
        _spec_list=[],
        _ce_strategy="weighted_mean",
        models=models,
        metadata=metadata,
        target_by_type={tt: {tn: np.zeros((20, 2))}},
        composite_target_discovery_config=None,
        target_name=tn,
        model_name="test",
        filtered_train_df=None,
        filtered_val_df=None,
        test_df_pd=None,
        filtered_train_idx=None,
        filtered_val_idx=None,
        test_idx=None,
        train_df_pd=None,
        val_df_pd=None,
        train_idx=None,
        val_idx=None,
        reporting_config=None,
        plot_file=None,
        _train_pred_cache=cache,
    )
    # 3 originals + 1 ensemble = 4.
    assert len(models[tt][tn]) == 4
    assert "mtr_ct_ensemble" in metadata


# --- E1: per-target K-grid charts ---------------------------------------------


def test_mtr_per_target_charts_render_k_files(tmp_path):
    """When plot_outputs + plot_file are set on the MTR report path,
    one chart file per target column is rendered under
    ``{base}_target{k}{ext}``."""
    from mlframe.training.reporting._reporting_regression import report_regression_model_perf

    rng = np.random.default_rng(0)
    n, k = 80, 3
    targets = rng.normal(size=(n, k)).astype(np.float32)
    preds = targets + 0.05 * rng.normal(size=(n, k)).astype(np.float32)
    plot_file = str(tmp_path / "mtr_chart.png")
    metrics = {}
    report_regression_model_perf(
        targets=targets, columns=["f1", "f2", "f3"],
        model_name="test_model", model=None, preds=preds,
        metrics=metrics, print_report=False, show_perf_chart=False,
        plot_outputs="matplotlib[png]", plot_file=plot_file,
    )
    expected = [tmp_path / f"mtr_chart_target{i}.png" for i in range(k)]
    rendered = sorted(p.name for p in tmp_path.iterdir())
    assert len(rendered) >= k, (
        f"expected at least {k} chart files; got {rendered}"
    )
    for ep in expected:
        assert ep.exists(), f"missing per-target chart {ep.name}; have {rendered}"
