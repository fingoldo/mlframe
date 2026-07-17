"""Regression sensors for the 2026-05-21 TVT MLP collapse incident.

Context: a 4M-row TVT regression run with mlp + linear in the suite produced

  Linear  MAE= 7.89  RMSE=11.63  R2= 1.00   (Ridge, perfect)
  MLP     MAE=1408.88 RMSE=1547.75 R2=-4.75 (catastrophic collapse)

Scatter plot showed MLP predictions tightly clustered at ~10000-10100 while
true values spanned 10000-13000 (a "constant-bias" failure mode). val_MSE
during training was 0.0602 (excellent on val), but test catastrophically
diverged. Root cause: ``MLPTorchModel`` default ``use_layernorm=True``
applies LayerNorm to the input features per-row, which destroys the
inter-row absolute-scale signal that the upstream pre-pipeline's
``StandardScaler`` carefully created. With a short time budget + group-
aware split + strong auto-regressive target signal (y ~= TVT_prev), the
MLP collapses to its final-layer bias and outputs near-constant
predictions.

Fix landed in trainer.py: ``use_layernorm=False`` in the suite-level
``mlp_network_params`` default. Users wanting LN can opt in via
``mlp_kwargs["network_params"]["use_layernorm"]=True``.

These tests:

1. ``test_mlp_ttr_no_collapse_on_autoregressive_data`` - MLP on synthetic
   group-split data with near-perfect AR signal must produce predictions
   with std >= 0.5 * y_std AND test R^2 >= 0.5. Pre-fix this failed with
   the default LayerNorm path; post-fix it passes consistently.

2. ``test_regression_collapse_sensor_fires`` - the sensor in
   ``_reporting.py`` must emit a HARD WARNING when a model returns
   collapsed predictions. We simulate this by constructing a stub with
   constant predictions and verifying the warning text mentions
   ``regression-collapse-sensor`` and the actionable fix hint.

3. ``test_mlp_default_use_layernorm_is_false`` - structural assertion
   that the suite-level MLP construction passes ``use_layernorm=False``
   so a silent default flip back to True would trip the gate.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def _ar_group_split_data():
    """Synthetic auto-regressive target with group structure.

    Mirrors the production TVT setup at small scale: 100 wells x 200 rows,
    y near-perfect linear in TVT_prev, group-aware split. Pre-fix this
    fixture's MLP test fails with R^2 < 0; post-fix passes.
    """
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n_groups, rpg = 100, 200
    group_ids = np.repeat(np.arange(n_groups), rpg)
    n = n_groups * rpg
    group_means = rng.uniform(10000, 13000, n_groups).astype(np.float32)
    tvt_prev = (group_means[group_ids] + rng.normal(0, 50, n)).astype(np.float32)
    y = (tvt_prev + rng.normal(0, 11, n)).astype(np.float32)
    extras = {f"f{i}": rng.normal(size=n).astype(np.float32) for i in range(20)}
    X_full = np.column_stack(
        [tvt_prev, *list(extras.values())],
    ).astype(np.float32)
    feat_cols = ["TVT_prev", *list(extras.keys())]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trv_idx, te_idx = next(gss.split(X_full, y, groups=group_ids))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    v_inner, tr_inner = next(gss2.split(X_full[trv_idx], y[trv_idx], groups=group_ids[trv_idx]))
    tr_idx = trv_idx[tr_inner]
    va_idx = trv_idx[v_inner]

    scaler = StandardScaler().fit(X_full[tr_idx])
    X_tr = scaler.transform(X_full[tr_idx]).astype(np.float32)
    X_va = scaler.transform(X_full[va_idx]).astype(np.float32)
    X_te = scaler.transform(X_full[te_idx]).astype(np.float32)
    return (
        pd.DataFrame(X_tr, columns=feat_cols),
        pd.DataFrame(X_va, columns=feat_cols),
        pd.DataFrame(X_te, columns=feat_cols),
        y[tr_idx],
        y[va_idx],
        y[te_idx],
    )


def test_mlp_ttr_no_collapse_on_autoregressive_data(_ar_group_split_data):
    """MLP+TTR must NOT collapse to near-constant predictions on a near-
    perfect AR target with group-aware split. Pre-2026-05-21 the default
    ``use_layernorm=True`` collapsed predictions to a tight cluster on
    test (std < 0.1 * y_std, R^2 < 0); post-fix predictions track y
    closely enough to beat the constant-mean baseline.
    """
    pytest.importorskip("lightning.pytorch")
    pytest.importorskip("torch")

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    from mlframe.training.neural.flat import (
        MLPNeuronsByLayerArchitecture,
        MLPTorchModel,
    )
    from mlframe.training.neural.base import PytorchLightningRegressor
    from mlframe.training.neural.data import TorchDataModule
    from mlframe.training.targets._ttr_eval_set_scaling import _TTRWithEvalSetScaling

    X_tr_df, X_va_df, X_te_df, y_tr, y_va, y_te = _ar_group_split_data

    raw_mlp = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params=dict(
            loss_fn=F.mse_loss,
            learning_rate=3e-3,
            l1_alpha=0.0,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={},
            lr_scheduler=None,
            lr_scheduler_kwargs={},
        ),
        network_params=dict(
            nlayers=4,
            first_layer_num_neurons=128,
            dropout_prob=0.0,
            inputs_dropout_prob=0.0,
            use_layernorm=False,  # the post-2026-05-21 suite-default
            activation_function=torch.nn.LeakyReLU,
            neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
            consec_layers_neurons_ratio=2.0,
        ),
        datamodule_class=TorchDataModule,
        datamodule_params=dict(
            read_fcn=None,
            data_placement_device=None,
            features_dtype=torch.float32,
            labels_dtype=torch.float32,
            dataloader_params=dict(
                num_workers=0,
                pin_memory=False,
                batch_size=1024,
                shuffle=False,
            ),
        ),
        trainer_params=dict(
            max_epochs=8,
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=False,
            accelerator="cpu",
        ),
        early_stopping_rounds=8,
    )
    mlp = _TTRWithEvalSetScaling(regressor=raw_mlp, transformer=StandardScaler())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(X_tr_df, y_tr, eval_set=(X_va_df, y_va))
    y_pred = mlp.predict(X_te_df)

    pred_std = float(np.std(y_pred))
    y_std = float(np.std(y_te))
    r2 = float(r2_score(y_te, y_pred))

    # The post-fix MLP gets R^2 ~ 0.98 on this fixture. The collapse mode
    # (pre-fix) gives R^2 < 0 with pred_std < 0.1 * y_std. Floor of 0.5
    # is loose enough to absorb run-to-run noise (different lightning
    # callback orderings, optimiser seed drift) without missing a
    # regression to the collapse mode.
    assert pred_std >= 0.5 * y_std, (
        f"predictions collapsed: pred_std={pred_std:.1f} "
        f"({100 * pred_std / y_std:.1f}% of y_std={y_std:.1f}). "
        f"The MLP is emitting near-constant values -- check use_layernorm "
        f"default + LN_in suitability for this data."
    )
    assert r2 >= 0.5, (
        f"MLP test R^2={r2:.3f} is below the 0.5 floor on a near-perfect "
        f"AR signal. Pre-2026-05-21 this dropped to R^2 ~= -4.75 with the "
        f"default ``use_layernorm=True``; post-fix the same fixture "
        f"reaches R^2 >= 0.95."
    )


def test_regression_collapse_sensor_fires(caplog):
    """The sensor in ``report_regression_model_perf`` must emit a HARD
    WARNING when predictions are collapsed (low std + R^2 < 0). Simulate
    by passing a near-constant ``preds`` array."""
    from mlframe.training.reporting._reporting import report_regression_model_perf

    rng = np.random.default_rng(42)
    n = 200
    targets = rng.normal(11500, 645, n).astype(np.float32)
    # Collapsed predictions: constant ~10050 with tiny jitter (matches
    # the TVT production scatter pattern: pred_std ~ 0.17 * y_std).
    preds = (10050 + rng.normal(0, 5, n)).astype(np.float32)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mlframe.training.reporting._reporting"):
        report_regression_model_perf(
            targets=targets,
            columns=["dummy"],
            model_name="collapsed_mlp_repro",
            model=None,
            preds=preds,
            show_perf_chart=False,
            print_report=False,
            verbose=False,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("regression-collapse-sensor" in m for m in msgs), f"Expected `[regression-collapse-sensor] ...` WARNING; got: {msgs}"
    # The sensor warning must include an actionable mitigation hint -- the
    # current message lists (a) composite-target discovery, (b) tree booster,
    # (c) group-aware split verification, so accept any of those keywords.
    # Pre-2026-05-22 the hint was "use_layernorm"; the rewrite swapped to a
    # more comprehensive set rooted in the actual production failure mode
    # (group-aware split + feature distribution shift drives the collapse,
    # NOT a missing layernorm).
    assert any(("composite-target" in m) or ("tree booster" in m) or ("group-aware split" in m) for m in msgs), (
        f"Sensor warning must hint at a fix; got: {msgs}"
    )


def test_regression_collapse_sensor_silent_on_healthy_predictions(caplog):
    """The sensor must NOT fire when the model produces reasonable
    predictions (R^2 > 0 OR pred_std close to y_std). Guards against
    sensor noise on legitimate runs."""
    from mlframe.training.reporting._reporting import report_regression_model_perf

    rng = np.random.default_rng(42)
    n = 200
    targets = rng.normal(11500, 645, n).astype(np.float32)
    # Healthy predictions: targets + small noise (R^2 ~ 0.99).
    preds = (targets + rng.normal(0, 50, n)).astype(np.float32)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mlframe.training.reporting._reporting"):
        report_regression_model_perf(
            targets=targets,
            columns=["dummy"],
            model_name="healthy_model_repro",
            model=None,
            preds=preds,
            show_perf_chart=False,
            print_report=False,
            verbose=False,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("regression-collapse-sensor" in m for m in msgs), f"Sensor should NOT fire on healthy predictions; got noise: {msgs}"


def test_mlp_suite_default_use_layernorm_is_false():
    """Structural pin: the suite-level MLP construction MUST set
    ``use_layernorm=False`` in its default ``network_params`` dict.
    Catches a future revert of the 2026-05-21 fix that would silently
    re-introduce the collapse mode on group-split tabular regression.

    The pin reads the source instead of running the suite because the
    suite path is non-trivial to invoke standalone and the default
    lives in a literal dict in trainer.py.
    """
    from pathlib import Path

    src = Path("src/mlframe/training/trainer.py").read_text(encoding="utf-8")
    # Find the mlp_network_params block (literal dict near line ~1262).
    assert "mlp_network_params = dict(" in src, (
        "trainer.py: mlp_network_params dict literal missing -- the suite-level MLP construction was refactored. Update the pin."
    )
    # Extract the dict block (between ``mlp_network_params = dict(`` and
    # the matching closing paren).
    start = src.index("mlp_network_params = dict(")
    # Walk forward to balanced close-paren.
    depth = 0
    end = start
    for i in range(start + len("mlp_network_params = dict"), len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    block = src[start : end + 1]
    assert "use_layernorm=False" in block, (
        f"trainer.py mlp_network_params block must contain "
        f"``use_layernorm=False`` (post-2026-05-21 fix). Pre-fix default "
        f"was True which collapsed MLP on group-split tabular regression "
        f"with strong AR signal. Block was:\n{block}"
    )
