"""biz_value tests for per-iteration full-metric-suite capture (meta-learning / HPO-from-early-observation).

These assert the measurable WIN of the feature:
  1. After a real lgb fit (flag ON) the booster exposes ``iteration_metrics_`` with the full suite at the
     stride-sampled rounds, and the values at a captured round are bit-identical to a direct full-suite computation
     on that round's val predictions.
  2. After a real MLP fit the neural estimator exposes ``iteration_metrics_`` per epoch (default-ON).
  3. The meta-learning signal: early-iteration val metrics CORRELATE with the final holdout metric across configs --
     the quantitative early-observation signal that the whole feature exists to provide.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics import compute_all_metrics
from mlframe.metrics.iteration_metrics import _BINARY_METRIC_KEYS


@pytest.fixture
def binary_split():
    """Binary split."""
    rng = np.random.default_rng(7)
    n = 3000
    X = rng.normal(0, 1, (n, 8)).astype(np.float32)
    logit = X[:, 0] * 1.3 + X[:, 1] * 0.8 - X[:, 2] * 0.6 + rng.normal(0, 0.5, n)
    y = (logit > 0).astype(np.int64)
    tr, va, te = slice(0, 1800), slice(1800, 2400), slice(2400, 3000)
    return dict(Xtr=X[tr], ytr=y[tr], Xva=X[va], yva=y[va], Xte=X[te], yte=y[te])


def _lgb(n_estimators, **kw):
    """Lgb."""
    from mlframe.training.lgb_shim import LGBMClassifierWithDatasetReuse

    return LGBMClassifierWithDatasetReuse(n_estimators=n_estimators, verbose=-1, n_jobs=1, **kw)


def test_biz_val_lgb_iteration_metrics_populated_and_bit_identical(binary_split):
    """Biz val lgb iteration metrics populated and bit identical."""
    d = binary_split
    stride = 4
    n_est = 40
    m = _lgb(n_est)
    m.fit(d["Xtr"], d["ytr"], eval_set=[(d["Xva"], d["yva"])], capture_iteration_metrics=True, iteration_metrics_stride=stride)

    im = m.iteration_metrics_
    assert im, "iteration_metrics_ must be populated when capture flag is ON"
    rounds = sorted(im)
    # Every captured round carries the full binary suite.
    for r in rounds:
        assert set(im[r]) == set(_BINARY_METRIC_KEYS)
    # Stride consistency: captured rounds are a subset of {0, stride, 2*stride, ...} plus the final round.
    expected = set(range(0, n_est, stride)) | {n_est - 1}
    assert set(rounds).issubset(expected)
    assert len(rounds) >= n_est // stride - 1  # most stride points captured

    # Bit-identity at a captured round: recompute the full suite directly on that round's val predictions.
    r = rounds[1]
    direct = compute_all_metrics(d["yva"], m._Booster.predict(d["Xva"], num_iteration=r + 1), "binary_classification")
    for k in direct:
        a, b = im[r][k], direct[k]
        assert (a == b) or (np.isnan(a) and np.isnan(b)), f"metric {k} diverged: {a} != {b}"


def test_biz_val_lgb_capture_off_by_default(binary_split):
    """Biz val lgb capture off by default."""
    d = binary_split
    m = _lgb(20)
    m.fit(d["Xtr"], d["ytr"], eval_set=[(d["Xva"], d["yva"])])  # no capture flag -> OFF
    assert not getattr(m, "iteration_metrics_", {})


def test_biz_val_early_iteration_predicts_final_holdout(binary_split):
    """Meta-learning signal: the early-iteration val ROC_AUC ranks configs the same way the FINAL holdout ROC_AUC
    does. A config that looks better at iteration K should tend to be better at the end -- this is exactly the
    signal HPO-from-early-observation exploits to prune bad configs early. We assert a strong rank correlation."""
    d = binary_split
    # Observe at the first ~25% of the budget (round 15 of 60) -- the regime HPO-from-early-observation prunes in.
    early_round = 15
    n_est = 60
    early_aucs, final_aucs = [], []
    # Sweep learning_rate, which strongly orders configs by how fast/well they fit at a fixed round budget.
    for lr in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        m = _lgb(n_est, num_leaves=15, learning_rate=lr)
        m.fit(d["Xtr"], d["ytr"], eval_set=[(d["Xva"], d["yva"])], capture_iteration_metrics=True, iteration_metrics_stride=1)
        im = m.iteration_metrics_
        assert early_round in im
        early_aucs.append(im[early_round]["ROC_AUC"])
        # Final honest holdout (model never saw the test split): direct full-suite computation.
        final = compute_all_metrics(d["yte"], m.predict_proba(d["Xte"])[:, 1], "binary_classification")
        final_aucs.append(final["ROC_AUC"])

    from scipy.stats import spearmanr

    rho, _ = spearmanr(early_aucs, final_aucs)
    # Measured rho ~1.0 (LR strictly orders early & final fit quality at a fixed round budget); floor at 0.6
    # absorbs seed noise while still catching a broken capture (rho ~0 would mean the early metric is garbage).
    assert rho >= 0.6, f"early-iter ROC_AUC must predict final holdout ranking; spearman rho={rho:.3f}"


def test_biz_val_mlp_iteration_metrics_populated():
    """Biz val mlp iteration metrics populated."""
    import torch

    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningClassifier,
        TorchDataModule,
    )

    rng = np.random.default_rng(3)
    n = 600
    X = rng.normal(0, 1, (n, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.normal(0, 0.4, n) > 0).astype(np.int64)
    Xtr, ytr, Xva, yva = X[:450], y[:450], X[450:], y[450:]

    clf = PytorchLightningClassifier(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.CrossEntropyLoss(), "learning_rate": 1e-2},
        network_params={
            "nlayers": 1,
            "first_layer_num_neurons": 16,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        datamodule_class=TorchDataModule,
        datamodule_params={
            "features_dtype": torch.float32,
            "labels_dtype": torch.int64,
            "dataloader_params": {"batch_size": 64, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 4,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
        random_state=0,
    )
    clf.fit(Xtr, ytr, eval_set=(Xva, yva))

    im = getattr(clf, "iteration_metrics_", None)
    assert im, "neural iteration_metrics_ must be populated by default (capture is ON for neural)"
    # Each captured epoch carries the binary suite.
    for metrics in im.values():
        assert set(metrics) == set(_BINARY_METRIC_KEYS)
        assert np.isfinite(metrics["log_loss"])
