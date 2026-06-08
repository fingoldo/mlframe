"""biz_value: learnable categorical entity embeddings recover a NON-MONOTONE category->target mapping that a single ordinal numeric column cannot.

Synthetic: ``y = lookup[cat] + noise`` where ``lookup`` is a random per-category value with NO ordinal structure (8 categories). The learnable-
embedding MLP (raw string cat + ``cat_features``) gets each category a free embedding vector; the naive baseline feeds the same category as ONE
ordinal numeric column, so a smooth MLP can only fit a monotone-ish function of the arbitrary code order and leaves most of the signal on the table.

Measured (3 seeds, n=2000, 40 epochs CPU): embedding R^2 ~0.982 vs ordinal R^2 ~0.68-0.72, delta 0.265-0.300. Floors set ~15-30% below the
minimum measured so a real regression (embeddings disabled / factorizer broken / hook unwired) trips the win, but seed variance does not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from sklearn.metrics import r2_score

from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor, TorchDataModule

N_CAT = 8
N = 2000
EPOCHS = 40


def _common():
    return dict(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 5e-3},
        network_params={"nlayers": 2, "first_layer_num_neurons": 32, "dropout_prob": 0.0, "activation_function": torch.nn.ReLU},
        datamodule_class=TorchDataModule,
        datamodule_params={"read_fcn": None, "data_placement_device": None, "features_dtype": torch.float32,
                            "labels_dtype": torch.float32, "dataloader_params": {"batch_size": 64, "num_workers": 0}},
        trainer_params={"max_epochs": EPOCHS, "enable_model_summary": False, "default_root_dir": None,
                        "log_every_n_steps": 1, "devices": 1, "logger": False, "accelerator": "cpu",
                        "enable_progress_bar": False},
        random_state=0,
    )


def _build(seed=0):
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, N_CAT, size=N)
    # Non-monotone per-category target value: a random lookup with no ordinal structure.
    lut = np.random.default_rng(123).normal(scale=3.0, size=N_CAT)
    y = (lut[cats] + rng.normal(scale=0.3, size=N)).astype(np.float32)
    labels = np.array([f"cat_{c}" for c in cats])
    idx = np.arange(N)
    np.random.default_rng(99).shuffle(idx)
    cut = int(N * 0.7)
    return cats, labels, y, idx[:cut], idx[cut:]


def test_biz_val_mlp_learnable_cat_embeddings_beat_ordinal_baseline():
    cats, labels, y, tr, te = _build(seed=0)

    # Learnable embeddings: raw string cat column, factorized + embedded end-to-end inside the estimator.
    X_emb = pd.DataFrame({"color": labels})
    reg_emb = PytorchLightningRegressor(**_common())
    reg_emb.fit(X_emb.iloc[tr], y[tr], cat_features=["color"], eval_set=(X_emb.iloc[te], y[te]))
    r2_emb = r2_score(y[te], np.asarray(reg_emb.predict(X_emb.iloc[te])))

    # Naive baseline: the same category as ONE ordinal numeric column (no cat_features -> plain numeric input, no embedding).
    X_ord = pd.DataFrame({"color_code": cats.astype(np.float32)})
    reg_ord = PytorchLightningRegressor(**_common())
    reg_ord.fit(X_ord.iloc[tr], y[tr], eval_set=(X_ord.iloc[te], y[te]))
    r2_ord = r2_score(y[te], np.asarray(reg_ord.predict(X_ord.iloc[te])))

    # Floor on the absolute embedding quality (measured ~0.982; floor 0.90) AND on the delta over the ordinal baseline (measured 0.265-0.300;
    # floor 0.18 ~= 30% below minimum). A regression that disables the embedding / breaks the factorizer drops r2_emb toward r2_ord, failing both.
    assert r2_emb >= 0.90, f"learnable-embedding R^2 should be >=0.90 on the non-monotone target, got {r2_emb:.4f}"
    assert r2_emb - r2_ord >= 0.18, (
        f"learnable embeddings should beat the ordinal-numeric baseline by >=0.18 R^2 "
        f"(measured 0.265-0.300); got emb={r2_emb:.4f} ord={r2_ord:.4f} delta={r2_emb - r2_ord:.4f}"
    )
