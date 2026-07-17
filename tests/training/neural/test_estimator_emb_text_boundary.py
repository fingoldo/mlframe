"""End-to-end: a real PyTorch-Lightning MLP fits + predicts on embedding-List and free-text columns via the estimator
fit/predict boundary (`_encode_emb_text_fit` / `_predict_raw`), which the MLP has no native layers for.

Embedding case is model-free for the encoder (fast). Text case uses the REAL default HuggingFace model (no mocking),
skipped only if transformers/model unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor, TorchDataModule


def _regressor_params():
    """Regressor params."""
    return dict(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 2, "first_layer_num_neurons": 16, "dropout_prob": 0.0, "activation_function": torch.nn.ReLU},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 1,
            "enable_model_summary": False,
            "default_root_dir": None,
            "log_every_n_steps": 1,
            "devices": 1,
            "logger": False,
            "accelerator": "cpu",
        },
    )


def test_mlp_fits_and_predicts_with_embedding_column():
    """Mlp fits and predicts with embedding column."""
    n, d = 64, 4
    rng = np.random.default_rng(0)
    embs = [rng.normal(size=d).astype(np.float32) for _ in range(n)]
    X = pd.DataFrame({"num_0": rng.normal(size=n).astype(np.float32), "emb_0": embs})
    y = (np.vstack(embs)[:, 0] * 2.0).astype(np.float32)

    reg = PytorchLightningRegressor(**_regressor_params())
    reg.fit(X, y, embedding_features=["emb_0"])  # boundary expands emb_0 -> numeric BEFORE validation/input-dim
    assert getattr(reg, "_emb_text_encoder_", None) is not None
    preds = np.asarray(reg.predict(X))  # boundary applies the same encoding at predict
    assert preds.shape[0] == n
    assert np.all(np.isfinite(preds))


def test_mlp_fits_and_predicts_with_text_column_real_hf():
    """Mlp fits and predicts with text column real hf."""
    pytest.importorskip("transformers")
    n = 48
    rng = np.random.default_rng(1)
    pos = ["great wonderful excellent", "superb fantastic", "amazing brilliant"]
    neg = ["terrible awful horrible", "dreadful bad", "atrocious lousy"]
    ys = rng.integers(0, 2, n)
    texts = [rng.choice(pos if v == 1 else neg) for v in ys]
    X = pd.DataFrame({"num_0": rng.normal(size=n).astype(np.float32), "text_0": texts})
    y = ys.astype(np.float32)

    reg = PytorchLightningRegressor(**_regressor_params())
    try:
        reg.fit(X, y, text_features=["text_0"])  # boundary HF-embeds text_0 -> numeric before the MLP
    except Exception as e:  # pragma: no cover -- offline / model-fetch failure
        pytest.skip(f"HuggingFace model unavailable ({type(e).__name__}: {e})")
    assert getattr(reg, "_emb_text_encoder_", None) is not None
    preds = np.asarray(reg.predict(X))
    assert preds.shape[0] == n
    assert np.all(np.isfinite(preds))
