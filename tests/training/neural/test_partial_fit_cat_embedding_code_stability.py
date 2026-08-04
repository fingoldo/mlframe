"""Regression test: partial_fit must NOT re-factorize categorical columns from scratch.

The CategoricalEmbedding table's cardinality/shape is frozen at the initial fit() call. Pre-fix,
``_fit_common`` called ``_factorize_cats_fit`` unconditionally on every partial_fit call too, which
rebuilds the value->code map via a fresh ``pd.factorize()`` over just that batch -- assigning codes
by THAT batch's own value order, desyncing every subsequent forward pass from the codes the
embedding table was actually trained against on the first fit() call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor, TorchDataModule


def _common():
    """Common fit-time kwargs for a tiny MLP regressor."""
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


def test_partial_fit_does_not_change_frozen_cat_code_map():
    """A second partial_fit call, with a DIFFERENT row order for the same category values, must
    not change the code map fixed by the initial fit() call."""
    rng = np.random.default_rng(0)
    n = 64
    # First batch: "blue" appears first (so pd.factorize would give it code 0 if re-run here).
    cats_1 = np.array((["blue"] * (n // 4)) + list(rng.choice(["red", "green", "yellow"], size=n - n // 4)))
    X1 = pd.DataFrame({"color": cats_1, "num": rng.normal(size=n).astype(np.float32)})
    y1 = rng.normal(size=n).astype(np.float32)

    reg = PytorchLightningRegressor(**_common())
    reg.fit(X1, y1, cat_features=["color"])
    code_maps_after_fit = {k: dict(v) for k, v in reg._cat_code_maps_.items()}
    cardinalities_after_fit = list(reg._cat_cardinalities_)

    # Second batch: "red" appears first this time (pre-fix, re-factorizing here would give "red"
    # code 0, "blue" some other code -- desynced from the embedding table trained on batch 1).
    cats_2 = np.array((["red"] * (n // 4)) + list(rng.choice(["blue", "green", "yellow"], size=n - n // 4)))
    X2 = pd.DataFrame({"color": cats_2, "num": rng.normal(size=n).astype(np.float32)})
    y2 = rng.normal(size=n).astype(np.float32)
    reg.partial_fit(X2, y2, cat_features=["color"])

    assert reg._cat_code_maps_["color"] == code_maps_after_fit["color"], "partial_fit must not change the frozen fit-time code map"
    assert list(reg._cat_cardinalities_) == cardinalities_after_fit, "partial_fit must not change the frozen fit-time cardinalities"

    preds = np.asarray(reg.predict(X1))
    assert preds.shape == (n,) and np.all(np.isfinite(preds))
