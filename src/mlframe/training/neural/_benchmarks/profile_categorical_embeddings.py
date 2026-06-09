"""cProfile harness for the learnable categorical-embedding path.

Profiles a full fit+predict of ``PytorchLightningRegressor`` with ``cat_features`` on a representative shape (n=20000, 6 cat cols + 10 numeric,
3 epochs CPU). Run:  PYTHONPATH=<worktree>/src python -m mlframe.training.neural._benchmarks.profile_categorical_embeddings

Conclusion (2026-06-08, this hardware): the categorical hotspots are NOT actionable. ``CategoricalEmbedding.forward`` is a per-cat Python loop
over a handful of columns (k=6) doing one ``nn.Embedding`` gather each -- it never appears in the top cumulative frames; the wall is dominated by
Lightning's training loop + DataLoader + the trunk GEMMs (the same hotspots as the no-cat MLP). The fit-boundary ``_factorize_cats_fit`` runs ONCE
per fit (``pd.factorize`` per cat column, vectorised) and is sub-millisecond at this shape -- profiling it standalone shows <1 ms, dwarfed by the
per-epoch cost. No numba/cupy ladder is warranted (k is tiny, the gather is already a fused torch kernel, and the factorize is a one-shot setup).
"""
from __future__ import annotations

import cProfile
import pstats

import numpy as np
import pandas as pd
import torch

from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor, TorchDataModule


def _make(n=20000, n_cat_cols=6, n_num=10, card=20, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    luts = []
    for j in range(n_cat_cols):
        codes = rng.integers(0, card, size=n)
        data[f"cat_{j}"] = np.array([f"v{c}" for c in codes])
        luts.append((codes, rng.normal(scale=2.0, size=card)))
    for j in range(n_num):
        data[f"num_{j}"] = rng.normal(size=n).astype(np.float32)
    X = pd.DataFrame(data)
    y = np.zeros(n, dtype=np.float32)
    for codes, lut in luts:
        y += lut[codes].astype(np.float32)
    y += X["num_0"].to_numpy()
    return X, y


def _estimator():
    return PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 3, "first_layer_num_neurons": 64, "dropout_prob": 0.0, "activation_function": torch.nn.ReLU},
        datamodule_class=TorchDataModule,
        datamodule_params={"read_fcn": None, "data_placement_device": None, "features_dtype": torch.float32,
                            "labels_dtype": torch.float32, "dataloader_params": {"batch_size": 256, "num_workers": 0}},
        trainer_params={"max_epochs": 3, "enable_model_summary": False, "default_root_dir": None,
                        "log_every_n_steps": 50, "devices": 1, "logger": False, "accelerator": "cpu",
                        "enable_progress_bar": False},
        random_state=0,
    )


def main() -> None:
    """Run the cProfile harness and print the top cumulative frames for a fit+predict of the categorical-embedding path."""
    X, y = _make()
    cat_cols = [c for c in X.columns if c.startswith("cat_")]

    def run():
        reg = _estimator()
        reg.fit(X, y, cat_features=cat_cols)
        reg.predict(X)

    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("cumulative")
    print("=== top 25 by cumulative time ===")
    st.print_stats(25)
    print("=== categorical-embedding frames ===")
    st.print_stats("categorical_embeddings|_factorize_cats|_apply_cat_codes")


if __name__ == "__main__":
    main()
