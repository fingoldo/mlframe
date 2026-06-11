"""cProfile harness for the learnable categorical-embedding path.

Profiles a full fit+predict of ``PytorchLightningRegressor`` with ``cat_features`` on a representative shape (n=20000, 6 cat cols + 10 numeric,
3 epochs CPU). Run:  PYTHONPATH=<worktree>/src python -m mlframe.training.neural._benchmarks.profile_categorical_embeddings

Conclusion (this hardware, n=20000 / 6 cat + 10 numeric / 3 epochs CPU; warm, 3-run mean): the categorical hotspots are NOT actionable. The three
cat-specific frames sum to under 2% of the fit+predict wall, well below the project's >5%-of-wall / >10ms actionability bar:
  * ``CategoricalEmbedding.forward`` -- 239 calls, ~1.4% cumulative (~0.12-0.17s). Its own body is a tiny per-cat Python loop over k=6 columns; the
    time inside it is all fused torch primitives the gather genuinely needs -- ``torch.embedding`` (the lookup), ``.long()`` cast, ``clamp_`` (unknown-row
    safety), and the final ``torch.cat``. There is no wasted Python overhead to strip and no per-element kernel to write (k is tiny, embedding is already fused).
  * ``_apply_cat_codes`` -- runs ONCE per predict, ~0.30% (~0.03s): vectorised ``Series.map`` + reorder, sub-millisecond per cat column.
  * ``_factorize_cats_fit`` -- runs ONCE per fit, ~0.15% (~0.01s): ``pd.factorize`` per cat column, vectorised one-shot setup.
The wall is dominated by the SAME frames as a no-cat MLP -- ``run_backward`` (~0.33s tottime), the Adam step (~0.12s), the trunk ``linear`` GEMMs (~0.10s),
dropout, and Lightning's per-step hook dispatch. No numba/cupy ladder is warranted: the only cat-side compute is one fused embedding gather over k=6 leading
columns plus a one-shot factorize. Re-profile only if k grows by an order of magnitude (then the per-cat Python loop in ``forward`` could be vectorised across
tables) or if the factorize moves into a per-batch path (it must stay a one-shot fit-boundary setup).
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
