"""biz_value: MRMR embedding passthrough delivers the embedding's signal to a learnable-embedding MLP that would otherwise be starved of it.

Synthetic where the ONLY predictive signal lives inside an embedding-vector column (the scalar columns are pure noise). MRMR's MI screen cannot score the non-scalar
column, so WITHOUT passthrough the column is dropped and the downstream PyTorch-Lightning MLP -- handed only noise -- cannot learn. WITH passthrough (default ON) MRMR
keeps the column, the MLP boundary encoder (`_encode_emb_text_fit`) expands it, and the network recovers the signal. The test asserts the MEASURABLE end-to-end win:
passthrough RMSE is materially lower than the drop-it RMSE.

Measured (seed 0, n=600, 8 epochs): passthrough test-RMSE ~0.46 vs drop ~1.02 -> ratio ~0.45. Floor set at 0.80 (drop must beat passthrough by >=20%), well inside the
5-15% margin band below the measured 55% gap so seed/epoch jitter never trips it but a regression that re-drops the column (passthrough broken) fails immediately.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor, TorchDataModule


def _mlp():
    """Helper that mlp."""
    return PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 5e-3},
        network_params={"nlayers": 2, "first_layer_num_neurons": 32, "dropout_prob": 0.0, "activation_function": torch.nn.ReLU},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 8,
            "enable_model_summary": False,
            "default_root_dir": None,
            "log_every_n_steps": 1,
            "devices": 1,
            "logger": False,
            "accelerator": "cpu",
        },
    )


def _make_data(n=600, d=4, seed=0):
    """Make data."""
    rng = np.random.default_rng(seed)
    embs = np.vstack([rng.normal(size=d) for _ in range(n)]).astype(np.float32)
    # Signal lives ENTIRELY in the embedding; the scalar columns are pure noise.
    y = (2.0 * embs[:, 0] - 1.5 * embs[:, 2]).astype(np.float32)
    df = pd.DataFrame(
        {
            "noise_0": rng.normal(size=n).astype(np.float32),
            "noise_1": rng.normal(size=n).astype(np.float32),
            "emb": [embs[i] for i in range(n)],
        }
    )
    return df, y


def _rmse_via_mrmr_passthrough(df, y):
    """End-to-end WITH passthrough: MRMR keeps the embedding, the MLP boundary encoder expands it."""
    tr_df, te_df, tr_y, te_y = _split(df, y)
    m = MRMR(max_runtime_mins=0.3, fe_max_steps=0, random_seed=0, embedding_passthrough=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(tr_df, tr_y)
        tr_sel, te_sel = m.transform(tr_df), m.transform(te_df)
    emb_in = [c for c in tr_sel.columns if c == "emb"]
    return _fit_predict_rmse(tr_sel, tr_y, te_sel, te_y, emb_in), emb_in


def _rmse_dropping_embedding(df, y):
    """Legacy baseline: the embedding is unhashable for MRMR's MI screen, so the suite drops it (with passthrough OFF
    MRMR.fit raises ``unhashable type: 'numpy.ndarray'``). Simulate that by dropping the column before fitting -- the MLP
    sees only the noise columns the screen could keep."""
    df_drop = df.drop(columns=["emb"])
    tr_df, te_df, tr_y, te_y = _split(df_drop, y)
    m = MRMR(max_runtime_mins=0.3, fe_max_steps=0, random_seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(tr_df, tr_y)
        tr_sel, te_sel = m.transform(tr_df), m.transform(te_df)
    return _fit_predict_rmse(tr_sel, tr_y, te_sel, te_y, [])


def _split(df, y):
    """Helper that split."""
    cut = int(len(df) * 0.75)
    return df.iloc[:cut], df.iloc[cut:], y[:cut], y[cut:]


def _fit_predict_rmse(tr_sel, tr_y, te_sel, te_y, emb_in):
    """Fit predict rmse."""
    reg = _mlp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(tr_sel, tr_y, embedding_features=emb_in)
        preds = np.asarray(reg.predict(te_sel)).reshape(-1)
    return float(np.sqrt(np.mean((preds - te_y) ** 2)))


def test_biz_val_mrmr_embedding_passthrough_legacy_drops_then_crashes():
    """Pin the bug the feature fixes: with passthrough OFF, MRMR.fit chokes on the unhashable ndarray cells."""
    df, y = _make_data(n=200)
    m = MRMR(max_runtime_mins=0.2, fe_max_steps=0, random_seed=0, embedding_passthrough=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match="unhashable"):
            m.fit(df, y)


def test_biz_val_mrmr_embedding_passthrough_beats_dropping_it():
    """Biz val mrmr embedding passthrough beats dropping it."""
    df, y = _make_data()

    rmse_keep, emb_keep = _rmse_via_mrmr_passthrough(df, y)
    rmse_drop = _rmse_dropping_embedding(df, y)

    assert emb_keep == ["emb"], "passthrough must keep the embedding column in the selected output"

    # Quantitative win: keeping the signal-bearing embedding must materially beat dropping it.
    assert rmse_keep <= 0.80 * rmse_drop, (
        f"embedding passthrough RMSE {rmse_keep:.3f} should be <= 0.80 x drop RMSE {rmse_drop:.3f} "
        f"(ratio {rmse_keep / max(rmse_drop, 1e-9):.3f}); the embedding carries all the signal"
    )
