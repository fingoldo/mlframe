"""biz_value: learnable categorical entity embeddings recover a NON-MONOTONE category->target mapping that a single ordinal numeric column cannot.

Synthetic: ``y = lookup[cat] + noise`` where ``lookup`` is a random per-category value with NO ordinal structure (8 categories). The learnable-
embedding MLP (raw string cat + ``cat_features``) gets each category a free embedding vector; the naive baseline feeds the same category as ONE
ordinal numeric column, so a smooth MLP can only fit a monotone-ish function of the arbitrary code order and leaves most of the signal on the table.

Measured (3 seeds, n=2000, 40 epochs CPU): embedding R^2 ~0.982 vs ordinal R^2 ~0.68-0.72, delta 0.265-0.300. Floors set ~15-30% below the
minimum measured so a real regression (embeddings disabled / factorizer broken / hook unwired) trips the win, but seed variance does not.

Two ordinal baselines are covered: a LOW-card (8-level) random-lookup target and a HIGH-card (120-level) smooth-latent-manifold target, so the
win is pinned across the cardinality range the feature targets. The baseline is ORDINAL, not one-hot, on purpose: a swept measurement (n<=2500,
50-200 levels, seen + unseen-category regimes) found one-hot is a STRONG baseline here -- on seen categories it ties/slightly beats the embedding,
and the embedding's edge under unseen/rare levels flips sign across seeds (a high-variance knife-edge). Ordinal is the baseline the embedding beats
ROBUSTLY (it discards the categorical's multi-dim structure by construction), so it is the defensible quantitative-win sensor. Do not re-add a
one-hot biz_value assertion without a measured, multi-seed regime where the embedding wins reliably.
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
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 64, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": EPOCHS,
            "enable_model_summary": False,
            "default_root_dir": None,
            "log_every_n_steps": 1,
            "devices": 1,
            "logger": False,
            "accelerator": "cpu",
            "enable_progress_bar": False,
        },
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


N_CAT_HIGH = 120


def _build_highcard_latent(seed=0):
    """High-cardinality (120-level) categorical whose levels sit on a SMOOTH 2-D latent manifold (a circle); the target is a smooth nonlinear
    function of each level's latent coordinate. A low-dim learnable embedding (embed_dim=4 here) can recover the 2-D manifold and so model the
    smooth coord->target surface, while a SINGLE ordinal numeric column collapses that 2-D structure onto one arbitrary code axis and cannot.
    """
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, N_CAT_HIGH, endpoint=False)
    lat = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # per-level latent (x, y) on the unit circle
    level_value = np.sin(2.0 * lat[:, 0]) + np.cos(1.5 * lat[:, 1]) + 0.5 * lat[:, 0] * lat[:, 1]
    cats = rng.integers(0, N_CAT_HIGH, size=N)
    y = (level_value[cats] + rng.normal(scale=0.15, size=N)).astype(np.float32)
    labels = np.array([f"c{c}" for c in cats])
    idx = np.arange(N)
    np.random.default_rng(99).shuffle(idx)
    cut = int(N * 0.6)
    return cats, labels, y, idx[:cut], idx[cut:]


def test_biz_val_mlp_highcard_latent_cat_embeddings_beat_ordinal_baseline():
    """High-cardinality (120-level) smooth-latent-manifold sibling of the low-card test above. With ~10 train rows/level, a low-dim (embed_dim=4)
    learnable embedding recovers the 2-D latent circle the levels live on and fits the smooth coord->target surface; the single ordinal numeric
    column flattens that manifold onto one arbitrary code axis and cannot. Measured (3 seeds, n=2000, 40 epochs CPU): emb R^2 ~0.970 vs ordinal
    R^2 ~0.86, delta ~0.104-0.111. Floors set ~25%/~18% below the minimum measured so a real regression (embedding disabled / factorizer broken /
    hook unwired) drops emb toward ord and trips the win, while seed variance does not.
    """
    cats, labels, y, tr, te = _build_highcard_latent(seed=0)

    X_emb = pd.DataFrame({"cat": labels})
    reg_emb = PytorchLightningRegressor(**_common(), categorical_embed_dim=4)
    reg_emb.fit(X_emb.iloc[tr], y[tr], cat_features=["cat"], eval_set=(X_emb.iloc[te], y[te]))
    r2_emb = r2_score(y[te], np.asarray(reg_emb.predict(X_emb.iloc[te])))

    X_ord = pd.DataFrame({"cat_code": cats.astype(np.float32)})
    reg_ord = PytorchLightningRegressor(**_common())
    reg_ord.fit(X_ord.iloc[tr], y[tr], eval_set=(X_ord.iloc[te], y[te]))
    r2_ord = r2_score(y[te], np.asarray(reg_ord.predict(X_ord.iloc[te])))

    assert r2_emb >= 0.90, f"high-card learnable-embedding R^2 should be >=0.90 on the smooth-latent target, got {r2_emb:.4f}"
    assert r2_emb - r2_ord >= 0.085, (
        f"high-card (120-level) learnable embeddings should beat the ordinal-numeric baseline by >=0.085 R^2 "
        f"(measured 0.104-0.111); got emb={r2_emb:.4f} ord={r2_ord:.4f} delta={r2_emb - r2_ord:.4f}"
    )
