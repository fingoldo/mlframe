"""Unit + biz_value tests for CompositePanelEstimator (panel / within-entity composite)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from mlframe.training.composite.panel import CompositePanelEstimator


def _make_panel(n_entities=40, per_entity=25, within_beta=0.7, fe_scale=20.0, seed=0):
    """Synthetic panel: target = large per-entity fixed effect + small within-entity feature effect."""
    rng = np.random.default_rng(seed)
    fe = rng.normal(0.0, fe_scale, size=n_entities)  # between-entity level (dominant variance)
    rows = []
    for e in range(n_entities):
        x = rng.normal(0.0, 1.0, size=per_entity)
        noise = rng.normal(0.0, 0.3, size=per_entity)
        y = fe[e] + within_beta * x + noise
        for i in range(per_entity):
            rows.append((e, x[i], y[i]))
    df = pd.DataFrame(rows, columns=["entity", "x", "y"])
    return df


# ----------------------------- unit tests -----------------------------


def test_within_transform_train_only_and_shrinkage():
    """Offsets are train-only shrunken means: large entity ~= own mean, single-obs ~= global."""
    df = pd.DataFrame(
        {
            "entity": ["A"] * 50 + ["B"],  # A has 50 rows, B a single row
            "x": np.zeros(51),
            "y": [10.0] * 50 + [100.0],
        }
    )
    est = CompositePanelEstimator(LinearRegression(), entity_column="entity", shrinkage_alpha=10.0)
    est.fit(df[["entity", "x"]], df["y"])

    global_mean = df["y"].mean()
    # A: w = 50/60 -> stays close to its own mean (10), only mildly pulled to global.
    w_a = 50 / (50 + 10)
    assert abs(est.entity_offsets_["A"] - (w_a * 10.0 + (1 - w_a) * global_mean)) < 1e-9
    # B (single obs): w = 1/11 -> shrunk hard toward the global mean, far from its raw 100.
    w_b = 1 / (1 + 10)
    assert abs(est.entity_offsets_["B"] - (w_b * 100.0 + (1 - w_b) * global_mean)) < 1e-9
    assert est.entity_offsets_["B"] < 50.0  # heavily shrunk, nowhere near raw 100
    assert est.entity_counts_ == {"A": 50, "B": 1}


def test_alpha_zero_gives_raw_means():
    """shrinkage_alpha=0 disables shrinkage -> offsets are the raw per-entity means."""
    df = _make_panel(n_entities=5, per_entity=20, seed=1)
    est = CompositePanelEstimator(LinearRegression(), entity_column="entity", shrinkage_alpha=0.0)
    est.fit(df[["entity", "x"]], df["y"])
    for e, grp in df.groupby("entity"):
        assert abs(est.entity_offsets_[e] - grp["y"].mean()) < 1e-9


def test_unseen_entity_fallback_finite():
    """An entity unseen at fit falls back to the global mean -> predictions stay finite."""
    df = _make_panel(n_entities=10, per_entity=20, seed=2)
    est = CompositePanelEstimator(Ridge(), entity_column="entity", shrinkage_alpha=5.0)
    est.fit(df[["entity", "x"]], df["y"])

    new = pd.DataFrame({"entity": [9999, 12345], "x": [0.5, -0.5]})
    pred = est.predict(new)
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))
    # unseen offset == global mean, so prediction = inner(x) + global_mean
    inner = est.inner_.predict(new[["x"]])
    assert np.allclose(pred, inner + est.global_mean_)


def test_predict_shape_and_entity_id_arg():
    """Entity id can be supplied as a separate array; predict returns one value per row."""
    df = _make_panel(n_entities=8, per_entity=15, seed=3)
    X = df[["x"]]
    est = CompositePanelEstimator(LinearRegression(), shrinkage_alpha=3.0)
    est.fit(X, df["y"], entity_id=df["entity"].to_numpy())
    pred = est.predict(X, entity_id=df["entity"].to_numpy())
    assert pred.shape == (len(df),)
    assert np.all(np.isfinite(pred))


def test_single_obs_entity_does_not_crash():
    """A panel where some entities have a single observation fits and predicts finitely."""
    rng = np.random.default_rng(4)
    ents = list(range(20)) + [100, 101, 102]  # 3 singletons
    rows = []
    for e in ents:
        k = 1 if e >= 100 else 10
        x = rng.normal(size=k)
        y = float(e) + 0.5 * x + rng.normal(0, 0.1, size=k)
        for i in range(k):
            rows.append((e, x[i], y[i]))
    df = pd.DataFrame(rows, columns=["entity", "x", "y"])
    est = CompositePanelEstimator(LinearRegression(), entity_column="entity", shrinkage_alpha=10.0)
    est.fit(df[["entity", "x"]], df["y"])
    pred = est.predict(df[["entity", "x"]])
    assert pred.shape == (len(df),)
    assert np.all(np.isfinite(pred))


def test_missing_entity_column_raises():
    df = _make_panel(n_entities=3, per_entity=5, seed=5)
    est = CompositePanelEstimator(LinearRegression(), entity_column="entity")
    est.fit(df[["entity", "x"]], df["y"])
    import pytest

    with pytest.raises(KeyError):
        est.predict(df[["x"]])  # entity column dropped


# ----------------------------- biz_value -----------------------------


def test_biz_val_panel_beats_pooled_on_fixed_effect_panel():
    """Panel composite beats a pooled tree regressor on OOS RMSE when variance is mostly the
    entity fixed effect + a small within-entity effect.

    A flexible pooled learner (GBDT that sees the entity id as a feature) must spend its
    finite capacity splitting on the entity id to recover per-entity offsets, and at limited
    depth it cannot perfectly memorise hundreds of levels -- so its within-entity signal is
    polluted by FE-recovery error. The panel composite demeans the entity level analytically
    (with shrinkage, train-only) so the SAME inner can devote all its capacity to the within
    signal. Measured improvement well over 25%; floor set conservatively at 10%.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    # Many entities, few obs each, dominant FE + small within effect -> the regime where the
    # within transform pays: the pooled tree cannot cleanly carve out every entity level.
    df = _make_panel(n_entities=120, per_entity=8, within_beta=0.8, fe_scale=25.0, seed=7)
    df = df.reset_index(drop=True)
    df["t"] = df.groupby("entity").cumcount()
    train = df[df["t"] < 6]
    test = df[df["t"] >= 6]

    def _mk():
        return HistGradientBoostingRegressor(max_depth=3, max_iter=120, random_state=0)

    # Panel composite: demean entity, inner GBDT on x only.
    panel = CompositePanelEstimator(_mk(), entity_column="entity", shrinkage_alpha=10.0)
    panel.fit(train[["entity", "x"]], train["y"])
    p_pred = panel.predict(test[["entity", "x"]])
    rmse_panel = float(np.sqrt(np.mean((p_pred - test["y"].to_numpy()) ** 2)))

    # Pooled baseline: same GBDT, entity id + x as features (sees the same information).
    pooled = _mk().fit(train[["entity", "x"]].astype(float), train["y"])
    pool_pred = pooled.predict(test[["entity", "x"]].astype(float))
    rmse_pool = float(np.sqrt(np.mean((pool_pred - test["y"].to_numpy()) ** 2)))

    assert rmse_panel < rmse_pool * 0.90, f"panel RMSE {rmse_panel:.4f} should beat pooled {rmse_pool:.4f} by >=10%"
    assert np.all(np.isfinite(p_pred))
