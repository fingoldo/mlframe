"""biz_value test for ``feature_engineering.panel_sequence_tensor.build_panel_sequence_tensor``.

Source: 5th_home-credit-default-risk.md -- a per-user "image" combining multiple source-table signals,
normalized by per-row (per-entity) max, fed to a sequence encoder. On a synthetic with an extreme dynamic
range of per-entity magnitude scales (1 to 1,000,000x), a distance-based downstream consumer (KNN, standing
in for any sequence-model layer that computes similarity/distance over the raw feature space) has its
neighbor structure dominated by absolute magnitude rather than trend SHAPE when fed the raw tensor; per-entity
normalization puts every entity on a comparable scale so distance reflects shape similarity instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from mlframe.feature_engineering.panel_sequence_tensor import build_panel_sequence_tensor


def _make_extreme_scale_trend_data(n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    labels = {}
    for e in range(n_entities):
        scale = 10 ** rng.uniform(0, 6)  # 1 to 1,000,000 -- extreme dynamic range across entities.
        trend_up = rng.random() < 0.5
        t = np.arange(8)
        shape = (t if trend_up else t[::-1]) / 7.0  # scale-free shape in [0, 1].
        base = shape * scale * 0.1 + scale
        noisy = base + rng.normal(scale=scale * 0.02, size=8)
        for tt in range(8):
            rows.append({"id": e, "t": tt, "x": noisy[tt]})
        labels[e] = int(trend_up)
    return pd.DataFrame(rows), pd.Series(labels).to_numpy()


def test_biz_val_normalized_tensor_beats_raw_tensor_for_distance_based_consumer():
    df, y = _make_extreme_scale_trend_data(n_entities=400, seed=5)

    tensor_norm = build_panel_sequence_tensor(df, "id", "t", ["x"], max_lags=8, normalize=True).reshape(400, -1)
    tensor_raw = build_panel_sequence_tensor(df, "id", "t", ["x"], max_lags=8, normalize=False).reshape(400, -1)

    rng = np.random.default_rng(5)
    idx = np.arange(400)
    rng.shuffle(idx)
    train_idx, test_idx = idx[:300], idx[300:]

    model_norm = KNeighborsClassifier(n_neighbors=5).fit(tensor_norm[train_idx], y[train_idx])
    model_raw = KNeighborsClassifier(n_neighbors=5).fit(tensor_raw[train_idx], y[train_idx])

    auc_norm = roc_auc_score(y[test_idx], model_norm.predict_proba(tensor_norm[test_idx])[:, 1])
    auc_raw = roc_auc_score(y[test_idx], model_raw.predict_proba(tensor_raw[test_idx])[:, 1])

    assert auc_norm >= 0.9, f"expected the per-entity-normalized tensor to cleanly separate trend direction, got auc={auc_norm:.4f}"
    assert auc_norm > auc_raw + 0.2, f"expected normalization to beat the raw-scale tensor by a wide margin for a distance-based consumer, got norm={auc_norm:.4f} raw={auc_raw:.4f}"


def _make_multi_scale_channel_data(n_entities: int, seed: int):
    """Two channels: "small" (e.g. a ratio, fixed native range ~[1, 2], ~1000x smaller than "big") carries the
    whole trend-direction label signal in its own shape; "big" (e.g. a dollar amount) carries NO label signal
    but its per-entity magnitude varies wildly (1 to 10,000x across entities) -- exactly the "different native
    scales/units" case the per-channel mode targets. Joint (whole-block) normalization divides "small" by each
    entity's overall max, which is always in "big" here, so "small" ends up rescaled by a label-independent,
    wildly entity-varying factor -- destroying its otherwise-clean signal; per-channel normalization scales
    "small" by its own (label-independent, near-constant) max, leaving its trend shape intact."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = {}
    t = np.arange(8)
    for e in range(n_entities):
        trend_up = rng.random() < 0.5
        shape = (t if trend_up else t[::-1]) / 7.0  # scale-free shape in [0, 1] -- the only label signal.
        small = shape * 1.0 + rng.normal(scale=0.005, size=8) + 1.0  # ratio-like, ~[1.0, 2.0], carries the label.
        big_scale = 10 ** rng.uniform(0, 6)  # 1 to 1,000,000x native-scale spread across entities, no label signal.
        big = rng.normal(loc=big_scale, scale=big_scale * 0.02, size=8)  # pure noise around a random level.
        for tt in range(8):
            rows.append({"id": e, "t": tt, "big": big[tt], "small": small[tt]})
        labels[e] = int(trend_up)
    return pd.DataFrame(rows), pd.Series(labels).to_numpy()


def test_biz_val_per_channel_normalize_preserves_small_scale_channel_signal():
    df, y = _make_multi_scale_channel_data(n_entities=400, seed=7)

    tensor_joint = build_panel_sequence_tensor(df, "id", "t", ["big", "small"], max_lags=8, normalize=True, per_channel_normalize=False).reshape(400, -1)
    tensor_per_channel = build_panel_sequence_tensor(
        df, "id", "t", ["big", "small"], max_lags=8, normalize=True, per_channel_normalize=True
    ).reshape(400, -1)

    rng = np.random.default_rng(7)
    idx = np.arange(400)
    rng.shuffle(idx)
    train_idx, test_idx = idx[:300], idx[300:]

    # Fit on the FULL (both-channel) tensor: the label lives entirely in "small"'s shape, "big" carries none.
    # Under joint normalization "big"'s huge, entity-varying magnitude dominates Euclidean distance, drowning
    # "small"'s contribution; per-channel normalization brings both channels to a comparable value range so
    # "small"'s own shape signal actually drives neighbor structure.
    model_joint = KNeighborsClassifier(n_neighbors=5).fit(tensor_joint[train_idx], y[train_idx])
    model_per_channel = KNeighborsClassifier(n_neighbors=5).fit(tensor_per_channel[train_idx], y[train_idx])

    auc_joint = roc_auc_score(y[test_idx], model_joint.predict_proba(tensor_joint[test_idx])[:, 1])
    auc_per_channel = roc_auc_score(y[test_idx], model_per_channel.predict_proba(tensor_per_channel[test_idx])[:, 1])

    assert auc_per_channel >= 0.9, f"expected per-channel normalization to preserve the small-scale channel's own trend signal, got auc={auc_per_channel:.4f}"
    assert auc_per_channel > auc_joint + 0.1, (
        f"expected per-channel normalization to beat joint normalization at preserving the small-scale channel's signal, "
        f"got per_channel={auc_per_channel:.4f} joint={auc_joint:.4f}"
    )

    # Default behavior (per_channel_normalize unused) must be bit-identical to the pre-existing joint path.
    tensor_default = build_panel_sequence_tensor(df, "id", "t", ["big", "small"], max_lags=8, normalize=True)
    np.testing.assert_array_equal(tensor_default, tensor_joint.reshape(400, 2, 8))


def test_build_panel_sequence_tensor_shape_and_alignment():
    df = pd.DataFrame({"id": [1, 1, 1, 2, 2], "t": [0, 1, 2, 0, 1], "x": [10.0, 20.0, 40.0, 5.0, 15.0], "y": [1.0, 2.0, 4.0, 0.5, 1.5]})
    tensor = build_panel_sequence_tensor(df, "id", "t", ["x", "y"], max_lags=3, normalize=True)

    assert tensor.shape == (2, 2, 3)
    # entity 1's most recent x value (40.0) is also its max -> normalized lag_0 == 1.0.
    np.testing.assert_allclose(tensor[0, 0, 0], 1.0)
    # entity 2 has only 2 observations -> lag_2 (oldest-of-3 slot) is unavailable -> NaN.
    assert np.isnan(tensor[1, 0, 2])
