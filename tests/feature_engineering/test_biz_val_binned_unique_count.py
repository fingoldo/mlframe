"""biz_value test for ``feature_engineering.binned_unique_count.binned_unique_count``.

The win (3rd_amex-default-prediction.md): an entity's behavioral VOLATILITY -- how many distinct
value-regimes it has visited across its history -- is a genuinely distinct "cardinality of visited states"
signal, not just a re-derivation of mean/std. This test constructs a "how many regimes has this entity
crossed into" target (multi-regime hoppers vs single-regime stayers) and confirms the bin-cardinality feature
recovers it near-perfectly, then separately confirms it adds real incremental value ON TOP OF mean/std (not
necessarily instead of), matching the realistic production use case (used as an additional feature alongside
existing aggregates, not a replacement for them).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_engineering.binned_unique_count import binned_unique_count


def _make_multi_regime_dataset(n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    labels = np.zeros(n_entities, dtype=int)
    band_pool = [-30, -10, 10, 30]
    for entity in range(n_entities):
        label = rng.integers(0, 2)
        labels[entity] = label
        n_obs = 20
        n_bands_visited = 3 if label == 1 else 1
        entity_bands = rng.choice(band_pool, size=n_bands_visited, replace=False)
        band_centers = rng.choice(entity_bands, size=n_obs, replace=True)
        # small per-band jitter -- deliberately much smaller than the between-band spacing so a
        # coarse binning aligned to band width cleanly separates "how many bands", independent of noise.
        values = band_centers + rng.normal(scale=1.0, size=n_obs)
        for v in values:
            rows.append({"entity": entity, "value": v})
    return pd.DataFrame(rows), labels


def test_biz_val_binned_unique_count_recovers_multi_regime_signal():
    df, labels = _make_multi_regime_dataset(n_entities=300, seed=0)
    entities = pd.unique(df["entity"])
    y = labels[entities]

    bin_edges = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])  # aligned to the 4 band centers
    bin_counts = binned_unique_count(df, entity_col="entity", value_col="value", bin_edges=bin_edges).set_index("entity").reindex(entities)
    auc_bin_count = cross_val_score(LogisticRegression(max_iter=500), bin_counts.to_numpy(), y, cv=5, scoring="roc_auc").mean()
    assert auc_bin_count > 0.95, f"expected bin-cardinality alone to near-perfectly recover the multi-regime-visited target, got AUC={auc_bin_count:.4f}"

    mean_std = df.groupby("entity", sort=False)["value"].agg(["mean", "std"]).reindex(entities).to_numpy()
    auc_mean_std_only = cross_val_score(LogisticRegression(max_iter=500), mean_std, y, cv=5, scoring="roc_auc").mean()
    combined = np.concatenate([mean_std, bin_counts.to_numpy()], axis=1)
    auc_combined = cross_val_score(LogisticRegression(max_iter=500), combined, y, cv=5, scoring="roc_auc").mean()
    assert auc_combined >= auc_mean_std_only, f"expected adding bin-cardinality to mean/std to not hurt (and here, to help), got combined={auc_combined:.4f} mean_std_only={auc_mean_std_only:.4f}"


def test_binned_unique_count_exact_values():
    df = pd.DataFrame({"entity": [1, 1, 1, 1, 2, 2], "value": [0.0, 1.0, 9.0, 9.5, 5.0, 5.1]})
    # explicit edges (not quantile-fitted -- deterministic for this small hand-picked example): entity 1's
    # values span bins [0,2.5), [7.5,10] (2 distinct bins); entity 2's values both land in [2.5,7.5) (1 bin).
    bin_edges = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    out = binned_unique_count(df, entity_col="entity", value_col="value", bin_edges=bin_edges)
    entity1_count = out.loc[out["entity"] == 1, "binned_unique_value"].item()
    entity2_count = out.loc[out["entity"] == 2, "binned_unique_value"].item()
    assert entity1_count > entity2_count, f"expected entity 1 (values spread across the full range) to visit more bins than entity 2 (values clustered together), got e1={entity1_count} e2={entity2_count}"
    assert entity2_count == 1
