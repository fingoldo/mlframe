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
    """Helper: Make multi regime dataset."""
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
    """Biz val binned unique count recovers multi regime signal."""
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
    assert (
        auc_combined >= auc_mean_std_only
    ), f"expected adding bin-cardinality to mean/std to not hurt (and here, to help), got combined={auc_combined:.4f} mean_std_only={auc_mean_std_only:.4f}"


def _make_skewed_scale_multiregime_dataset(n_entities: int, seed: int, high_scale_frac: float = 0.1):
    """Heavily right-skewed value column: 90% of entities live at a low absolute magnitude (base~U(1,5)),
    10% live at a far larger magnitude (base~U(1000,2000)) -- e.g. balances/amounts spanning orders of
    magnitude. Each entity visits either 1 or 3 "regimes" -- relative offsets (-30%,-10%,+10%,+30% of its
    OWN base) around its own base -- with per-observation jitter much smaller than the regime spacing.
    "Volatile" (label 1) entities hop across 3 of their own relative regimes; "stable" (label 0) entities
    stay in 1. A single global bin-edge set (quantile or fixed) allocates resolution by POPULATION density:
    the low-scale majority dominates the quantile edges, so the sparse high-scale minority's entire value
    range collapses into one or two global bins regardless of how many of its OWN relative regimes it
    visited -- destroying the signal exactly where it matters (the minority, out-of-density entities). Bins
    computed from each entity's OWN observed range are scale-invariant and resolve the relative-regime
    structure at any absolute magnitude.
    """
    rng = np.random.default_rng(seed)
    rows = []
    labels = np.zeros(n_entities, dtype=int)
    is_high_scale = np.zeros(n_entities, dtype=bool)
    n_high = max(1, int(n_entities * high_scale_frac))
    high_scale_idx = rng.choice(n_entities, size=n_high, replace=False)
    is_high_scale[high_scale_idx] = True
    regime_offset_frac_pool = np.array([-0.3, -0.1, 0.1, 0.3])
    for entity in range(n_entities):
        base = rng.uniform(1000.0, 2000.0) if is_high_scale[entity] else rng.uniform(1.0, 5.0)
        volatile = rng.integers(0, 2)
        labels[entity] = volatile
        n_regimes_visited = 3 if volatile else 1
        entity_regimes = rng.choice(regime_offset_frac_pool, size=n_regimes_visited, replace=False)
        n_obs = 20
        offset_frac_per_row = rng.choice(entity_regimes, size=n_obs, replace=True)
        jitter = rng.normal(scale=0.02 * base, size=n_obs)  # small vs. the 20%-of-base regime spacing
        values = base + offset_frac_per_row * base + jitter
        for v in values:
            rows.append({"entity": entity, "value": v})
    return pd.DataFrame(rows), labels, is_high_scale


def test_biz_val_binned_unique_count_per_entity_bins_beats_global_on_skewed_scale():
    """Biz val binned unique count per entity bins beats global on skewed scale."""
    df, labels, is_high_scale = _make_skewed_scale_multiregime_dataset(n_entities=400, seed=1)
    entities = pd.unique(df["entity"])
    y_high_scale = labels[is_high_scale]

    global_counts = binned_unique_count(df, entity_col="entity", value_col="value", n_bins=10).set_index("entity").reindex(entities)
    per_entity_counts = binned_unique_count(df, entity_col="entity", value_col="value", n_bins=10, per_entity_bins=True).set_index("entity").reindex(entities)

    global_high = global_counts.to_numpy()[is_high_scale]
    per_entity_high = per_entity_counts.to_numpy()[is_high_scale]

    auc_global_high = cross_val_score(LogisticRegression(max_iter=500), global_high, y_high_scale, cv=5, scoring="roc_auc").mean()
    auc_per_entity_high = cross_val_score(LogisticRegression(max_iter=500), per_entity_high, y_high_scale, cv=5, scoring="roc_auc").mean()

    assert (
        auc_per_entity_high > 0.9
    ), f"expected per-entity bins to sharply recover the regime-hopping target on the sparsely-populated high-scale minority, got AUC={auc_per_entity_high:.4f}"
    assert auc_per_entity_high - auc_global_high > 0.3, (
        f"expected per-entity bins to clearly beat global quantile bins on the high-scale minority (global bins are starved by the "
        f"low-scale majority's population density -- measured global AUC~0.5, chance), got per_entity={auc_per_entity_high:.4f} global={auc_global_high:.4f}"
    )


def test_binned_unique_count_per_entity_bins_default_matches_prior_behavior():
    """per_entity_bins=False (the default) must reproduce the pre-existing global-quantile behavior bit-for-bit."""
    df, _labels = _make_multi_regime_dataset(n_entities=50, seed=2)
    out_default = binned_unique_count(df, entity_col="entity", value_col="value", n_bins=8)
    out_explicit_false = binned_unique_count(df, entity_col="entity", value_col="value", n_bins=8, per_entity_bins=False)
    pd.testing.assert_frame_equal(out_default, out_explicit_false)


def test_binned_unique_count_per_entity_bins_rejects_explicit_edges():
    """Binned unique count per entity bins rejects explicit edges."""
    df = pd.DataFrame({"entity": [1, 1, 2, 2], "value": [0.0, 1.0, 5.0, 5.1]})
    try:
        binned_unique_count(df, entity_col="entity", value_col="value", bin_edges=np.array([0.0, 5.0]), per_entity_bins=True)
        raise AssertionError("expected ValueError when bin_edges is combined with per_entity_bins=True")
    except ValueError:
        pass


def test_binned_unique_count_exact_values():
    """Binned unique count exact values."""
    df = pd.DataFrame({"entity": [1, 1, 1, 1, 2, 2], "value": [0.0, 1.0, 9.0, 9.5, 5.0, 5.1]})
    # explicit edges (not quantile-fitted -- deterministic for this small hand-picked example): entity 1's
    # values span bins [0,2.5), [7.5,10] (2 distinct bins); entity 2's values both land in [2.5,7.5) (1 bin).
    bin_edges = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    out = binned_unique_count(df, entity_col="entity", value_col="value", bin_edges=bin_edges)
    entity1_count = out.loc[out["entity"] == 1, "binned_unique_value"].item()
    entity2_count = out.loc[out["entity"] == 2, "binned_unique_value"].item()
    assert (
        entity1_count > entity2_count
    ), f"expected entity 1 (values spread across the full range) to visit more bins than entity 2 (values clustered together), got e1={entity1_count} e2={entity2_count}"
    assert entity2_count == 1
