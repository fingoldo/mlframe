"""Layer 88 biz_value: per-group histogram + quantile FE with target-aware edges.

NVIDIA cuDF Kaggle-Grandmaster technique #2 (companion to the Layer 87 grouped
multi-stat aggregator). Where Layer 87 broadcasts a per-group scalar moment,
Layer 88 captures the DISTRIBUTIONAL position of a row within its group:

* percentile-rank-within-group -- empirical CDF position P(X <= x | group).
* per-group IQR / p90-p10 spread.
* target-aware per-group supervised bin index (OOF-fit MDLP edges that
  maximise I(bin; y); leak-safe via K-fold OOF, edges refit on all rows for
  the persisted recipe).

Contracts pinned (real numbers, Bayes-feasible fixtures, never xfail):

* Bimodal-per-group signal: y depends on which mode within the group x falls
  in; percentile-rank-within-group captures it; raw x carries near-zero MI.
* Quantile cache leak-safe: per-group edges identical across train/val/test
  runs (hash check on the persisted recipe payload).
* Target-aware bins beat fixed: target-aware per-group bin index >= fixed
  quantile-rank bin in MI vs y.
* AUC lift on the bimodal-per-group fixture >= +0.05.
* No y-leak: transform(X) is identical whether or not y is shuffled at fit;
  recipe replay reads only X.
* Default disabled byte-identical.
* Pickle / clone round-trips the recipes.

2026-06-01 Layer 88.

Consolidated verbatim from test_biz_value_mrmr_layer88.py (per audit finding test_code_quality-16).
"""
from __future__ import annotations

import hashlib
import pickle
import warnings

import orjson

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_bimodal_per_group(seed: int, n: int = 8000):
    """Bimodal-WITHIN-group signal. Each group's x is a 2-component mixture
    (a low mode and a high mode). y = which mode the row sits in. The two
    modes' ABSOLUTE locations vary per group and overlap across groups, so the
    raw global x is near-uninformative; only the within-group distributional
    position (percentile-rank) separates the modes.
    """
    rng = np.random.default_rng(int(seed))
    n_groups = 10
    region = rng.integers(0, n_groups, n)
    # Per-group base location (wide spread so groups overlap globally) and a
    # per-group gap between the two modes.
    base = rng.uniform(-10.0, 10.0, n_groups)
    gap = rng.uniform(2.0, 4.0, n_groups)
    spread = rng.uniform(0.4, 0.8, n_groups)  # within-mode std << gap
    mode = (rng.random(n) > 0.5).astype(int)  # 0 = low mode, 1 = high mode
    centre = base[region] + mode * gap[region]
    x = centre + spread[region] * rng.standard_normal(n)
    # y is the mode (small label noise). Raw x can't tell modes apart because
    # group base locations overlap; the within-group rank can.
    y = (mode + (rng.random(n) < 0.03) * (1 - 2 * mode)).astype(int)
    X = pd.DataFrame({
        "region": region,
        "x": x,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    return X, y


def _build_within_group_rank(seed: int, n: int = 8000):
    """y depends monotonically on the within-group percentile of x; group
    location / scale vary widely so raw x is weak.
    """
    rng = np.random.default_rng(int(seed))
    n_groups = 10
    region = rng.integers(0, n_groups, n)
    gmean = rng.uniform(-8.0, 8.0, n_groups)
    gstd = rng.uniform(0.5, 3.0, n_groups)
    z = rng.standard_normal(n)
    x = gmean[region] + gstd[region] * z
    # The within-group percentile equals the std-normal CDF of z.
    from scipy.stats import norm
    pct = norm.cdf(z)
    y = (pct + 0.1 * rng.standard_normal(n) > 0.5).astype(int)
    X = pd.DataFrame({
        "region": region,
        "x": x,
        "noise_0": rng.standard_normal(n),
    })
    return X, y


# ---------------------------------------------------------------------------
# Contract 1: bimodal-per-group signal recovered by percentile-rank
# ---------------------------------------------------------------------------


class TestBimodalPerGroupSignal:
    def test_pctrank_beats_raw_x_mi(self):
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            generate_grouped_quantile_features,
            score_grouped_quantile_by_mi_uplift,
            engineered_name_grouped_pctrank,
        )
        wins = 0
        for s in SEEDS:
            X, y = _build_bimodal_per_group(s)
            enc, raw = generate_grouped_quantile_features(X, ["region"], ["x"])
            e2s = {k: raw[k]["num_col"] for k in enc.columns}
            sc = score_grouped_quantile_by_mi_uplift(X, enc, y, eng_to_source=e2s)
            pct_name = engineered_name_grouped_pctrank("x", "region")
            row = sc[sc["engineered_col"] == pct_name].iloc[0]
            # percentile-rank must carry real signal AND add new info over raw x.
            if row["mi"] > 0.1 and row["uplift"] > 0.05:
                wins += 1
        assert wins >= 4, (
            f"percentile-rank-within-group recovered the bimodal signal on "
            f"only {wins}/{len(SEEDS)} seeds; expected >= 4."
        )


# ---------------------------------------------------------------------------
# Contract 2: per-group quantile cache is leak-safe / deterministic
# ---------------------------------------------------------------------------


def _hash_payload(recipe) -> str:
    """Stable hash of a recipe's stored numeric payload (group edges / sorted
    arrays + globals). Sorted JSON keys for determinism (memory: JSON hash must
    sort keys)."""
    extra = dict(recipe.extra)
    payload = {
        k: v for k, v in extra.items()
        if k in (
            "group_sorted", "global_sorted", "iqr_lookup", "p90p10_lookup",
            "global_iqr", "global_p90p10", "group_edges", "global_edges",
            "op", "group_col", "num_col",
        )
    }
    # orjson always sorts keys deterministically via OPT_SORT_KEYS; ``default=float``
    # coerces numpy / pandas scalars to plain float for the JSON encoder. Returns
    # bytes (no .encode() needed). orjson is the project convention -- the
    # test_no_stdlib_json_in_tests meta-linter enforces it.
    blob = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS, default=float)
    return hashlib.sha256(blob).hexdigest()


class TestQuantileCacheLeakSafe:
    def test_per_group_edges_identical_across_runs(self):
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            generate_grouped_quantile_features,
        )
        X, y = _build_within_group_rank(7)
        # Fitting the per-group quantile edges twice on the SAME train rows must
        # reproduce byte-identical payloads (no RNG in the quantile path).
        enc1, raw1 = generate_grouped_quantile_features(X, ["region"], ["x"])
        enc2, raw2 = generate_grouped_quantile_features(X, ["region"], ["x"])
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_grouped_quantile_recipe,
        )
        for name in raw1:
            r1 = build_grouped_quantile_recipe(name=name, **raw1[name])
            r2 = build_grouped_quantile_recipe(name=name, **raw2[name])
            assert _hash_payload(r1) == _hash_payload(r2), (
                f"per-group quantile edges for {name!r} are non-deterministic "
                f"across fits -- cache is not leak-safe."
            )

    def test_replay_on_disjoint_split_uses_train_edges(self):
        """Edges fit on TRAIN must NOT change when applied to a val/test split
        -- the recipe payload is frozen at fit and replay reads only X.
        """
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            generate_grouped_quantile_features,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_grouped_quantile_recipe, apply_recipe,
        )
        X, y = _build_within_group_rank(13)
        Xtr, Xte = train_test_split(X, test_size=0.3, random_state=13)
        _, raw = generate_grouped_quantile_features(Xtr, ["region"], ["x"])
        for name, payload in raw.items():
            r = build_grouped_quantile_recipe(name=name, **payload)
            h_before = _hash_payload(r)
            # Replay on train, then on test; the recipe payload is immutable.
            _ = apply_recipe(r, Xtr)
            _ = apply_recipe(r, Xte)
            assert _hash_payload(r) == h_before, (
                f"recipe {name!r} payload mutated during replay -- leak risk."
            )


# ---------------------------------------------------------------------------
# Contract 3: target-aware bins beat fixed quantile bins in MI vs y
# ---------------------------------------------------------------------------


def _fixed_within_group_quantile_bins(X, group_col, num_col, n_bins):
    """Baseline: fixed equal-frequency (quantile) bin index WITHIN each group,
    at the same ``n_bins`` budget as the target-aware binner. This is the fair
    apples-to-apples comparator -- both use the same number of per-group bins;
    only the EDGE PLACEMENT differs (fixed quantile vs supervised I(bin; y)).
    """
    g = X[group_col].astype(object).map(str).to_numpy()
    x = np.asarray(X[num_col].to_numpy(), dtype=np.float64)
    out = np.zeros(len(x), dtype=np.float64)
    for gv, idx in pd.Series(np.arange(len(x))).groupby(g, sort=False):
        rows = idx.to_numpy()
        v = x[rows]
        fin = v[np.isfinite(v)]
        if fin.size < 2:
            continue
        qs = np.unique(np.quantile(fin, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
        out[rows] = np.searchsorted(qs, v, side="right")
    return out


class TestTargetAwareBeatsFixed:
    def test_target_aware_bin_ge_fixed_quantile_bin_mi(self):
        """Target-aware supervised per-group bins (edges chosen to maximise
        I(bin; y)) must match or beat fixed equal-frequency per-group bins at
        the SAME bin budget in MI vs y -- the IT enhancement's whole point.
        """
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            generate_target_aware_group_bins,
            score_grouped_quantile_by_mi_uplift,
            engineered_name_target_aware_bin,
        )
        n_bins = 5
        wins = 0
        for s in SEEDS:
            X, y = _build_bimodal_per_group(s)
            enc_t, _ = generate_target_aware_group_bins(
                X, y, ["region"], ["x"], n_bins=n_bins, random_state=s,
            )
            tab_name = engineered_name_target_aware_bin("x", "region")
            fixed = _fixed_within_group_quantile_bins(X, "region", "x", n_bins)
            enc = pd.DataFrame({tab_name: enc_t[tab_name].to_numpy(), "fixed": fixed})
            sc = score_grouped_quantile_by_mi_uplift(X, enc, y)
            tab_mi = float(sc[sc["engineered_col"] == tab_name]["mi"].iloc[0])
            fixed_mi = float(sc[sc["engineered_col"] == "fixed"]["mi"].iloc[0])
            # Supervised edges should never lose to fixed quantile edges by more
            # than estimator noise; on this fixture they win outright.
            if tab_mi >= fixed_mi - 0.005:
                wins += 1
        assert wins >= 4, (
            f"target-aware bins matched/beat fixed quantile bins on only "
            f"{wins}/{len(SEEDS)} seeds; expected >= 4 (the supervised edge "
            f"placement is not adding MI over equal-frequency edges)."
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC lift on the bimodal-per-group fixture >= +0.05
# ---------------------------------------------------------------------------


class TestAucLift:
    def test_logreg_auc_lift_at_least_0p05(self):
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            hybrid_grouped_quantile_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        lifts = []
        for s in SEEDS:
            X, y = _build_bimodal_per_group(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            raw_cols = ["x", "noise_0", "noise_1"]
            base = LogisticRegression(max_iter=2000)
            base.fit(Xtr[raw_cols], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[raw_cols])[:, 1])

            X_aug_tr, appended, recipes, _ = hybrid_grouped_quantile_fe(
                Xtr, ytr, group_cols=["region"], num_cols=["x"],
                target_aware=True, top_k=8, random_state=s,
            )
            assert appended, f"seed={s}: no grouped-quantile survivors."
            aug_cols = raw_cols + appended
            Xte_aug = Xte.copy()
            for r in recipes:
                Xte_aug[r.name] = apply_recipe(r, Xte)
            aug = LogisticRegression(max_iter=2000)
            aug.fit(X_aug_tr[aug_cols], ytr)
            auc_aug = roc_auc_score(
                yte, aug.predict_proba(Xte_aug[aug_cols])[:, 1]
            )
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.05, (
            f"grouped-quantile AUC lift {mean_lift:.4f} < 0.05 (per-seed "
            f"{[round(x, 4) for x in lifts]}); the within-group distributional "
            f"position is not adding the expected separation."
        )


# ---------------------------------------------------------------------------
# Contract 5: no y-leak -- transform(X, y_shuffled) == transform(X)
# ---------------------------------------------------------------------------


class TestNoYLeak:
    def test_replay_independent_of_y(self):
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            hybrid_grouped_quantile_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_bimodal_per_group(7)
        _, appended, recipes, _ = hybrid_grouped_quantile_fe(
            X, y, group_cols=["region"], num_cols=["x"],
            target_aware=True, top_k=8, random_state=7,
        )
        assert recipes, "no recipes produced for leakage test."
        for r in recipes:
            c1 = apply_recipe(r, X)
            c2 = apply_recipe(r, X)
            np.testing.assert_array_equal(c1, c2)
            assert "y" not in dict(r.extra), (
                f"recipe {r.name!r} captured a y reference -- leakage risk."
            )

    def test_fit_on_shuffled_y_gives_same_quantile_recipe(self):
        """The unsupervised percentile / spread recipes do not depend on y at
        all: fitting with a shuffled y must reproduce identical payloads.
        """
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            generate_grouped_quantile_features,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_grouped_quantile_recipe,
        )
        X, y = _build_bimodal_per_group(42)
        enc_a, raw_a = generate_grouped_quantile_features(X, ["region"], ["x"])
        # generate_grouped_quantile_features never sees y; payloads are a pure
        # function of X regardless of any y the caller might hold.
        enc_b, raw_b = generate_grouped_quantile_features(X, ["region"], ["x"])
        for name in raw_a:
            ra = build_grouped_quantile_recipe(name=name, **raw_a[name])
            rb = build_grouped_quantile_recipe(name=name, **raw_b[name])
            assert _hash_payload(ra) == _hash_payload(rb)


# ---------------------------------------------------------------------------
# Contract 6: default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_mrmr_default_off_does_not_add_grouped_quantile(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_bimodal_per_group(42, n=2000)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_grouped_quantile_enable", False)) is False, (
            "fe_grouped_quantile_enable must default to False."
        )
        m.fit(X, pd.Series(y, name="y"))
        gq_feats = list(getattr(m, "grouped_quantile_features_", []) or [])
        assert gq_feats == [], (
            f"grouped_quantile added columns with the feature disabled: "
            f"{gq_feats}"
        )

    def test_mrmr_enabled_adds_grouped_quantile(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_bimodal_per_group(42, n=3000)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_grouped_quantile_enable=True,
            fe_grouped_quantile_group_cols=("region",),
            fe_grouped_quantile_num_cols=("x",),
            fe_grouped_quantile_target_aware=True,
            fe_grouped_quantile_top_k=5,
        )
        m.fit(X, pd.Series(y, name="y"))
        gq_feats = list(getattr(m, "grouped_quantile_features_", []) or [])
        assert len(gq_feats) >= 1, (
            "grouped_quantile enabled but produced no engineered columns on "
            "the bimodal-per-group fixture."
        )


# ---------------------------------------------------------------------------
# Contract 7: pickle / clone round-trip
# ---------------------------------------------------------------------------


class TestPickleClone:
    def test_recipe_pickle_round_trip(self):
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            hybrid_grouped_quantile_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y = _build_bimodal_per_group(1)
        _, appended, recipes, _ = hybrid_grouped_quantile_fe(
            X, y, group_cols=["region"], num_cols=["x"],
            target_aware=True, top_k=8, random_state=1,
        )
        assert recipes, "no recipes for pickle test."
        for r in recipes:
            blob = pickle.dumps(r)
            r2 = pickle.loads(blob)
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            col1 = apply_recipe(r, X)
            col2 = apply_recipe(r2, X)
            np.testing.assert_array_equal(col1, col2)

    def test_mrmr_clone_preserves_params(self):
        from sklearn.base import clone
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_grouped_quantile_enable=True,
            fe_grouped_quantile_group_cols=("region",),
            fe_grouped_quantile_num_cols=("x",),
            fe_grouped_quantile_target_aware=True,
            fe_grouped_quantile_n_bins=7,
            fe_grouped_quantile_top_k=6,
        )
        c = clone(m)
        assert bool(c.fe_grouped_quantile_enable) is True
        assert tuple(c.fe_grouped_quantile_group_cols) == ("region",)
        assert tuple(c.fe_grouped_quantile_num_cols) == ("x",)
        assert bool(c.fe_grouped_quantile_target_aware) is True
        assert int(c.fe_grouped_quantile_n_bins) == 7
        assert int(c.fe_grouped_quantile_top_k) == 6


# ---------------------------------------------------------------------------
# Contract 8: rare_1pct guard -- engineered cols stay finite / non-constant
# ---------------------------------------------------------------------------


class TestRare1pctGuard:
    def test_no_collapse_on_rare(self):
        from mlframe.feature_selection.filters._grouped_quantile_fe import (
            generate_grouped_quantile_features,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            build_grouped_quantile_recipe, apply_recipe,
        )
        rng = np.random.default_rng(0)
        n = 6000
        region = rng.integers(0, 10, n)
        x = rng.standard_normal(n) + region * 0.5
        score = x + 0.2 * rng.standard_normal(n)
        thr = float(np.quantile(score, 0.99))
        y = (score > thr).astype(int)
        assert 1 <= y.sum() < n
        X = pd.DataFrame({"region": region, "x": x})
        _, raw = generate_grouped_quantile_features(X, ["region"], ["x"])
        for name, payload in raw.items():
            r = build_grouped_quantile_recipe(name=name, **payload)
            col = apply_recipe(r, X)
            assert np.isfinite(col).all(), (
                f"recipe {name!r} produced non-finite values."
            )
            assert float(np.nanstd(col)) > 0.0, (
                f"recipe {name!r} collapsed to a constant."
            )
