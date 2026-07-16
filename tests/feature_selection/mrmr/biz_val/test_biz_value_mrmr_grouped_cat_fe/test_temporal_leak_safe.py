"""Layer 92 biz_value: TEMPORAL LEAK-SAFE GROUPED AGGREGATIONS.

Layer 87 grouped aggregations compute the per-group statistic over the WHOLE
train fold. For time-series / transaction data that statistic peeks at a row's
own FUTURE (the per-entity mean includes the entity's later rows), so a random
CV scores artificially high while a true forward holdout collapses. Layer 92
keys aggregations on a TIME column and only ever sees the strict past
(expanding / rolling / lag), making them leak-safe by construction.

Contracts pinned (real AUC numbers, Bayes-feasible fixtures, never xfail):

* PHASE-0 leak proof: a look-ahead target (y = function of the entity's
  FULL-series mean, which the whole-fold agg knows but the past cannot)
  produces leaky-random-CV AUC >> forward-holdout AUC, gap > 0.10. This IS the
  Layer 87 leak.
* L92 is leak-safe: on a CAUSAL past-driven target, expanding-agg train-CV
  approximately equals forward-holdout (gap < 0.05).
* L92 recovers the past-mean signal: y = f(expanding_mean(value | entity));
  L92 expanding-mean AUC >= 0.80 on the forward holdout.
* Rolling window: a time-windowed rolling stat recovers a windowed signal.
* Lag features: lag-1 captures an autoregressive y = f(value_{t-1}) signal.
* Transform leak-safe: replaying on a time-prefix of the frame yields the SAME
  values as replaying on the full frame -- appending future rows never changes
  an earlier row (no test-future peeking).
* Default disabled byte-identical.
* Pickle / clone round-trips the recipes.

2026-06-01 Layer 92.

Consolidated verbatim from test_biz_value_mrmr_layer92.py (per audit finding test_code_quality-16).
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_lookahead_leak(seed: int, n_ent: int = 200, per: int = 12):
    """Look-ahead target: y_t = 1 iff value_t is BELOW the entity's FULL-SERIES
    mean. By construction the target is a function of the whole-fold mean (which
    peeks at the future), so the Layer 87 whole-fold aggregate scores perfectly
    under a random CV but cannot generalise to a forward holdout. Per-entity
    series is a random walk so the full-series mean is genuinely unknowable from
    the past alone.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    for e in range(n_ent):
        ts = np.sort(rng.uniform(0.0, 1000.0, per))
        vals = np.cumsum(rng.normal(0.0, 1.0, per)) + rng.uniform(-3.0, 3.0)
        rows.extend((e, ts[k], vals[k]) for k in range(per))
    df = pd.DataFrame(rows, columns=["entity", "ts", "value"])
    df = df.sort_values("ts").reset_index(drop=True)
    full_mean = df.groupby("entity")["value"].transform("mean").to_numpy()
    y = (df["value"].to_numpy() < full_mean).astype(int)
    return df, y


def _build_past_mean_causal(seed: int, n_ent: int = 200, per: int = 12):
    """Causal target knowable from the past alone: y_t depends on the entity's
    EXPANDING (past) mean of value. Each entity has a stable hidden level, so
    the past-mean is informative and the future adds nothing the past lacks.
    """
    rng = np.random.default_rng(int(seed))
    rows = []
    for e in range(n_ent):
        lvl = rng.uniform(-2.5, 2.5)
        ts = np.sort(rng.uniform(0.0, 1000.0, per))
        vals = lvl + rng.normal(0.0, 1.0, per)
        rows.extend((e, ts[k], vals[k]) for k in range(per))
    df = pd.DataFrame(rows, columns=["entity", "ts", "value"])
    df = df.sort_values("ts").reset_index(drop=True)
    pm = df.groupby("entity")["value"].transform(lambda s: s.expanding().mean().shift(1)).bfill().to_numpy()
    y = (pm + 0.3 * rng.normal(size=len(df)) > 0.0).astype(int)
    return df, y


def _build_lag1_autoregressive(seed: int, n_ent: int = 150, per: int = 20):
    """Autoregressive target: y_t = 1 iff value_{t-1} > 0 within the entity."""
    rng = np.random.default_rng(int(seed))
    rows = []
    base = pd.Timestamp("2020-01-01")
    for e in range(n_ent):
        days = np.sort(rng.uniform(0.0, 300.0, per))
        ts = base + pd.to_timedelta(days, unit="D")
        vals = rng.normal(0.0, 1.0, per)
        rows.extend((e, ts[k], vals[k]) for k in range(per))
    df = pd.DataFrame(rows, columns=["entity", "ts", "value"])
    df = df.sort_values("ts").reset_index(drop=True)
    prev = df.groupby("entity")["value"].shift(1).bfill().to_numpy()
    y = (prev > 0.0).astype(int)
    return df, y


def _build_rolling_window(seed: int, n_ent: int = 120, per: int = 25):
    """Per-entity stable level; the rolling time-window mean recovers it.
    Target = entity-level sign (knowable from any windowed past summary)."""
    rng = np.random.default_rng(int(seed))
    rows = []
    base = pd.Timestamp("2020-01-01")
    for e in range(n_ent):
        lvl = rng.uniform(-2.0, 2.0)
        days = np.sort(rng.uniform(0.0, 200.0, per))
        ts = base + pd.to_timedelta(days, unit="D")
        vals = lvl + rng.normal(0.0, 0.7, per)
        rows.extend((e, ts[k], vals[k]) for k in range(per))
    df = pd.DataFrame(rows, columns=["entity", "ts", "value"])
    df = df.sort_values("ts").reset_index(drop=True)
    lvl_map = df.groupby("entity")["value"].transform("mean").to_numpy()
    y = (lvl_map > 0.0).astype(int)
    return df, y


def _forward_split(df: pd.DataFrame, y: np.ndarray, frac: float = 0.6):
    """Split by row order (time-sorted) into an early-train / late-holdout pair."""
    cut = int(len(df) * frac)
    df = df.reset_index(drop=True)
    tr = np.asarray(df.index < cut)
    te = np.asarray(df.index >= cut)
    return df[tr], df[te], y[tr], y[te]


# ---------------------------------------------------------------------------
# PHASE 0 (mandatory): prove the Layer 87 whole-fold leak EXISTS
# ---------------------------------------------------------------------------


class TestL87WholeFoldLeaksOnTimeSeries:
    """The motivation contract: Layer 87's whole-fold grouped_mean peeks at the
    future on time-series data -> high leaky CV, poor forward holdout."""

    def test_L87_wholefold_agg_leaks_on_timeseries(self):
        """Whole-fold grouped-mean scores far higher under random CV than a forward holdout."""
        from mlframe.feature_selection.filters._grouped_agg_fe import (
            generate_grouped_agg_features,
            engineered_name_grouped_agg,
        )

        gaps = []
        for s in SEEDS:
            df, y = _build_lookahead_leak(s)
            gm = engineered_name_grouped_agg("value", "entity", "mean")

            # Leaky evaluation: whole-fold grouped_mean (Layer 87) scored with a
            # random 5-fold CV -- every fold's per-entity mean still saw the
            # held-out rows, so the look-ahead leaks straight into the OOF score.
            enc_full, _ = generate_grouped_agg_features(df, ["entity"], ["value"])
            feat = np.column_stack([df["value"].to_numpy(), enc_full[gm].to_numpy()])
            oof = cross_val_predict(
                LogisticRegression(max_iter=1000), feat, y, cv=5,
                method="predict_proba",
            )[:, 1]
            auc_leaky = roc_auc_score(y, oof)

            # Honest evaluation: train on the time-early fold, recompute the
            # whole-fold mean independently on the forward holdout (as one MUST
            # at deployment). The look-ahead signal evaporates.
            Xtr, Xte, ytr, yte = _forward_split(df, y)
            enc_tr, _ = generate_grouped_agg_features(Xtr, ["entity"], ["value"])
            m = LogisticRegression(max_iter=1000).fit(
                np.column_stack([Xtr["value"], enc_tr[gm]]), ytr,
            )
            enc_te, _ = generate_grouped_agg_features(Xte, ["entity"], ["value"])
            auc_fwd = roc_auc_score(
                yte,
                m.predict_proba(np.column_stack([Xte["value"], enc_te[gm]]))[:, 1],
            )
            gaps.append(auc_leaky - auc_fwd)

        mean_gap = float(np.mean(gaps))
        assert mean_gap > 0.10, (
            f"Layer 87 whole-fold leak not demonstrated: mean leaky-CV minus "
            f"forward-holdout AUC gap {mean_gap:.4f} <= 0.10 "
            f"(per-seed {[round(g, 4) for g in gaps]}). The look-ahead leak "
            f"should be large and positive."
        )


# ---------------------------------------------------------------------------
# Contract: L92 expanding aggregations are LEAK-SAFE
# ---------------------------------------------------------------------------


class TestL92LeakSafe:
    """L92 expanding aggregations must not leak the future into train-time CV."""

    def test_expanding_train_cv_approx_forward_holdout(self):
        """Train-CV AUC on the expanding-mean feature closely tracks the forward-holdout AUC."""
        from mlframe.feature_selection.filters._temporal_agg_fe import (
            hybrid_temporal_agg_fe,
            engineered_name_expanding,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        gaps = []
        for s in SEEDS:
            df, y = _build_past_mean_causal(s)
            Xtr, Xte, ytr, yte = _forward_split(df, y)
            X_aug, appended, recipes, _ = hybrid_temporal_agg_fe(
                Xtr, ytr, entity_cols=["entity"], value_cols=["value"],
                time_col="ts", stats=("mean", "std", "count"), lags=(1,),
                top_k=10,
            )
            en = engineered_name_expanding("value", "entity", "mean")
            assert en in appended, f"seed={s}: expanding-mean not produced."
            rec = next(r for r in recipes if r.name == en)
            ftr = X_aug[en].to_numpy().reshape(-1, 1)
            oof = cross_val_predict(
                LogisticRegression(max_iter=1000), ftr, ytr, cv=5,
                method="predict_proba",
            )[:, 1]
            auc_cv = roc_auc_score(ytr, oof)
            m = LogisticRegression(max_iter=1000).fit(ftr, ytr)
            fte = apply_recipe(rec, Xte).reshape(-1, 1)
            auc_fwd = roc_auc_score(yte, m.predict_proba(fte)[:, 1])
            gaps.append(abs(auc_cv - auc_fwd))
        mean_gap = float(np.mean(gaps))
        assert mean_gap < 0.05, (
            f"L92 expanding agg is not leak-safe: |train-CV - forward-holdout| " f"AUC gap {mean_gap:.4f} >= 0.05 " f"(per-seed {[round(g, 4) for g in gaps]})."
        )


# ---------------------------------------------------------------------------
# Contract: L92 recovers the past-mean signal (high forward-holdout AUC)
# ---------------------------------------------------------------------------


class TestL92RecoversPastMeanSignal:
    """The expanding-mean feature must recover the past-mean-driven signal on the forward holdout."""

    def test_expanding_mean_forward_auc_at_least_0p80(self):
        """Forward-holdout AUC using the expanding-mean feature is at least 0.80."""
        from mlframe.feature_selection.filters._temporal_agg_fe import (
            hybrid_temporal_agg_fe,
            engineered_name_expanding,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        aucs = []
        for s in SEEDS:
            df, y = _build_past_mean_causal(s)
            Xtr, Xte, ytr, yte = _forward_split(df, y)
            X_aug, _appended, recipes, _ = hybrid_temporal_agg_fe(
                Xtr, ytr, entity_cols=["entity"], value_cols=["value"],
                time_col="ts", stats=("mean", "std", "count"), lags=(1,),
                top_k=10,
            )
            en = engineered_name_expanding("value", "entity", "mean")
            rec = next(r for r in recipes if r.name == en)
            m = LogisticRegression(max_iter=1000).fit(
                X_aug[en].to_numpy().reshape(-1, 1), ytr,
            )
            fte = apply_recipe(rec, Xte).reshape(-1, 1)
            aucs.append(roc_auc_score(yte, m.predict_proba(fte)[:, 1]))
        mean_auc = float(np.mean(aucs))
        assert mean_auc >= 0.80, (
            f"L92 expanding-mean forward-holdout AUC {mean_auc:.4f} < 0.80 "
            f"(per-seed {[round(a, 4) for a in aucs]}); did not recover the "
            f"past-mean signal."
        )


# ---------------------------------------------------------------------------
# Contract: rolling window captures a windowed signal
# ---------------------------------------------------------------------------


class TestRollingWindow:
    """A time-windowed rolling stat must recover a per-entity level signal."""

    def test_rolling_window_recovers_level(self):
        """Rolling 30D window mean has AUC >= 0.80 (symmetric) against the entity-level signal on most seeds."""
        from mlframe.feature_selection.filters._temporal_agg_fe import (
            generate_rolling_window_agg_features,
        )

        wins = 0
        for s in SEEDS:
            df, y = _build_rolling_window(s)
            enc, _ = generate_rolling_window_agg_features(
                df, ["entity"], ["value"], "ts", windows=("30D",),
                stats=("mean",),
            )
            assert enc.shape[1] >= 1, f"seed={s}: no rolling columns produced."
            rn = next(iter(enc.columns))
            auc = roc_auc_score(y, enc[rn].to_numpy())
            # AUC can be < 0.5 if anti-correlated; use the symmetric measure.
            if max(auc, 1.0 - auc) >= 0.80:
                wins += 1
        assert wins >= 4, f"rolling 30D window recovered the level signal on only " f"{wins}/{len(SEEDS)} seeds; expected >= 4."


# ---------------------------------------------------------------------------
# Contract: lag features capture an autoregressive signal
# ---------------------------------------------------------------------------


class TestLagFeatures:
    """Lag-1 features must capture an autoregressive dependency on the prior value."""

    def test_lag1_captures_autoregressive_signal(self):
        """Lag-1 feature AUC against y = f(value_{t-1}) is at least 0.80."""
        from mlframe.feature_selection.filters._temporal_agg_fe import (
            generate_lag_features,
            engineered_name_lag,
        )

        aucs = []
        for s in SEEDS:
            df, y = _build_lag1_autoregressive(s)
            enc, _ = generate_lag_features(
                df, ["entity"], ["value"], "ts", lags=(1,),
            )
            ln = engineered_name_lag("value", "entity", 1)
            assert ln in enc.columns, f"seed={s}: lag-1 not produced."
            aucs.append(roc_auc_score(y, enc[ln].to_numpy()))
        mean_auc = float(np.mean(aucs))
        assert mean_auc >= 0.80, (
            f"lag-1 feature AUC {mean_auc:.4f} < 0.80 "
            f"(per-seed {[round(a, 4) for a in aucs]}); did not capture the "
            f"autoregressive y = f(value_t-1) signal."
        )


# ---------------------------------------------------------------------------
# Contract: transform is leak-safe (no test-future peeking)
# ---------------------------------------------------------------------------


class TestTransformLeakSafe:
    """Recipe replay must never let future rows change an earlier row's computed stat."""

    def test_replay_prefix_matches_full(self):
        """Replaying a recipe on a time-prefix of the frame must yield the SAME
        values as replaying on the full frame for those prefix rows -- i.e.
        appending future rows never changes an earlier row's stat. This is the
        no-test-future-peeking guarantee.
        """
        from mlframe.feature_selection.filters._temporal_agg_fe import (
            generate_expanding_agg_features,
            generate_lag_features,
            generate_rolling_window_agg_features,
            build_temporal_expanding_recipe,
            build_temporal_lag_recipe,
            build_temporal_rolling_recipe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        df, _ = _build_lag1_autoregressive(7)
        # Sort by time so a row-prefix is a genuine TIME prefix.
        df = df.sort_values("ts").reset_index(drop=True)
        prefix = df.iloc[: int(len(df) * 0.7)].copy()

        _enc_e, raw_e = generate_expanding_agg_features(
            df, ["entity"], ["value"], "ts", stats=("mean", "count"),
        )
        _enc_l, raw_l = generate_lag_features(
            df, ["entity"], ["value"], "ts", lags=(1, 2),
        )
        _enc_r, raw_r = generate_rolling_window_agg_features(
            df, ["entity"], ["value"], "ts", windows=("30D",), stats=("mean",),
        )

        builders = []
        for name, payload in raw_e.items():
            builders.append(build_temporal_expanding_recipe(name=name, **payload))
        for name, payload in raw_l.items():
            builders.append(build_temporal_lag_recipe(name=name, **payload))
        for name, payload in raw_r.items():
            builders.append(build_temporal_rolling_recipe(name=name, **payload))

        n_pref = len(prefix)
        for rec in builders:
            full = apply_recipe(rec, df)
            part = apply_recipe(rec, prefix)
            assert len(part) == n_pref
            # The prefix rows must be identical whether or not the future rows
            # were present in the frame being transformed.
            np.testing.assert_allclose(
                full[:n_pref], part, rtol=0, atol=1e-9, equal_nan=True,
                err_msg=(
                    f"recipe {rec.name!r}: prefix values changed when future "
                    f"rows were appended -> test-future leak."
                ),
            )

    def test_replay_independent_of_y(self):
        """Temporal recipes carry no stored y reference and replay deterministically from df alone."""
        from mlframe.feature_selection.filters._temporal_agg_fe import (
            hybrid_temporal_agg_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        df, y = _build_past_mean_causal(13)
        _, _appended, recipes, _ = hybrid_temporal_agg_fe(
            df, y, entity_cols=["entity"], value_cols=["value"],
            time_col="ts", stats=("mean", "std", "count"), lags=(1,),
            top_k=10,
        )
        assert recipes, "no recipes produced."
        for r in recipes:
            assert "y" not in dict(r.extra), f"recipe {r.name!r} captured a y reference -- leakage risk."
            c1 = apply_recipe(r, df)
            c2 = apply_recipe(r, df)
            np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# Contract: default disabled byte-identical + enabled wiring
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_temporal_agg_enable defaults to False and, when enabled, produces temporal_agg columns."""

    def test_mrmr_default_off_does_not_add_temporal_agg(self):
        """With the family disabled by default, MRMR.fit adds no temporal_agg_features_."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        df, y = _build_past_mean_causal(42, n_ent=80, per=8)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_temporal_agg_enable", False)) is False, "fe_temporal_agg_enable must default to False."
        m.fit(df, pd.Series(y, name="y"))
        ta = list(getattr(m, "temporal_agg_features_", []) or [])
        assert ta == [], f"temporal_agg added columns with the feature disabled: {ta}"

    def test_mrmr_enabled_adds_temporal_agg(self):
        """Enabling the family produces at least one temporal_agg engineered column."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        df, y = _build_past_mean_causal(42, n_ent=150, per=10)
        m = MRMR(
            max_runtime_mins=1.0,
            fe_temporal_agg_enable=True,
            fe_temporal_agg_entity_cols=("entity",),
            fe_temporal_agg_value_cols=("value",),
            fe_temporal_agg_time_col="ts",
            fe_temporal_agg_stats=("mean", "std", "count"),
            fe_temporal_agg_lags=(1,),
            fe_temporal_agg_top_k=5,
        )
        m.fit(df, pd.Series(y, name="y"))
        ta = list(getattr(m, "temporal_agg_features_", []) or [])
        assert len(ta) >= 1, "temporal_agg enabled but produced no engineered columns on the " "past-mean causal fixture."


# ---------------------------------------------------------------------------
# Contract: pickle / clone round-trip
# ---------------------------------------------------------------------------


class TestPickleClone:
    """Temporal recipes and MRMR params must survive pickle/clone round-trips intact."""

    def test_recipe_pickle_round_trip(self):
        """Recipe pickle round-trip preserves equality and replay output."""
        from mlframe.feature_selection.filters._temporal_agg_fe import (
            hybrid_temporal_agg_fe,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )

        df, y = _build_lag1_autoregressive(1)
        _, _appended, recipes, _ = hybrid_temporal_agg_fe(
            df, y, entity_cols=["entity"], value_cols=["value"],
            time_col="ts", stats=("mean", "std", "count"),
            windows=("30D",), lags=(1, 2), top_k=20,
        )
        assert recipes, "no recipes for pickle test."
        for r in recipes:
            r2 = pickle.loads(pickle.dumps(r))  # nosec B301 -- round-trip of a locally-created, trusted object
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            c1 = apply_recipe(r, df)
            c2 = apply_recipe(r2, df)
            np.testing.assert_array_equal(c1, c2)

    def test_mrmr_clone_preserves_params(self):
        """sklearn clone() copies every fe_temporal_agg_* ctor param without fitted state."""
        from sklearn.base import clone
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR(
            fe_temporal_agg_enable=True,
            fe_temporal_agg_entity_cols=("entity",),
            fe_temporal_agg_value_cols=("value",),
            fe_temporal_agg_time_col="ts",
            fe_temporal_agg_windows=("7D", "30D"),
            fe_temporal_agg_lags=(1, 2, 3),
            fe_temporal_agg_top_k=7,
        )
        c = clone(m)
        assert bool(c.fe_temporal_agg_enable) is True
        assert tuple(c.fe_temporal_agg_entity_cols) == ("entity",)
        assert tuple(c.fe_temporal_agg_value_cols) == ("value",)
        assert c.fe_temporal_agg_time_col == "ts"
        assert tuple(c.fe_temporal_agg_windows) == ("7D", "30D")
        assert tuple(c.fe_temporal_agg_lags) == (1, 2, 3)
        assert int(c.fe_temporal_agg_top_k) == 7
