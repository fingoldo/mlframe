"""Layer 97 biz_value: DEFAULT-ON FLIP of the genuinely-safe MRMR mechanism
``fe_local_mi_gate`` (Layer 91 Tier 1).

The gate is a PURE corrective: after a recipe-emitting FE mechanism (L33 target
encoding, L34 count/freq/cat-num, L37 missingness, L38 ratio/grouped-delta/
lagged-diff) generates its candidate columns, it drops any whose marginal
MI(col; y) is below the RAW-baseline noise floor and keeps the top-K survivors.
It can NEVER touch a raw input feature, NEVER drops a genuinely predictive
engineered column (that column clears the raw floor by construction), and is a
strict NO-OP unless one of the four FE mechanisms is also enabled. Flipping its
default to True therefore shrinks the engineered candidate pool (speed +
downstream-MRMR precision) for users who enable an FE mechanism without reading
the docstring, with no accuracy downside.

Coverage:
* Ctor default is now True (and survives clone / pickle round-trip).
* recommend_enabled_fe classifies the flag as flip_safe and never as risky.
* Signal fixture: enabling the now-default gate does NOT drop a real engineered
  signal that the un-gated path selected.
* Noise fixture: the now-default gate DOES shrink the engineered pool.
* Full-suite smoke: representative layers (L21, L87, L91) import + a minimal fit
  runs clean with the new defaults.

NEVER xfail. Real MI / count numbers.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

SEEDS = (1, 7, 13)


def _make_mrmr(**overrides):
    from mlframe.feature_selection.filters.mrmr import MRMR
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _build_bounded_many_cat_signal(seed: int, n: int = 3000, n_noise_cats: int = 40):
    """One genuinely predictive categorical (its COUNT drives y) plus
    ``n_noise_cats`` independent random categoricals whose count carries no
    signal. Cardinality bounded under the raw cat-FE high-card ceiling so the
    augmented frame feeds straight into MRMR.fit."""
    rng = np.random.default_rng(seed)
    levels = np.array([f"P{k:02d}" for k in range(30)])
    w = np.geomspace(400.0, 5.0, 30)
    pred_cat = rng.choice(levels, size=n, p=w / w.sum())
    cnt = pd.Series(pred_cat).map(pd.Series(pred_cat).value_counts()).to_numpy().astype(float)
    z = np.log1p(cnt) - np.median(np.log1p(cnt))
    p = 1.0 / (1.0 + np.exp(-1.6 * z))
    y = (rng.random(n) < p).astype(int)
    data = {"pred_cat": pred_cat}
    for j in range(n_noise_cats):
        card = int(rng.integers(8, 40))
        nl = np.array([f"N{j}_{k:03d}" for k in range(card)])
        data[f"noise_cat_{j:02d}"] = rng.choice(nl, size=n)
    return pd.DataFrame(data), pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Ctor defaults flipped
# ---------------------------------------------------------------------------


class TestFlippedDefaults:
    def test_fe_local_mi_gate_default_true(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert m.fe_local_mi_gate is True, (
            "Layer 97: fe_local_mi_gate must default to True (corrective gate)"
        )
        # top_k unchanged.
        assert m.fe_local_mi_gate_top_k == 20

    def test_unified_second_pass_gate_stays_opt_in(self):
        """The Tier-2 unified CMI pass is NOT a pure corrective (it has a
        min_gain cost and CAN drop columns), so it must stay default-False."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert m.fe_unified_second_pass_gate is False

    def test_clone_preserves_flipped_default(self):
        m = _make_mrmr()
        m2 = clone(m)
        assert m2.get_params()["fe_local_mi_gate"] is True

    def test_pickle_round_trip_default(self):
        m = _make_mrmr()
        m2 = pickle.loads(pickle.dumps(m))
        assert m2.fe_local_mi_gate is True


# ---------------------------------------------------------------------------
# recommend_enabled_fe classification
# ---------------------------------------------------------------------------


class TestRecommendEnabledFe:
    def test_local_mi_gate_classified_flip_safe(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        rec = MRMR.recommend_enabled_fe()
        assert "fe_local_mi_gate" in rec["flip_safe"]
        # A pure corrective is never a risky generator.
        assert "fe_local_mi_gate" not in rec["flip_risky"]

    def test_generators_classified_risky(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        rec = MRMR.recommend_enabled_fe()
        for gen in (
            "fe_grouped_agg_enable", "fe_cat_pair_enable", "fe_hybrid_orth_enable",
            "fe_count_encoding_enable", "fe_unified_second_pass_gate",
        ):
            assert gen in rec["flip_risky"], f"{gen} should be flip_risky"

    def test_already_default_listed(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        rec = MRMR.recommend_enabled_fe()
        assert "dcd_enable" in rec["already_default"]
        assert "cardinality_bias_correction" in rec["already_default"]

    def test_recommended_enable_is_stub(self):
        """L98 Param-Oracle deliverable; the L97 stub returns an empty list."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rec = MRMR.recommend_enabled_fe(X=None, y=None)
        assert rec["recommended_enable"] == []

    def test_no_flag_in_two_buckets(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        rec = MRMR.recommend_enabled_fe()
        safe, default, risky = (
            set(rec["flip_safe"]), set(rec["already_default"]), set(rec["flip_risky"]),
        )
        assert not (safe & risky)
        assert not (safe & default)
        assert not (default & risky)


# ---------------------------------------------------------------------------
# Signal fixture: now-default gate does NOT drop a real engineered signal
# ---------------------------------------------------------------------------


class TestSignalSurvivesNewDefault:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_predictive_count_survives_default_gate(self, seed: int):
        """With the gate at its NEW default (True), the genuinely predictive
        ``pred_cat__count`` is still selected -- the same column the explicitly
        un-gated path selects. The gate only removed noise, not signal."""
        X, y = _build_bounded_many_cat_signal(seed, n=3000, n_noise_cats=40)

        # Explicit OFF (pre-Layer-97 behaviour).
        m_off = _make_mrmr(
            fe_count_encoding_enable=True, fe_local_mi_gate=False,
            fe_ntop_features=5,
        )
        m_off.fit(X, y)

        # NEW default (gate True, not passed explicitly).
        m_def = _make_mrmr(
            fe_count_encoding_enable=True, fe_ntop_features=5,
        )
        m_def.fit(X, y)
        assert m_def.fe_local_mi_gate is True

        assert "pred_cat__count" in m_off.count_encoding_features_, (
            f"seed={seed}: ungated path should select the predictive count"
        )
        assert "pred_cat__count" in m_def.count_encoding_features_, (
            f"seed={seed}: NEW-default gate dropped the real signal "
            f"pred_cat__count; survivors={m_def.count_encoding_features_}"
        )


# ---------------------------------------------------------------------------
# Noise fixture: now-default gate shrinks the engineered pool (the win)
# ---------------------------------------------------------------------------


class TestPoolShrinksUnderNewDefault:
    @pytest.mark.parametrize("seed", (1, 7))
    def test_default_gate_shrinks_count_pool(self, seed: int):
        """41 cats (1 predictive + 40 noise). The ungated count-encoding emits
        one column per cat; the NEW-default gate (top_k=20) keeps strictly
        fewer, and never more than top_k -- the speed / precision win."""
        X, y = _build_bounded_many_cat_signal(seed, n=3000, n_noise_cats=40)

        m_off = _make_mrmr(
            fe_count_encoding_enable=True, fe_local_mi_gate=False,
            fe_ntop_features=5,
        )
        m_off.fit(X, y)
        n_off = len(m_off.count_encoding_features_)

        m_def = _make_mrmr(
            fe_count_encoding_enable=True, fe_ntop_features=5,
        )
        m_def.fit(X, y)
        n_def = len(m_def.count_encoding_features_)

        assert n_def <= m_def.fe_local_mi_gate_top_k, (
            f"seed={seed}: default-gated pool {n_def} exceeds top_k"
        )
        assert n_def < n_off, (
            f"seed={seed}: default gate did NOT shrink the pool "
            f"(off={n_off}, default={n_def})"
        )


# ---------------------------------------------------------------------------
# Full-suite smoke: representative layers still import + fit with new defaults
# ---------------------------------------------------------------------------


class TestRepresentativeLayersSmoke:
    def test_representative_layer_modules_import(self):
        import importlib
        for layer in (21, 87, 91):
            mod = importlib.import_module(
                f"tests.feature_selection.test_biz_value_mrmr_layer{layer}"
            )
            assert mod is not None

    def test_minimal_fit_under_new_defaults(self):
        """A plain numeric fit (no FE mechanism on) is byte-stable under the new
        defaults because the gate is a strict no-op when no mechanism runs."""
        rng = np.random.default_rng(0)
        n = 1500
        x_sig = rng.standard_normal(n)
        y = (x_sig + 0.3 * rng.standard_normal(n) > 0).astype(int)
        X = pd.DataFrame({
            "sig": x_sig,
            "noise0": rng.standard_normal(n),
            "noise1": rng.standard_normal(n),
        })
        ys = pd.Series(y, name="y")
        m = _make_mrmr(fe_ntop_features=3)
        m.fit(X, ys)
        assert "sig" in list(m.get_feature_names_out())
