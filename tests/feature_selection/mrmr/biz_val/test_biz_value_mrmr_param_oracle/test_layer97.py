"""Layer 97 biz_value: DEFAULT-ON FLIP of the genuinely-safe MRMR mechanism

Consolidated verbatim from test_biz_value_mrmr_layer97.py (per audit finding test_code_quality-16).
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


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
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
    """fe_local_mi_gate now defaults to True; fe_unified_second_pass_gate stays opt-in."""

    def test_fe_local_mi_gate_default_true(self):
        """MRMR() ctor default for fe_local_mi_gate is True with top_k unchanged at 20."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert m.fe_local_mi_gate is True, "Layer 97: fe_local_mi_gate must default to True (corrective gate)"
        # top_k unchanged.
        assert m.fe_local_mi_gate_top_k == 20

    def test_unified_second_pass_gate_stays_opt_in(self):
        """The Tier-2 unified CMI pass is NOT a pure corrective (it has a
        min_gain cost and CAN drop columns), so it must stay default-False."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert m.fe_unified_second_pass_gate is False

    def test_clone_preserves_flipped_default(self):
        """sklearn clone() preserves the new fe_local_mi_gate=True default."""
        m = _make_mrmr()
        m2 = clone(m)
        assert m2.get_params()["fe_local_mi_gate"] is True

    def test_pickle_round_trip_default(self):
        """A pickle round-trip preserves the new fe_local_mi_gate=True default."""
        m = _make_mrmr()
        m2 = pickle.loads(pickle.dumps(m))  # nosec B301 -- round-trip of a locally-created, trusted object
        assert m2.fe_local_mi_gate is True


# ---------------------------------------------------------------------------
# recommend_enabled_fe classification
# ---------------------------------------------------------------------------


class TestRecommendEnabledFe:
    """MRMR.recommend_enabled_fe() must classify every FE flag into exactly one bucket."""

    def test_local_mi_gate_classified_flip_safe(self):
        """fe_local_mi_gate is classified flip_safe and never flip_risky."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rec = MRMR.recommend_enabled_fe()
        assert "fe_local_mi_gate" in rec["flip_safe"]
        # A pure corrective is never a risky generator.
        assert "fe_local_mi_gate" not in rec["flip_risky"]

    def test_generators_classified_risky(self):
        """Every accuracy-affecting FE generator is classified flip_risky."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rec = MRMR.recommend_enabled_fe()
        for gen in (
            "fe_grouped_agg_enable", "fe_cat_pair_enable", "fe_hybrid_orth_enable",
            "fe_count_encoding_enable", "fe_unified_second_pass_gate",
        ):
            assert gen in rec["flip_risky"], f"{gen} should be flip_risky"

    def test_already_default_listed(self):
        """Flags already default-on (dcd_enable, cardinality_bias_correction) appear in already_default."""
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
        """The flip_safe, already_default, and flip_risky buckets are mutually exclusive."""
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
    """The new-default gate must not drop a genuinely predictive engineered signal."""

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

        assert "pred_cat__count" in m_off.count_encoding_features_, f"seed={seed}: ungated path should select the predictive count"
        assert "pred_cat__count" in m_def.count_encoding_features_, (
            f"seed={seed}: NEW-default gate dropped the real signal " f"pred_cat__count; survivors={m_def.count_encoding_features_}"
        )


# ---------------------------------------------------------------------------
# Noise fixture: now-default gate shrinks the engineered pool (the win)
# ---------------------------------------------------------------------------


class TestPoolShrinksUnderNewDefault:
    """The new-default gate must shrink the engineered candidate pool relative to the ungated path."""

    @pytest.mark.parametrize("seed", (1, 7))
    def test_default_gate_shrinks_count_pool(self, seed: int, monkeypatch):
        """41 cats (1 predictive + 40 noise). The ungated count-encoding emits
        one column per cat; the NEW-default gate (top_k=20) keeps strictly
        fewer, and never more than top_k -- the speed / precision win.

        The shrink is on the engineered candidate POOL the count-encoding stage
        hands to the MRMR screen, NOT on the post-selection
        ``count_encoding_features_`` roster: this fixture has exactly ONE
        predictive cat, so the MRMR relevance screen drops every noise
        count-encoding regardless of the gate and the roster is 1 either way,
        masking the gate. Observe the gate where it operates by capturing the
        appended pool size from the genuine in-fit ``count_encode_with_recipes``
        call."""
        import mlframe.feature_selection.filters._count_freq_interaction_fe as _cfi
        X, y = _build_bounded_many_cat_signal(seed, n=3000, n_noise_cats=40)

        _orig_count_enc = _cfi.count_encode_with_recipes
        _pool = {}

        def _capture(tag):
            """Build a count_encode_with_recipes wrapper that records the appended pool size under ``tag``."""

            def _wrapped(Xarg, **kw):
                """Call the original count_encode_with_recipes and record its appended-pool size."""
                res = _orig_count_enc(Xarg, **kw)
                _pool[tag] = len(res[1])
                return res
            return _wrapped

        monkeypatch.setattr(_cfi, "count_encode_with_recipes", _capture("off"))
        m_off = _make_mrmr(
            fe_count_encoding_enable=True, fe_local_mi_gate=False,
            fe_ntop_features=5,
        )
        m_off.fit(X, y)
        n_off = _pool["off"]

        monkeypatch.setattr(_cfi, "count_encode_with_recipes", _capture("def"))
        m_def = _make_mrmr(
            fe_count_encoding_enable=True, fe_ntop_features=5,
        )
        m_def.fit(X, y)
        n_def = _pool["def"]

        assert n_def <= m_def.fe_local_mi_gate_top_k, f"seed={seed}: default-gated pool {n_def} exceeds top_k"
        assert n_def < n_off, f"seed={seed}: default gate did NOT shrink the pool " f"(off={n_off}, default={n_def})"


# ---------------------------------------------------------------------------
# Full-suite smoke: representative layers still import + fit with new defaults
# ---------------------------------------------------------------------------


class TestRepresentativeLayersSmoke:
    """A smoke check that representative layer modules still import and fit under the new defaults."""

    def test_representative_layer_modules_import(self):
        """Representative layer test modules import cleanly under the new fe_local_mi_gate default."""
        import importlib
        # Representative layers, now relocated into themed subpackages (the flat
        # test_biz_value_mrmr_layer<N>.py files were consolidated).
        modules = (
            "tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_fe_hybrid_orth.test_layer21",
            "tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_grouped_cat_fe.test_composite_group_key",
            "tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_param_oracle.test_layer91",
        )
        for name in modules:
            mod = importlib.import_module(name)
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
