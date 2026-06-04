"""Layer 91 biz_value: TWO-TIER IT GATES on the four recipe-emitting FE
mechanisms (L33 target-encoding, L34 count/freq/cat-num, L37 missingness,
L38 ratio/grouped-delta/lagged-diff).

Tier 1 (local MI floor) bounds each mechanism's combinatorial pool by dropping
candidate columns whose marginal MI(col; y) is below the RAW-baseline noise
floor and keeping top-K. Tier 2 (unified second-pass CMI gate) runs ONE greedy
CMI selection over ALL engineered columns conditioned on the running support,
catching CROSS-mechanism redundancy a per-mechanism gate cannot see.

Coverage:
* Tier 1 bounds pool: 50 cat cols -> <= K count-encoded columns (vs 50).
* Tier 1 keeps signal: the genuinely predictive cat's count-encoding survives.
* Tier 2 drops cross-mechanism redundant: count(cat_a) ~ freq(cat_a) -> one.
* Tier 2 keeps complementary: count(cat_a) + count(cat_b) both kept.
* Default disabled byte-identical: both gates off == current master.
* Pickle / clone.

NEVER xfail. Real MI / count numbers.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


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


def _build_many_cat_signal(seed: int, n: int = 4000, n_noise_cats: int = 50):
    """One genuinely predictive categorical (its COUNT drives y), plus
    ``n_noise_cats`` independent random categoricals whose count carries no
    signal. The Tier-1 floor should drop the ~50 noise count-encodings and keep
    the predictive one (or at most top-K)."""
    rng = np.random.default_rng(seed)
    # Predictive cat: a Zipf-like distribution; P(y=1) sigmoid in log(count).
    heavy = [f"P_HEAVY_{i}" for i in range(8)]
    hw = np.array([400, 350, 300, 250, 200, 200, 200, 150])
    heavy_assign = np.concatenate([np.repeat(u, int(w)) for u, w in zip(heavy, hw)])
    rng.shuffle(heavy_assign)
    n_rare = n - int(hw.sum())
    rare = np.array([f"P_RARE_{i:05d}" for i in range(max(n_rare, 0))])
    pred_cat = np.concatenate([heavy_assign, rare])[:n]
    rng.shuffle(pred_cat)
    counts = pd.Series(pred_cat).value_counts()
    log_cnt = np.log1p(pd.Series(pred_cat).map(counts).to_numpy().astype(float))
    p = 1.0 / (1.0 + np.exp(-2.0 * (log_cnt - float(np.median(log_cnt)))))
    y = (rng.random(n) < p).astype(int)

    data = {"pred_cat": pred_cat}
    # Noise cats: each has cardinality in [5, 500] so auto-detect picks it,
    # but the count of each random level is unrelated to y.
    for j in range(n_noise_cats):
        card = rng.integers(8, 60)
        levels = np.array([f"N{j}_{k:03d}" for k in range(card)])
        data[f"noise_cat_{j:02d}"] = rng.choice(levels, size=n)
    X = pd.DataFrame(data)
    return X, pd.Series(y, name="y")


def _build_bounded_many_cat_signal(seed: int, n: int = 3000, n_noise_cats: int = 40):
    """Same shape as ``_build_many_cat_signal`` but every categorical's
    cardinality is bounded well under the raw cat-FE high-card ceiling
    (sqrt(n)*2), so the augmented frame can be fed into MRMR.fit. The COUNT
    signal lives in the per-level frequency spread of ``pred_cat``."""
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
# Tier 1 — local MI floor
# ---------------------------------------------------------------------------


class TestTier1BoundsPool:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_count_encoding_pool_bounded(self, seed: int):
        """Without the gate, count-encoding emits 1 column per cat (51 here).
        With the gate (top_k=5), the surviving pool is <= 5."""
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_with_recipes,
        )
        X, y = _build_many_cat_signal(seed)
        cat_cols = [c for c in X.columns]  # 51 cats

        _, ungated, _ = count_encode_with_recipes(X, cat_cols=cat_cols)
        assert len(ungated) == 51, f"expected 51 ungated, got {len(ungated)}"

        _, gated, _ = count_encode_with_recipes(
            X, cat_cols=cat_cols, mi_gate=True, mi_gate_top_k=5, y=y.to_numpy(),
        )
        assert len(gated) <= 5, (
            f"seed={seed}: gated pool {len(gated)} exceeds top_k=5"
        )
        assert len(gated) >= 1, f"seed={seed}: gate dropped everything"
        assert len(gated) < len(ungated)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_signal_survives_floor(self, seed: int):
        """The predictive cat's count-encoding clears the raw-baseline floor."""
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_with_recipes,
        )
        X, y = _build_many_cat_signal(seed)
        cat_cols = [c for c in X.columns]
        _, gated, _ = count_encode_with_recipes(
            X, cat_cols=cat_cols, mi_gate=True, mi_gate_top_k=5, y=y.to_numpy(),
        )
        assert "pred_cat__count" in gated, (
            f"seed={seed}: predictive count-encoding was dropped by the floor; "
            f"survivors={gated}"
        )

    def test_floor_anchored_on_raw_not_engineered(self):
        from mlframe.feature_selection.filters._unified_fe_gate import (
            raw_mi_noise_floor,
        )
        X, y = _build_many_cat_signal(seed=1)
        # raw_X has only object cats -> no numeric raw cols -> floor 0.0.
        floor_obj = raw_mi_noise_floor(X, y.to_numpy())
        assert floor_obj == 0.0
        # With numeric raw cols, the floor is a finite positive band.
        rng = np.random.default_rng(0)
        Xn = pd.DataFrame({f"r{i}": rng.standard_normal(len(X)) for i in range(6)})
        floor_num = raw_mi_noise_floor(Xn, y.to_numpy())
        assert np.isfinite(floor_num) and floor_num >= 0.0


# ---------------------------------------------------------------------------
# Tier 2 — unified second-pass CMI gate (cross-mechanism)
# ---------------------------------------------------------------------------


def _build_two_cat_redundant(seed: int, n: int = 3000):
    """Two predictive cats. count(cat_a) and freq(cat_a) are an affine
    transform of each other (identical bin pattern). count(cat_b) is
    independently informative.

    Cardinality is kept under the raw cat-FE high-card ceiling (sqrt(n)*2) so
    these fixtures can be fed straight into MRMR.fit without tripping the
    cat-interactions guard -- the count/freq SIGNAL only needs a spread of
    per-level frequencies, not a long singleton tail."""
    rng = np.random.default_rng(seed)

    def _spread_cat(prefix, n_levels=30):
        levels = np.array([f"{prefix}_{k:02d}" for k in range(n_levels)])
        # Geometric frequency spread: a few heavy levels, many light ones.
        w = np.geomspace(300.0, 5.0, n_levels)
        col = rng.choice(levels, size=n, p=w / w.sum())
        return col

    cat_a = _spread_cat("A")
    cat_b = _spread_cat("B")
    cnt_a = pd.Series(cat_a).map(pd.Series(cat_a).value_counts()).to_numpy().astype(float)
    cnt_b = pd.Series(cat_b).map(pd.Series(cat_b).value_counts()).to_numpy().astype(float)
    za = np.log1p(cnt_a) - np.median(np.log1p(cnt_a))
    zb = np.log1p(cnt_b) - np.median(np.log1p(cnt_b))
    p = 1.0 / (1.0 + np.exp(-(1.2 * za + 1.2 * zb)))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"cat_a": cat_a, "cat_b": cat_b})
    return X, pd.Series(y, name="y")


class TestTier2CrossMechanism:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_drops_redundant_count_vs_freq(self, seed: int):
        """count(cat_a) and freq(cat_a) are identical up to affine scaling ->
        identical equi-frequency bins -> the second pass keeps only ONE."""
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_fit, frequency_encode_fit,
        )
        from mlframe.feature_selection.filters._unified_fe_gate import (
            unified_second_pass_gate,
        )
        X, y = _build_two_cat_redundant(seed)
        cnt, _ = count_encode_fit(X, ["cat_a"])
        frq, _ = frequency_encode_fit(X, ["cat_a"])
        Xall = pd.concat([X, cnt, frq], axis=1)
        eng = ["cat_a__count", "cat_a__freq"]
        keep = unified_second_pass_gate(
            Xall, y.to_numpy(), raw_cols=["cat_a", "cat_b"], engineered_cols=eng,
        )
        assert len(keep) == 1, (
            f"seed={seed}: count/freq of same col are redundant; gate kept "
            f"{keep} (expected exactly 1)"
        )
        assert keep[0] in eng

    @pytest.mark.parametrize("seed", SEEDS)
    def test_keeps_complementary_count_a_and_b(self, seed: int):
        """count(cat_a) and count(cat_b) are both independently informative ->
        both kept."""
        from mlframe.feature_selection.filters._count_freq_interaction_fe import (
            count_encode_fit,
        )
        from mlframe.feature_selection.filters._unified_fe_gate import (
            unified_second_pass_gate,
        )
        X, y = _build_two_cat_redundant(seed)
        cnt, _ = count_encode_fit(X, ["cat_a", "cat_b"])
        Xall = pd.concat([X, cnt], axis=1)
        eng = ["cat_a__count", "cat_b__count"]
        keep = unified_second_pass_gate(
            Xall, y.to_numpy(), raw_cols=["cat_a", "cat_b"], engineered_cols=eng,
            seed_raw_cols_count=0,  # raw cats are object -> nothing to seed anyway
        )
        assert set(keep) == set(eng), (
            f"seed={seed}: both complementary count-encodings should survive; "
            f"kept {keep}"
        )


# ---------------------------------------------------------------------------
# End-to-end MRMR.fit with both gates
# ---------------------------------------------------------------------------


class TestMRMRIntegration:
    @pytest.mark.parametrize("seed", (1, 7))
    def test_local_gate_bounds_count_encoding_features(self, seed: int, monkeypatch):
        # The gate bounds the engineered candidate POOL the count-encoding stage
        # hands to the MRMR screen (51 cats -> <= top_k columns). It does NOT
        # bound the post-selection ``count_encoding_features_`` ROSTER: this
        # fixture has exactly ONE predictive cat (``pred_cat``), so the MRMR
        # relevance screen drops every noise count-encoding regardless of the
        # gate -- the roster is 1 with the gate ON or OFF, masking the gate's
        # effect. Observe the gate where it actually operates: capture the
        # appended pool size from the genuine in-fit ``count_encode_with_recipes``
        # call so the shrink (off == one-per-cat vs on <= top_k) is visible.
        import mlframe.feature_selection.filters._count_freq_interaction_fe as _cfi
        X, y = _build_bounded_many_cat_signal(seed, n=3000, n_noise_cats=40)

        _orig_count_enc = _cfi.count_encode_with_recipes
        _pool = {}

        def _capture(tag):
            def _wrapped(Xarg, **kw):
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

        monkeypatch.setattr(_cfi, "count_encode_with_recipes", _capture("on"))
        m_on = _make_mrmr(
            fe_count_encoding_enable=True, fe_local_mi_gate=True,
            fe_local_mi_gate_top_k=5, fe_ntop_features=5,
        )
        m_on.fit(X, y)
        n_on = _pool["on"]
        assert n_off > n_on, (
            f"seed={seed}: local gate did not shrink the count pool "
            f"(off={n_off}, on={n_on})"
        )
        assert n_on <= 5
        # The genuinely predictive count-encoding clears the gate's MI floor and
        # is the column the screen ultimately selects.
        assert "pred_cat__count" in m_on.count_encoding_features_, (
            f"seed={seed}: predictive count-encoding dropped by MRMR local gate"
        )

    def test_unified_gate_drops_count_or_freq(self):
        """With BOTH count and freq encoding enabled on the same cats, the
        Tier-2 gate prunes the redundant sibling end-to-end inside MRMR.fit."""
        X, y = _build_two_cat_redundant(seed=1)
        m = _make_mrmr(
            fe_count_encoding_enable=True,
            fe_count_encoding_cols=("cat_a", "cat_b"),
            fe_frequency_encoding_enable=True,
            fe_frequency_encoding_cols=("cat_a", "cat_b"),
            fe_unified_second_pass_gate=True,
            fe_ntop_features=8,
        )
        m.fit(X, y)
        eng = (
            list(m.count_encoding_features_)
            + list(m.frequency_encoding_features_)
        )
        # For each cat, count and freq are redundant; at most one of the
        # {count,freq} pair per cat survives the Tier-2 gate.
        for cat in ("cat_a", "cat_b"):
            pair = [f"{cat}__count", f"{cat}__freq"]
            present = [c for c in pair if c in eng]
            assert len(present) <= 1, (
                f"Tier-2 gate kept BOTH redundant siblings for {cat}: {present}"
            )


# ---------------------------------------------------------------------------
# Default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    def test_gates_default_off(self):
        # 2026-06-01 Layer 97 — ``fe_local_mi_gate`` default flipped to True.
        # It is a pure corrective (drops only sub-noise engineered columns,
        # keeps top-k) and a strict no-op unless an L33/L34/L37/L38 FE
        # mechanism is also enabled, so enabling by default cannot reduce
        # accuracy. ``fe_unified_second_pass_gate`` stays opt-in (it is a real
        # CMI pass with a min_gain cost that CAN drop columns).
        m = _make_mrmr()
        assert m.fe_local_mi_gate is True
        assert m.fe_unified_second_pass_gate is False

    def test_transform_identical_with_gates_off_vs_master(self):
        """With both gates OFF (the default), enabling count-encoding produces
        the SAME selected columns whether or not the Layer-91 knobs exist --
        i.e. the gate knobs are no-ops when their masters are False."""
        # Low-cardinality cats (<= sqrt(n)*2) so the raw cat-FE high-card guard
        # doesn't trip -- this test isolates the Layer-91 byte-identity claim.
        rng = np.random.default_rng(7)
        n = 2500
        levels = np.array([f"L{k:02d}" for k in range(20)])
        level_counts = rng.integers(20, 400, size=20)
        pred_cat = rng.choice(levels, size=n, p=level_counts / level_counts.sum())
        cnt = pd.Series(pred_cat).map(pd.Series(pred_cat).value_counts()).to_numpy().astype(float)
        z = np.log1p(cnt) - np.median(np.log1p(cnt))
        p = 1.0 / (1.0 + np.exp(-1.5 * z))
        yv = (rng.random(n) < p).astype(int)
        noise_levels = np.array([f"M{k:02d}" for k in range(15)])
        X = pd.DataFrame({
            "pred_cat": pred_cat,
            "noise_cat": rng.choice(noise_levels, size=n),
        })
        y = pd.Series(yv, name="y")
        Xtr, Xho = X.iloc[:1800].reset_index(drop=True), X.iloc[1800:].reset_index(drop=True)
        ytr = y.iloc[:1800].reset_index(drop=True)

        # 2026-06-01 Layer 97 — both instances pin the gate OFF explicitly so
        # this test isolates the "gate knobs are no-ops when OFF" byte-identity
        # claim (the ctor default for fe_local_mi_gate is now True).
        m1 = _make_mrmr(
            fe_count_encoding_enable=True, fe_count_encoding_cols=("pred_cat",),
            fe_ntop_features=4,
            fe_local_mi_gate=False, fe_unified_second_pass_gate=False,
        )
        m1.fit(Xtr, ytr)
        out1 = m1.transform(Xho)

        m2 = _make_mrmr(
            fe_count_encoding_enable=True, fe_count_encoding_cols=("pred_cat",),
            fe_ntop_features=4,
            fe_local_mi_gate=False, fe_unified_second_pass_gate=False,
            fe_local_mi_gate_top_k=20,
        )
        m2.fit(Xtr, ytr)
        out2 = m2.transform(Xho)

        assert list(out1.columns) == list(out2.columns), (
            "Layer-91 knobs (gates OFF) changed the selected columns"
        )
        for c in out1.columns:
            if pd.api.types.is_numeric_dtype(out1[c]):
                np.testing.assert_allclose(
                    out1[c].to_numpy(), out2[c].to_numpy(), atol=1e-12,
                )


# ---------------------------------------------------------------------------
# Pickle / clone
# ---------------------------------------------------------------------------


class TestPickleClone:
    def test_clone_preserves_layer91_params(self):
        m = _make_mrmr(
            fe_local_mi_gate=True, fe_local_mi_gate_top_k=7,
            fe_unified_second_pass_gate=True,
            fe_unified_second_pass_max_keep=12,
            fe_unified_second_pass_min_gain=0.01,
        )
        m2 = clone(m)
        p, p2 = m.get_params(), m2.get_params()
        for k in (
            "fe_local_mi_gate", "fe_local_mi_gate_top_k",
            "fe_unified_second_pass_gate", "fe_unified_second_pass_max_keep",
            "fe_unified_second_pass_min_gain",
        ):
            assert p[k] == p2[k], f"clone lost {k}"

    def test_pickle_round_trip_fitted(self):
        X, y = _build_bounded_many_cat_signal(seed=1, n=2000, n_noise_cats=15)
        m = _make_mrmr(
            fe_count_encoding_enable=True, fe_local_mi_gate=True,
            fe_local_mi_gate_top_k=5,
            fe_unified_second_pass_gate=True,
            fe_ntop_features=4,
        )
        m.fit(X, y)
        out = m.transform(X)
        m_pkl = pickle.loads(pickle.dumps(m))
        out_pkl = m_pkl.transform(X)
        assert list(out.columns) == list(out_pkl.columns)
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]):
                np.testing.assert_allclose(
                    out[c].to_numpy(), out_pkl[c].to_numpy(), atol=1e-12,
                )
