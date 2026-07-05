"""Unit + biz_value tests for the HybridSelector tree-importance member (round-4 A1-3/A3-1).

The member adds a cheap shallow-GBM signal the MI-filter members structurally miss: it folds co-occurrence PRODUCT
columns into the shared augmented frame, gated by the shared honest FI so it self-regulates by regime. Tests:
  - gate logic (_admit_tree_products): synergy / relevant_median / raw_median, on a synthetic FI dict (deterministic).
  - _tree_signals: returns an importance ranking + co-occurrence pairs incl. the true interaction operands.
  - _augment: tree product columns are present and REPLAYED leakage-free at transform time (pure raw[a]*raw[b]).
  - pickle: tree attrs survive a roundtrip (needed to replay products); transient _Xaug_/_y_ are dropped.
  - biz_value: on interaction-heavy data the member LIFTS held-out AUC vs use_tree_member=False (quantitative win).
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_selector import HybridSelector


def _interaction_frame(n=2000, seed=0, p_noise=25):
    """y driven by a PURE interaction a*b (zero-marginal operands) + a linear term, amid p_noise noise columns.

    p_noise controls the regime: with the total raw count > MRMR's fe_synergy_screen_max_features (60), MRMR's
    synergy bootstrap is SKIPPED and its marginal greedy never engineers a*b (the operands have ~0 marginal MI) --
    the madelon-like regime where the tree member is the ONLY source of the product. Below the cap MRMR engineers
    a*b itself, so the tree member is redundant there (use small p_noise for the cheap structural tests)."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    noise = rng.normal(size=(n, p_noise))
    y = ((a * b + 0.6 * c + 0.3 * rng.normal(size=n)) > 0).astype(int)
    X = pd.DataFrame(np.column_stack([a, b, c, noise]), columns=["a", "b", "c"] + [f"n{i}" for i in range(p_noise)])
    return X, pd.Series(y)


# ----------------------------------------------------------------- gate logic (deterministic, no model fit)
def _gate(gate, fi, pairs, relevant, raw_cols):
    h = HybridSelector(tree_prod_gate=gate)
    h.fi_ = fi
    h._tree_prod_pairs_ = pairs
    h._tree_prod_names_ = [f"tprod_{i}" for i in range(len(pairs))]
    return set(h._admit_tree_products(relevant, raw_cols))


def test_gate_synergy_admits_only_superadditive_products():
    # tprod_0 = a*b is super-additive (FI 0.9 > max(0.1, 0.1)); tprod_1 = c*d is NOT (FI 0.05 < max(0.4, 0.3)).
    fi = {"a": 0.1, "b": 0.1, "c": 0.4, "d": 0.3, "tprod_0": 0.9, "tprod_1": 0.05}
    raw = ["a", "b", "c", "d"]
    admitted = _gate("synergy", fi, [("a", "b"), ("c", "d")], raw + ["tprod_0", "tprod_1"], raw)
    assert admitted == {"tprod_0"}


def test_gate_raw_median_is_looser_than_synergy_on_noise_heavy_frame():
    # many zero-FI noise raw cols -> raw median ~0 -> a weak product (FI 0.02) clears raw_median but FAILS synergy.
    fi = {"a": 0.5, "b": 0.5, "tprod_0": 0.02}
    fi.update({f"n{i}": 0.0 for i in range(20)})
    raw = ["a", "b"] + [f"n{i}" for i in range(20)]
    rel = raw + ["tprod_0"]
    assert _gate("raw_median", fi, [("a", "b")], rel, raw) == {"tprod_0"}  # loose: admitted
    assert _gate("synergy", fi, [("a", "b")], rel, raw) == set()  # synergy: rejected (0.02 < 0.5)


def test_gate_relevant_median_uses_survivor_bar_not_raw():
    # relevant survivors {a,b} have FI 0.5 -> median 0.5; product at 0.3 fails relevant_median but passes raw_median.
    fi = {"a": 0.5, "b": 0.5, "tprod_0": 0.3}
    fi.update({f"n{i}": 0.0 for i in range(10)})
    raw = ["a", "b"] + [f"n{i}" for i in range(10)]
    assert _gate("relevant_median", fi, [("a", "b")], ["a", "b", "tprod_0"], raw) == set()
    assert _gate("raw_median", fi, [("a", "b")], ["a", "b", "tprod_0"], raw) == {"tprod_0"}


def test_gate_empty_when_no_products():
    h = HybridSelector(); h.fi_ = {}; h._tree_prod_pairs_ = []; h._tree_prod_names_ = []
    assert h._admit_tree_products(["a"], ["a"]) == []


# ----------------------------------------------------------------- tree signals + augment + pickle
def test_tree_signals_finds_interaction_pair():
    X, y = _interaction_frame()
    h = HybridSelector(use_tree_member=True, tree_cooccur_pairs=12)
    h.random_state = 0
    h._tree_signals(X, y)
    assert h._tree_ranked_, "tree should rank some features"
    assert {"a", "b"}.issubset(set(h._tree_ranked_[:6])), "the interaction operands should rank high"
    flat = {frozenset(p) for p in h._tree_prod_pairs_}
    assert frozenset({"a", "b"}) in flat, "the true a*b pair should be among co-occurrence pairs"


def test_augment_replays_tree_ops_leakage_free():
    from mlframe.feature_selection.hybrid_selector import _TREE_OPS
    X, y = _interaction_frame()
    h = HybridSelector(use_tree_member=True).fit(X, y)
    # tree op columns present in the augmented frame (named t{op}_N: tmul_/tabsd_/tsign_/trat_)
    aug = h._augment(X)
    tcols = [c for c in aug.columns if any(c.startswith(f"t{op}_") for op in _TREE_OPS)]
    assert tcols, "augmented frame should carry tree co-occurrence op columns"
    # replay on a fresh slice equals the exact op of raw[a], raw[b] (pure function of X, no leakage)
    for nm in h._tree_prod_names_:
        if nm in aug.columns:
            a, b, op = h._tree_op_[nm]
            expected = np.nan_to_num(_TREE_OPS[op](X[a].values.astype(float), X[b].values.astype(float)), nan=0.0, posinf=0.0, neginf=0.0)
            np.testing.assert_allclose(aug[nm].values[:50], expected[:50], rtol=1e-6)


def test_tree_signals_expands_rich_ops():
    """Default tree_rich_ops engineers one candidate column PER operator per co-occurrence pair, each tagged with
    its (a, b, op) in _tree_op_ so _augment replays it. products-only (tree_rich_ops=('mul',)) yields just tmul_."""
    X, y = _interaction_frame()
    h = HybridSelector(use_tree_member=True, tree_rich_ops=("mul", "absd", "sign", "rat"))
    h.random_state = 0
    h._tree_signals(X, y)
    ops_seen = {h._tree_op_[nm][2] for nm in h._tree_prod_names_}
    assert ops_seen == {"mul", "absd", "sign", "rat"}, f"all rich ops should be engineered, got {ops_seen}"
    h2 = HybridSelector(use_tree_member=True, tree_rich_ops=("mul",)); h2.random_state = 0
    h2._tree_signals(X, y)
    assert {h2._tree_op_[nm][2] for nm in h2._tree_prod_names_} == {"mul"}


def test_pickle_keeps_tree_attrs_drops_transient():
    import pickle
    X, y = _interaction_frame()
    h = HybridSelector(use_tree_member=True).fit(X, y)
    h2 = pickle.loads(pickle.dumps(h))
    assert h2._tree_prod_pairs_ == h._tree_prod_pairs_  # needed to replay tree op cols at transform
    assert h2._tree_prod_names_ == h._tree_prod_names_
    assert h2._tree_op_ == h._tree_op_  # the (a,b,op) spec per column
    assert not hasattr(h2, "_Xaug_") and not hasattr(h2, "_y_")  # transient training data dropped
    # transform still works after roundtrip
    assert list(h2.transform(X.iloc[:10]).columns) == list(h.transform(X.iloc[:10]).columns)


# ----------------------------------------------------------------- biz_value: quantitative win on interaction data
def test_bizvalue_tree_member_lifts_interaction_auc():
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    # VERY-WIDE frame (303 raw cols > the hybrid's mrmr_synergy_cap=250): MRMR's synergy bootstrap is SKIPPED, so the
    # MRMR member cannot engineer a*b (zero-marginal operands) and the tree member is the ONLY source of the product --
    # the madelon-like regime. (At <=250 cols MRMR's own bootstrap now engineers it too, so the tree member's marginal
    # lift shrinks -- which is why this frame must exceed the cap to isolate the tree member's contribution.)
    X, y = _interaction_frame(n=3000, seed=1, p_noise=300)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)

    def auc(sel):
        sel.fit(Xtr, ytr)
        Ztr, Zte = sel.transform(Xtr), sel.transform(Xte)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ztr, ytr)
        return roc_auc_score(yte, m.predict_proba(Zte)[:, 1])

    a_off = auc(HybridSelector(vote=1, use_fe=True, use_tree_member=False, random_state=1))
    a_on = auc(HybridSelector(vote=1, use_fe=True, use_tree_member=True, random_state=1))
    # The pure-interaction a*b is invisible to a LINEAR model unless the product is engineered; in the wide regime
    # only the tree member supplies it, so off-AUC is near chance for logit and on-AUC recovers it -- a clear lift.
    assert a_on >= a_off + 0.03, f"tree member should lift linear-downstream AUC on wide interaction data: {a_off:.3f} -> {a_on:.3f}"


# ----------------------------------------------------------------- mrmr_synergy_cap (default-raise of MRMR's bootstrap cap)
def test_mrmr_synergy_cap_default_is_250():
    import inspect
    assert inspect.signature(HybridSelector.__init__).parameters["mrmr_synergy_cap"].default == 250


def test_mrmr_synergy_cap_does_not_regress_on_wide_interaction_frame():
    """The default-raise (60 -> 250) enables MRMR's synergy bootstrap on moderate-width frames. On a wide (>60 col)
    interaction frame it must NOT regress vs MRMR's own default cap of 60 (the 3-seed bench shows +0.030 on
    hard_synth; here the safety property: raising the cap never meaningfully hurts)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    X, y = _interaction_frame(n=2500, seed=2, p_noise=75)   # 78 cols > 60 -> default would skip the bootstrap
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=2, stratify=y)

    def auc(cap):
        h = HybridSelector(vote=1, use_fe=True, mrmr_synergy_cap=cap, random_state=2).fit(Xtr, ytr)
        Ztr, Zte = h.transform(Xtr), h.transform(Xte)
        m = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(Ztr, ytr)
        return roc_auc_score(yte, m.predict_proba(Zte)[:, 1])

    a60, a250 = auc(60), auc(250)
    assert a250 >= a60 - 0.01, f"raising mrmr_synergy_cap must not regress on a wide interaction frame: cap60={a60:.3f} cap250={a250:.3f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov", "-p", "no:cacheprovider", "-p", "no:randomly"]))
