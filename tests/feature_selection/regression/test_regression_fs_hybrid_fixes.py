"""Regression tests for the FS-hybrid fixes:

A. RFECV.fit accepts a bare ndarray X (previously crashed with a KeyError when y was a
   pandas Series carrying a non-RangeIndex: positional fold indices hit Series LABEL lookup).
   Also covers the split_into_train_test helper's ndarray-X + pandas-y branch directly.
D. MRMR exposes the additional-RFECV rescue knobs (additional_rfecv_selection_rule defaulting
   to the parsimonious 'one_se_min', and additional_rfecv_kwargs) as real constructor params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


# ----------------------------------------------------------------------------- Fix A: helper
def test_split_into_train_test_ndarray_x_pandas_y_positional():
    """ndarray X + pandas Series y with a SHUFFLED index must slice y positionally (not by label)."""
    from mlframe.feature_selection.wrappers._helpers import split_into_train_test

    X = np.arange(20).reshape(10, 2).astype(float)
    # y index deliberately non-RangeIndex (mimics a post-train_test_split Series).
    y = pd.Series(np.arange(100, 110), index=[7, 3, 9, 1, 5, 8, 0, 6, 2, 4])
    train_index = np.array([0, 2, 4, 6])
    test_index = np.array([1, 3])

    X_tr, y_tr, X_te, y_te = split_into_train_test(X, y, train_index, test_index)
    # POSITIONAL slice expected: y.iloc[[0,2,4,6]] == [100,102,104,106]
    assert list(np.asarray(y_tr)) == [100, 102, 104, 106]
    assert list(np.asarray(y_te)) == [101, 103]
    assert X_tr.shape == (4, 2) and X_te.shape == (2, 2)


# ----------------------------------------------------------------------------- Fix A: RFECV
def test_rfecv_accepts_ndarray_x_with_pandas_y():
    """RFECV.fit(ndarray, pandas-Series-with-shuffled-index) must not raise and must select features."""
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig
    from sklearn.model_selection import train_test_split

    Xa, ya = make_classification(n_samples=400, n_features=12, n_informative=5, random_state=0)
    X_df = pd.DataFrame(Xa, columns=[f"c{i}" for i in range(12)])
    y_s = pd.Series(ya)
    Xtr, _, ytr, _ = train_test_split(X_df, y_s, test_size=0.4, random_state=0, stratify=y_s)
    # ytr now carries a shuffled (non-Range) index; pass X as a bare ndarray -> the pre-fix crash path.
    r = RFECV(
        estimator=RandomForestClassifier(n_estimators=40, max_depth=5, random_state=0),
        cv=2,
        scoring=None,
        verbose=0,
        fi_config=FIConfig(importance_getter="auto", n_features_selection_rule="one_se_min"),
        search_config=SearchConfig(max_refits=6, max_runtime_mins=1),
        random_state=0,
    )
    r.fit(Xtr.to_numpy(), ytr)  # ndarray X + pandas y -> previously KeyError
    assert r.n_features_ >= 1
    names = list(r.get_feature_names_out())
    # ndarray columns get legacy-compatible stringified positional names ("0".."N-1").
    assert all(str(n).isdigit() for n in names)
    # transform works on both ndarray and the equivalent DataFrame.
    Zt_arr = r.transform(Xtr.to_numpy())
    assert Zt_arr.shape[0] == Xtr.shape[0] and Zt_arr.shape[1] == r.n_features_


# ----------------------------------------------------------------------------- Fix D: MRMR rescue knobs
def test_mrmr_exposes_additional_rfecv_rescue_knobs():
    """The rescue selection-rule + kwargs override knobs are real, get_params-visible constructor params."""
    from mlframe.feature_selection.filters import MRMR

    m = MRMR(fe_max_steps=0)
    params = m.get_params()
    assert "additional_rfecv_selection_rule" in params
    assert "additional_rfecv_kwargs" in params
    # Parsimonious default so the rescue does not re-admit the whole discarded pool.
    assert m.additional_rfecv_selection_rule == "one_se_min"
    assert m.additional_rfecv_kwargs is None

    m2 = MRMR(fe_max_steps=0, additional_rfecv_selection_rule="argmax", additional_rfecv_kwargs={"max_refits": 25})
    assert m2.additional_rfecv_selection_rule == "argmax"
    assert m2.additional_rfecv_kwargs == {"max_refits": 25}


# ----------------------------------------------------------------------------- Fix C + D: rescue pool excludes cluster members, uses one_se_min
def test_mrmr_rescue_excludes_cluster_members_and_uses_parsimony(monkeypatch):
    """The additional-RFECV rescue must (D) run with one_se_min and (C) NOT reconsider features already
    represented by a cluster aggregate / DCD PC1 swap. We stub the inner RFECV to record the pool it
    receives, so the test is fast and deterministic (no CatBoost) and asserts the exclusion + the rule.
    """
    import mlframe.feature_selection.filters.mrmr as mrmr_mod
    from mlframe.feature_selection.filters import MRMR

    rec: dict = {}

    class _StubRFECV:
        def __init__(self, *a, **k):
            rec["params"] = k

        def fit(self, X, y):
            rec["pool"] = list(X.columns)
            self.n_features_ = 0
            self.support_ = np.zeros(X.shape[1], dtype=bool)
            return self

    monkeypatch.setattr(mrmr_mod, "RFECV", _StubRFECV)

    # Tight redundant clusters around 3 informative columns so DCD/cluster-aggregate populates members.
    rng = np.random.default_rng(0)
    n = 1500
    z = rng.standard_normal((n, 3))
    y = (z[:, 0] + 0.8 * z[:, 1] - z[:, 2] + 0.3 * rng.standard_normal(n) > 0).astype(int)
    cols = {}
    for i in range(3):
        for j in range(4):  # 4 near-identical copies per informative latent -> obvious clusters
            cols[f"g{i}_{j}"] = z[:, i] + 0.05 * rng.standard_normal(n)
    for k in range(8):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)

    m = MRMR(
        verbose=0, fe_max_steps=0, n_jobs=1, random_seed=0, cluster_aggregate_enable=True, cluster_aggregate_mode="replace", run_additional_rfecv_minutes=0.2
    )
    m.fit(X, pd.Series(y))

    assert "pool" in rec, "rescue RFECV did not run"
    # Fix D: parsimonious rule is the default for the rescue.
    assert rec["params"].get("n_features_selection_rule") == "one_se_min"

    pool = set(rec["pool"])
    # Fix C: anything folded into a cluster aggregate or a DCD cluster must be absent from the rescue pool.
    removed = set(getattr(m, "_cluster_aggregate_removals_", None) or [])
    cluster_members = set()
    cm = getattr(m, "cluster_members_", None)
    if isinstance(cm, dict):
        for anchor, members in cm.items():
            cluster_members.add(anchor)
            if isinstance(members, (list, tuple, set)):
                cluster_members.update(members)
    assert not (removed & pool), f"cluster-aggregate-removed members leaked into rescue pool: {sorted(removed & pool)}"
    assert not (cluster_members & pool), f"DCD cluster members leaked into rescue pool: {sorted(cluster_members & pool)}"
    # Guard against a vacuous test: this fixture is designed to produce clusters.
    assert removed or cluster_members, "no cluster members discovered; fixture no longer exercises the exclusion"


# --------------------------------------------------------- ShapProxiedFS measured-optimal parsimony default
def test_shap_proxied_parsimony_tol_is_a_recall_vs_precision_dial():
    """ShapProxiedFS ships within_cluster_refine=True with the PRECISION-tuned default parsimony_tol=0.02.

    parsimony_tol is a recall-vs-precision dial, not a one-way win. The standalone default is 0.02 because this
    selector's native contract is excluding noise + redundancy: on the biz_val bed 0.02 keeps 0 noise while a
    looser 0.005 admits 1-2 (it prunes less). The opposite holds when optimising DOWNSTREAM AUC across models:
    on the fs_hybrid synthetic 0.005 recovers more (5.0/7 vs 4.3/7, mean AUC 0.795 vs 0.792). So AUC-optimising
    callers (the fs_hybrid ShapSel wrapper + HybridSelector) set parsimony_tol=0.005 EXPLICITLY, while the global
    default stays 0.02. This guards against re-flipping the global default to the recall setting (which regressed
    the noise-exclusion biz_val contract).
    """
    import inspect
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    p = inspect.signature(ShapProxiedFS.__init__).parameters
    assert p["within_cluster_refine"].default is True, "refine must stay ON by default"
    assert p["parsimony_tol"].default == 0.02, (
        "global parsimony_tol default must stay the precision-tuned 0.02; the recall-tuned 0.005 belongs only in "
        "AUC-optimising callers (hybrid/benchmark), set there explicitly"
    )
    s = ShapProxiedFS()
    assert s.within_cluster_refine is True and s.parsimony_tol == 0.02
