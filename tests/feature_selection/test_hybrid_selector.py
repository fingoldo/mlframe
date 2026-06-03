"""Unit tests for the compute-once-share-many HybridSelector (feature_selection/_benchmarks/fs_hybrid).

The cluster-aware combine rule (Stage 3) is a pure function of the shared artifacts (FI / clusters) and the
per-member selections, so the bulk of the behaviour is tested by injecting that state into ``_combine`` directly
-- no expensive MRMR/SHAP/Boruta fits. A single end-to-end test (generous timeout) exercises the real plumbing:
the three shared artifacts are populated once and the members run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

COLS = ["inf_0", "red_0", "red_1", "inf_1", "inf_2", "noise_0"]


def _selector(**kw):
    """A HybridSelector with the shared state injected (clusters + FI), ready for _combine without fitting.

    Clusters: inf_0 + red_0 + red_1 form one redundant cluster (rep inf_0); inf_1 / inf_2 / noise_0 are singletons.
    Shared FI: informative high, inf_2 weak, noise ~0.
    """
    from mlframe.feature_selection._benchmarks.fs_hybrid.hybrid_selector import HybridSelector
    kw.setdefault("anchor_fe", False)  # these tests exercise the PLAIN cluster-vote; the anchored path has its own test
    h = HybridSelector(**kw)
    h.members_ = {"inf_0": ["inf_0", "red_0", "red_1"], "inf_1": ["inf_1"], "inf_2": ["inf_2"], "noise_0": ["noise_0"]}
    h.cluster_of_ = {f: r for r, ms in h.members_.items() for f in ms}
    h.fi_ = {"inf_0": 0.10, "red_0": 0.09, "red_1": 0.08, "inf_1": 0.06, "inf_2": 0.02, "noise_0": 0.001}
    return h


# member sub-selections: inf_0 cluster gets 3 votes (mrmr+shap+boruta), inf_1 gets 2 (mrmr+boruta),
# inf_2 gets 1 (shap), noise_0 gets 1 (boruta). boruta picks red_0 (an inf_0-cluster member) not inf_0 itself.
MEMBER_SEL = {"mrmr": ["inf_0", "inf_1"], "shap": ["inf_0", "inf_2"], "boruta": ["red_0", "inf_1", "noise_0"]}


def test_vote1_keeps_all_clusters_and_dedups_redundant():
    """vote=1 (default): every voted cluster kept; redundant inf_0 cluster collapses to its best-FI member."""
    h = _selector(vote=1, fi_guard=False, expand_clusters=False)
    sel = set(h._combine(MEMBER_SEL, COLS))
    assert {"inf_0", "inf_1", "inf_2"} <= sel          # all informative recovered (incl. single-member inf_2)
    assert "noise_0" in sel                            # vote=1 admits the single-member noise too
    assert "red_0" not in sel and "red_1" not in sel   # redundant copies de-duplicated to the rep inf_0


def test_vote2_drops_single_member_clusters_noise_and_weak_signal():
    """vote=2 (majority) drops EVERY single-member cluster -- noise_0 AND the real-but-weak inf_2: the recall
    cost that makes vote=1 the default."""
    h = _selector(vote=2, fi_guard=False)
    sel = set(h._combine(MEMBER_SEL, COLS))
    assert {"inf_0", "inf_1"} <= sel
    assert "noise_0" not in sel
    assert "inf_2" not in sel  # consensus requirement sacrifices the feature only one member caught


def test_fi_guard_cuts_noise_but_also_the_weak_base_feature():
    """fi_guard=True admits a single-member cluster only above the consensus-FI median (0.08 here): it drops the
    noise (FI 0.001) AND the weak base inf_2 (FI 0.02) -- the measured recall-for-precision trade that keeps it
    OFF by default."""
    h = _selector(vote=1, fi_guard=True)
    sel = set(h._combine(MEMBER_SEL, COLS))
    assert {"inf_0", "inf_1"} <= sel
    assert "noise_0" not in sel
    assert "inf_2" not in sel


def test_expand_clusters_re_emits_redundant_members():
    h = _selector(vote=1, fi_guard=False, expand_clusters=True)
    sel = set(h._combine(MEMBER_SEL, COLS))
    assert {"inf_0", "red_0", "red_1"} <= sel  # all members of a kept cluster re-emitted


def test_fi_guard_default_is_off():
    import inspect
    from mlframe.feature_selection._benchmarks.fs_hybrid.hybrid_selector import HybridSelector
    p = inspect.signature(HybridSelector.__init__).parameters
    assert p["fi_guard"].default is False and p["vote"].default == 1


def test_anchored_combine_keeps_mrmr_substrate_and_only_adds():
    """anchor_fe (the default): the FE substrate (MRMR's picks) is kept VERBATIM and the other members can only ADD
    clusters MRMR missed -- never drop an MRMR pick. So the selection is a superset of mrmr_selected."""
    h = _selector(vote=1, anchor_fe=True)
    sel = set(h._combine(MEMBER_SEL, COLS))
    assert {"inf_0", "inf_1"} <= sel                  # MRMR's picks kept verbatim
    assert {"inf_2", "noise_0"} <= sel                # others ADD their vote>=1 clusters MRMR missed
    assert set(MEMBER_SEL["mrmr"]) <= sel             # superset of the MRMR substrate (the anchor guarantee)
    # vote=2: each add-cluster has only ONE other voter -> no additions -> exactly the MRMR substrate
    h2 = _selector(vote=2, anchor_fe=True)
    assert set(h2._combine(MEMBER_SEL, COLS)) == {"inf_0", "inf_1"}


def test_anchor_fe_default_is_on():
    import inspect
    from mlframe.feature_selection._benchmarks.fs_hybrid.hybrid_selector import HybridSelector
    assert inspect.signature(HybridSelector.__init__).parameters["anchor_fe"].default is True


def _linear_dataset(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-(z @ np.array([1.6, -1.3, 1.1, 0.9]))))).astype(int))
    cols = {f"inf_{i}": z[:, i] for i in range(4)}
    for j in range(3):
        cols[f"red_{j}"] = z[:, 0] + 0.02 * rng.standard_normal(n)
    for k in range(8):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), y, [f"inf_{i}" for i in range(4)]


@pytest.mark.timeout(900)
def test_end_to_end_no_fe_shared_artifacts_and_recovery():
    """use_fe=False: the three shared artifacts are computed once over the raw frame and the members run; the
    hybrid recovers the informative block and de-duplicates the redundant cluster. (Deterministic raw-only path.)"""
    from mlframe.feature_selection._benchmarks.fs_hybrid.hybrid_selector import HybridSelector
    X, y, base = _linear_dataset()
    h = HybridSelector(vote=1, use_fe=False, random_state=0).fit(X, y)
    assert set(h.fi_) == set(X.columns)                       # raw-only FI when FE off
    assert any(len(ms) > 1 for ms in h.members_.values())     # redundant copies clustered
    assert set(h.member_selections_) == {"mrmr", "shap", "boruta"}
    assert h.n_engineered_ == 0 and list(h.transform(X).columns) == list(h.raw_selected_)
    assert len(set(h.raw_selected_) & set(base)) >= 3


@pytest.mark.timeout(900)
def test_fe_default_augments_and_transform_replays():
    """use_fe=True (the default): the MRMR member may engineer columns shared via X_aug; the shared FI then covers
    raw+engineered, n_engineered_ counts engineered survivors, and transform REPLAYS engineering on fresh data so
    its columns match the selected set (the leakage-free recipe replay). FE is data-dependent, so we assert the
    plumbing invariants rather than a fixed engineered count."""
    from mlframe.feature_selection._benchmarks.fs_hybrid.hybrid_selector import HybridSelector
    # interaction-bearing target so MRMR-FE has a product worth engineering
    rng = np.random.default_rng(0)
    n = 1500
    z = rng.standard_normal((n, 4))
    logit = 1.4 * z[:, 0] * z[:, 1] + 0.9 * z[:, 2] - 0.8 * z[:, 3]
    y = pd.Series((rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int))
    cols = {f"inf_{i}": z[:, i] for i in range(4)}
    for k in range(6):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)

    h = HybridSelector(vote=1, use_fe=True, random_state=0).fit(X, y)
    assert inspect_default("use_fe") is True                  # FE is the shipped default
    assert set(X.columns) <= set(h.fi_)                       # shared FI spans raw (+ any engineered)
    eng_cols = set(h._eng_rename.values())
    assert h.n_engineered_ == len([c for c in h.raw_selected_ if c in eng_cols])
    Z = h.transform(X)                                        # transform must replay engineering, not KeyError
    assert list(Z.columns) == [c for c in h.raw_selected_ if c in Z.columns]
    assert Z.shape[0] == X.shape[0]


def inspect_default(param):
    import inspect
    from mlframe.feature_selection._benchmarks.fs_hybrid.hybrid_selector import HybridSelector
    return inspect.signature(HybridSelector.__init__).parameters[param].default
