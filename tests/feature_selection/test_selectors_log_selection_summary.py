"""Every selector logs ONE INFO line at the end of fit saying how many / which features it kept (like RFECV's fit summary).

Behavioural: each selector is really fitted on a tiny synthetic and the captured log records are inspected -- K and N in the line must match
the selector's own fitted state, the line must be emitted once (wrapped / nested selectors stay quiet), and names are truncated.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from mlframe.feature_selection._selection_log import format_name_list, log_selection, logs_selection, quiet_nested
from tests.feature_selection._selector_factories import SELECTOR_SPECS, selected_names

N_FEATURES = 10
_SUMMARY_RE = re.compile(r"^(?P<label>[A-Za-z_()]+): (?P<what>selected|kept|dropped|flagged unstable) (?P<k>[\d_]+) of (?P<n>[\d_]+) (?:original )?features")


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=300, n_features=N_FEATURES, n_informative=3, n_redundant=0, random_state=0, shuffle=False)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)]), y


def _rf():
    return RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0, n_jobs=1)


def _summaries(caplog):
    """(label, k, n) for every INFO record shaped like a selection summary."""
    out = []
    for rec in caplog.records:
        if rec.levelno != logging.INFO:
            continue
        m = _SUMMARY_RE.match(rec.getMessage())
        if m and "after" not in rec.getMessage().split(":")[1][:80]:  # RFECV's own (richer) summary has its own tests
            out.append((m["label"], int(m["k"].replace("_", "")), int(m["n"].replace("_", "")), rec.getMessage()))
    return out


def _one(caplog, label):
    found = [s for s in _summaries(caplog) if s[0] == label]
    assert len(found) == 1, f"expected exactly one {label!r} summary, got {[s[3] for s in _summaries(caplog)]}"
    return found[0]


# label emitted by each contract-registry selector; MRMR/Hybrid report raw-selected count, the wrapper reports original-feature count.
_LABELS = {
    "MRMR": "MRMR",
    "ShapProxiedFS": "ShapProxiedFS",
    "BorutaShap": "BorutaShap",
    "HybridSelector": "HybridSelector",
    "ACE": "ACESelector",
    "ForwardSelect": "ForwardSelectSelector",
    "GreedyBackwardElimination": "GreedyBackwardEliminationSelector",
    "ZeroImportancePruning": "ZeroImportancePruningSelector",
    "CascadeSelect": "CascadeSelectSelector",
    "GroupAware(RFECV)": "CorrelatedFeaturesSelector",
}


def _registry_params():
    out = []
    for key in _LABELS:
        spec = SELECTOR_SPECS[key]
        marks = []
        if spec.slow:
            marks.append(pytest.mark.slow)
        out.append(pytest.param(key, id=key, marks=marks))
    return out


@pytest.mark.parametrize("key", _registry_params())
def test_registered_selector_logs_single_info_summary_with_correct_counts(key, data, caplog):
    X, y = data
    if SELECTOR_SPECS[key].needs_shap:
        pytest.importorskip("shap")
    sel = SELECTOR_SPECS[key].make("binary")
    caplog.set_level(logging.INFO)
    sel.fit(X, y)
    label, k, n, msg = _one(caplog, _LABELS[key])
    assert n == N_FEATURES
    assert k == len(selected_names(sel)), msg
    for name in selected_names(sel)[:5]:
        assert name in msg
    # a wrapped / composed selector must not leak its members' own summary lines next to its own
    assert len(_summaries(caplog)) == 1, [s[3] for s in _summaries(caplog)]


def test_rfecv_summary_still_emitted(data, caplog):
    X, y = data
    sel = SELECTOR_SPECS["RFECV"].make("binary")
    caplog.set_level(logging.INFO)
    sel.fit(X, y)
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and r.getMessage().startswith("RFECV: selected")]
    assert len(msgs) == 1


@pytest.mark.slow
def test_stability_mrmr_logs_one_summary_not_one_per_bootstrap(data, caplog):
    from mlframe.feature_selection.filters.stability import StabilityMRMR
    from tests.feature_selection._selector_factories import _make_mrmr

    X, y = data
    caplog.set_level(logging.INFO)
    sel = StabilityMRMR(_make_mrmr(), n_bootstraps=2, support_threshold=0.5, random_state=0).fit(X, y)
    label, k, n, _ = _one(caplog, "StabilityMRMR")
    assert (k, n) == (len(sel.support_), N_FEATURES)
    assert len(_summaries(caplog)) == 1


@pytest.mark.slow
def test_stability_fe_selector_logs_one_summary(data, caplog):
    from mlframe.feature_selection.filters._stability_fe import StabilityFESelector

    X, y = data
    params = dict(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False, full_npermutations=3, random_seed=0, min_features_fallback=1, verbose=False)
    caplog.set_level(logging.INFO)
    sel = StabilityFESelector(params, n_bootstraps=2, support_threshold=0.5, random_state=0).fit(X, y)
    label, k, n, msg = _one(caplog, "StabilityFESelector")
    assert n == N_FEATURES
    assert k == int(np.asarray(sel.full_mrmr_.support_).size)
    assert len(_summaries(caplog)) == 1


def test_oracle_scorer_selector_logs_recommendation(data, caplog, tmp_path):
    from mlframe.feature_selection.filters._oracle_scorer_select import OracleScorerSelector

    X, y = data
    sel = OracleScorerSelector(store_path=str(tmp_path / "oracle.parquet"))
    caplog.set_level(logging.INFO)
    scorer = sel.recommend_scorer(X, y)
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("OracleScorerSelector: recommended scorer")]
    assert len(msgs) == 1 and repr(scorer) in msgs[0]


# ------------------------------------------------------------------------------------------------------ functional selectors / filters
def _fwd(X, y):
    from mlframe.feature_selection import forward_select

    r = forward_select(X, y, lambda: LogisticRegression(max_iter=200), scoring="accuracy", cv=2, max_features=3)
    return len(r)


def _bwd(X, y):
    from mlframe.feature_selection import greedy_backward_elimination

    return len(greedy_backward_elimination(_rf(), X, y, accuracy_score, cv=KFold(2), min_features=8))


def _zero(X, y):
    from mlframe.feature_selection import iterative_zero_importance_pruning

    return len(iterative_zero_importance_pruning(_rf(), X, y, accuracy_score, cv=KFold(2), max_rounds=2))


def _casc(X, y):
    from mlframe.feature_selection import cascade_select

    return len(cascade_select(X, y, _rf, n_boruta_iterations=5, cv=2)["final_selected"])


def _casc_stable(X, y):
    from mlframe.feature_selection import cascade_select_stable

    return len(cascade_select_stable(X, y, _rf, n_bootstrap=2, stability_threshold=0.5, n_boruta_iterations=5, cv=2)["stable_selected"])


def _ace(X, y):
    from mlframe.feature_selection import ace_select

    return len(ace_select(X, y, _rf(), n_replicates=5).selected_features)


def _near_noise(X, y):
    from mlframe.feature_selection import drop_near_noise_univariate_auc

    return len(drop_near_noise_univariate_auc(X, y, tolerance=0.1))


def _vs_reference(X, y):
    from mlframe.feature_selection import drop_noninformative_vs_reference

    mask = np.asarray(y) == 0
    return len(drop_noninformative_vs_reference(X, mask, alpha=0.5))


def _raw_after_embedding(X, y):
    from mlframe.feature_selection import drop_raw_after_embedding

    return len(X.columns) - len(drop_raw_after_embedding(X, {"f0": ["f1"], "f5": ["f6"]}).columns)


def _unanimous(X, y):
    from mlframe.feature_selection import unanimous_permutation_prune

    splits = list(KFold(2, shuffle=True, random_state=0).split(X))
    return len(unanimous_permutation_prune(X, y, _rf, splits, n_repeats=2, max_iterations=2))


def _ridge_prefilter(X, y):
    from mlframe.feature_selection import ridge_coefficient_prefilter

    return len(ridge_coefficient_prefilter(X.to_numpy(), y, list(X.columns), cv=2, is_classifier=True))


def _bandit(X, y):
    from mlframe.feature_selection import stochastic_bandit_selection

    return len(stochastic_bandit_selection(_rf(), X, y, accuracy_score, subset_size=3, n_epochs=6, cv=KFold(2)))


def _bandit_ensemble(X, y):
    from mlframe.feature_selection import stochastic_bandit_selection_ensemble

    return len(stochastic_bandit_selection_ensemble(_rf(), X, y, accuracy_score, subset_size=3, seeds=[0, 1], n_epochs=6, cv=KFold(2)).union_top_feats)


def _hetero(X, y):
    from mlframe.feature_selection import heterogeneous_relevance_vote

    accepted, _ = heterogeneous_relevance_vote(X, y, models={"rf": _rf()}, n_shadow_trials=1)
    return len(accepted)


def _pre_screen(X, y):
    from mlframe.feature_selection.pre_screen import compute_unsupervised_drops

    df = X.copy()
    df["const"] = 1.0
    df["allnull"] = np.nan
    return len(compute_unsupervised_drops(df)), df.shape[1]


def _boruta_fn(X, y):
    from mlframe.feature_selection.filters import boruta_select

    def imp(Xa, ya):
        m = _rf().fit(np.asarray(Xa), ya)
        return m.feature_importances_

    res = boruta_select(X, y, imp, n_iterations=6)
    return sum(1 for d in res["decision"] if d == "confirmed")


def _null_importance(X, y):
    from mlframe.feature_selection.filters import null_importance_filter

    def imp(Xa, ya):
        return _rf().fit(np.asarray(Xa), ya).feature_importances_

    return int(null_importance_filter(X, y, imp, n_shuffles=5)["keep_mask"].sum())


def _monotonic(X, y):
    from mlframe.feature_selection.filters import monotonic_deviation_stability_filter

    df = X.copy()
    df["grp"] = np.arange(len(df)) % 20
    rep = monotonic_deviation_stability_filter(df, y, "grp", n_subsamples=5)
    return int(rep["stable"].sum()), len(rep)


def _ks(X, y):
    from mlframe.feature_selection.filters import ks_stability_filter

    rep = ks_stability_filter(X.iloc[:150], X.iloc[150:], p_value_threshold=0.5)
    return int((~rep["stable"]).sum())


# (label, runner returning K or (K, N), expected N)
_FUNCTION_CASES = [
    ("forward_select", _fwd, N_FEATURES),
    ("greedy_backward_elimination", _bwd, N_FEATURES),
    ("iterative_zero_importance_pruning", _zero, N_FEATURES),
    ("cascade_select", _casc, N_FEATURES),
    ("cascade_select_stable", _casc_stable, N_FEATURES),
    ("ace_select", _ace, N_FEATURES),
    ("drop_near_noise_univariate_auc", _near_noise, N_FEATURES),
    ("drop_noninformative_vs_reference", _vs_reference, N_FEATURES),
    ("drop_raw_after_embedding", _raw_after_embedding, N_FEATURES),
    ("unanimous_permutation_prune", _unanimous, N_FEATURES),
    ("ridge_coefficient_prefilter", _ridge_prefilter, N_FEATURES),
    ("stochastic_bandit_selection", _bandit, N_FEATURES),
    ("stochastic_bandit_selection_ensemble", _bandit_ensemble, N_FEATURES),
    ("heterogeneous_relevance_vote", _hetero, N_FEATURES),
    ("pre_screen", _pre_screen, N_FEATURES + 2),
    ("boruta_select", _boruta_fn, N_FEATURES),
    ("null_importance_filter", _null_importance, N_FEATURES),
    ("monotonic_deviation_stability_filter", _monotonic, N_FEATURES),
    ("ks_stability_filter", _ks, N_FEATURES),
]


@pytest.mark.parametrize("label,runner,n_expected", _FUNCTION_CASES, ids=[c[0] for c in _FUNCTION_CASES])
def test_functional_selector_logs_single_info_summary(label, runner, n_expected, data, caplog):
    X, y = data
    caplog.set_level(logging.INFO)
    out = runner(X, y)
    k_expected = out[0] if isinstance(out, tuple) else out
    _, k, n, msg = _one(caplog, label)
    assert (k, n) == (k_expected, n_expected), msg
    assert len(_summaries(caplog)) == 1, [s[3] for s in _summaries(caplog)]  # nested selector functions stay quiet


def test_varying_size_top_k_subsets_logs_summary(caplog):
    from mlframe.feature_selection import varying_size_top_k_subsets

    ranked = [f"f{i}" for i in range(10)]
    caplog.set_level(logging.INFO)
    subsets = varying_size_top_k_subsets(ranked, [2, 4, 6])
    _, k, n, msg = _one(caplog, "varying_size_top_k_subsets")
    assert (k, n) == (6, 10) and len(subsets) == 3 and "3 subset(s)" in msg


# ------------------------------------------------------------------------------------------------------ shared helper behaviour
def test_log_selection_truncates_names_to_first_thirty(caplog):
    names = [f"col{i}" for i in range(500)]
    caplog.set_level(logging.INFO)
    log_selection(logging.getLogger("t"), "Sel", 500, 800, names, elapsed=1.25)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert msg.startswith("Sel: selected 500 of 800 features in 1.2 s")
    assert "col29" in msg and "col30" not in msg and "(+470 more)" in msg
    assert msg == "Sel: selected 500 of 800 features in 1.2 s: [" + format_name_list(names) + "]"


def test_logs_selection_decorator_is_silent_when_nested(caplog):
    @logs_selection("inner", lambda res, b: (res, 5, None))
    def inner():
        return ["a", "b"]

    caplog.set_level(logging.INFO)
    with quiet_nested():
        assert inner() == ["a", "b"]
    assert not caplog.records
    assert inner() == ["a", "b"]
    assert [r.getMessage().split(" features")[0] for r in caplog.records] == ["inner: selected 2 of 5"]


def test_log_selection_respect_quiet_only_inside_quiet_scope(caplog):
    caplog.set_level(logging.INFO)
    lg = logging.getLogger("t2")
    with quiet_nested():
        log_selection(lg, "X", 1, 2, ["a"], respect_quiet=True)
        log_selection(lg, "Y", 1, 2, ["a"])
    log_selection(lg, "Z", 1, 2, ["a"], respect_quiet=True)
    assert [r.getMessage().split(":")[0] for r in caplog.records] == ["Y", "Z"]


# ------------------------------------------------------------------------------------------------------ suite-level line
def test_suite_selector_retention_line_lists_dropped_columns(caplog):
    from mlframe.training.pipeline._pipeline_selector_log import log_selector_retention

    class DummySelector:
        pass

    cols = [f"c{i}" for i in range(40)]
    out = pd.DataFrame(np.zeros((3, 5)), columns=cols[:5])
    caplog.set_level(logging.INFO)
    log_selector_retention(DummySelector(), None, cols, out)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert msg.startswith("feature selector DummySelector: 40 -> 5 columns (dropped 35: [c5, c6")
    assert "(+5 more)" in msg  # 35 dropped names, first 30 shown


def test_suite_selector_retention_line_polars_and_engineered(caplog):
    pl = pytest.importorskip("polars")
    from mlframe.training.pipeline._pipeline_selector_log import log_selector_retention

    out = pl.DataFrame({"a": [1, 2], "eng(a)": [3, 4]})
    caplog.set_level(logging.INFO)
    log_selector_retention(object(), None, ["a", "b", "c"], out)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert msg == "feature selector object: 3 -> 2 columns (dropped 2: [b, c]) (+1 new columns)"


@pytest.mark.slow
def test_mrmr_tree_rescued_final_summary_matches_rescued_support(data, caplog):
    from mlframe.feature_selection.filters import MRMRTreeRescued

    X, y = data
    caplog.set_level(logging.INFO)
    sel = MRMRTreeRescued(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False, full_npermutations=3, random_seed=0, min_features_fallback=1, verbose=False)
    sel.fit(X, y)
    lines = [s for s in _summaries(caplog) if s[0] == "MRMR"]
    assert lines, "MRMRTreeRescued must emit the MRMR summary"
    assert lines[-1][1] == int(np.asarray(sel.support_).size), lines[-1][3]
