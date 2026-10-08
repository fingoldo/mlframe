"""Every selector that cross-validates / holds out rows internally uses the suite's shared split policy (time-ordered folds on a time-shuffled frame)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from mlframe.feature_selection.cv_policy import CVPolicy, TimestampOrderedSplit, get_cv_policy
from mlframe.feature_selection.functional_adapters import (
    ForwardSelectSelector, GreedyBackwardEliminationSelector, ZeroImportancePruningSelector,
)
from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines

N = 240
TS = np.random.default_rng(0).permutation(N)


def _frame():
    """Build N rows whose every column carries the row timestamp plus tiny noise, so a call reveals which rows it received, with a binary target."""
    rng = np.random.default_rng(1)
    # every column carries the row's timestamp (plus tiny noise) so a fit / predict call reveals which rows it was handed
    X = pd.DataFrame(TS[:, None] + 0.001 * rng.normal(size=(N, 4)), columns=list("abcd"))
    y = (rng.normal(size=N) > 0).astype(int)
    return X, y


class _TimeSpy(DecisionTreeClassifier):
    """Records whether any scored (test) row is older than the newest row the model was fit on."""

    violations: list = []
    fits: list = []

    def fit(self, X, y, **kw):
        """Record the newest training timestamp and fit normally."""
        arr = np.asarray(X)
        type(self).fits.append(arr[:, 0].max())
        self._train_max = arr[:, 0].max()
        return super().fit(X, y, **kw)

    def _check(self, X):
        """Record a violation when prediction rows are older than the newest training row."""
        train_max = getattr(self, "_train_max", None)
        if train_max is not None and np.asarray(X)[:, 0].min() < train_max:
            type(self).violations.append((train_max, np.asarray(X)[:, 0].min()))

    def predict(self, X):
        """Check row ordering, then predict."""
        self._check(X)
        return super().predict(X)

    def predict_proba(self, X):
        """Check row ordering, then predict probabilities."""
        self._check(X)
        return super().predict_proba(X)


def _reset():
    """Clear the spy's recorded violations and fits."""
    _TimeSpy.violations = []
    _TimeSpy.fits = []


def _temporal_policy():
    """Temporal CV policy over the module's timestamps."""
    return CVPolicy("temporal", "test", TS)


def _build_raw(**flags):
    """Build the pre-pipelines with only the requested selectors enabled, returning pipelines and names."""
    flags.setdefault("use_mrmr_fs", False)
    return _build_pre_pipelines(use_ordinary_models=False, rfecv_models=[], rfecv_models_params={}, mrmr_kwargs=None, **flags)


def _build(**flags):
    """Build the pre-pipelines under the temporal policy and return them keyed by stripped name."""
    pipes, names = _build_raw(cv_policy=_temporal_policy(), target_type="binary_classification", **flags)
    return dict(zip([n.strip() for n in names], pipes))


def test_build_pre_pipelines_swaps_cv_param_selectors_to_timestamp_ordered_split():
    """Build pre pipelines swaps cv param selectors to timestamp ordered split."""
    built = _build(
        use_forward_select_fs=True, use_greedy_backward_elimination_fs=True, use_zero_importance_pruning_fs=True, use_cascade_select_fs=True,
    )
    for name in ("ForwardSelect", "GreedyBackwardElimination", "ZeroImportancePruning", "CascadeSelect"):
        assert isinstance(built[name].cv, TimestampOrderedSplit), name
        assert get_cv_policy(built[name]).kind == "temporal"
    assert built["ForwardSelect"].cv.n_splits == 5 and built["GreedyBackwardElimination"].cv.n_splits == 5


def test_build_pre_pipelines_stamps_holdout_selectors_and_mrmr():
    """Build pre pipelines stamps holdout selectors and mrmr."""
    built = _build(use_mrmr_fs=True, use_ace_fs=True, use_boruta_shap=True, use_shap_proxied_fs=True)
    for name in ("MRMR", "ACE", "BorutaShap", "ShapProxiedFS"):
        assert get_cv_policy(built[name]).kind == "temporal", name


def test_build_pre_pipelines_explicit_user_cv_wins():
    """Build pre pipelines explicit user cv wins."""
    own = GroupKFold(n_splits=3)
    built = _build(use_forward_select_fs=True, forward_select_kwargs={"cv": own},
                   use_zero_importance_pruning_fs=True, zero_importance_pruning_kwargs={"cv": own})
    assert built["ForwardSelect"].cv is own and built["ZeroImportancePruning"].cv is own
    built = _build(use_forward_select_fs=True, forward_select_kwargs={"cv": 3})
    assert isinstance(built["ForwardSelect"].cv, TimestampOrderedSplit) and built["ForwardSelect"].cv.n_splits == 3


def test_build_pre_pipelines_without_policy_keeps_selector_defaults():
    """Build pre pipelines without policy keeps selector defaults."""
    pipes, _ = _build_raw(use_forward_select_fs=True, use_zero_importance_pruning_fs=True)
    assert pipes[0].cv == 5 and pipes[1].cv is None and all(get_cv_policy(p) is None for p in pipes)


@pytest.mark.parametrize(
    "factory",
    [
        lambda pol: ForwardSelectSelector(_TimeSpy(random_state=0), cv=TimestampOrderedSplit(n_splits=3, timestamps=pol.timestamps), max_features=2),
        lambda pol: GreedyBackwardEliminationSelector(_TimeSpy(random_state=0), cv=TimestampOrderedSplit(n_splits=3, timestamps=pol.timestamps)),
        lambda pol: ZeroImportancePruningSelector(_TimeSpy(random_state=0), cv=TimestampOrderedSplit(n_splits=3, timestamps=pol.timestamps)),
    ],
)
def test_functional_selectors_score_only_on_later_rows_than_they_fit(factory):
    """Functional selectors score only on later rows than they fit."""
    _reset()
    X, y = _frame()
    factory(_temporal_policy()).fit(X, y)
    assert _TimeSpy.fits, "the spy estimator was never fit"
    assert not _TimeSpy.violations


@pytest.mark.parametrize(
    "kind,kwargs",
    [
        ("ForwardSelect", dict(use_forward_select_fs=True, forward_select_kwargs={"max_features": 2})),
        ("GreedyBackwardElimination", dict(use_greedy_backward_elimination_fs=True)),
        ("ZeroImportancePruning", dict(use_zero_importance_pruning_fs=True)),
    ],
)
def test_suite_built_selector_scores_only_on_later_rows_than_it_fits(kind, kwargs):
    """End to end through the suite builder: fail-before is the shuffled KFold default, which scores on rows older than the fit rows."""
    _reset()
    X, y = _frame()
    sel = _build(**kwargs)[kind]
    sel.set_params(estimator=_TimeSpy(random_state=0))
    sel.fit(X, y)
    assert _TimeSpy.fits and not _TimeSpy.violations
    _reset()
    unwired = {"ForwardSelect": ForwardSelectSelector, "GreedyBackwardElimination": GreedyBackwardEliminationSelector,
               "ZeroImportancePruning": ZeroImportancePruningSelector}[kind](_TimeSpy(random_state=0))
    unwired.fit(X, y)
    assert _TimeSpy.violations, "control: the default shuffled CV should have scored on older rows"


def test_greedy_backward_elimination_ignores_n_repeats_under_time_ordered_cv(caplog):
    """Greedy backward elimination ignores n repeats under time ordered cv."""
    from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination

    _reset()
    X, y = _frame()
    with caplog.at_level("INFO"):
        greedy_backward_elimination(
            _TimeSpy(random_state=0), X, y, lambda yt, yp: float(np.mean(yt == yp)),
            cv=TimestampOrderedSplit(n_splits=3, timestamps=TS), n_repeats=3,
        )
    assert not _TimeSpy.violations
    assert any("ignoring n_repeats" in r.getMessage() for r in caplog.records)


def test_cascade_select_hands_time_ordered_cv_to_cascade_select(monkeypatch):
    """Cascade select hands time ordered cv to cascade select."""
    import mlframe.feature_selection.functional_adapters as adapters

    seen = {}

    def spy_cascade(X, y, factory, **kw):
        """Capture the cv argument passed to cascade_select and return an empty selection."""
        seen["cv"] = kw["cv"]
        return {"final_selected": []}

    monkeypatch.setattr(adapters, "cascade_select", spy_cascade)
    X, y = _frame()
    sel = _build(use_cascade_select_fs=True)["CascadeSelect"]
    sel.fit(X, y)
    assert isinstance(seen["cv"], TimestampOrderedSplit)


def test_ace_permutation_holdout_is_newest_rows_under_temporal_policy():
    """Ace permutation holdout is newest rows under temporal policy."""
    from mlframe.feature_selection.ace import _pfi_split

    rng = np.random.default_rng(0)
    fit_idx, score_idx = _pfi_split(N, np.zeros(N), rng, _temporal_policy())
    assert TS[fit_idx].max() < TS[score_idx].min()
    fit_idx, score_idx = _pfi_split(N, (np.arange(N) % 2), rng, None)
    assert TS[fit_idx].max() > TS[score_idx].min(), "control: no policy keeps the random split"


def test_ace_selector_fit_uses_policy_holdout(monkeypatch):
    """Ace selector fit uses policy holdout."""
    import mlframe.feature_selection.ace as ace_mod

    calls = []
    real = ace_mod._pfi_split
    monkeypatch.setattr(
        ace_mod, "_pfi_split", lambda n, y, rng, cv_policy=None, split_seed=0: (calls.append(cv_policy), real(n, y, rng, cv_policy, split_seed))[1]
    )
    X, y = _frame()
    sel = _build(use_ace_fs=True, ace_kwargs={"importance": "permutation", "n_replicates": 3, "n_masking_rounds": 1})["ACE"]
    sel.fit(X, y)
    assert calls and all(c is not None and c.kind == "temporal" for c in calls)


def test_boruta_shap_holdout_is_newest_rows_under_temporal_policy():
    """Boruta shap holdout is newest rows under temporal policy."""
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _frame()
    bs = BorutaShap(model=RandomForestClassifier(n_estimators=5, random_state=0), importance_measure="gini", classification=True, train_or_test="test", random_state=0)
    bs.X_boruta_, bs.y_ = X, pd.Series(y)
    bs.Train_model = lambda Xt, yt: None
    bs._mlframe_cv_policy_ = _temporal_policy()
    bs.Check_if_chose_train_or_test_and_train_model()
    assert bs.X_boruta_train_["a"].max() < bs.X_boruta_test_["a"].min()
    assert len(bs.X_boruta_test_) == round(0.3 * N) and len(bs.y_test_) == len(bs.X_boruta_test_)


def test_shap_proxied_fs_holdout_and_oof_partition_follow_temporal_policy(monkeypatch):
    """Shap proxied fs holdout and oof partition follow temporal policy."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain as explain_mod

    captured = {}

    class _Stop(Exception):
        """Sentinel raised to abort once the policy has been captured."""
        pass

    def spy(*a, **kw):
        """Capture the cv_policy passed in, then abort with _Stop."""
        captured["policy"] = kw.get("cv_policy")
        raise _Stop

    monkeypatch.setattr(explain_mod, "compute_shap_matrix", spy)
    rng = np.random.default_rng(3)
    n = 400
    ts = rng.permutation(n)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f1"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    sel = ShapProxiedFS(classification=True, random_state=0, verbose=False, holdout_size=0.25, n_splits=3)
    sel._mlframe_cv_policy_ = CVPolicy("temporal", "test", ts)
    with pytest.raises(_Stop):
        sel.fit(X, y)
    pol = captured["policy"]
    assert pol is not None and pol.kind == "temporal" and len(pol.timestamps) == 300
    assert np.asarray(pol.timestamps).max() < np.sort(ts)[-100], "search rows must all be older than the holdout rows"


def test_mrmr_heldout_gate_probe_uses_newest_third_under_temporal_policy(monkeypatch):
    """Mrmr heldout gate probe uses newest third under temporal policy."""
    import mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._heldout_gate as gate

    captured = {}

    def fake_scorer(cols, y, tr, va):
        """Capture the train and validation index arrays and return a zero-scoring closure."""
        captured["tr"], captured["va"] = tr.copy(), va.copy()
        return lambda extra=None: 0.0

    monkeypatch.setattr(gate, "heldout_r2_scorer", fake_scorer)
    y = np.random.default_rng(0).normal(size=N)
    gate.build_heldout_incr_probe(y_gate=y, sel_value_cols=[TS.astype(float)], random_seed=0, cv_policy=_temporal_policy())
    assert TS[captured["tr"]].max() < TS[captured["va"]].min()
    gate.build_heldout_incr_probe(y_gate=y, sel_value_cols=[TS.astype(float)], random_seed=0)
    assert TS[captured["tr"]].max() > TS[captured["va"]].min(), "control: no policy keeps the seeded shuffle"


def test_mrmr_stability_vote_folds_are_time_blocks_under_temporal_policy():
    """Mrmr stability vote folds are time blocks under temporal policy."""
    from mlframe.feature_selection.filters._fe_stability_vote import _vote_folds

    folds = _vote_folds(_temporal_policy(), N, 5, np.random.default_rng(0))
    assert len(folds) == 5 and sum(len(f) for f in folds) == N
    assert all(TS[earlier].max() < TS[later].min() for earlier, later in zip(folds[:-1], folds[1:]))
    shuffled = _vote_folds(None, N, 5, np.random.default_rng(0))
    assert TS[shuffled[0]].max() > TS[shuffled[1]].min(), "control: no policy keeps the seeded shuffled partition"
