"""Real small RFECV fits around the futility stop: it cuts the search short where the full set is the pick and never changes the selection."""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from mlframe.feature_selection.wrappers import RFECV


def _dense(n=3000, p=14, informative=14, seed=0):
    X, y = make_regression(n, p, n_informative=informative, noise=20.0, random_state=seed, shuffle=False)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


def _fit(X, y, **kw):
    return RFECV(estimator=Ridge(1.0), cv=5, random_state=0, verbose=0, **kw).fit(X, y)


@pytest.fixture(scope="module")
def dense_fits():
    """One un-stopped and one stopped fit on a dense-signal frame, shared by the tests below."""
    X, y = _dense()
    return _fit(X, y, futility_stop=False), _fit(X, y, futility_stop=True)


def test_stops_before_exhausting_the_search_and_keeps_the_same_set(dense_fits):
    full, stopped = dense_fits
    assert len(full.eval_trace_) == 14
    assert len(stopped.eval_trace_) < len(full.eval_trace_)
    assert stopped.futility_verdict_.stop
    assert list(stopped.get_feature_names_out()) == list(full.get_feature_names_out())
    assert stopped.n_features_ == full.n_features_ == 14


def test_on_by_default_with_an_explicit_opt_out(dense_fits):
    full, _ = dense_fits
    assert full.futility_verdict_ is None and len(full.eval_trace_) == 14
    assert RFECV(estimator=Ridge()).futility_stop is True


def test_fit_summary_reports_the_futility_stop_reason_with_evidence(dense_fits, caplog):
    _, stopped = dense_fits
    from mlframe.feature_selection.wrappers.rfecv._fit_summary import build_rfecv_fit_summary

    msg = build_rfecv_fit_summary(stopped, stop_reason=stopped.futility_verdict_.describe(), n_iters=len(stopped.eval_trace_), elapsed_s=1.0)
    assert "stopped: futility (" in msg and "paired gain" in msg and "upper bound" in msg


def test_pure_noise_extra_features_keep_the_selection_identical_whether_or_not_the_search_is_cut():
    X, y = _dense(p=14, informative=3)
    full, stopped = _fit(X, y, futility_stop=False), _fit(X, y, futility_stop=True)
    assert len(stopped.eval_trace_) <= len(full.eval_trace_)
    assert list(stopped.get_feature_names_out()) == list(full.get_feature_names_out())


@pytest.mark.parametrize("kw", [{"n_features_selection_rule": "argmax"}, {"n_features_selection_rule": "one_se_min"}, {"feature_cost": 1e-4}, {"max_nfeatures": 10}])
def test_not_armed_for_rules_where_shrinking_can_still_change_the_pick(kw):
    X, y = _dense()
    sel = _fit(X, y, futility_stop=True, **kw)
    assert sel.futility_verdict_ is None


def test_grouped_config_carries_the_knobs():
    from mlframe.feature_selection.wrappers.rfecv._configs import SearchConfig

    sel = RFECV(estimator=Ridge(), search_config=SearchConfig(futility_stop=True, futility_min_iters=7, futility_alpha=0.1))
    assert sel.futility_stop is True and sel.futility_min_iters == 7 and sel.futility_alpha == pytest.approx(0.1)


def test_bad_anchor_is_rejected_at_construction():
    with pytest.raises(ValueError, match="futility_anchor"):
        RFECV(estimator=Ridge(), futility_anchor="nope")
