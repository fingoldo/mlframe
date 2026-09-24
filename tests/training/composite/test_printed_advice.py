"""Every message in composite/ that tells the reader to do something is followed here, and the promised effect is observed.

Advice is read only once something has gone wrong, so advice that names a renamed knob or one that changes nothing sends
the reader in circles with no test noticing. ``PRINTED_ADVICE_TESTS`` maps each advising message (found by
``py_ci_shared.printed_advice``, keyed by file and enclosing function) to the test below that performs it; the meta test in
tests/test_meta/test_printed_advice_wired.py fails on an unregistered or stale key.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

_C = "src/mlframe/training/composite/"

PRINTED_ADVICE_TESTS = {
    _C + "discovery/_filter.py::_filter_features#1": "test_a_corr_dropped_base_comes_back_by_threshold_or_by_name",
    _C + "discovery/_fit.py::fit#1": "test_bin_mi_becomes_active_with_a_larger_sample_or_fewer_bins",
    _C + "discovery/_knn_budget.py::maybe_downgrade_knn_estimator#1": "test_the_knn_downgrade_is_switched_off_by_its_flag_or_budget",
    _C + "discovery/_screening_tiny.py::_tiny_cv_rmse_raw_y#1": "test_the_group_split_holds_with_fewer_folds_or_more_groups",
    _C + "discovery/_screening_tiny_perbin.py::_tiny_cv_rmse_y_scale#1": "test_the_group_split_holds_with_fewer_folds_or_more_groups",
    _C + "ensemble/__init__.py::_oof_holdout_predictions_with_rows#1": "test_kfold_one_uses_the_external_holdout",
    _C + "ensemble/_cross_target.py::CompositeCrossTargetEnsemble.from_uniform_weights#1": "test_the_ensemble_builders_accept_what_their_errors_ask_for",
    _C + "ensemble/_cross_target.py::CompositeCrossTargetEnsemble.from_train_metrics#1": "test_an_oof_baseline_is_used_and_silences_the_scale_warning",
    _C + "ensemble/_cross_target.py::CompositeCrossTargetEnsemble.from_train_metrics#2": "test_the_ensemble_builders_accept_what_their_errors_ask_for",
    _C + "ensemble/_cross_target.py::CompositeCrossTargetEnsemble.from_train_metrics#3": "test_oof_rmses_replace_the_train_rmse_ranking",
    _C + "estimator/_estimator.py::CompositeTargetEstimator.fit#1": "test_drop_invalid_rows_fits_through_domain_violations",
    _C + "grouped_block_stacking.py::GroupedBlockStacker.fit#1": "test_auto_discovered_blocks_replace_missing_feature_groups",
    _C + "multi_output.py::CompositeMultiOutputEstimator._build_column_estimator#1": "test_the_estimators_accept_what_their_errors_ask_for",
    _C + "panel.py::_resolve_entity_ids#1": "test_the_estimators_accept_what_their_errors_ask_for",
    _C + "sklearn_compat.py::make_composite_regressor#1": "test_the_estimators_accept_what_their_errors_ask_for",
    _C + "sklearn_compat.py::CompositeTargetTransformer._resolve_base#1": "test_the_target_transformer_takes_x_or_an_explicit_base",
    _C + "sklearn_compat.py::CompositeTargetTransformer._resolve_base#2": "test_the_target_transformer_takes_x_or_an_explicit_base",
    _C + "suite_features.py::CompositeFeatureGenerator._make_wrapper#1": "test_the_feature_generator_accepts_a_wrapper_factory",
    _C + "suite_features.py::CompositeFeatureGenerator.transform#1": "test_fit_final_on_all_enables_transform",
    _C + "transforms/interaction_bases.py::generate_interaction_bases#1": "test_a_train_mask_silences_the_eps_scale_warning",
    _C + "venn_abers.py::_isotonic_envelopes#1": "test_rounding_scores_brings_venn_abers_under_the_warning",
}


def _warnings(caplog) -> list[str]:
    """The WARNING-or-worse messages captured so far."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def _disc(**cfg):
    """A discovery object with a small config."""
    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    d = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(enabled=True, random_state=0, **cfg))
    d._target_col = "y"  # set by fit(); the filter reads it to skip the target column
    return d


def test_a_corr_dropped_base_comes_back_by_threshold_or_by_name():
    """ "raise forbidden_base_corr_threshold or pass it via base_candidates=[...]": both keep a near-copy of y as a base."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(10.0, 2.0, n)
    df = pd.DataFrame({"lag": y + rng.normal(0.0, 1e-4, n), "x": rng.normal(size=n), "y": y})
    idx = np.arange(n)
    d = _disc()
    assert "lag" not in d._filter_features(df, ["lag", "x"], y, idx), "fixture: the default threshold must drop the near-copy"
    # 1 - 1e-10 still drops an exact copy of y; the near-copy (|corr| 1 - 1.3e-9) used to score 1.00000018 in float32.
    raised = _disc(forbidden_base_corr_threshold=1.0 - 1e-10)
    assert "lag" in raised._filter_features(df, ["lag", "x"], y, idx)
    assert "copy" not in raised._filter_features(df.assign(copy=y), ["copy"], y, idx)
    named = _disc(base_candidates=["lag"])
    usable = named._filter_features(df, ["lag", "x"], y, idx)
    assert named._resolve_base_candidates(df, "y", usable, y, idx) == ["lag"]


def test_bin_mi_becomes_active_with_a_larger_sample_or_fewer_bins(caplog):
    """ "Raise mi_sample_n or lower mi_nbins": either removes the inactive-bin-MI warning from the fit."""
    rng = np.random.default_rng(0)
    n = 400
    b = rng.uniform(1.0, 10.0, n)
    df = pd.DataFrame({"b": b, "x": rng.normal(size=n), "y": 2.0 * b + rng.normal(0.0, 0.3, n)})

    def warned(**cfg) -> bool:
        """Fit with this config and report whether the bin-MI advice was logged."""
        caplog.clear()
        with caplog.at_level(logging.WARNING), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _disc(base_candidates=["b"], transforms=["diff"], mi_estimator="bin", **cfg).fit(df, "y", ["b", "x"], np.arange(n))
        return any("bin-MI is inactive" in m for m in _warnings(caplog))

    assert warned(mi_sample_n=50, mi_nbins=12), "fixture: 50 rows < 5 x 12 bins must warn"
    assert not warned(mi_sample_n=None, mi_nbins=12)
    assert not warned(mi_sample_n=50, mi_nbins=8)


def test_the_knn_downgrade_is_switched_off_by_its_flag_or_budget():
    """ "set knn_mi_auto_downgrade=False or raise the budget to keep knn": both keep the knn estimator."""
    from mlframe.training.composite.discovery._knn_budget import maybe_downgrade_knn_estimator

    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    y = rng.normal(size=n)
    feats = list(df.columns)

    def estimator(**cfg) -> str:
        """Return the MI estimator left after the kNN downgrade check."""
        d = _disc(mi_estimator="knn", **cfg)
        maybe_downgrade_knn_estimator(d, df, feats, np.arange(n), y)
        return d.config.mi_estimator

    assert estimator(knn_mi_budget_seconds=1e-9) == "bin", "fixture: a near-zero budget must downgrade"
    assert estimator(knn_mi_budget_seconds=1e-9, knn_mi_auto_downgrade=False) == "knn"
    assert estimator(knn_mi_budget_seconds=1e9) == "knn"


def test_the_group_split_holds_with_fewer_folds_or_more_groups(caplog):
    """ "reduce cv_folds or supply more groups": either keeps the grouped split in both tiny-CV paths (no downgrade warning)."""
    from mlframe.training.composite.discovery._screening_tiny import _tiny_cv_rmse_raw_y
    from mlframe.training.composite.discovery._screening_tiny_perbin import _tiny_cv_rmse_y_scale
    from mlframe.training.composite.transforms import get_transform

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 2))
    base = rng.uniform(1.0, 10.0, n)
    y = 2.0 * base + x[:, 0] + rng.normal(0.0, 0.3, n)
    t = get_transform("diff")
    kw = dict(family="lightgbm", n_estimators=5, num_leaves=7, learning_rate=0.1, random_state=0)

    def downgraded(folds: int, n_groups: int) -> bool:
        """Run the tiny CV and report whether the group split was downgraded."""
        caplog.clear()
        g = np.arange(n) % n_groups
        with caplog.at_level(logging.WARNING):
            _tiny_cv_rmse_raw_y(y, x, cv_folds=folds, groups=g, **kw)
            _tiny_cv_rmse_y_scale(y, base, t, t.fit(y, base), x, cv_folds=folds, groups=g, **kw)
        return sum("falling back" in m for m in _warnings(caplog)) == 2

    assert downgraded(3, 2), "fixture: 2 groups under 3 folds must downgrade in both paths"
    assert not any("falling back" in m for m in (downgraded(2, 2) or _warnings(caplog)))
    assert not any("falling back" in m for m in (downgraded(3, 6) or _warnings(caplog)))


def test_kfold_one_uses_the_external_holdout(caplog):
    """ "Pass kfold=1 to use the external holdout": the holdout then is the external frame's rows."""
    from mlframe.training.composite.ensemble import compute_oof_holdout_predictions

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=300)})
    y = 2.0 * X["a"].to_numpy() + rng.normal(0.0, 0.1, 300)
    Xv = pd.DataFrame({"a": rng.normal(size=80)})
    yv = 2.0 * Xv["a"].to_numpy()
    model = LinearRegression().fit(X, y)
    kw = dict(
        component_models=[model],
        component_names=["m"],
        component_specs=[None],
        train_X=X,
        y_train_full=y,
        base_train_full_per_spec={},
        holdout_frac=0.2,
        random_state=0,
        external_holdout_X=Xv,
        external_holdout_y=yv,
    )
    with caplog.at_level(logging.WARNING):
        _p, y_hold, _n = compute_oof_holdout_predictions(kfold=3, **kw)
    assert any("IGNORED" in m for m in _warnings(caplog)) and y_hold.size != yv.size
    _p, y_hold, _n = compute_oof_holdout_predictions(kfold=1, **kw)
    np.testing.assert_array_equal(y_hold, yv)


def test_the_ensemble_builders_accept_what_their_errors_ask_for():
    """ "supply at least one component" / "supply either component_oof_rmse or component_train_rmse": doing so builds."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    m = LinearRegression()
    with pytest.raises(ValueError, match="supply at least"):
        CompositeCrossTargetEnsemble.from_uniform_weights(component_models=[], component_names=[])
    assert CompositeCrossTargetEnsemble.from_uniform_weights(component_models=[m], component_names=["a"]).component_names == ["a"]
    with pytest.raises(ValueError, match="supply either"):
        CompositeCrossTargetEnsemble.from_train_metrics(component_models=[m, m], component_names=["a", "b"])
    for kw in ({"component_oof_rmse": [1.0, 2.0]}, {"component_train_rmse": [1.0, 2.0]}):
        CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=[m, m], component_names=["a", "b"], baseline_oof_rmse=3.0, baseline_train_rmse=3.0, **kw
        )


def test_an_oof_baseline_is_used_and_silences_the_scale_warning(caplog):
    """ "Pass baseline_oof_rmse=... for a real gain-over-naive OOF weighting": the weights then follow the gain over it."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    m = [LinearRegression(), LinearRegression()]
    with caplog.at_level(logging.WARNING):
        CompositeCrossTargetEnsemble.from_train_metrics(component_models=m, component_names=["a", "b"], component_oof_rmse=[1.0, 1.5], baseline_train_rmse=0.5)
    assert any("IGNORING" in m_ for m_ in _warnings(caplog))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ens = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=m, component_names=["a", "b"], component_oof_rmse=[1.0, 1.5], baseline_oof_rmse=2.0
        )
    assert not _warnings(caplog)
    w = np.asarray(ens.weights, dtype=float)
    np.testing.assert_allclose(w / w.sum(), np.array([1.0, 0.5]) / 1.5, rtol=1e-9)  # gains over 2.0: 1.0 and 0.5


def test_oof_rmses_replace_the_train_rmse_ranking(caplog):
    """ "Pass component_oof_rmse=... for an honest cross-validated weighting": the ranking follows the OOF numbers."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble

    m = [LinearRegression(), LinearRegression()]
    with caplog.at_level(logging.WARNING):
        CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=m, component_names=["a", "b"], component_train_rmse=[1.0, 1.5], baseline_train_rmse=2.0
        )
    assert any("biased optimistic" in m_ for m_ in _warnings(caplog))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ens = CompositeCrossTargetEnsemble.from_train_metrics(
            component_models=m, component_names=["a", "b"], component_train_rmse=[1.0, 1.5], component_oof_rmse=[1.5, 1.0], baseline_oof_rmse=2.0
        )
    assert not _warnings(caplog)
    w = np.asarray(ens.weights, dtype=float)
    assert w[1] > w[0], "the OOF numbers, not the train ones, must decide the weights"


def test_drop_invalid_rows_fits_through_domain_violations():
    """ "Set drop_invalid_rows=True to drop them automatically": the fit then succeeds on the valid rows."""
    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.composite.estimator import DomainViolationError

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"b": rng.uniform(1.0, 5.0, 200), "x": rng.normal(size=200)})
    y = X["b"].to_numpy() * np.exp(0.1 * X["x"].to_numpy())
    y[:5] = -1.0  # log(y / b) is undefined on these rows
    kw = dict(base_estimator=LinearRegression(), transform_name="logratio", base_column="b")
    with pytest.raises(DomainViolationError, match="drop_invalid_rows=True"):
        CompositeTargetEstimator(drop_invalid_rows=False, **kw).fit(X, y)
    est = CompositeTargetEstimator(drop_invalid_rows=True, **kw).fit(X, y)
    assert np.all(np.isfinite(est.predict(X)))


def test_auto_discovered_blocks_replace_missing_feature_groups():
    """ "(or set auto_discover_blocks=True)": the stacker then fits without hand-written groups."""
    from mlframe.training.composite.grouped_block_stacking import GroupedBlockStacker

    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    X = pd.DataFrame({"a1": a, "a2": a + rng.normal(0.0, 0.1, 300), "b1": rng.normal(size=300)})
    y = X.sum(axis=1).to_numpy()
    with pytest.raises(ValueError, match="auto_discover_blocks=True"):
        GroupedBlockStacker(submodel_factory=LinearRegression, meta_estimator=LinearRegression()).fit(X, y)
    st = GroupedBlockStacker(submodel_factory=LinearRegression, meta_estimator=LinearRegression(), auto_discover_blocks=True).fit(X, y)
    assert st.feature_groups_ and np.all(np.isfinite(st.predict(X)))


def test_the_estimators_accept_what_their_errors_ask_for():
    """ "supply one [base_estimator]", "supply the entity id via entity_column or entity_id=", "Pass base_column='<col>'"."""
    from mlframe.training.composite.multi_output import CompositeMultiOutputEstimator
    from mlframe.training.composite.panel import CompositePanelEstimator
    from mlframe.training.composite.sklearn_compat import make_composite_regressor

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"b": rng.uniform(1.0, 5.0, 200), "x": rng.normal(size=200), "e": np.arange(200) % 5})
    Y = np.column_stack([X["b"] + X["x"], 2.0 * X["b"]])
    with pytest.raises(ValueError, match="supply one"):
        CompositeMultiOutputEstimator(column_specs={"transform_name": "diff", "base_column": "b"}, skip_failed_columns=False).fit(X, Y)
    CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(), column_specs={"transform_name": "diff", "base_column": "b"}, skip_failed_columns=False
    ).fit(X, Y)
    y = Y[:, 0]
    with pytest.raises(ValueError, match="supply the entity id"):
        CompositePanelEstimator(LinearRegression()).fit(X[["x"]], y)
    CompositePanelEstimator(LinearRegression(), entity_column="e").fit(X, y)
    CompositePanelEstimator(LinearRegression()).fit(X[["x"]], y, entity_id=X["e"].to_numpy())
    with pytest.raises(ValueError, match="Pass base_column="):
        make_composite_regressor(LinearRegression(), transform_name="diff")
    make_composite_regressor(LinearRegression(), transform_name="diff", base_column="b").fit(X, y)


def test_the_target_transformer_takes_x_or_an_explicit_base():
    """ "call fit(y, X) or pass base=<array>": both fit, and they agree."""
    from mlframe.training.composite.sklearn_compat import CompositeTargetTransformer

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"b": rng.uniform(1.0, 5.0, 100), "b2": rng.uniform(1.0, 5.0, 100)})
    y = 2.0 * X["b"].to_numpy() + 1.0
    for kw in ({"base_column": "b"}, {"base_columns": ["b", "b2"]}):
        name = "linear_residual" if "base_column" in kw else "linear_residual_multi"
        with pytest.raises(ValueError, match="call fit\\(y, X\\) or pass base="):
            CompositeTargetTransformer(transform_name=name, **kw).fit(y)
        CompositeTargetTransformer(transform_name=name, **kw).fit(y, X)
    a = CompositeTargetTransformer(transform_name="linear_residual", base_column="b").fit(y, X).transform(y)
    c = CompositeTargetTransformer(transform_name="linear_residual", base=X["b"].to_numpy()).fit(y).transform(y)
    np.testing.assert_allclose(a, c)


def test_the_feature_generator_accepts_a_wrapper_factory():
    """ "supply either `spec` or `wrapper_factory`": a factory alone builds the feature."""
    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.composite.suite_features import CompositeFeatureGenerator

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"b": rng.uniform(1.0, 5.0, 200), "x": rng.normal(size=200)})
    y = X["b"].to_numpy() + X["x"].to_numpy()
    with pytest.raises(ValueError, match="supply either"):
        CompositeFeatureGenerator().fit_transform(X, y)
    gen = CompositeFeatureGenerator(
        wrapper_factory=lambda: CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="diff", base_column="b"), column_name="f", n_splits=3
    )
    assert "f" in gen.fit_transform(X, y).columns


def test_fit_final_on_all_enables_transform():
    """ "set fit_final_on_all=True to enable transform on new data"."""
    from sklearn.exceptions import NotFittedError

    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.composite.suite_features import CompositeFeatureGenerator

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"b": rng.uniform(1.0, 5.0, 200), "x": rng.normal(size=200)})
    y = X["b"].to_numpy() + X["x"].to_numpy()
    factory = lambda: CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="diff", base_column="b")
    off = CompositeFeatureGenerator(wrapper_factory=factory, column_name="f", n_splits=3, fit_final_on_all=False)
    off.fit_transform(X, y)
    with pytest.raises(NotFittedError, match="fit_final_on_all=True"):
        off.transform(X)
    on = CompositeFeatureGenerator(wrapper_factory=factory, column_name="f", n_splits=3, fit_final_on_all=True)
    on.fit_transform(X, y)
    assert "f" in on.transform(X).columns


def test_a_train_mask_silences_the_eps_scale_warning(caplog):
    """ "Pass train_mask=<train rows>": the warning goes and the floor follows the train rows only (the leakage canary checks the latter)."""
    from mlframe.training.composite.transforms.interaction_bases import generate_interaction_bases

    rng = np.random.default_rng(0)
    c = {"a": rng.uniform(1.0, 2.0, 100), "b": rng.uniform(1.0, 2.0, 100)}
    with caplog.at_level(logging.WARNING):
        generate_interaction_bases(c, top_k=2)
    assert any("Pass train_mask=" in m for m in _warnings(caplog))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        generate_interaction_bases(c, top_k=2, train_mask=np.arange(100) < 80)
    assert not _warnings(caplog)


def test_rounding_scores_brings_venn_abers_under_the_warning(caplog):
    """ "rounding scores to reduce cardinality": rounded scores leave the warning band and still give valid envelopes."""
    from mlframe.training.composite.venn_abers import _VENN_ABERS_G_WARN, _isotonic_envelopes

    rng = np.random.default_rng(0)
    s = np.sort(rng.uniform(size=_VENN_ABERS_G_WARN + 500))
    y = (rng.uniform(size=s.size) < s).astype(float)
    with caplog.at_level(logging.WARNING):
        _isotonic_envelopes(s, y)
    assert any("reduce cardinality" in m for m in _warnings(caplog))
    caplog.clear()
    r = np.round(s, 3)
    with caplog.at_level(logging.WARNING):
        grid, p0, p1 = _isotonic_envelopes(r, y)
    assert not _warnings(caplog) and grid.size <= 1001 and np.all(p0 <= p1 + 1e-12)
