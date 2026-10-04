"""sklearn protocol contracts for the public selectors and estimators: params stay verbatim, fitted state is underscore-named and pickles lean."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.feature_selection.boruta_shap import BorutaShap
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from mlframe.feature_selection.wrappers.rfecv import RFECV


@pytest.fixture(scope="module")
def clf_data():
    """Small binary-classification frame where columns a and b carry the signal."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(240, 6)), columns=list("abcdef"))
    y = (X["a"] + X["b"] + 0.3 * rng.normal(size=240) > 0).astype(int)
    return X, y


@pytest.fixture(scope="module")
def fitted_boruta(clf_data):
    """A BorutaShap fitted once on the shared frame."""
    X, y = clf_data
    return BorutaShap(n_trials=8, verbose=False, importance_measure="gini", random_state=0).fit(X, y)


def test_rfecv_fit_does_not_overwrite_the_scoring_constructor_param(clf_data):
    """Fitting with scoring=None leaves get_params()['scoring'] None and stores the resolved scorer in scoring_."""
    X, y = clf_data
    sel = RFECV(estimator=LogisticRegression(max_iter=200), cv=3, max_refits=2, verbose=0, random_state=0)
    sel.fit(X, y)
    assert sel.get_params()["scoring"] is None
    assert sel.scoring_ is not None
    assert clone(sel).get_params()["scoring"] is None


def test_rfecv_refit_on_a_regression_task_re_resolves_the_default_scorer(clf_data):
    """A classifier fit must not leave its scorer behind to be reused by a later regression fit of the same instance."""
    X, y = clf_data
    sel = RFECV(estimator=LinearRegression(), cv=3, max_refits=2, verbose=0, random_state=0, skip_retraining_on_same_shape=False)
    sel.fit(X, y.astype(float))
    first = sel.scoring_
    sel.set_params(estimator=LogisticRegression(max_iter=200))
    sel.fit(X, y)
    assert sel.scoring is None
    assert sel.scoring_ is not first


def test_boruta_shap_fitted_state_uses_trailing_underscores(fitted_boruta):
    """The results and history live under underscore names, and the bare legacy names stay readable aliases of them."""
    bs = fitted_boruta
    for name in ("accepted_", "rejected_", "tentative_", "history_x_", "hits_", "order_"):
        assert name in vars(bs), name
    for bare in ("accepted", "rejected", "tentative", "history_x", "hits", "order", "X", "y", "X_shadow"):
        assert bare not in vars(bs), bare
    assert bs.accepted is bs.accepted_
    assert set(bs.accepted_) <= set(bs.selected_features_)


def test_boruta_shap_pickle_drops_training_data_copies(fitted_boruta, clf_data):
    """The pickle carries no X/y/shadow frames, still transforms, and round-trips the selection."""
    X, _ = clf_data
    restored = pickle.loads(pickle.dumps(fitted_boruta))  # nosec B301 - round-trip of an object this test just pickled
    for name in ("X_", "y_", "starting_X_", "X_shadow_", "X_boruta_", "shap_values_"):
        assert name not in vars(restored), name
    assert list(restored.transform(X).columns) == list(fitted_boruta.transform(X).columns)
    assert sorted(restored.accepted) == sorted(fitted_boruta.accepted)
    assert hasattr(fitted_boruta, "X_")


def test_boruta_shap_pickle_does_not_grow_with_the_row_count():
    """Training-data copies were pickled before; the pickle size must now be independent of the number of rows."""
    sizes = []
    for n in (200, 4000):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
        y = (X["a"] > 0).astype(int)
        bs = BorutaShap(n_trials=6, verbose=False, importance_measure="gini", random_state=0, model=None).fit(X, y)
        bs.model_ = None
        sizes.append(len(pickle.dumps(bs)))
    assert sizes[1] < sizes[0] * 1.5, sizes


def test_boruta_shap_set_output_pandas_and_feature_names_out(fitted_boruta, clf_data):
    """set_output works and get_feature_names_out reports the columns transform emits."""
    X, _ = clf_data
    bs = clone(fitted_boruta)
    with pytest.raises(NotFittedError):
        bs.get_feature_names_out()
    names = fitted_boruta.get_feature_names_out()
    assert list(names) == list(fitted_boruta.transform(X).columns)
    fitted_boruta.set_output(transform="pandas")
    assert isinstance(fitted_boruta.transform(X), pd.DataFrame)


def test_boruta_shap_loads_a_pickle_that_stored_the_bare_attribute_names(fitted_boruta, clf_data):
    """A pickle from before the rename keys its state by the bare names; loading it must expose them through the new names."""
    state = fitted_boruta.__getstate__()
    legacy = dict(state)
    legacy["accepted"] = legacy.pop("accepted_")
    legacy["history_x"] = legacy.pop("history_x_")
    legacy.pop("auto_dispatch_diagnostics_", None)
    restored = BorutaShap.__new__(BorutaShap)
    restored.__setstate__(legacy)
    assert restored.accepted_ == fitted_boruta.accepted_
    assert restored.history_x_ is legacy["history_x"]
    assert restored.auto_dispatch_diagnostics_ is None


def test_rfecv_setstate_backfills_auxiliary_fitted_attributes_but_not_core_state(clf_data):
    """An old pickle lacking newer auxiliary attributes gets defaults; one lacking support_ stays unfitted."""
    X, y = clf_data
    sel = RFECV(estimator=LogisticRegression(max_iter=200), cv=3, max_refits=2, verbose=0, random_state=0).fit(X, y)
    state = sel.__getstate__()
    old = {k: v for k, v in state.items() if k not in ("scoring_", "cv_results_", "provenance_", "estimators_")}
    restored = RFECV.__new__(RFECV)
    restored.__setstate__(old)
    assert restored.scoring_ is None and restored.provenance_ is None
    assert restored.cv_results_ == {} and restored.estimators_ == []
    unfitted = RFECV.__new__(RFECV)
    unfitted.__setstate__({k: v for k, v in old.items() if k != "support_"})
    assert not hasattr(unfitted, "scoring_")


def test_shap_proxied_fs_setstate_backfills_report_on_old_fitted_pickle():
    """A fitted ShapProxiedFS pickle written before shap_proxy_report_ existed loads with the attribute defaulted."""
    old = ShapProxiedFS.__new__(ShapProxiedFS)
    old.__setstate__({"support_": np.array([True]), "selected_features_": ["a"]})
    assert old.shap_proxy_report_ is None
    fresh = ShapProxiedFS.__new__(ShapProxiedFS)
    fresh.__setstate__({"n_features_in_": 3})
    assert not hasattr(fresh, "shap_proxy_report_")


def test_neural_embedding_text_encoder_is_cloneable_and_keeps_params_verbatim():
    """The constructor keeps list params as given (clone requires identity); normalisation happens at fit time on a copy."""
    pytest.importorskip("torch")
    from mlframe.training.neural.feature_prep import NeuralEmbeddingTextEncoder

    emb = ["e"]
    enc = NeuralEmbeddingTextEncoder(embedding_features=emb, text_features=None)
    assert enc.embedding_features is emb and enc.text_features is None
    cloned = clone(enc)
    assert cloned.get_params()["embedding_features"] == ["e"]
    clone(NeuralEmbeddingTextEncoder())
    df = pd.DataFrame({"e": [[1.0, 2.0], [3.0, 4.0]], "x": [1, 2]})
    out = enc.fit(df).transform(df)
    assert list(out.columns) == ["x", "e__e0", "e__e1"]
    assert emb == ["e"]


def test_mlranker_queries_per_batch_is_not_coerced_by_the_constructor():
    """get_params must return exactly what was passed; the clamp to at least one query happens when the batch size is used."""
    pytest.importorskip("torch")
    from mlframe.training.neural.ranker import MLPRanker

    assert MLPRanker(queries_per_batch=0).get_params()["queries_per_batch"] == 0
    assert clone(MLPRanker(queries_per_batch=7.0)).get_params()["queries_per_batch"] == 7.0


def test_shap_proxied_numeric_params_are_not_coerced_by_the_constructor():
    """Float-valued ints survive clone, which the int() coercion in __init__ used to break."""
    est = ShapProxiedFS(interaction_proxy_top_k=3.0, su_seeded_top_k=5.0)
    assert est.get_params()["interaction_proxy_top_k"] == 3.0 and isinstance(est.get_params()["su_seeded_top_k"], float)
    clone(est)


def test_mrmr_tree_rescued_numeric_params_are_not_coerced_by_the_constructor():
    """tree_rescue_top_k is stored as passed so get_params and clone round-trip float-valued ints."""
    from mlframe.feature_selection.filters._mrmr_tree_rescue import MRMRTreeRescued

    est = MRMRTreeRescued(tree_rescue_top_k=20.0)
    assert isinstance(est.get_params()["tree_rescue_top_k"], float)


def test_post_shim_default_constructed_shim_can_be_cloned():
    """clone() of a shim with no inner model, or a non-estimator model, keeps the model as is instead of raising."""
    from mlframe.training.composite.post_shim import PrePipelinePredictShim

    assert clone(PrePipelinePredictShim()).model is None
    sentinel = object()
    assert clone(PrePipelinePredictShim(model=sentinel)).model is sentinel
    inner = LinearRegression()
    cloned = clone(PrePipelinePredictShim(model=inner))
    assert cloned.model is not inner and isinstance(cloned.model, LinearRegression)


def test_gaussian_mixture_classifier_predict_before_fit_raises_not_fitted():
    """Unfitted predict must raise NotFittedError, not AttributeError on classes_."""
    from mlframe.competition.gmm_classifier import GaussianMixtureClassifier

    with pytest.raises(NotFittedError):
        GaussianMixtureClassifier().predict(np.zeros((3, 2)))


def test_import_optional_names_the_extra_in_the_error():
    """A missing optional package raises ImportError whose message tells the user which extra to install."""
    from mlframe._optional_imports import import_optional

    with pytest.raises(ImportError, match=r"mlframe\[boosting\]"):
        import_optional("definitely_not_an_installed_package_xyz", "boosting", "a test")
    assert import_optional("json", "none").dumps({}) == "{}"


def test_suite_metadata_writer_stamps_the_shared_schema_version(monkeypatch):
    """The per-model schema record carries the shared version constant, so a bump of it reaches freshly written bundles."""
    from mlframe.training.core import _phase_train_one_target_schema as mod

    monkeypatch.setattr(mod, "CURRENT_SCHEMA_VERSION", 99)
    monkeypatch.setattr(mod, "_maybe_render_friend_graph", lambda *a, **k: None)
    metadata: dict = {}
    ctx = SimpleNamespace(_fs_report_cache={})
    mod._build_and_record_model_schema(
        ctx, metadata, "file.dump", "lgb", "uniform", None, SimpleNamespace(), "t", np.array([0, 1, 0, 1]), np.array([0, 1, 2, 3]), None, "pp",
        None, "h", {"cols": []}, lambda **k: {}, lambda s: "p", lambda s: s,
    )
    assert metadata["model_schemas"]["file.dump"]["schema_version"] == 99
