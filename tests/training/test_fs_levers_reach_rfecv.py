"""The first-class ``rfecv_*`` levers and ``rfecv_kwargs`` of FeatureSelectionConfig reach the RFECV the suite builds.

They used to fold into ``rfecv_kwargs`` and stop there: nothing read the dict, so ``rfecv_must_include`` and the other seven
levers changed the config but not the selector.
"""

from __future__ import annotations

import pytest
from sklearn.linear_model import Ridge

from mlframe.feature_selection.wrappers import RFECV
from mlframe.training.configs import FeatureSelectionConfig
from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines


def _built_rfecv(fsc: FeatureSelectionConfig):
    """The bare RFECV inside the suite's pre-pipeline for ``fsc``, built exactly as ``_phase_train_one_target_model_setup`` calls it."""
    pre, _ = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=["cb_rfecv"],
        rfecv_models_params={"cb_rfecv": RFECV(estimator=Ridge())},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        rfecv_cluster_reduce=False,
        rfecv_overrides=fsc.rfecv_kwargs,
    )
    return pre[0]


def test_every_lever_reaches_the_rfecv_instance():
    """Each lever lands on the constructor parameter it folds into."""
    fsc = FeatureSelectionConfig(
        rfecv_models=["cb"],
        rfecv_must_include=["a"],
        rfecv_must_exclude=["b"],
        rfecv_feature_groups={"g": ["c", "d"]},
        rfecv_n_features_selection_rule="one_se_max",
        rfecv_enable_stability_selection=True,
        rfecv_enable_permutation_importance=True,
        rfecv_prescreen="univariate_ht",
        rfecv_swap_top_k=2,
    )
    rfecv = _built_rfecv(fsc)
    assert list(rfecv.must_include) == ["a"]
    assert list(rfecv.must_exclude) == ["b"]
    assert rfecv.feature_groups == {"g": ["c", "d"]}
    assert rfecv.n_features_selection_rule == "one_se_max"
    assert rfecv.stability_selection is True
    assert rfecv.importance_getter == "permutation"
    assert rfecv.prescreen == "univariate_ht"
    assert rfecv.swap_top_k == 2


def test_rfecv_kwargs_reach_the_instance_and_cv_n_splits_becomes_cv():
    """A plain constructor key is applied, and the config-level fold count becomes ``cv=<int>``."""
    rfecv = _built_rfecv(FeatureSelectionConfig(rfecv_models=["cb"], rfecv_kwargs={"max_runtime_mins": 7.0, "cv_n_splits": 4}))
    assert rfecv.max_runtime_mins == 7.0
    assert rfecv.cv == 4


def test_an_explicit_cv_wins_over_cv_n_splits():
    """``cv_n_splits`` only fills in a missing ``cv``."""
    rfecv = _built_rfecv(FeatureSelectionConfig(rfecv_models=["cb"], rfecv_kwargs={"cv": 3, "cv_n_splits": 9}))
    assert rfecv.cv == 3


def test_an_unset_config_leaves_the_instance_alone():
    """No lever, no override: the RFECV keeps its constructor defaults."""
    rfecv = _built_rfecv(FeatureSelectionConfig(rfecv_models=["cb"]))
    assert rfecv.must_include is None and rfecv.swap_top_k == 0 and rfecv.n_features_selection_rule == "auto"


def test_a_key_the_instance_does_not_accept_raises_naming_it():
    """A key RFECV does not know is an error at build time, not an attribute silently stuck on the object."""
    pre = dict(
        use_ordinary_models=False, rfecv_models=["cb_rfecv"], rfecv_models_params={"cb_rfecv": RFECV(estimator=Ridge())},
        use_mrmr_fs=False, mrmr_kwargs={}, rfecv_cluster_reduce=False,
    )
    with pytest.raises(ValueError, match="not_a_param"):
        _build_pre_pipelines(rfecv_overrides={"not_a_param": 1}, **pre)


def _selector_seed(selector):
    """``random_state`` of a selector, looking through the cluster-reduce wrapper some registry entries put around it."""
    if hasattr(selector, "random_state"):
        return selector.random_state
    seeds = {v for k, v in selector.get_params(deep=True).items() if k.endswith("random_state")}
    assert len(seeds) == 1, f"expected exactly one inner random_state, found {seeds}"
    return seeds.pop()


@pytest.mark.parametrize(
    ("flag", "kwargs_name"),
    [
        ("use_boruta_shap", "boruta_shap_kwargs"),
        ("use_shap_proxied_fs", "shap_proxied_fs_kwargs"),
        ("use_ace_fs", "ace_kwargs"),
        ("use_forward_select_fs", "forward_select_kwargs"),
        ("use_greedy_backward_elimination_fs", "greedy_backward_elimination_kwargs"),
        ("use_zero_importance_pruning_fs", "zero_importance_pruning_kwargs"),
        ("use_cascade_select_fs", "cascade_select_kwargs"),
    ],
)
def test_every_selector_is_seeded_from_the_suite_seed_unless_pinned(flag, kwargs_name):
    """Selectors whose ``random_state`` used to stay at its constructor default now follow the suite seed; an explicit one wins."""
    base = dict(use_ordinary_models=False, rfecv_models=[], rfecv_models_params={}, use_mrmr_fs=False, mrmr_kwargs={})
    seeded, _ = _build_pre_pipelines(**base, **{flag: True}, fs_random_seed=123)
    assert _selector_seed(seeded[0]) == 123
    pinned, _ = _build_pre_pipelines(**base, **{flag: True, kwargs_name: {"random_state": 7}}, fs_random_seed=123)
    assert _selector_seed(pinned[0]) == 7
    unseeded, _ = _build_pre_pipelines(**base, **{flag: True})
    assert _selector_seed(unseeded[0]) != 123
