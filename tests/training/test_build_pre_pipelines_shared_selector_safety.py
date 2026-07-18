"""Guards for the shared-mutable-object bug class fixed in ``strategy.build_pipeline`` (see
``tests/training/test_strategy_imputer_propagation.py::TestBuildPipelineClonesSharedComponents``).

``_build_pre_pipelines`` is called ONCE PER TARGET (``_phase_train_one_target_model_setup.py``) but
pulls its RFECV instance(s) straight out of the CALLER's ``rfecv_models_params`` dict, which is a
single suite-level config object reused across every target -- so the SAME RFECV instance is
returned (by reference, not cloned) by every call across the whole suite. This sharing is only safe
because ``_build_pre_pipelines`` never calls ``.fit()`` on that instance itself: the actual per-model
fit always goes through ``clone(base_pipeline)`` at the call site in ``_phase_train_one_target_body.py``
("Clone the base_pipeline per model" comment). If a future edit to ``_build_pre_pipelines`` ever fit
the shared instance directly (e.g. a "pre-screen the selector once" optimization), every other target
sharing that same config would silently see the LAST target's fitted state -- the exact bug class this
file guards against, one layer up from the imputer/scaler fix.
"""

from __future__ import annotations

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines
from mlframe.training.pipeline._pipeline_helpers import _is_fitted


def _assert_unfitted(estimator) -> None:
    """Raise if ``estimator`` carries genuine sklearn-fitted state.

    Uses the suite's own marker-aware ``_is_fitted`` (not raw ``sklearn.utils.validation.check_is_fitted``):
    ``_build_pre_pipelines`` stamps trailing-underscore suite markers (``_mlframe_selector_kind_`` etc.)
    on the selector, which trips sklearn's default fitted-heuristic (ANY trailing-underscore attribute)
    even though no real ``.fit()`` ever ran -- exactly the false positive ``_is_fitted``'s own docstring
    documents guarding against.
    """
    if _is_fitted(estimator):
        raise AssertionError(f"{type(estimator).__name__} instance is already fitted -- _build_pre_pipelines must never fit the shared selector it returns.")


class TestBuildPrePipelinesNeverFitsSharedSelector:
    """``_build_pre_pipelines`` must only CONFIGURE (set_params/setattr hyperparameters), never FIT,
    the RFECV instance it pulls from the caller's ``rfecv_models_params`` -- that instance is the same
    object across every target in the suite (see module docstring)."""

    def test_returned_rfecv_instance_is_unfitted(self):
        """The RFECV pre_pipeline entry must come back unfitted.

        ``rfecv_cluster_reduce=False`` keeps the bare RFECV as the returned entry -- the default-ON
        ``GroupAwareMRMR`` wrapper reports ``check_is_fitted`` == True right after construction (its
        own, unrelated sklearn-convention quirk: it sets a trailing-underscore attribute in
        ``__init__``, not in ``fit``), which is orthogonal to what this test pins.
        """
        rfecv = RFECV(estimator=LinearRegression(), min_features_to_select=1)
        pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
            use_ordinary_models=False,
            rfecv_models=["lin"],
            rfecv_models_params={"lin": rfecv},
            use_mrmr_fs=False,
            mrmr_kwargs={},
            rfecv_cluster_reduce=False,
        )
        assert pre_pipeline_names == ["lin "]
        assert pre_pipelines[0] is rfecv
        _assert_unfitted(pre_pipelines[0])

    def test_same_shared_config_across_two_simulated_targets_stays_unfitted_and_identity_stable(self):
        """Two ``_build_pre_pipelines`` calls (simulating two targets in one suite run) sharing the
        SAME ``rfecv_models_params`` dict must both return the identical, still-UNFITTED instance --
        pinning that the sharing-by-reference contract is safe only because neither call fits it."""
        rfecv = RFECV(estimator=LinearRegression(), min_features_to_select=1)
        shared_params = {"lin": rfecv}

        pre_pipelines_a, _ = _build_pre_pipelines(
            use_ordinary_models=False, rfecv_models=["lin"], rfecv_models_params=shared_params,
            use_mrmr_fs=False, mrmr_kwargs={}, rfecv_cluster_reduce=False,
        )
        pre_pipelines_b, _ = _build_pre_pipelines(
            use_ordinary_models=False, rfecv_models=["lin"], rfecv_models_params=shared_params,
            use_mrmr_fs=False, mrmr_kwargs={}, rfecv_cluster_reduce=False,
        )
        assert pre_pipelines_a[0] is pre_pipelines_b[0] is rfecv, "documents the by-design cross-target sharing"
        _assert_unfitted(pre_pipelines_a[0])
        _assert_unfitted(pre_pipelines_b[0])
