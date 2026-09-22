"""The ensemble's OOF refit wrappers must carry their training base range, as the deployed wrapper does.

The deployed composite wrapper captures its train base range and soft-shrinks deep out-of-range rows. The OOF refits
that weight the ensemble built their wrappers without a base, so they scored a model without that guard - a different
model from the one served - on exactly the unseen-range rows the guard exists for.
"""

from __future__ import annotations

from mlframe.training.composite.estimator import CompositeTargetEstimator


def test_every_oof_refit_passes_its_training_base(monkeypatch):
    """Record ``base_train`` on every ``from_fitted_inner`` call the OOF computation makes."""
    from tests.training.composite.test_composite_integration import (
        _LEAN_OUTPUT_CONFIG_KWARGS,
        _LEAN_REPORTING_CONFIG_KWARGS,
        _build_minimal_fte,
        _tvt_dataset,
    )
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite
    import tempfile
    import pathlib

    seen: list = []
    real = CompositeTargetEstimator.from_fitted_inner.__func__

    def recording(cls, *args, **kwargs):
        """Record whether a base was supplied, then build as usual."""
        seen.append(kwargs.get("base_train") is not None)
        return real(cls, *args, **kwargs)

    monkeypatch.setattr(CompositeTargetEstimator, "from_fitted_inner", classmethod(recording))
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["TVT_prev"], transforms=["diff", "linear_residual"], mi_sample_n=200,
        top_k_after_mi=2, eps_mi_gain=-1.0, cross_target_ensemble_strategy="oof_weighted",
        # The fixture's gains sit below the default ship floor; only a spec that is actually trained gets OOF refits (an
        # untrained spec left in the metadata used to be refit too, which is what this test once observed).
        min_honest_gain_to_train=None,
    )
    tmp = pathlib.Path(tempfile.mkdtemp())
    train_mlframe_models_suite(
        df=_tvt_dataset(n=800), target_name="target", model_name="oofbase",
        features_and_targets_extractor=_build_minimal_fte(), mlframe_models=["linear"],
        output_config={"data_dir": str(tmp / "data"), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
        reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
    )
    assert seen, "no composite wrapper was built through from_fitted_inner"
    assert all(seen), f"{seen.count(False)} of {len(seen)} wrappers were built without their training base"
