"""The uncertainty / calibration / conformal helpers are importable from the public ``mlframe.training`` API."""

from __future__ import annotations


def test_public_uncertainty_api_importable():
    """Public uncertainty api importable."""
    import mlframe.training as training_mod

    names = [
        "conformal_regression_report",
        "conformal_classification_report",
        "infer_split_structure",
        "PointRecalibrator",
        "fit_point_recalibrator",
        "RecalibratedRegressor",
        "DistributionalRecalibrator",
        "duan_log_smearing_factor",
        "smearing_predict",
        "tta_predict",
        "tta_predict_spread",
        "mc_dropout_predict",
        "predictive_entropy",
        "NoiseAugmentedEnsemble",
    ]
    for n in names:
        assert hasattr(training_mod, n), f"mlframe.training missing public symbol {n}"
        assert n in training_mod.__all__, f"{n} not in mlframe.training.__all__"
