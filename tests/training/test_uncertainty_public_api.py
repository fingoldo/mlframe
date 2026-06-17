"""The uncertainty / calibration / conformal helpers are importable from the public ``mlframe.training`` API."""

from __future__ import annotations


def test_public_uncertainty_api_importable():
    import mlframe.training as T

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
        assert hasattr(T, n), f"mlframe.training missing public symbol {n}"
        assert n in T.__all__, f"{n} not in mlframe.training.__all__"
