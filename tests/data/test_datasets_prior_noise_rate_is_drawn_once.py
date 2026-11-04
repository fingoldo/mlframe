"""A prior-valued corruption rate is drawn once per target, so the calibration, the recorded ceiling and the labels share one law and later knob draws do not depend on the calibration."""
import numpy as np

from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, NoiseSpec, Prior, TargetSpec


def _spec(calibrated_auc: float) -> DatasetSpec:
    """Linear bed with a uniform-flip corruption whose rate is a prior."""
    return DatasetSpec(
        name="priornoise",
        n_samples=3000,
        root_seed=5,
        features=(FeatureSpec(name="x1"), FeatureSpec(name="x2"), FeatureSpec(name="n1")),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.35,
                link=LinkSpec(kind="logistic", coefficients={"x1": 1.0, "x2": 0.7}),
                calibrate_to=CeilingTarget(metric="auc", value=calibrated_auc),
                noise=NoiseSpec(kind="uniform_flip", rate=Prior(kind="uniform", low=0.0, high=0.4), true_prob_update="uniform_flip"),
            ),
        ),
        edges=(EdgeSpec(source="x1", target="y"), EdgeSpec(source="x2", target="y")),
    )


def test_recorded_auc_matches_the_ceiling_of_the_shipped_probabilities():
    """Each bisection step used a different flip rate, so the achieved AUC described a law other than the one the labels were drawn from."""
    ds = generate(_spec(0.75))
    assert abs(float(ds.calibration["achieved_auc"]) - float(ds.calibration["bayes_auc"])) < 0.01


def test_changing_the_calibration_target_does_not_change_the_noise_rate_drawn():
    """The flip rate is the same draw whatever the number of bisection steps was; its trace is the recorded true probability floor/ceiling spread."""
    a, b = generate(_spec(0.75)), generate(_spec(0.80))
    lo_a, hi_a = float(np.min(a.truth.true_prob)), float(np.max(a.truth.true_prob))
    lo_b, hi_b = float(np.min(b.truth.true_prob)), float(np.max(b.truth.true_prob))
    # uniform_flip keeps p within [rate, 1 - rate]: the same drawn rate gives the same floor and ceiling whatever the link scale
    assert abs(lo_a - lo_b) < 0.02 and abs(hi_a - hi_b) < 0.02
