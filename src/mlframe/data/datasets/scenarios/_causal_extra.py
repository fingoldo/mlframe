"""Four more causal shapes, each one a structure where a reasonable method does the wrong thing.

The existing causal beds cover a blanket member no marginal method can reach and a mediator whose
correctness depends on which answer key is declared. These four cover shapes that are just as common and
that no bed in this suite currently contains.

* **M-bias.** ``U1 -> X``, ``U1 -> Z``, ``U2 -> Z``, ``U2 -> Y``. What is marginally independent here is
  ``X`` and ``Y`` -- NOT ``Z``, which is associated with both through its latent parents and which a
  marginal filter will therefore happily select. Conditioning on ``Z`` then *creates* an association
  between ``X`` and ``Y`` that the graph does not contain. Measured on this bed: marginal correlation of
  ``X`` with the target -0.008, partial correlation given ``Z`` of -0.216. This is the one shape where
  adding a column makes a selector worse rather than merely wider, and the column it adds is one every
  marginal score recommends.
* **Confounder.** ``U -> X``, ``U -> Y``. The classic backdoor: ``X`` is associated with the target and
  causes none of it. Every predictive method should take ``X`` and every causal reading of that choice is
  wrong, which is the same lesson the mediator bed teaches from the other side.
* **Instrument.** ``I -> X -> Y``, with ``I`` affecting the target only through ``X``. Once ``X`` is
  selected, ``I`` adds nothing -- it is screened off -- so a method that keeps both is paying for a column
  whose information it already has. It is the cheapest available test of whether a method notices
  redundancy that is structural rather than correlational.
* **Proxy attenuation.** The true cause is measured noisily and a cleaner proxy of it sits beside it.
  Scored on prediction the proxy is the better column and a method preferring it is right; scored on
  causal parents it is exactly wrong. The bed exists because the preference is not a bug in the method --
  it follows from what the method was asked to optimise.

Every one of these is declared as edges rather than built by correlating columns until the numbers look
convincing. That matters most for M-bias: the whole property depends on ``Z`` having two parents and no
path to the target, and a construction that merely produced a weakly correlated column would not have it.
"""

from __future__ import annotations

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LatentSpec, LinkSpec, TargetSpec

__all__ = ["m_bias_spec", "confounder_spec", "instrument_spec", "proxy_attenuation_spec"]


def m_bias_spec(n_noise: int = 28, n_samples: int = 8000, seed: int = 0) -> DatasetSpec:
    """Return the M-bias bed: a column every marginal score recommends and that poisons what follows.

    ``m_collider`` has two latent parents, one shared with the cause and one shared with the target, so it
    is associated with BOTH -- 0.58 with the cause and 0.28 with the target as this bed is built. A
    marginal filter will select it, and it looks like a reasonable pick.

    It is not. The cause and the target are marginally independent (-0.008 here), and conditioning on the
    collider manufactures an association between them (-0.216). Anything reasoning conditionally after
    admitting this column is reasoning through a path the graph does not contain.

    An earlier version of this docstring said the collider was marginally independent of both, which is
    the standard way the shape is misdescribed and is not what the structure does; the numbers above are
    measured, not asserted.

    The bed is large by this suite's standards because the effect lives in the difference between two
    conditional associations, and at small n that difference is inside the noise.
    """
    return DatasetSpec(
        name="m_bias",
        n_samples=n_samples,
        root_seed=seed,
        features=(
            FeatureSpec(name="x_cause"),
            FeatureSpec(name="m_collider"),
            FeatureSpec(name="y_parent"),
            *probes(n_noise),
        ),
        latents=(
            # Two independent latents, each feeding one arm of the M, and UNOBSERVED on purpose: if either
            # were a column, a method could condition on it and the collider would stop being one.
            LatentSpec(name="u_left", reflections=("x_cause",), loadings=(0.9,), noise_sd=0.45),
            LatentSpec(name="u_right", reflections=("y_parent",), loadings=(0.9,), noise_sd=0.45),
        ),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                # Only `y_parent` drives the target. `x_cause` reaches it through nothing at all; the
                # association a method finds after conditioning on the collider is manufactured.
                link=LinkSpec(kind="logistic", coefficients={"y_parent": 1.2}),
                calibrate_to=CeilingTarget(metric="auc", value=0.78),
            ),
        ),
        edges=(
            EdgeSpec(source="y_parent", target="y"),
            # The collider takes BOTH latents as parents, which is the shape and cannot be expressed as a
            # reflection: a reflection overwrites its column, so a column named by two latents ends up a
            # reflection of whichever ran last and one arm of the M vanishes without a word.
            EdgeSpec(source="u_left", target="m_collider", kind="latent"),
            EdgeSpec(source="u_right", target="m_collider", kind="latent"),
        ),
        provenance={"family": "causal", "purpose": "conditioning on a marginally independent column CREATES an association that is not there"},
    )


def confounder_spec(n_noise: int = 30, n_samples: int = 6000, seed: int = 0) -> DatasetSpec:
    """Return the classic backdoor: a column associated with the target that causes none of it.

    Every predictive method should select ``x_confounded`` and every causal reading of that selection is
    wrong. The bed is here so the suite can say that plainly rather than leaving it implicit in the
    mediator bed's more complicated shape.
    """
    return DatasetSpec(
        name="confounder_backdoor",
        n_samples=n_samples,
        root_seed=seed,
        features=(
            FeatureSpec(name="x_confounded"),
            FeatureSpec(name="x_true_cause"),
            *probes(n_noise),
        ),
        latents=(LatentSpec(name="u_common", reflections=("x_confounded", "x_true_cause"), loadings=(0.85, 0.6), noise_sd=0.55),),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(kind="logistic", coefficients={"x_true_cause": 1.1}),
                calibrate_to=CeilingTarget(metric="auc", value=0.78),
            ),
        ),
        edges=(EdgeSpec(source="x_true_cause", target="y"),),
        provenance={"family": "causal", "purpose": "a column predictive of the target that causes none of it"},
    )


def instrument_spec(n_noise: int = 30, n_samples: int = 6000, seed: int = 0) -> DatasetSpec:
    """Return the instrument bed: ``I -> X -> Y``, where ``I`` reaches the target only through ``X``.

    Once ``X`` is in the set, ``I`` is screened off and adds nothing. A method that keeps both is paying
    for information it already holds -- redundancy that is STRUCTURAL rather than correlational, which is
    the kind a correlation-threshold filter is not built to notice.
    """
    return DatasetSpec(
        name="instrument_screened_off",
        n_samples=n_samples,
        root_seed=seed,
        features=(
            FeatureSpec(name="instrument"),
            FeatureSpec(name="x_cause"),
            *probes(n_noise),
        ),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                link=LinkSpec(kind="logistic", coefficients={"x_cause": 1.2}),
                calibrate_to=CeilingTarget(metric="auc", value=0.80),
            ),
        ),
        edges=(
            EdgeSpec(source="instrument", target="x_cause", kind="instrument"),
            EdgeSpec(source="x_cause", target="y"),
        ),
        provenance={"family": "causal", "purpose": "a column screened off by one already selected: structural redundancy, not correlational"},
    )


def proxy_attenuation_spec(n_noise: int = 30, n_samples: int = 6000, seed: int = 0) -> DatasetSpec:
    """Return the bed where a cleaner proxy out-predicts the noisily measured cause.

    The true cause is UNOBSERVED. ``x_measured`` is how it reaches the dataset -- attenuated by measurement
    error -- and ``proxy`` is a cleaner reflection of the same thing. Because the target is driven by the
    latent rather than by either column, the proxy is the better predictor: measured on this bed, 0.359
    against 0.264.

    That ordering is the bed. A method optimising prediction prefers the proxy and is behaving correctly;
    a causal reading of the same choice is wrong, since the proxy is a symptom and ``x_measured`` is the
    instrument-of-record for the cause. The difference is not a defect in the method, it follows from what
    the method was asked to optimise.

    The first version of this bed drove the target from ``x_measured`` directly, which made the measured
    column the strongest predictor by construction and inverted the very ordering the bed exists to show.
    """
    return DatasetSpec(
        name="proxy_attenuation",
        n_samples=n_samples,
        root_seed=seed,
        features=(
            FeatureSpec(name="x_measured"),
            FeatureSpec(name="proxy"),
            *probes(n_noise),
        ),
        latents=(
            # One latent, two reflections of very different quality: the measurement of record is the noisy
            # one, the proxy is the clean one, and that asymmetry is what makes the trade bite.
            LatentSpec(name="u_truth", reflections=("x_measured", "proxy"), loadings=(0.5, 0.95), noise_sd=0.8),
        ),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.4,
                # Driven by the LATENT. Neither observed column causes the target; one is a noisy measure of
                # what does and the other is a clean one.
                link=LinkSpec(kind="logistic", coefficients={"u_truth": 1.3}),
                calibrate_to=CeilingTarget(metric="auc", value=0.78),
            ),
        ),
        # Only the edge into the target. The two columns are REFLECTIONS -- the latent layer builds them
        # with their declared loadings -- and adding latent edges for them would hand the job to the derived
        # layer instead, which rebuilds both identically and erases the asymmetry this bed is made of.
        edges=(EdgeSpec(source="u_truth", target="y", kind="latent"),),
        provenance={"family": "causal", "purpose": "a clean proxy out-predicts the noisy measurement of the real cause"},
    )
