"""Linear beds: the easy case, and the one place a difficulty sweep is unambiguous.

A linear-logistic bed with independent Gaussian columns is the case every method handles, which is exactly
why it is worth having: it is the control that says a method is not broken, and it is the only family where
the recovery-versus-difficulty curve is uncontaminated by structure.

Two variants earn their place beyond the control role:

* ``linear_gaussian_lowdim_n200`` -- few rows, few columns. At ``n = 200`` a t-statistic beats a binned
  mutual-information estimate, because the binning throws away most of what little data there is. Any
  method that discretises before scoring is expected to lose here, and if it does not, the claim that it
  handles small samples is stronger than its authors could have known.
* ``linear_ceiling_sweep`` -- the same structure at a grid of achievable ceilings. The point where two
  methods cross on that curve is a finding; their ranking at one hand-picked signal level is a choice made
  by whoever picked it.
"""

from __future__ import annotations

from typing import Tuple

from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["linear_spec", "linear_lowdim_spec", "CEILING_GRID"]

#: Ceilings the recovery curve is measured along. The bottom of the range is where methods diverge; the top
#: is where they all succeed and the benchmark learns nothing, which is why a single high-SNR point is not
#: an experiment.
CEILING_GRID: Tuple[float, ...] = (0.60, 0.70, 0.80, 0.90, 0.97)


def linear_spec(
    n_informative: int = 5,
    n_noise: int = 45,
    n_samples: int = 5000,
    ceiling: float = 0.80,
    prevalence: float = 0.35,
    seed: int = 0,
) -> DatasetSpec:
    """Return a linear-logistic bed with independent columns and a calibrated ceiling.

    Coefficients decay geometrically across the informative columns, so the bed contains both an obvious
    signal and one that sits near the detection threshold. A flat coefficient vector would make recovery a
    step function of the ceiling and hide every difference between methods.
    """
    informative = tuple(FeatureSpec(name=f"s{i}") for i in range(n_informative))
    noise = tuple(FeatureSpec(name=f"n{i:03d}") for i in range(n_noise))
    weights = {f"s{i}": float(1.2 * (0.72**i)) for i in range(n_informative)}
    return DatasetSpec(
        name=f"linear_k{n_informative}_p{n_informative + n_noise}_auc{int(ceiling * 100)}",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + noise,
        targets=(
            TargetSpec(
                name="y",
                prevalence=prevalence,
                link=LinkSpec(kind="logistic", coefficients=weights),
                calibrate_to=CeilingTarget(metric="auc", value=ceiling),
            ),
        ),
        edges=tuple(EdgeSpec(source=name, target="y") for name in weights),
        provenance={"family": "linear", "purpose": "control, and the difficulty sweep"},
    )


def linear_lowdim_spec(n_samples: int = 200, seed: int = 0) -> DatasetSpec:
    """Return the small-sample linear bed where discretising estimators are expected to lose.

    Two hundred rows and twenty columns. A binned mutual-information score needs enough rows per bin to
    estimate anything, and at this size it does not have them, whereas a t-statistic uses every row.
    """
    spec = linear_spec(n_informative=4, n_noise=16, n_samples=n_samples, ceiling=0.75, seed=seed)
    return spec.model_copy(update={"name": "linear_gaussian_lowdim_n200", "provenance": {"family": "linear", "purpose": "small-n: t-statistic versus binned MI"}})
