"""Beds where the STRUCTURE is trivial and the marginal distribution is the whole difficulty.

Every bed in the linear family draws standard normals, which is the one marginal on which every estimator
behaves. These beds keep exactly that structure -- a handful of informative columns, a calibrated ceiling,
independent probes -- and change only the shape of the columns. A method that loses here loses to the
distribution and not to the dependence, and separating those two is the point: "MRMR does badly on this
dataset" is advice nobody can act on, "a binned estimator loses resolution once a column has fewer
distinct values than it wants bins" is.

Four shapes, each attacking a different assumption:

* **heavy tails.** A Student-t with four degrees of freedom has infinite kurtosis, so a Pearson
  correlation is dominated by a handful of rows and a rank statistic is not. The two families of method
  separate by construction.
* **outliers.** A small share of the rows displaced far from the bulk. Distinct from heavy tails: the
  contamination is a fixed count rather than a property of the law, so a method robust to one is not
  automatically robust to the other.
* **zero inflation.** A point mass at zero on top of a continuous law. Equal-mass binning cannot split a
  point mass, so the requested bin count is not the delivered one and the estimator's resolution silently
  drops to whatever the mass leaves over.
* **quantisation.** The informative columns are rounded onto a coarse grid, as an instrument reporting to
  two decimal places would. Ties everywhere, which is the same attack on equal-mass binning from the
  other direction.

The probes always share the informative columns' marginal. A bed whose probes were standard normal while
its signal columns were heavy-tailed would be solvable without ever looking at the target, and the
varsortability tripwire exists because that mistake is easy to make.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["heavy_tail_spec", "outlier_contaminated_spec", "zero_inflated_spec", "quantized_spec"]


def _marginal_bed(
    name: str,
    family: str,
    params: Optional[Dict[str, float]] = None,
    n_informative: int = 5,
    n_noise: int = 30,
    n_samples: int = 5000,
    ceiling: float = 0.80,
    seed: int = 0,
    outlier_fraction: float = 0.0,
    quantize_levels: Optional[int] = None,
    purpose: str = "",
) -> DatasetSpec:
    """Build a linear bed whose every column -- informative and probe alike -- carries one marginal shape."""
    shared: Dict[str, Any] = {"family": family, "params": dict(params or {}), "outlier_fraction": outlier_fraction}
    if quantize_levels is not None:
        shared["quantize_levels"] = quantize_levels
    informative = tuple(FeatureSpec(name=f"s{i}", **shared) for i in range(n_informative))
    probes = tuple(FeatureSpec(name=f"n{i:03d}", **shared) for i in range(n_noise))
    weights = {f"s{i}": float(1.2 * (0.72**i)) for i in range(n_informative)}
    return DatasetSpec(
        name=name,
        n_samples=n_samples,
        root_seed=seed,
        features=informative + probes,
        targets=(TargetSpec(name="y", prevalence=0.35, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "marginals", "marginal": family, "purpose": purpose},
    )


def heavy_tail_spec(n_samples: int = 5000, seed: int = 0, df: float = 4.0) -> DatasetSpec:
    """Return the heavy-tailed bed: a Student-t marginal on a linear structure.

    At four degrees of freedom the kurtosis is infinite, so a Pearson correlation is decided by a handful
    of extreme rows while a rank statistic is not. Any claim that a method is "robust" has a measurable
    meaning on this bed and on almost none of the others.
    """
    return _marginal_bed(f"heavy_tail_t{int(df)}", "student_t", {"df": df}, n_samples=n_samples, seed=seed, purpose="infinite kurtosis: Pearson decided by a few rows, rank statistics unaffected")


def outlier_contaminated_spec(n_samples: int = 5000, seed: int = 0, fraction: float = 0.02) -> DatasetSpec:
    """Return the contaminated bed: normal columns with a fixed share of rows displaced far from the bulk.

    Deliberately separate from the heavy-tail bed. There the extremes come from the law and scale with the
    sample; here they are a fixed contamination rate, which is what a data-entry error or a broken sensor
    actually looks like. A method robust to one is not automatically robust to the other, and a suite with
    only one of them cannot tell.
    """
    return _marginal_bed(f"outliers_{int(fraction * 1000):03d}permille", "normal", n_samples=n_samples, seed=seed, outlier_fraction=fraction, purpose="fixed-rate contamination rather than a heavy law")


def zero_inflated_spec(n_samples: int = 5000, seed: int = 0, inflation: float = 0.4) -> DatasetSpec:
    """Return the zero-inflated bed: a point mass at zero on top of a continuous law.

    Equal-mass binning cannot split a point mass. With forty per cent of the rows on one value, an
    estimator asking for ten equal-mass bins gets far fewer usable ones, and its resolution drops without
    anything reporting that it did. The degradation is silent, which is what makes it worth a bed.
    """
    return _marginal_bed(
        f"zero_inflated_{int(inflation * 100)}pct",
        "zero_inflated_normal",
        {"inflation": inflation},
        n_samples=n_samples,
        seed=seed,
        purpose="a point mass equal-mass binning cannot split, so the delivered bin count is not the requested one",
    )


def quantized_spec(n_samples: int = 5000, seed: int = 0, levels: int = 6) -> DatasetSpec:
    """Return the quantised bed: every column rounded onto a coarse grid, as a low-resolution instrument does.

    Ties everywhere. An estimator that wants ten equal-mass bins from six distinct values cannot have them,
    and a rank-based statistic has to decide what to do with large tie groups. Both are real, both are
    common, and neither is visible on a bed of continuous normals.
    """
    return _marginal_bed(f"quantized_{levels}levels", "normal", n_samples=n_samples, seed=seed, quantize_levels=levels, purpose="fewer distinct values than an estimator wants bins")
