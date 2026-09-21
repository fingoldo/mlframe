"""Beds with categorical columns, skewed cardinality, and the identifier trap.

Every other bed in this suite is all-numeric, which quietly excludes the failure mode that costs the most
in practice: an impurity-based importance prefers a high-cardinality column over a low-cardinality one at
equal information, because more distinct values mean more places to split. That is a property of the
measure, not of the data, and it is invisible on a bed where every column is continuous and unique.

Three beds:

* **graded cardinality.** The same informative signal carried by columns of two, five, twenty and a
  hundred levels. Any method that ranks them by anything other than the signal they carry is ranking by
  cardinality, and the bed says which.
* **the identifier trap.** One column whose value is unique per row and carries no information whatsoever.
  It is the single most common real-world leak, it maximises impurity-based importance by construction,
  and a method that selects it has failed in a way no aggregate score would reveal -- the downstream model
  then has a column it cannot generalise from at all.
* **Zipf levels.** A categorical whose level frequencies follow a power law, so a handful of levels hold
  most of the rows and a long tail holds one each. Target encoding has almost no data for the tail, and
  equal-mass binning cannot balance levels it cannot split.

The identifier column is declared as a PROBE and not as a cause, which is the whole point: it is
maximally attractive to an impurity measure and worth exactly nothing.
"""

from __future__ import annotations

from typing import Tuple

from mlframe.data.datasets.scenarios._common import probes
from mlframe.data.datasets.spec import CeilingTarget, DatasetSpec, EdgeSpec, FeatureSpec, LinkSpec, TargetSpec

__all__ = ["graded_cardinality_spec", "id_trap_spec", "zipf_levels_spec", "CARDINALITIES"]

#: The level counts the graded bed carries, spanning from binary to nearly continuous. Each column gets the
#: SAME link weight, so any difference in how methods rank them is a difference in how they treat
#: cardinality and not in how much signal is there.
CARDINALITIES: Tuple[int, ...] = (2, 5, 20, 100)


def _levels(count: int) -> Tuple[str, ...]:
    """Return explicit level labels, which the generator needs in order to keep a stable category dtype."""
    return tuple(f"L{index:03d}" for index in range(count))


def graded_cardinality_spec(n_noise: int = 20, n_samples: int = 6000, ceiling: float = 0.80, seed: int = 0) -> DatasetSpec:
    """Return the bed where the same signal is carried at four different cardinalities.

    The columns are drawn as categorical codes and enter the link as those codes, so each carries a
    monotone signal of equal weight. A method ranking ``k100`` above ``k2`` is expressing a preference for
    split opportunities, which is exactly the bias this bed exists to measure.
    """
    informative = tuple(FeatureSpec(name=f"k{levels}", family="categorical", params={"n_levels": float(levels)}, dtype="category", levels=_levels(levels)) for levels in CARDINALITIES)
    # Probe categoricals at matched cardinalities, so "prefers many levels" and "prefers informative
    # columns" cannot be confused: every cardinality present among the signals is also present among the
    # probes.
    probe_categoricals = tuple(
        FeatureSpec(name=f"p{levels}_{index}", family="categorical", params={"n_levels": float(levels)}, dtype="category", levels=_levels(levels)) for levels in CARDINALITIES for index in range(2)
    )
    weights = {f"k{levels}": 1.0 for levels in CARDINALITIES}
    return DatasetSpec(
        name="graded_cardinality",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + probe_categoricals + probes(n_noise),
        targets=(TargetSpec(name="y", prevalence=0.4, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "mixed_types", "purpose": "equal signal at four cardinalities: does a method rank by information or by split opportunities"},
    )


def id_trap_spec(n_noise: int = 25, n_samples: int = 6000, ceiling: float = 0.80, seed: int = 0) -> DatasetSpec:
    """Return the identifier-trap bed: a unique-per-row column carrying nothing.

    A row identifier maximises impurity-based importance by construction -- every split it offers is pure --
    while carrying no generalisable information at all. Selecting it is not a small error: the downstream
    model gets a column whose every test-set value is unseen.

    The identifier is declared as a probe. There is no edge from it to the target, because there is no
    relationship; a method that selects it has recorded a false positive of the most expensive kind.
    """
    identifier = FeatureSpec(name="row_id", family="uniform", params={"low": 0.0, "high": 1.0e6}, dtype="int", standardize=False)
    informative = tuple(FeatureSpec(name=f"s{i}") for i in range(4))
    weights = {f"s{i}": float(1.1 * (0.8**i)) for i in range(4)}
    return DatasetSpec(
        name="id_trap",
        n_samples=n_samples,
        root_seed=seed,
        features=(identifier, *informative, *probes(n_noise)),
        targets=(TargetSpec(name="y", prevalence=0.4, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "mixed_types", "purpose": "a unique-per-row column that maximises impurity importance and is worth nothing"},
    )


def zipf_levels_spec(n_noise: int = 20, n_samples: int = 8000, ceiling: float = 0.78, seed: int = 0) -> DatasetSpec:
    """Return the power-law categorical bed: a few levels hold most rows, a long tail holds one each.

    Real categorical columns almost never have balanced levels, and two things break on the tail. Target
    encoding has a handful of rows per level, so its estimate is mostly its own prior. Equal-mass binning
    cannot balance levels it is not allowed to split, so the requested bin count is not the delivered one.
    Both degrade quietly, which is why the bed states the shape rather than leaving it to chance.
    """
    informative = tuple(FeatureSpec(name=f"z{i}", family="zipf", params={"a": 2.0, "max_value": 200.0}, dtype="int", standardize=False) for i in range(3))
    weights = {f"z{i}": float(0.02 * (0.85**i)) for i in range(3)}
    return DatasetSpec(
        name="zipf_levels",
        n_samples=n_samples,
        root_seed=seed,
        features=informative + tuple(FeatureSpec(name=f"zp{i}", family="zipf", params={"a": 2.0, "max_value": 200.0}, dtype="int", standardize=False) for i in range(3)) + probes(n_noise),
        targets=(TargetSpec(name="y", prevalence=0.4, link=LinkSpec(kind="logistic", coefficients=weights), calibrate_to=CeilingTarget(metric="auc", value=ceiling)),),
        edges=tuple(EdgeSpec(source=column, target="y") for column in weights),
        provenance={"family": "mixed_types", "purpose": "power-law level frequencies: a long tail with almost no rows per level"},
    )
