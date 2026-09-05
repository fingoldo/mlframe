"""Ground truth: what the generator knows about a dataset, and what the benchmark scores against.

Why this is not ``informative: Tuple[str, ...]``
------------------------------------------------
A flat list of "informative" columns is not a definition, and on the flagship scenarios of this very
benchmark it is ambiguous in at least five ways at once:

* the operands of a parity (XOR) target have marginal mutual information of exactly zero, yet the target
  is a deterministic function of them;
* a regionally gated feature is informative on 30% of the input space and inert elsewhere;
* a mediator is screened off by the variable it mediates through, so it is informative until you condition
  and useless afterwards;
* five redundant copies are individually sufficient and jointly carry one bit;
* a DESCENDANT of the target is strongly predictive while having exactly zero causal effect.

The direct consequence, and the reason this matters rather than being pedantry: the natural verification
criterion "true MI ranks the informative features above the probes, Spearman >= 0.9" CANNOT pass on the
parity scenario the suite is built around. The criterion is not merely hard there, it is false. So truth
is stored as a per-feature ROLE plus several NAMED TARGET SETS, and every score declares which set it is
scored against.

Three target sets, all reported, one pre-registered as primary
--------------------------------------------------------------
=====================  ==========================================  =================================
set                    definition                                  uniqueness
=====================  ==========================================  =================================
``markov_blanket``     parents, children and spouses of Y          unique under faithfulness+positivity
``minimal_sufficient`` smallest subset attaining the ceiling       NOT unique: an equivalence partition
``causal_parents``     non-zero direct causal effect               unique, but not identifiable from
                                                                   observational data alone
=====================  ==========================================  =================================

``markov_blanket`` is primary because feature selection in this repository feeds predictive pipelines, the
Markov blanket IS the optimal predictive set, and it is unique - so it is a valid scoring key.
``minimal_sufficient`` is scored by CLASS COVERAGE (did the arm take exactly one representative of each
equivalence class?) rather than by set equality, because set equality against a non-unique target
penalises a correct answer for choosing a different representative. ``causal_parents`` is reported for
causal scenarios only and never in a headline.

That this distinction changes winners rather than wording is easy to demonstrate: on a spouse/collider
scenario (``Y -> C``, ``S -> C``, ``S`` independent of ``Y``), a marginal-MI selector takes ``C`` first and
is RIGHT under ``markov_blanket``, while an arm that refuses descendants of the target wins under
``causal_parents``. Same data, same arms, opposite verdicts, decided entirely by a definition that a
benchmark can forget to state.

Cheap truth vs expensive truth
------------------------------
Structural truth - roles, edges, redundancy groups, target sets - is derived from the graph and is always
present. Statistical truth - the Bayes ceiling and the reference mutual information - is a Monte-Carlo
estimate over a large oracle sample and costs orders of magnitude more, so it lives behind the lazy
accessors :meth:`GroundTruth.ceiling` and :meth:`GroundTruth.mi_reference`, never in the constructor. Those
accessors are declared here and implemented in a later changeset; the memo cache they will populate is
already in place so that adding the implementation does not change this class's shape.

Estimates are never bare floats. :class:`MIEstimate` carries ``(value, estimator, n_bins, n_samples)`` and
:class:`Ceiling` carries its standard error, its method and what it is conditional on, because a bare
float is exactly what lets a backend swap through unnoticed - and the reference estimator is required to
come from a family that no scored arm uses, or "rank correlation with the reference MI" degenerates into
measuring a method's agreement with itself.

This module is a frozen ``dataclass`` layer, not pydantic: it carries ndarrays, and pydantic v2 validation
of an ndarray either opts out of validation (losing the benefit) or copies the array (losing the memory
discipline this repository holds itself to).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from mlframe.data.datasets.spec import GateSpec

#: Name of the pre-registered primary target set. Scoring code reads this constant rather than
#: re-spelling the string, so the primary set cannot drift between the report and the registry lock.
PRIMARY_TARGET_SET = "markov_blanket"


class FeatureRole(str, Enum):
    """What a column IS with respect to the target, derived from the structural causal model.

    Inherits :class:`str` so a role serialises to a plain string in a manifest and compares equal to its
    own name without a conversion step.
    """

    CAUSAL_PARENT = "causal_parent"
    CAUSAL_ANCESTOR = "causal_ancestor"
    MEDIATOR = "mediator"
    SPOUSE = "spouse"
    CHILD = "child"
    PROXY = "proxy"
    INSTRUMENT = "instrument"
    # Conditioning on an M-collider OPENS a path between two otherwise independent variables, so the
    # correct action is to NOT select it - the one role where inclusion is actively harmful rather than
    # merely wasteful, and the structural vulnerability of every greedy conditional selector.
    M_COLLIDER = "m_collider"
    REDUNDANT_EXACT = "redundant_exact"
    REDUNDANT_NOISY = "redundant_noisy"
    # Zero marginal association, non-zero conditional association: the parity operands.
    CONDITIONAL_ONLY = "conditional_only"
    REGIONAL = "regional"
    SHIFT_NUISANCE = "shift_nuisance"
    PROBE = "probe"


@dataclass(frozen=True)
class Edge:
    """One arrow of the structural causal model, on the truth side.

    The counterpart input form is ``spec.EdgeSpec``; the two are separate because this side lives in a
    dataclass layer that also hosts ndarrays. Conversion is one function in ``_scm``.
    """

    source: str
    target: str
    kind: str = "direct"
    weight: float = 1.0


@dataclass(frozen=True)
class MIEstimate:
    """One mutual-information number together with everything needed to defend it.

    A bare float invites the silent-backend-swap failure documented in ``CLAUDE.md``: two numbers computed
    with different estimators, bin counts or sample sizes are not comparable, and nothing in a float says
    so. ``estimator`` must name a family disjoint from the MI backends of the arms being scored.
    """

    value: float
    estimator: str
    n_bins: Optional[int] = None
    n_samples: Optional[int] = None
    se: Optional[float] = None


@dataclass(frozen=True)
class MIBundle:
    """Several estimates of the same quantity, plus the reliability verdict their spread implies.

    Binning a huge sample is a biased, variable ESTIMATE, not truth - hence ``mi_reference``, not
    ``true_mi``. When the spread across estimators exceeds the effect being measured, the honest output is
    ``unreliable=True`` and a SUPPRESSED rank-correlation metric, not a number nobody can defend.
    """

    estimates: Tuple[MIEstimate, ...]
    exact: Optional[MIEstimate] = None
    unreliable: bool = False
    caveats: Tuple[str, ...] = ()

    def spread(self) -> float:
        """Return the max-minus-min spread across the bundled estimates.

        Returns:
            The spread, or ``0.0`` when fewer than two estimates are present (a single estimate has no
            measurable disagreement, which is not the same as agreement and is why ``unreliable`` is a
            separate flag rather than a threshold on this value).
        """
        if len(self.estimates) < 2:
            return 0.0
        values = [estimate.value for estimate in self.estimates]
        return float(max(values) - min(values))


@dataclass(frozen=True)
class Ceiling:
    """Best achievable performance on a dataset, with the provenance that makes it usable.

    ``method`` distinguishes a closed form from a Monte-Carlo estimate over the realised ``true_prob``.
    The rule the plan settles on inverts the naive one: MC is the PRODUCT (it always exists, because every
    corruption is required to supply a ``true_prob`` update), and the closed form, where one exists, is the
    CROSS-CHECK - MC must agree with it to within four standard errors.

    ``conditional_on`` separates the honest comparator (the ceiling on the realised X that the arms
    actually saw) from the population ceiling. ``caveats`` carries the cases where MC is biased or noisy:
    at very low prevalence the oracle sample must be sized by the MINORITY count rather than by total rows,
    a deterministic gate with ``p`` in ``{0, 1}`` makes the sample log-loss ceiling zero and attainable only
    in the limit, and under MNAR the observed-data ceiling differs from the complete-data one.
    """

    value: float
    se: float
    method: str
    conditional_on: str
    n_oracle: int
    metric: str = "auc"
    caveats: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RedundancyGroup:
    """A set of columns spanning one shared subspace, with the rank of that span.

    ``rank`` is what separates the two cases the benchmark keeps confusing: five exact copies span a rank-1
    subspace and one representative suffices, whereas replicates carrying private deltas span a
    higher-rank subspace, so collapsing them to their mean DESTROYS information a correct selector keeps.
    ``exact`` records which of the two this group is.
    """

    members: Tuple[str, ...]
    rank: int
    exact: bool
    source: Optional[str] = None
    caveats: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetSet:
    """One named answer key, its definition, and whether it is unique.

    ``classes`` is the equivalence partition: for a unique set every member is its own singleton class,
    while for ``minimal_sufficient`` each class holds interchangeable representatives and an arm is scored
    by covering each class exactly once rather than by matching ``members`` exactly.
    """

    name: str
    members: Tuple[str, ...]
    classes: Tuple[Tuple[str, ...], ...]
    definition: str
    unique: bool
    caveats: Tuple[str, ...] = ()

    def covers(self, selected: Tuple[str, ...]) -> bool:
        """Report whether ``selected`` takes at least one representative of every equivalence class.

        Args:
            selected: The columns an arm selected.

        Returns:
            ``True`` when every class intersects ``selected``. This is the correct scoring predicate for a
            non-unique target set; exact set equality would fail a correct arm for preferring a different
            representative of the same class.
        """
        chosen = set(selected)
        return all(bool(set(equivalence_class) & chosen) for equivalence_class in self.classes)


@dataclass(frozen=True)
class FeatureTruth:
    """Everything known about one column.

    ``conditional_mi_given_rest`` is the operational statement of Markov-blanket membership, and it is the
    form in which the suite's verification criterion is actually true: "conditional MI ranks blanket
    members above probes" holds on parity scenarios, where the marginal-MI version is false by
    construction. Both MI fields are ``Optional`` because they are statistical truth, filled in lazily by
    the oracle rather than at construction.
    """

    role: FeatureRole
    marginal_mi: Optional[MIEstimate] = None
    conditional_mi_given_rest: Optional[MIEstimate] = None
    region: Optional[GateSpec] = None
    source: Optional[str] = None
    r2_to_source: Optional[float] = None
    cost: float = 1.0
    pre_standardization_scale: Optional[float] = None


@dataclass(frozen=True)
class GroundTruth:
    """The full truth record for one generated dataset.

    Structural fields are populated by the SCM layer at generation time and are cheap. Statistical truth is
    reached only through :meth:`ceiling` and :meth:`mi_reference`, which memoise into ``_memo``.

    ``pre_standardization_scale`` on each feature exists because of varsortability: in an additively
    generated SCM the marginal variance grows with topological depth, so sorting features by variance
    recovers the causal order and a "variance sort" control arm would win for a reason that has nothing to
    do with feature selection. The mitigation is to standardise every column to unit variance and record
    the original scale here rather than to discard it.
    """

    features: Dict[str, FeatureTruth]
    redundancy_groups: Tuple[RedundancyGroup, ...] = ()
    target_sets: Dict[str, TargetSet] = field(default_factory=dict)
    graph: Tuple[Edge, ...] = ()
    target_name: str = "y"
    true_prob: Optional[np.ndarray] = None
    true_mean: Optional[np.ndarray] = None
    caveats: Tuple[str, ...] = ()
    # Lazy-accessor memo. Excluded from equality and repr: it is a cache, not state, so two truth records
    # that differ only in what has been computed so far must still compare equal.
    _memo: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def roles(self) -> Dict[str, FeatureRole]:
        """Return the per-column role map.

        Returns:
            A new dict mapping column name to :class:`FeatureRole`; a copy, so a caller cannot mutate the
            truth record through it (the dataclass is frozen, but its dict field is not).
        """
        return {name: truth.role for name, truth in self.features.items()}

    def names_with_role(self, role: FeatureRole) -> Tuple[str, ...]:
        """Return the columns carrying one role, in the record's own feature order.

        Args:
            role: The role to filter by.

        Returns:
            Matching column names as a tuple.
        """
        return tuple(name for name, truth in self.features.items() if truth.role is role)

    def primary_target_set(self) -> TargetSet:
        """Return the pre-registered primary answer key.

        Returns:
            The :class:`TargetSet` named by :data:`PRIMARY_TARGET_SET`.

        Raises:
            KeyError: If the record was built without the primary set, which means the scoring key is
                undefined and a silent fallback to another set would change the winner.
        """
        if PRIMARY_TARGET_SET not in self.target_sets:
            raise KeyError(f"ground truth for target {self.target_name!r} has no {PRIMARY_TARGET_SET!r} target set; " f"present: {sorted(self.target_sets)}")
        return self.target_sets[PRIMARY_TARGET_SET]

    def ceiling(self, metric: str = "auc") -> Ceiling:
        """Return the Bayes ceiling for ``metric`` (lazy; expensive; memoised).

        Monte-Carlo over the realised ``true_prob`` is the product, and any closed form is the cross-check
        that MC must match to within four standard errors. The oracle sample is sized by the minority-class
        count rather than by total rows, because at prevalence 0.01 the standard error of the AUC ceiling
        over a million rows is already comparable to the effects being measured.

        Args:
            metric: Metric the ceiling is expressed in; ``"auc"``, ``"logloss"`` or ``"accuracy"``.

        Returns:
            The computed :class:`Ceiling`, memoised per metric.

        Raises:
            NotImplementedError: Always, for now. The oracle that computes it is a later changeset; the
                accessor and its memo slot are declared here so that adding the implementation does not
                change this class's shape or its callers.
        """
        raise NotImplementedError(
            f"Bayes ceiling for metric={metric!r} is computed by the oracle changeset; "
            "GroundTruth declares the accessor so callers and the memo slot are already in place"
        )

    def mi_reference(self) -> Dict[str, MIBundle]:
        """Return the per-column reference mutual information (lazy; expensive; memoised).

        Deliberately NOT named ``true_mi``: binning a huge sample is a biased, variable estimate. The
        bundle reports an exact value where a closed form exists (jointly Gaussian, fully discrete links,
        or any scenario with independent features and an available ``true_prob``) alongside several
        estimators, and treats their spread as the error bar. The estimator family must be disjoint from
        every scored arm's MI backend.

        Returns:
            Column name to :class:`MIBundle`, memoised.

        Raises:
            NotImplementedError: Always, for now - implemented by the oracle changeset.
        """
        raise NotImplementedError(
            "reference MI is computed by the oracle changeset; GroundTruth declares the accessor so the "
            "estimator-disjointness assertion has a single place to live"
        )
