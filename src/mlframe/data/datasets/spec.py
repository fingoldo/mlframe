"""Declarative specification of a synthetic dataset: what to draw, how to link it, how to corrupt it.

A :class:`DatasetSpec` is the complete, content-hashable, serialisable input to the generator. Nothing about a
dataset may be decided inside the generator that is not written down here: a knob that lives only in code
is a knob nobody can audit, and the benchmark this package feeds is run by the author of one of the arms
it scores.

Design decisions worth stating, because they are easy to undo by accident:

**Only ``typing`` annotation forms.** Every model below annotates with ``Optional[X]``, ``Dict[str, X]``,
``Tuple[X, ...]`` and never with PEP-604 ``X | None`` or the bare builtin generics ``dict[...]`` /
``tuple[...]``. pydantic v2 EVALUATES annotations at class-creation time, so a PEP-604 annotation in a
``BaseModel`` raises ``TypeError`` on import under Python 3.9 - which this project's ``requires-python``
still allows and which CI genuinely runs. mypy here targets 3.10 and will NOT flag it, and a frozen
``dataclass`` under ``from __future__ import annotations`` would not either, which is exactly why the
mistake survives review. The AST guard in ``tests/data/datasets/test_datasets_spec_py39_annotations.py``
is the mechanical check.

**Free knobs are priors, not points.** A hand-picked ``corr=0.9`` is a hand-picked verdict; MI-based and
correlation-based rankings agree at high SNR and separate in a narrow band, so choosing the point chooses
the winner. Every knob the plan wants swept is typed ``Union[float, Prior]`` (:data:`Knob`) so a scenario
can declare ``corr ~ U(0.5, 0.99)`` and have the aggregate integrate over it.

**``FeatureSpec.cost`` exists from day one.** The Pareto cost/quality leg of the benchmark needs it, and
retrofitting a field onto the spec changes its hash and therefore invalidates every cached dataset.

**Corruptions must be able to update ``true_prob``.** :class:`NoiseSpec` carries an explicit
``true_prob_update`` tag rather than leaving it to the generator to notice: a corruption that cannot say
how it transforms the true conditional probability destroys the Bayes ceiling, and the ceiling is the
only thing that makes "regret" mean anything.

The spec layer is a leaf: it imports nothing from this package. :mod:`mlframe.data.datasets.ground_truth`
imports it (for :class:`GateSpec`), never the other way round.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Distribution families a Prior may declare. Kept as a Literal rather than an Enum so a spec round-trips
# through JSON as plain strings (the registry lock file stores spec hashes, not pickles).
PriorKind = Literal["uniform", "loguniform", "uniform_int", "choice"]

#: Edge semantics the SCM layer understands. ``direct`` is plain structural causation; the remaining kinds
#: annotate relationships whose ROLE the raw arrow direction cannot determine on its own (a redundant copy
#: and a mediator are both "an arrow from a feature", yet they score completely differently).
EdgeKind = Literal[
    "direct",
    "latent",
    "proxy",
    "instrument",
    "redundant_exact",
    "redundant_noisy",
    "conditional_only",
    "shift",
]


class _DatasetSubSpec(BaseModel):
    """Shared base for every spec model: unknown fields are an error and instances are immutable.

    ``extra="forbid"`` turns a typo'd knob into a construction-time exception instead of a silently
    ignored field that makes a scenario quietly different from what its author wrote; ``frozen=True``
    makes a spec immutable, so a scenario cannot be edited after the run that used it. Same
    configuration, and for the same reasons, as
    ``mlframe.feature_selection.filters.mrmr._mrmr_config_dataclasses._MRMRSubConfig``.

    Cache keys go through :meth:`content_hash`, not through ``hash()``: a model carrying a ``Dict`` field
    is unhashable even when frozen, and a hash derived from unsorted JSON is not stable across processes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def content_hash(self) -> str:
        """Return a stable content hash of this spec, suitable as a dataset-cache key.

        Sorted-key JSON is what makes the hash stable: dict ordering is insertion order, so two specs that
        declare the same parameters in a different order must still key the same cached dataset.

        Default-valued fields are excluded, which is what keeps the hash a statement about the BED rather
        than about the spec language. Adding a new field -- a nonlinear basis term, a cost, a missingness
        knob -- otherwise moves the hash of every spec in the registry at once, and the scenario lock then
        reports twelve beds as structurally changed when none of them were. That is not a harmless false
        positive: the lock exists so that a bed edited after its results were seen is visible in the diff,
        and a check that cries wolf on every unrelated field addition is one nobody reads.

        The cost is that a spec explicitly setting a field to its default hashes the same as one leaving
        it out. Those two specs generate identical data, so they are the same bed.

        Returns:
            A 32-character hex digest of the canonicalised JSON form.
        """
        canonical = json.dumps(self.model_dump(mode="json", exclude_defaults=True), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


class Prior(_DatasetSubSpec):
    """A declared distribution over one free knob, so the sweep is auditable instead of hand-picked.

    Exactly one of the two parameterisations is used: ``low``/``high`` for the continuous and integer
    families, ``choices`` for ``"choice"``. ``loguniform`` requires a strictly positive support.
    """

    kind: PriorKind
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Tuple[float, ...]] = None

    @model_validator(mode="after")
    def _check_support(self) -> "Prior":
        """Reject parameterisations that do not describe a sampleable distribution.

        Returns:
            ``self``, unchanged, when the support is well formed.

        Raises:
            ValueError: If a bounded family is missing bounds, the bounds are not ordered, a log-uniform
                support touches zero, or ``"choice"`` is given an empty / absent option list.
        """
        if self.kind == "choice":
            if not self.choices:
                raise ValueError("Prior(kind='choice') requires a non-empty `choices` tuple")
            return self
        if self.choices is not None:
            raise ValueError(f"`choices` is only meaningful for kind='choice', not {self.kind!r}")
        if self.low is None or self.high is None:
            raise ValueError(f"Prior(kind={self.kind!r}) requires both `low` and `high`")
        if not self.low < self.high:
            raise ValueError(f"Prior bounds must satisfy low < high, got low={self.low}, high={self.high}")
        if self.kind == "loguniform" and self.low <= 0.0:
            raise ValueError(f"Prior(kind='loguniform') needs a strictly positive support, got low={self.low}")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one value from the declared distribution.

        Args:
            rng: Generator to draw from, normally obtained from
                :func:`mlframe.data.datasets._rng.stream_for` so the draw is addressed by name.

        Returns:
            The sampled value as a float (integer families return an integral float, so a knob's type does
            not change depending on which prior a scenario declared for it).
        """
        if self.kind == "choice":
            options = self.choices or ()
            return float(options[int(rng.integers(0, len(options)))])
        low = float(self.low if self.low is not None else 0.0)
        high = float(self.high if self.high is not None else 1.0)
        if self.kind == "uniform":
            return float(rng.uniform(low, high))
        if self.kind == "uniform_int":
            return float(rng.integers(int(low), int(high) + 1))
        return float(np.exp(rng.uniform(float(np.log(low)), float(np.log(high)))))


#: A knob that may be pinned to a value or declared as a distribution to integrate over.
Knob = Union[float, Prior]


def resolve_knob(knob: Knob, rng: np.random.Generator) -> float:
    """Return a knob's value, drawing it when the scenario declared a distribution rather than a point.

    Args:
        knob: Either a pinned value or a :class:`Prior` to integrate over.
        rng: Stream the prior is drawn from; unused for a pinned value, so a caller can pass any stream.

    Returns:
        The resolved value as a float.
    """
    if isinstance(knob, Prior):
        return knob.sample(rng)
    return float(knob)


class GateSpec(_DatasetSubSpec):
    """A region of one column over which a regional effect is active.

    Referenced from ground truth (``FeatureTruth.region``) so a feature that is informative on 30% of the
    input space is recorded as regional rather than as a weak global effect - the two are different
    findings and only one of them is a failure of the selector.
    """

    column: str = Field(min_length=1)
    low: Optional[float] = None
    high: Optional[float] = None
    fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_interval(self) -> "GateSpec":
        """Reject an inverted or wholly unspecified interval.

        Returns:
            ``self``, unchanged, when the gate describes a usable region.

        Raises:
            ValueError: If the gate describes no region at all, mixes the two ways of describing one, or
                gives an inverted interval.
        """
        bounded = self.low is not None or self.high is not None
        if not bounded and self.fraction is None:
            raise ValueError(f"GateSpec on {self.column!r} needs either `fraction` or at least one of `low` / `high`")
        # The two forms are alternatives, not layers. An explicit interval names a region in the column's
        # own units; a fraction names the top share of its distribution, which is what keeps a region the
        # same SIZE across marginal families. Accepting both would leave it undefined which one applies,
        # and the earlier version of this validator rejected the fraction form outright -- making it
        # unreachable, and with it the heteroscedastic path in the generator, whose only trigger is a
        # region declared as a fraction.
        if bounded and self.fraction is not None:
            raise ValueError(f"GateSpec on {self.column!r} declares both `fraction` and an explicit interval; they are alternative ways to name a region, so pick one")
        if self.low is not None and self.high is not None and not self.low < self.high:
            raise ValueError(f"GateSpec bounds must satisfy low < high, got low={self.low}, high={self.high}")
        return self


class FeatureSpec(_DatasetSubSpec):
    """One observed column: its marginal family, its declared type, and what it costs to acquire.

    ``cost`` is present from the first version deliberately. The cost/quality Pareto leg of the benchmark
    needs it, and adding it later would change every spec hash and invalidate every cached dataset.
    ``levels`` must be given for categorical columns because polars re-derives a ``Categorical``
    dictionary on every slice, so a train/test split of one generated frame yields two incompatible
    dictionaries; the levels are known to the generator by construction, so the column is built as a
    ``pl.Enum`` with this explicit list.
    """

    name: str = Field(min_length=1)
    family: str = Field(default="normal", min_length=1)
    params: Dict[str, float] = Field(default_factory=dict)
    dtype: Literal["float", "int", "category"] = "float"
    levels: Optional[Tuple[str, ...]] = None
    cost: float = Field(default=1.0, ge=0.0)
    standardize: bool = True
    outlier_fraction: Knob = 0.0
    #: Round the column onto this many equally spaced levels before standardising, or ``None`` to leave it
    #: continuous. Quantisation is a real and common corruption -- a sensor reporting to two decimal
    #: places, a price rounded to the nearest cent, an age in whole years -- and it is the one that breaks
    #: equal-mass binning specifically: once a column has fewer distinct values than the estimator wants
    #: bins, the bin edges collapse onto ties and the binning no longer carries the mass it was asked for.
    quantize_levels: Optional[int] = Field(default=None, ge=2)

    @model_validator(mode="after")
    def _check_levels(self) -> "FeatureSpec":
        """Tie ``levels`` to ``dtype='category'`` in both directions.

        Returns:
            ``self``, unchanged, when the declaration is consistent.

        Raises:
            ValueError: If a categorical column has no levels, or a numeric column has some.
        """
        if self.dtype == "category" and not self.levels:
            raise ValueError(f"categorical feature {self.name!r} must declare its `levels` explicitly")
        if self.dtype != "category" and self.levels:
            raise ValueError(f"feature {self.name!r} declares levels but dtype is {self.dtype!r}")
        return self


class CopulaSpec(_DatasetSubSpec):
    """A dependence structure imposed on a group of columns, separate from their marginals.

    Correlation and tail dependence are different properties and only a copula separates them. A Cholesky
    factor -- the usual way to make correlated synthetic columns -- is a Gaussian copula, which has zero tail
    dependence at every correlation below one: conditional on one column being extreme, the chance the other
    is extreme too vanishes as the threshold moves out. Declaring the family explicitly makes that a choice
    rather than an accident, and makes the alternative expressible.
    """

    columns: Tuple[str, ...] = Field(min_length=2)
    family: Literal["gaussian", "t", "clayton"] = "gaussian"
    rho: Knob = 0.7
    df: Knob = 4.0
    theta: Knob = 2.0
    margin: Literal["normal", "uniform"] = "normal"


class LatentSpec(_DatasetSubSpec):
    """An unobserved variable plus the observed reflections drawn from it.

    ``distinct_sd`` is the private per-reflection deviation. It is the whole point of the
    ``latent_replicates_private_delta`` scenario: with ``distinct_sd > 0`` the reflections carry private
    information that cluster-averaging destroys, so a redundancy-collapsing selector loses signal that an
    honest one keeps. It is a :data:`Knob` because the interesting question is where on the sweep the
    collapse starts costing, not what happens at one chosen value.
    """

    name: str = Field(min_length=1)
    family: str = Field(default="normal", min_length=1)
    reflections: Tuple[str, ...] = ()
    loadings: Tuple[float, ...] = ()
    distinct_sd: Knob = 0.0
    noise_sd: Knob = 0.0

    @model_validator(mode="after")
    def _check_loadings(self) -> "LatentSpec":
        """Require one loading per reflection when loadings are given at all.

        Returns:
            ``self``, unchanged, when the two tuples are compatible.

        Raises:
            ValueError: If loadings are present but their count differs from the reflection count.
        """
        if self.loadings and len(self.loadings) != len(self.reflections):
            raise ValueError(f"latent {self.name!r} has {len(self.reflections)} reflections but {len(self.loadings)} loadings")
        return self


class BasisTerm(_DatasetSubSpec):
    """One named nonlinear term in the link, stated rather than approximated.

    The additive and multiplicative parts of a link cover a lot, but not the shapes the literature's
    reference beds are DEFINED by: Friedman's first function opens with ``10 sin(pi x1 x2)`` and follows it
    with ``20 (x3 - 0.5)^2``, and a benchmark that substituted a product for the sine would no longer be
    comparable to any published number -- which is the only reason those beds are worth having.

    Each kind is a closed form with a stated derivative behaviour, because that is what decides which
    methods can see it:

    * ``sin_product`` -- ``sin(frequency * prod(columns))``. Non-monotone in every operand and, over a
      symmetric input range, weakly correlated with each; the operands are individually visible only
      because the range is one period or less.
    * ``centered_square`` -- ``(x - center)^2``. Symmetric about the centre, so a linear statistic scores
      it at zero while a binned one finds it immediately. The cleanest separator between the two families
      there is.
    * ``abs_deviation`` -- ``|x - center|``. The same symmetry with a kink instead of curvature, which
      distinguishes a method that needs smoothness from one that needs only non-monotonicity.
    * ``ratio`` -- ``numerator / max(|denominator|, floor)`` over exactly two columns. Neither operand is
      informative alone at a fixed marginal, and the floor keeps a near-zero denominator from producing a
      single row that dominates every scale computed downstream.
    * ``identity`` -- the column itself, so a bed can express a weighted linear term in the same list as
      its nonlinear ones instead of splitting one formula across two fields.
    * ``group_effect`` -- one value drawn per DISTINCT LEVEL of a column and shared by every row of that
      level. It is the only term here that makes rows non-independent, which is the point: two rows of the
      same group share an offset no feature explains, so a random split puts correlated rows on both
      sides of it and every estimate taken across that split is optimistic.
    """

    kind: Literal["sin_product", "centered_square", "abs_deviation", "ratio", "identity", "group_effect"] = "identity"
    columns: Tuple[str, ...] = Field(min_length=1)
    weight: float = 1.0
    #: Kind-specific constants: ``frequency`` for ``sin_product``, ``center`` for the centred kinds,
    #: ``floor`` for ``ratio``. Unknown keys are ignored rather than rejected, so a term can carry a note
    #: for a reader without changing what it computes.
    params: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_arity(self) -> "BasisTerm":
        """Reject an operand count the kind cannot use.

        Returns:
            ``self``, unchanged, when the arity is usable.

        Raises:
            ValueError: If the kind needs exactly one or exactly two columns and got something else. A
                silently ignored extra operand would change the formula from the one the bed's name
                promises, which for a reference bed is the whole of its value.
        """
        if self.kind in ("centered_square", "abs_deviation", "identity", "group_effect") and len(self.columns) != 1:
            raise ValueError(f"basis term {self.kind!r} takes exactly one column; got {self.columns!r}")
        if self.kind == "ratio" and len(self.columns) != 2:
            raise ValueError(f"basis term 'ratio' takes exactly two columns (numerator, denominator); got {self.columns!r}")
        if self.kind == "sin_product" and len(self.columns) < 1:
            raise ValueError("basis term 'sin_product' needs at least one column")
        return self


class LinkSpec(_DatasetSubSpec):
    """How the features combine into the latent score that drives the target.

    ``coefficients`` covers the additive part; ``interactions`` lists the multiplicative / parity terms as
    tuples of column names with their own weights in ``interaction_weights``. ``scale`` is the knob the
    ceiling calibration bisects on, which is why it is a :data:`Knob`: a coefficient of 0.8 means something
    completely different under a logistic link than under a parity gate, so difficulty is set by calibrating
    to a target ceiling rather than by choosing coefficients.
    """

    kind: Literal["linear", "logistic", "parity", "threshold", "polynomial", "tail_gate"] = "logistic"
    coefficients: Dict[str, float] = Field(default_factory=dict)
    interactions: Tuple[Tuple[str, ...], ...] = ()
    interaction_weights: Tuple[float, ...] = ()
    #: Named nonlinear terms, added to the additive and interaction parts. Empty for every bed that does
    #: not need one, so adding the field leaves every existing spec hash untouched.
    basis_terms: Tuple[BasisTerm, ...] = ()
    intercept: float = 0.0
    scale: Knob = 1.0
    region: Optional[GateSpec] = None
    # For ``kind="tail_gate"``: the per-column quantile every operand of a term must exceed for the term to
    # fire. The signal then lives in the JOINT tail, which is where a Gaussian copula has none and an
    # equal-mass binned estimator has one cell.
    tail_quantile: float = Field(default=0.9, gt=0.0, lt=1.0)
    # Which tail a ``tail_gate`` term fires in. ``both`` cancels the marginal channel by symmetry and is
    # what a bed wants when it is testing joint structure. A one-sided gate is for the asymmetric copulas:
    # Clayton's dependence lives in the LOWER tail only, so including the upper tail -- where it is
    # asymptotically independent -- dilutes exactly the contrast such a bed exists to measure.
    tail_direction: Literal["both", "lower", "upper"] = "both"

    @model_validator(mode="after")
    def _check_interactions(self) -> "LinkSpec":
        """Validate interaction arity and the weight-count match.

        Returns:
            ``self``, unchanged, when every interaction term is usable.

        Raises:
            ValueError: If an interaction has fewer than two operands, or the weight count disagrees.
        """
        for term in self.interactions:
            if len(term) < 2:
                raise ValueError(f"interaction term {term!r} needs at least two operands")
        if self.interaction_weights and len(self.interaction_weights) != len(self.interactions):
            raise ValueError(f"{len(self.interactions)} interaction terms but {len(self.interaction_weights)} weights")
        return self


class NoiseSpec(_DatasetSubSpec):
    """A corruption applied to the target, together with how it transforms ``true_prob``.

    The generator refuses to apply a corruption that cannot supply a ``true_prob`` update, which is what
    keeps the Bayes ceiling exact by construction instead of merely plausible. The update rules are all
    cheap: a uniform label flip is ``p' = p(1 - f) + (1 - p) f / (K - 1)``; a feature-dependent flip is the
    same formula applied elementwise; post-hoc binning pushes the noise CDF through the bin edges; and
    injecting outliers into a FEATURE before the link does not change ``p`` at all (after the link it is
    forbidden, because there is then no update to supply).
    """

    kind: Literal["uniform_flip", "feature_dependent_flip", "binning", "none"] = "none"
    rate: Knob = 0.0
    true_prob_update: Literal[
        "uniform_flip",
        "feature_dependent_flip",
        "binning_pushforward",
        "identity_pre_link",
    ] = "identity_pre_link"
    gate: Optional[GateSpec] = None

    @model_validator(mode="after")
    def _check_update_available(self) -> "NoiseSpec":
        """Enforce the ceiling invariant at spec-construction time rather than mid-generation.

        Returns:
            ``self``, unchanged, when the declared corruption carries a matching ``true_prob`` update.

        Raises:
            ValueError: If an active corruption claims the identity update, or a feature-dependent flip
                arrives without the gate that says where it applies.
        """
        if self.kind != "none" and self.true_prob_update == "identity_pre_link":
            raise ValueError(
                f"corruption kind={self.kind!r} must declare how it updates true_prob; " "a corruption without an update destroys the Bayes ceiling"
            )
        if self.kind == "feature_dependent_flip" and self.gate is None:
            raise ValueError("feature_dependent_flip needs a `gate` describing where the flip rate applies")
        return self


class CeilingTarget(_DatasetSubSpec):
    """The achievable-performance point a scenario calibrates its link scale to.

    Calibrating to a ceiling rather than picking coefficients is what makes difficulty comparable across
    link families, and what turns the headline into a recovery-vs-ceiling CURVE (the crossing point is a
    finding) instead of a point estimate at a chosen SNR (a choice).
    """

    metric: Literal["auc", "logloss", "accuracy"] = "auc"
    value: float = Field(ge=0.0, le=1.0)
    tolerance: float = Field(default=0.005, gt=0.0, le=0.5)


class TargetSpec(_DatasetSubSpec):
    """One target column: its type, its link, its corruption, and its difficulty calibration."""

    name: str = Field(default="y", min_length=1)
    kind: Literal["binary", "multiclass", "ordinal", "count", "multilabel"] = "binary"
    n_classes: int = Field(default=2, ge=2)
    prevalence: Knob = 0.5
    link: LinkSpec = Field(default_factory=LinkSpec)
    noise: NoiseSpec = Field(default_factory=NoiseSpec)
    calibrate_to: Optional[CeilingTarget] = None

    @field_validator("n_classes")
    @classmethod
    def _check_n_classes(cls, value: int) -> int:
        """Reject a class count that no supported target kind can use.

        Args:
            value: The proposed number of classes.

        Returns:
            ``value``, unchanged.

        Raises:
            ValueError: If more than 1000 classes are requested, which is a spec typo in every realistic
                scenario and would otherwise allocate a huge probability matrix.
        """
        if value > 1000:
            raise ValueError(f"n_classes={value} is implausible for a benchmark scenario")
        return value

    @model_validator(mode="after")
    def _check_binary_arity(self) -> "TargetSpec":
        """Keep ``kind`` and ``n_classes`` consistent.

        Returns:
            ``self``, unchanged, when the arity matches the declared kind.

        Raises:
            ValueError: If a binary target declares more than two classes.
        """
        if self.kind == "binary" and self.n_classes != 2:
            raise ValueError(f"binary target {self.name!r} cannot have n_classes={self.n_classes}")
        return self


class EdgeSpec(_DatasetSubSpec):
    """One arrow of the structural causal model, as declared in the spec.

    This is the pydantic-validated INPUT form. Its counterpart on the truth side,
    ``ground_truth.Edge``, is a frozen dataclass with the same fields; the two are deliberately separate
    because ground truth also hosts ndarrays, for which pydantic validation would either be skipped
    (losing the benefit) or copy the array (losing the memory discipline this repository holds itself to).
    """

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: EdgeKind = "direct"
    weight: float = 1.0

    @model_validator(mode="after")
    def _check_not_self_loop(self) -> "EdgeSpec":
        """Reject self-loops, which no acyclic SCM can host.

        Returns:
            ``self``, unchanged, when source and target differ.

        Raises:
            ValueError: If the edge points a node at itself.
        """
        if self.source == self.target:
            raise ValueError(f"self-loop on {self.source!r}: an SCM edge must connect two distinct nodes")
        return self


class MissingnessSpec(_DatasetSubSpec):
    """Holes punched in an OBSERVED column, after the link has already consumed the complete values.

    The ordering is the whole design, and it is what keeps the Bayes ceiling honest. ``true_prob`` is the
    law relating the COMPLETE feature values to the target, and that law is what the generator calibrated
    and what the truth record reports. Masking values afterwards does not change it; it changes what a
    method can SEE. So the ceiling on the emitted frame is lower than the recorded one, by an amount that
    depends on the mechanism, and the truth carries a caveat saying so rather than a number nobody
    computed.

    Three mechanisms, which are genuinely different problems and not degrees of one:

    * ``mcar`` -- missing completely at random. The observed rows are a fair sample, so a complete-case
      analysis is unbiased and only loses power. The easy case.
    * ``mar`` -- missingness depends on ANOTHER observed column. A complete-case analysis is biased, but
      the bias is recoverable because the thing driving it is in the data.
    * ``mnar`` -- missingness depends on the masked column's OWN value: the largest values go missing
      precisely because they are large. Nothing observable identifies the bias, and a method that imputes
      from the observed distribution imputes from the wrong one.
    """

    column: str = Field(min_length=1)
    mechanism: Literal["mcar", "mar", "mnar"] = "mcar"
    rate: Knob = 0.2
    #: The column whose value decides missingness under ``mar``. Ignored by the other two mechanisms:
    #: ``mcar`` depends on nothing and ``mnar`` depends on the masked column itself.
    driver: Optional[str] = None

    @model_validator(mode="after")
    def _check_driver(self) -> "MissingnessSpec":
        """Require a driver for the one mechanism that needs one, and refuse a self-referential MAR.

        Returns:
            ``self``, unchanged, when the declaration is usable.

        Raises:
            ValueError: If ``mar`` has no driver, or names the masked column as its own driver -- which is
                ``mnar`` wearing the wrong label, and the difference decides whether the bias is
                recoverable at all.
        """
        if self.mechanism == "mar":
            if not self.driver:
                raise ValueError(f"missingness on {self.column!r} declares 'mar' but names no driver column")
            if self.driver == self.column:
                raise ValueError(f"missingness on {self.column!r} declares 'mar' driven by itself, which is 'mnar'")
        return self


class DatasetSpec(_DatasetSubSpec):
    """The complete, content-hashable description of one synthetic dataset.

    ``root_seed`` addresses every stream in the dataset through
    :func:`mlframe.data.datasets._rng.stream_for`; the reserved-range discipline of the benchmark
    (development seeds vs report-only seeds) is enforced on the values scenarios put here, not inside
    this model, so that a spec remains a pure description.
    """

    name: str = Field(min_length=1)
    n_samples: int = Field(ge=1)
    root_seed: int = Field(default=0, ge=0)
    features: Tuple[FeatureSpec, ...] = ()
    copulas: Tuple[CopulaSpec, ...] = ()
    latents: Tuple[LatentSpec, ...] = ()
    targets: Tuple[TargetSpec, ...] = ()
    missingness: Tuple[MissingnessSpec, ...] = ()
    edges: Tuple[EdgeSpec, ...] = ()
    provenance: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_names_and_references(self) -> "DatasetSpec":
        """Reject duplicate node names and edges referencing nodes the spec never declares.

        Catching a dangling edge here rather than in the SCM layer means a scenario author sees the typo
        at spec-construction time, before a single row has been generated.

        Returns:
            ``self``, unchanged, when every name is unique and every edge endpoint is declared.

        Raises:
            ValueError: On a duplicated node name, or an edge endpoint that is neither a feature, a
                latent, nor a target.
        """
        names = [feature.name for feature in self.features] + [latent.name for latent in self.latents] + [target.name for target in self.targets]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate node names in spec {self.name!r}: {duplicates}")
        known = set(names)
        for edge in self.edges:
            unknown = {edge.source, edge.target} - known
            if unknown:
                raise ValueError(f"edge {edge.source!r}->{edge.target!r} references undeclared node(s) {sorted(unknown)}")
        return self

    def feature_names(self) -> Tuple[str, ...]:
        """Return the observed feature names in declaration order.

        Returns:
            Feature names in the order the spec lists them, which is also the column order of the
            generated frame; the generator never shuffles columns, because a shuffle is how
            ``sklearn.make_classification`` loses its own ground truth.
        """
        return tuple(feature.name for feature in self.features)

    def total_cost(self) -> float:
        """Return the summed acquisition cost of every declared feature.

        Returns:
            The cost of the all-features arm, which is the denominator of the cost/quality Pareto plot
            and the reference point for the "does FS pay for itself" null hypothesis.
        """
        return float(sum(feature.cost for feature in self.features))
