"""The scenario library: named beds, each declaring what it exists to break.

A scenario is not just a spec. It carries the two things that keep a benchmark honest once its author also
writes its arms:

* ``expected_to_break`` -- the methods the bed is designed to defeat, declared BEFORE the run. A suite where
  every bed is expected to break nothing is a suite that cannot produce a negative result, and the
  meta-tests require each arm to appear in at least two beds' lists.
* ``primary_target_set`` -- which of the three answer keys this bed is scored against. On the causal beds
  the choice decides the winner outright, so leaving it implicit would report a preference as a finding.

The registry is hashed into ``REGISTRY.lock.json``. Adding a scenario after looking at results is allowed --
it is often the right response to a surprise -- but it bumps the lock, shows up in the diff, and is reported
as a post-hoc addition rather than blending into the pre-registered set.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from mlframe.data.datasets.ground_truth import PRIMARY_TARGET_SET
from mlframe.data.datasets.spec import DatasetSpec

from ._causal import mediator_chain_spec, spouse_collider_spec
from ._causal_extra import confounder_spec, instrument_spec, m_bias_spec, proxy_attenuation_spec
from ._corrupted import coarsened_label_spec, feature_dependent_flip_spec, uniform_flip_spec
from ._economics import expensive_signal_spec, graded_redundancy_spec
from ._interactions import parity_plus_decoy_spec, parity_spec
from ._linear import linear_lowdim_spec, linear_spec
from ._marginals import heavy_tail_spec, outlier_contaminated_spec, quantized_spec, zero_inflated_spec
from ._mixed_types import graded_cardinality_spec, id_trap_spec, zipf_levels_spec
from ._observation import concept_shift_spec, covariate_shift_spec, missingness_trio_spec, rare_class_spec
from ._null import null_spec
from ._redundant import exact_redundancy_spec, private_delta_spec
from ._structure import grouped_rows_spec, simpson_reversal_spec
from ._reference import friedman1_spec, friedman2_spec, friedman3_spec, weston_guyon_spec
from ._tails import gaussian_tail_control_spec, tail_dependence_spec, tail_isolation_spec
from ._targets import count_spec, multiclass_spec, ordinal_spec

logger = logging.getLogger(__name__)

__all__ = [
    "Scenario",
    "SCENARIOS",
    "LOCK_PATH",
    "get",
    "names",
    "build_lock",
    "load_lock",
    "lock_differences",
    "scenario_names",
]

LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REGISTRY.lock.json")


@dataclass(frozen=True)
class Scenario:
    """One named bed: how to build it, what it should break, and how it is scored."""

    name: str
    family: str
    builder: Callable[..., DatasetSpec]
    expected_to_break: Tuple[str, ...]
    purpose: str
    primary_target_set: str = PRIMARY_TARGET_SET
    defaults: Dict[str, Any] = field(default_factory=dict)
    #: Whether sorting columns by raw variance recovers this bed's answer key. False everywhere it can be:
    #: a bed where the unsupervised control wins is normally a BROKEN bed, because in an additively
    #: generated SCM the variance grows with depth in topological order and the control then recovers the
    #: causal order without ever looking at the target (Reisach, Seiler and Drton, NeurIPS 2021). Columns
    #: are standardised to unit variance precisely to remove that channel. A bed setting this True is
    #: declaring that its unequal scales are part of the published design rather than an accident, and it
    #: owes the reason in ``varsortable_reason``.
    varsortable: bool = False
    varsortable_reason: str = ""

    def __post_init__(self) -> None:
        """Refuse a varsortability declaration with no stated reason.

        Raises:
            ValueError: When a bed marks itself varsortable without saying why. The flag exempts the bed
                from the suite's strongest anti-rigging check, so an unexplained one is how a broken bed
                would get through.
        """
        if self.varsortable and not self.varsortable_reason.strip():
            raise ValueError(f"scenario {self.name!r} declares varsortable=True without a reason; the flag waives an anti-rigging check")

    def build(self, seed: int = 0, **overrides: Any) -> DatasetSpec:
        """Return this scenario's spec at one seed.

        Args:
            seed: Root seed; the development and reserved ranges are the caller's discipline, not this
                function's, so a scenario stays a pure description.
            **overrides: Builder arguments overriding the registered defaults.

        Returns:
            The dataset specification.
        """
        kwargs = dict(self.defaults)
        kwargs.update(overrides)
        return self.builder(seed=seed, **kwargs)


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        name="null_p100",
        family="null",
        builder=null_spec,
        defaults={"width": 100},
        expected_to_break=("select-fdr", "skb-f", "skb-mi", "univariate-mi", "bandit", "bandit-ensemble", "unsupervised-prescreen"),
        purpose="false-discovery discipline: nothing is relevant, so anything selected is a false positive",
    ),
    Scenario(
        name="null_p1000",
        family="null",
        builder=null_spec,
        defaults={"width": 1000},
        expected_to_break=("select-fdr", "skb-f", "skb-mi", "boruta", "rfecv", "sfm-lgbm", "noise-floor", "bandit", "bandit-ensemble"),
        purpose="the same discipline where the best-looking noise column is genuinely convincing",
    ),
    Scenario(
        name="linear_k5_p50",
        family="linear",
        builder=linear_spec,
        defaults={"n_informative": 5, "n_noise": 45},
        expected_to_break=("variance-sort", "zero-importance", "unsupervised-prescreen"),
        purpose="control: every method should succeed here, except the one that never looks at the target",
    ),
    Scenario(
        name="linear_gaussian_lowdim_n200",
        family="linear",
        builder=linear_lowdim_spec,
        expected_to_break=("mrmr", "univariate-mi", "skb-mi", "ace", "it-relax", "mrmr-pld", "mrmr-relax", "mrmr-tree-rescued", "mrmr-stability", "greedy-backward"),
        purpose="small n: a t-statistic uses every row, a binned MI estimate throws most of them away",
    ),
    Scenario(
        name="redundant_exact_k5",
        family="redundant",
        builder=exact_redundancy_spec,
        defaults={"n_copies": 5},
        expected_to_break=("boruta-shap", "sfm-lgbm", "ace", "catboost-loss", "shap-proxied", "it-mim", "mrmr-grouped-expand", "unanimous-permutation", "null-importance", "permutation-topk"),
        purpose="importance splits across identical copies, so each looks weaker than the direction is",
    ),
    Scenario(
        name="latent_replicates_private_delta",
        family="redundant",
        builder=private_delta_spec,
        defaults={"n_members": 3},
        expected_to_break=("mrmr", "variance-sort", "univariate-mi", "it-cmim", "mrmr-grouped", "rfecv-registry", "boruta-shap-registry", "unanimous-permutation", "permutation-topk"),
        purpose="the members are jointly necessary, so collapsing the cluster destroys signal",
    ),
    Scenario(
        name="xor3",
        family="interactions",
        builder=parity_spec,
        defaults={"order": 3},
        expected_to_break=("mrmr", "univariate-mi", "skb-f", "skb-mi", "lars-order", "select-fdr", "variance-sort", "rank-vote", "byproduct-ensemble", "it-mim", "it-cmim", "it-jmim", "it-relax", "mrmr-pld", "mrmr-relax", "mrmr-stability", "cascade", "cascade-stable", "ridge-prefilter", "forward-select", "near-noise-auc", "ksg-mi", "relevance-table"),
        purpose="operands with zero marginal association: invisible to any one-column-at-a-time ranking",
    ),
    Scenario(
        name="xor3_plus_marginal_decoy",
        family="interactions",
        builder=parity_plus_decoy_spec,
        defaults={"order": 3},
        expected_to_break=("mrmr", "univariate-mi", "skb-f", "skb-mi", "boruta", "forward-select"),
        purpose="separates finding nothing from confidently finding the wrong thing",
    ),
    Scenario(
        name="joint_tail_t4",
        family="tails",
        builder=tail_dependence_spec,
        expected_to_break=("lars-order", "select-fdr", "skb-f"),
        purpose="the signal fires only where both columns share a tail, so no linear marginal statistic sees it",
    ),
    Scenario(
        name="joint_tail_gaussian_control",
        family="tails",
        builder=gaussian_tail_control_spec,
        expected_to_break=("lars-order", "variance-sort", "skb-f"),
        purpose="same correlation and gate, no tail dependence: isolates a tail failure from a gate failure",
    ),
    Scenario(
        name="friedman1",
        family="reference",
        builder=friedman1_spec,
        expected_to_break=("skb-f", "select-fdr", "lars-order", "variance-sort", "ridge-prefilter"),
        purpose="published bed: a sine interaction and a centred square that no linear statistic can see",
    ),
    Scenario(
        name="friedman2",
        family="reference",
        builder=friedman2_spec,
        expected_to_break=("skb-mi", "univariate-mi", "variance-sort"),
        purpose="published bed whose column ranges differ by orders of magnitude, by design rather than by accident",
    ),
    Scenario(
        name="friedman3",
        family="reference",
        builder=friedman3_spec,
        expected_to_break=("skb-f", "lars-order", "variance-sort"),
        purpose="published bed where a ratio is the entire signal, so neither operand is informative at a fixed marginal",
    ),
    Scenario(
        name="weston_guyon_k4_p100",
        family="reference",
        builder=weston_guyon_spec,
        expected_to_break=("select-fdr", "skb-f", "skb-mi", "boruta", "rfecv", "noise-floor"),
        purpose="published bed: equal-weight informative columns against same-marginal probes, so partial credit is unavailable",
    ),
    Scenario(
        name="tail_isolation_clayton_vs_gaussian",
        family="tails",
        builder=tail_isolation_spec,
        expected_to_break=("skb-f", "skb-mi", "univariate-mi", "select-fdr", "lars-order", "mrmr", "rank-vote", "byproduct-ensemble"),
        purpose="two pairs at matched rank correlation, one tail-dependent: the only bed here where the copula is the whole difference",
    ),
    Scenario(
        name="heavy_tail_t4",
        family="marginals",
        builder=heavy_tail_spec,
        expected_to_break=("skb-f", "select-fdr", "lars-order"),
        purpose="infinite kurtosis: a Pearson statistic is decided by a handful of rows and a rank statistic is not",
    ),
    Scenario(
        name="outliers_020permille",
        family="marginals",
        builder=outlier_contaminated_spec,
        expected_to_break=("skb-f", "lars-order", "variance-sort"),
        purpose="fixed-rate contamination rather than a heavy law, which is what a broken sensor actually produces",
    ),
    Scenario(
        name="zero_inflated_40pct",
        family="marginals",
        builder=zero_inflated_spec,
        expected_to_break=("skb-mi", "univariate-mi", "mrmr"),
        purpose="a point mass equal-mass binning cannot split, so the delivered bin count is silently not the requested one",
    ),
    Scenario(
        name="quantized_6levels",
        family="marginals",
        builder=quantized_spec,
        expected_to_break=("skb-mi", "univariate-mi", "mrmr", "knockoffs", "it-jmim"),
        purpose="fewer distinct values than an estimator wants bins, so ties dominate the ranking",
    ),
    Scenario(
        name="graded_cardinality",
        family="mixed_types",
        builder=graded_cardinality_spec,
        expected_to_break=("sfm-lgbm", "boruta", "boruta-shap", "catboost-shap", "catboost-predictions", "boruta-shap-registry", "hetero-vote", "null-importance"),
        purpose="equal signal at four cardinalities: a method ranking them apart is ranking by split opportunities",
    ),
    Scenario(
        name="id_trap",
        family="mixed_types",
        builder=id_trap_spec,
        expected_to_break=("sfm-lgbm", "boruta-shap", "rfecv", "variance-sort", "catboost-shap", "catboost-loss", "catboost-predictions", "shap-proxied", "mrmr-tree-rescued", "rfecv-registry", "boruta-shap-registry", "cascade", "cascade-stable", "hetero-vote", "zero-importance"),
        purpose="a unique-per-row column that maximises impurity importance and generalises to nothing",
    ),
    Scenario(
        name="zipf_levels",
        family="mixed_types",
        builder=zipf_levels_spec,
        expected_to_break=("skb-mi", "univariate-mi", "knockoffs"),
        purpose="power-law level frequencies: a long tail with almost no rows per level",
    ),
    Scenario(
        name="missingness_trio_30pct",
        family="observation",
        builder=missingness_trio_spec,
        expected_to_break=("skb-f", "select-fdr", "lars-order", "knockoffs"),
        purpose="three missingness mechanisms on sibling columns, so one run separates recoverable bias from unrecoverable",
    ),
    Scenario(
        name="rare_class_010permille",
        family="observation",
        builder=rare_class_spec,
        expected_to_break=("skb-mi", "univariate-mi", "mrmr", "boruta"),
        purpose="a one-per-cent positive rate reached by intercept shift, so imbalance is not confounded with sample size",
    ),
    Scenario(
        name="shift_covariate",
        family="observation",
        builder=covariate_shift_spec,
        expected_to_break=("variance-sort", "skb-mi"),
        purpose="P(x) moves and P(y|x) does not: paired with the concept-shift bed, which differs in exactly one declaration",
    ),
    Scenario(
        name="shift_concept",
        family="observation",
        builder=concept_shift_spec,
        expected_to_break=("skb-f", "select-fdr", "univariate-mi", "lars-order"),
        purpose="P(y|x) rotates while P(x) stays put: the half of drift that reweighting cannot fix",
    ),
    Scenario(
        name="multiclass_k3",
        family="targets",
        builder=multiclass_spec,
        expected_to_break=("skb-f", "select-fdr", "lars-order", "knockoffs"),
        purpose="each class depends on a different rotation of the weights, so the answer key is the union over classes",
    ),
    Scenario(
        name="ordinal_k4",
        family="targets",
        builder=ordinal_spec,
        expected_to_break=("skb-mi", "univariate-mi", "variance-sort"),
        purpose="ordered classes on one latent axis, paired with the multiclass bed to separate class structure from class count",
    ),
    Scenario(
        name="count_poisson",
        family="targets",
        builder=count_spec,
        expected_to_break=("skb-f", "skb-mi", "select-fdr", "lars-order", "rank-vote"),
        purpose="a Poisson target whose variance is its mean, so a constant-variance scorer is mis-weighted where it matters",
    ),
    Scenario(
        name="label_flip_uniform_15pct",
        family="corrupted",
        builder=uniform_flip_spec,
        expected_to_break=("boruta", "rfecv", "sfm-lgbm", "byproduct-ensemble", "greedy-backward"),
        purpose="symmetric label noise: the declared ceiling is unreachable and the gap is the information the flip destroyed",
    ),
    Scenario(
        name="label_flip_regional_35pct",
        family="corrupted",
        builder=feature_dependent_flip_spec,
        expected_to_break=("skb-f", "skb-mi", "univariate-mi", "select-fdr", "rank-vote"),
        purpose="label reliability depends on a column, which nothing scoring against the observed label can see",
    ),
    Scenario(
        name="label_coarsened",
        family="corrupted",
        builder=coarsened_label_spec,
        expected_to_break=("skb-mi", "univariate-mi", "knockoffs"),
        purpose="the label is a lossy function of the truth rather than a noisy copy of it",
    ),
    Scenario(
        name="m_bias",
        family="causal",
        builder=m_bias_spec,
        expected_to_break=("mrmr", "ace", "boruta", "rfecv", "lars-order"),
        purpose="a column every marginal score recommends, whose admission manufactures an association the graph does not contain",
    ),
    Scenario(
        name="confounder_backdoor",
        family="causal",
        builder=confounder_spec,
        expected_to_break=("variance-sort", "rank-vote"),
        purpose="a column predictive of the target that causes none of it: right under prediction, wrong under causation",
    ),
    Scenario(
        name="instrument_screened_off",
        family="causal",
        builder=instrument_spec,
        expected_to_break=("skb-f", "skb-mi", "univariate-mi", "select-fdr", "rank-vote"),
        purpose="structural redundancy rather than correlational: a column screened off by one already selected",
    ),
    Scenario(
        name="proxy_attenuation",
        family="causal",
        builder=proxy_attenuation_spec,
        expected_to_break=("lars-order", "knockoffs", "byproduct-ensemble"),
        purpose="a clean proxy out-predicts the noisy measurement of the real cause, so prediction and causation disagree",
    ),
    Scenario(
        name="expensive_signal",
        family="economics",
        builder=expensive_signal_spec,
        expected_to_break=("skb-f", "skb-mi", "sfm-lgbm", "boruta", "rank-vote"),
        purpose="the informative columns cost a hundred times the probes, so a cost-blind selection is unaffordable",
    ),
    Scenario(
        name="redundancy_graded",
        family="economics",
        builder=graded_redundancy_spec,
        expected_to_break=("mrmr", "knockoffs", "boruta-shap", "sfm-lgbm", "mrmr-grouped", "mrmr-grouped-expand"),
        purpose="four correlation levels in one bed, so a threshold's breaking point is a curve rather than a guess",
    ),
    Scenario(
        name="simpson_sign_reversal",
        family="structure",
        builder=simpson_reversal_spec,
        expected_to_break=("skb-f", "skb-mi", "univariate-mi", "select-fdr", "lars-order", "rank-vote", "byproduct-ensemble", "it-mim", "near-noise-auc", "ksg-mi", "relevance-table"),
        purpose="a strong column whose marginal association is zero because its sign flips between subgroups",
    ),
    Scenario(
        name="grouped_rows_40",
        family="structure",
        builder=grouped_rows_spec,
        expected_to_break=("variance-sort", "skb-mi", "knockoffs"),
        purpose="rows of one group share an offset no feature explains, so a row-wise split reports a number it should not",
    ),
    Scenario(
        name="mb_spouse_collider",
        family="causal",
        builder=spouse_collider_spec,
        expected_to_break=("univariate-mi", "skb-f", "skb-mi", "select-fdr", "lars-order", "rank-vote", "byproduct-ensemble"),
        purpose="a blanket member that is invisible until one conditions on the collider",
    ),
    Scenario(
        name="mediator_chain_with_proxy",
        family="causal",
        builder=mediator_chain_spec,
        expected_to_break=("rfecv", "mrmr", "boruta-shap"),
        purpose="the declared answer key, not the method, decides who wins here",
    ),
)

_BY_NAME: Dict[str, Scenario] = {scenario.name: scenario for scenario in SCENARIOS}


def names() -> Tuple[str, ...]:
    """Return every registered scenario name."""
    return tuple(_BY_NAME)


#: Exported under a qualified name on the package facade, where a bare ``names`` would say nothing.
scenario_names = names


def get(name: str) -> Scenario:
    """Return one scenario by name.

    Raises:
        KeyError: If no scenario carries that name, listing what is registered so a typo is one read away
            from being fixed.
    """
    if name not in _BY_NAME:
        raise KeyError(f"unknown scenario {name!r}; registered: {sorted(_BY_NAME)}")
    return _BY_NAME[name]


def build_lock(seed: int = 0) -> Dict[str, Any]:
    """Return the lock contents: each scenario's family, structural hash, key and expectations.

    The hash is of the SPEC at a fixed seed, so it moves when the structure changes and stays put when a
    scenario is merely run at a different seed. That is the distinction the lock exists to make: editing a
    bed after seeing results is visible, running it again is not an edit.
    """
    return {
        "schema_version": 1,
        "seed": seed,
        "scenarios": {
            scenario.name: {
                "family": scenario.family,
                "spec_hash": scenario.build(seed=seed).content_hash(),
                "expected_to_break": list(scenario.expected_to_break),
                "primary_target_set": scenario.primary_target_set,
                "purpose": scenario.purpose,
            }
            for scenario in SCENARIOS
        },
    }


def load_lock() -> Dict[str, Any]:
    """Return the committed lock, or an empty mapping when it has not been written yet."""
    try:
        with open(LOCK_PATH, encoding="utf-8") as handle:
            return dict(json.load(handle))
    except (OSError, ValueError) as exc:
        logger.info("no usable scenario lock at %s: %s", LOCK_PATH, exc)
        return {}


def lock_differences(seed: int = 0) -> List[str]:
    """Return the differences between the committed lock and the code, newest state described first.

    An empty list means the registry on disk is the registry that ran. Anything else is a sentence a report
    has to carry, because it means a bed changed after the document that described it was committed.
    """
    committed = load_lock()
    if not committed:
        return ["no scenario lock is committed, so no structural change can be detected"]

    current = build_lock(seed=seed)
    notes: List[str] = []
    old: Dict[str, Any] = dict(committed.get("scenarios", {}))
    new: Dict[str, Any] = dict(current["scenarios"])

    notes.extend(f"POST-HOC SCENARIO: {name!r} is in the code but not in the lock" for name in sorted(set(new) - set(old)))
    notes.extend(f"REMOVED SCENARIO: {name!r} is in the lock but not in the code" for name in sorted(set(old) - set(new)))
    for name in sorted(set(old) & set(new)):
        if old[name].get("spec_hash") != new[name]["spec_hash"]:
            notes.append(f"CHANGED STRUCTURE: {name!r} no longer matches its locked spec hash")
        if list(old[name].get("expected_to_break", ())) != list(new[name]["expected_to_break"]):
            notes.append(f"CHANGED EXPECTATIONS: {name!r} declares different arms as expected to break")
    return notes
