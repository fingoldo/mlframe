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
from ._interactions import parity_plus_decoy_spec, parity_spec
from ._linear import linear_lowdim_spec, linear_spec
from ._null import null_spec
from ._redundant import exact_redundancy_spec, private_delta_spec
from ._tails import gaussian_tail_control_spec, tail_dependence_spec

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
        expected_to_break=("select-fdr", "skb-f", "skb-mi", "univariate-mi"),
        purpose="false-discovery discipline: nothing is relevant, so anything selected is a false positive",
    ),
    Scenario(
        name="null_p1000",
        family="null",
        builder=null_spec,
        defaults={"width": 1000},
        expected_to_break=("select-fdr", "skb-f", "skb-mi", "boruta", "rfecv", "sfm-lgbm"),
        purpose="the same discipline where the best-looking noise column is genuinely convincing",
    ),
    Scenario(
        name="linear_k5_p50",
        family="linear",
        builder=linear_spec,
        defaults={"n_informative": 5, "n_noise": 45},
        expected_to_break=("variance-sort",),
        purpose="control: every method should succeed here, except the one that never looks at the target",
    ),
    Scenario(
        name="linear_gaussian_lowdim_n200",
        family="linear",
        builder=linear_lowdim_spec,
        expected_to_break=("mrmr", "univariate-mi", "skb-mi", "ace"),
        purpose="small n: a t-statistic uses every row, a binned MI estimate throws most of them away",
    ),
    Scenario(
        name="redundant_exact_k5",
        family="redundant",
        builder=exact_redundancy_spec,
        defaults={"n_copies": 5},
        expected_to_break=("boruta-shap", "sfm-lgbm", "ace"),
        purpose="importance splits across identical copies, so each looks weaker than the direction is",
    ),
    Scenario(
        name="latent_replicates_private_delta",
        family="redundant",
        builder=private_delta_spec,
        defaults={"n_members": 3},
        expected_to_break=("mrmr", "variance-sort", "univariate-mi"),
        purpose="the members are jointly necessary, so collapsing the cluster destroys signal",
    ),
    Scenario(
        name="xor3",
        family="interactions",
        builder=parity_spec,
        defaults={"order": 3},
        expected_to_break=("mrmr", "univariate-mi", "skb-f", "skb-mi", "lars-order", "select-fdr", "variance-sort"),
        purpose="operands with zero marginal association: invisible to any one-column-at-a-time ranking",
    ),
    Scenario(
        name="xor3_plus_marginal_decoy",
        family="interactions",
        builder=parity_plus_decoy_spec,
        defaults={"order": 3},
        expected_to_break=("mrmr", "univariate-mi", "skb-f", "skb-mi", "boruta"),
        purpose="separates finding nothing from confidently finding the wrong thing",
    ),
    Scenario(
        name="tail_dependence_t4",
        family="tails",
        builder=tail_dependence_spec,
        expected_to_break=("skb-mi", "univariate-mi", "mrmr"),
        purpose="the dependence lives in the joint tail, which one cell of an equal-mass histogram cannot resolve",
    ),
    Scenario(
        name="tail_control_gaussian",
        family="tails",
        builder=gaussian_tail_control_spec,
        expected_to_break=("variance-sort", "skb-mi"),
        purpose="same correlation and gate, no tail dependence: isolates a tail failure from a gate failure",
    ),
    Scenario(
        name="mb_spouse_collider",
        family="causal",
        builder=spouse_collider_spec,
        expected_to_break=("univariate-mi", "skb-f", "skb-mi", "select-fdr", "lars-order"),
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
