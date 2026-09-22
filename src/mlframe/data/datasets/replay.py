"""Rebuilding the exact dataset a recorded result was measured on, months later.

A benchmark cell records a scenario name and a seed. That is enough to regenerate the data ONLY while the
scenario still builds the same spec, and a scenario library is edited: a bed gains a column, a weight
changes, a corruption is added. Then the same name and the same seed produce different data, every number
recorded against them describes something that no longer exists, and nothing says so.

The fix is not to freeze the library. Editing a bed after seeing a result is often the right response to a
surprise; what must not happen is the edit being invisible. So a replay carries the spec HASH it expects,
compares it against what the library builds today, and reports the three states separately:

* **exact** -- the library builds the same spec, so the data is bit-identical and the recorded numbers
  still describe it.
* **changed** -- the library builds a different spec under the same name. The data is regenerated and
  RETURNED, because looking at it is usually how the difference gets diagnosed, but it is not the data the
  result was measured on and the record says which hash was wanted and which arrived.
* **absent** -- no scenario by that name. Nothing can be rebuilt, and no partial answer is offered.

Bit-identity is not assumed, it follows from the generator being a pure function of the spec: every stream
is addressed by name through blake2b, so the same spec and the same seed produce the same bytes on any
machine and any run order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["ReplayResult", "replay", "replay_from_record", "replay_report", "spec_hash_for"]


@dataclass(frozen=True)
class ReplayResult:
    """One replay attempt: what was asked for, what came back, and whether they are the same thing."""

    scenario: str
    seed: int
    status: str
    expected_hash: Optional[str]
    actual_hash: Optional[str]
    dataset: Optional[Any]
    reason: str

    def is_exact(self) -> bool:
        """True only when the regenerated data is the data the result was measured on."""
        return self.status == "exact"

    def as_dict(self) -> Dict[str, Any]:
        """Return the record shape a report stores, without the data itself."""
        return {"scenario": self.scenario, "seed": self.seed, "status": self.status, "expected_hash": self.expected_hash, "actual_hash": self.actual_hash, "reason": self.reason}


def spec_hash_for(scenario: str, seed: int = 0) -> Optional[str]:
    """Return the structural hash of a registered scenario at one seed, or ``None`` when it is not registered."""
    from mlframe.data.datasets import scenarios as registry

    try:
        return registry.get(scenario).build(seed=seed).content_hash()
    except KeyError:
        return None


def replay(scenario: str, seed: int, expected_hash: Optional[str] = None) -> ReplayResult:
    """Regenerate one bed and say whether it is the bed a recorded result was measured on.

    Args:
        scenario: Registered scenario name.
        seed: The dataset seed the result was recorded against.
        expected_hash: The spec hash recorded with the result. Without it the replay can still produce
            data but CANNOT certify it, and says so: a replay whose status is "unverified" is a different
            claim from one whose status is "exact", and collapsing them is the whole failure this module
            exists to prevent.

    Returns:
        A :class:`ReplayResult`.
    """
    from mlframe.data.datasets import scenarios as registry
    from mlframe.data.datasets.generator import generate

    try:
        entry = registry.get(scenario)
    except KeyError:
        return ReplayResult(scenario=scenario, seed=seed, status="absent", expected_hash=expected_hash, actual_hash=None, reason=f"no scenario named {scenario!r} is registered, so nothing can be rebuilt", dataset=None)

    spec = entry.build(seed=seed)
    actual = spec.content_hash()
    dataset = generate(spec)

    if expected_hash is None:
        return ReplayResult(scenario=scenario, seed=seed, status="unverified", expected_hash=None, actual_hash=actual, dataset=dataset, reason="the record carries no spec hash, so this data cannot be certified as the data the result was measured on")
    if actual == expected_hash:
        return ReplayResult(scenario=scenario, seed=seed, status="exact", expected_hash=expected_hash, actual_hash=actual, dataset=dataset, reason="the library builds the same spec, so this is the data the result was measured on")
    logger.warning("scenario %r has changed since the recorded result: expected spec hash %s, built %s", scenario, expected_hash, actual)
    return ReplayResult(
        scenario=scenario,
        seed=seed,
        status="changed",
        expected_hash=expected_hash,
        actual_hash=actual,
        dataset=dataset,
        reason=f"the bed was edited after the result was recorded: expected spec hash {expected_hash}, the library now builds {actual}",
    )


def replay_from_record(record: Dict[str, Any]) -> ReplayResult:
    """Replay the bed one stored cell was run on, reading the scenario, seed and hash out of the record."""
    return replay(str(record.get("scenario", "")), int(record.get("dataset_seed", 0)), record.get("spec_hash"))


def replay_report(records: List[Dict[str, Any]]) -> List[str]:
    """Render, per scenario, whether the beds behind a results file can still be rebuilt exactly.

    One line per scenario rather than per cell: every cell of a scenario at one seed rebuilds identically
    or not at all, so a per-cell table would repeat one fact a hundred times and bury the one scenario
    where it is false.
    """
    seen: Dict[str, ReplayResult] = {}
    for record in records:
        scenario = str(record.get("scenario", ""))
        if scenario and scenario not in seen:
            seen[scenario] = replay_from_record(record)

    lines = ["", "=" * 100, "CAN THESE RESULTS' BEDS STILL BE REBUILT?", "=" * 100, ""]
    if not seen:
        lines.append("no scenarios in these records, so there is nothing to rebuild")
        return lines
    for scenario, result in sorted(seen.items()):
        lines.append(f"{scenario:<36}{result.status:<12}{result.reason}")
    changed = [name for name, result in seen.items() if result.status == "changed"]
    if changed:
        lines.append("")
        lines.append(f"{len(changed)} bed(s) were edited after this result was recorded: {', '.join(sorted(changed))}")
        lines.append("Every number this file reports for them describes data the library no longer produces.")
    return lines
