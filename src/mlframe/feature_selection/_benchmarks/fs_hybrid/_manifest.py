"""Run manifest: what was declared before a run, so what is claimed after it can be checked against it.

A results file records what happened. It cannot record what was SUPPOSED to happen, and the difference is
where a benchmark run by the author of one of its arms goes wrong -- not through fabrication, but through
the small freedoms nobody writes down: another handful of seeds when the first twenty looked unconvincing,
a pre-registration edited after seeing the tables, a comparison of timings across two different machines.

The manifest is written when a run starts, next to its results file, and the analysis checks the run back
against it:

* ``n_seeds_declared`` versus the seeds actually present -- optional stopping is not forbidden here, it is
  reported, which is the only version of the rule that survives a resumable runner.
* the pre-registration's hash at run time versus its hash now -- a document that changed after the run
  cannot bind the run, and the report says so instead of citing it.
* the environment tuple -- quality metrics are hardware independent, timings are not, so an aggregate that
  compares wall-clock across two tuples is refusing to be compared, not being conservative.

Nothing here blocks a run. It records the facts a reader would otherwise have to take on trust.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "manifest_path_for",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "environment_tuple",
    "arm_family",
    "check_run_against_manifest",
]

MANIFEST_SCHEMA_VERSION = 1
_PREREG_RELATIVE = os.path.join("docs", "BENCHMARK_PREREGISTRATION.md")


def manifest_path_for(results_path: str) -> str:
    """Return the manifest path that belongs to a results file."""
    return f"{results_path}.manifest.json"


def arm_family(arm: str) -> str:
    """Return the declared FAMILY of an arm name.

    The matched-cardinality control is named after the cardinality it matches -- `random-5` on a five-column
    answer key, `random-250` on a wide bed -- so its literal name is fixed by the bed, not chosen by whoever
    declared the run. Comparing raw names would report every bed but the first as carrying an undeclared arm,
    which is a false alarm that trains a reader to ignore the real ones.
    """
    if arm.startswith("random-") and arm[len("random-") :].isdigit():
        return "random-<k>"
    return arm


def _git_sha() -> str:
    """Return the current commit, or a marker when git cannot answer.

    A missing SHA is recorded as such rather than omitted: "this run cannot be tied to a revision" is a fact
    about the run, and an absent field would read as an oversight.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("git sha unavailable: %s", exc)
        return "unavailable"
    sha = out.stdout.strip()
    return sha or "unavailable"


def _repo_root() -> str:
    """Return the repository root inferred from this file's location."""
    here = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(here, *([os.pardir] * 6)))


def _file_sha256(path: str) -> Optional[str]:
    """Return a file's SHA-256, or ``None`` when it is not there to hash."""
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        logger.warning("cannot hash %s: %s", path, exc)
        return None


def _versions() -> Dict[str, str]:
    """Return versions of the libraries whose behaviour can move a result."""
    out: Dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "lightgbm", "scipy"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version:
            out[name] = str(version)
    return out


def environment_tuple() -> Dict[str, Any]:
    """Return the tuple timings are only comparable within."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "versions": _versions(),
    }


def build_manifest(
    results_path: str,
    scenarios: Sequence[str],
    arms: Sequence[str],
    dataset_seeds: Sequence[int],
    cv_seeds: Sequence[int],
    protocol_version: str,
    mode: str = "unspecified",
) -> Dict[str, Any]:
    """Assemble the manifest for one run."""
    prereg = os.path.join(_repo_root(), _PREREG_RELATIVE)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results_file": os.path.basename(results_path),
        "protocol_version": protocol_version,
        "mode": mode,
        "git_sha": _git_sha(),
        "preregistration_sha256": _file_sha256(prereg),
        "n_seeds_declared": len(list(dataset_seeds)),
        "dataset_seeds": [int(s) for s in dataset_seeds],
        "cv_seeds": [int(s) for s in cv_seeds],
        "scenarios": [str(s) for s in scenarios],
        "arms": [str(a) for a in arms],
        "arm_families": sorted({arm_family(str(a)) for a in arms}),
        "environment": environment_tuple(),
    }


def write_manifest(results_path: str, manifest: Dict[str, Any]) -> str:
    """Write the manifest beside its results file and return the path it was written to.

    Written through a temporary file and replaced atomically, so a run interrupted mid-write leaves the
    previous manifest intact rather than a truncated one that would fail to parse later.
    """
    path = manifest_path_for(results_path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load_manifest(results_path: str) -> Optional[Dict[str, Any]]:
    """Return the manifest beside a results file, or ``None`` when the run predates manifests."""
    path = manifest_path_for(results_path)
    try:
        with open(path, encoding="utf-8") as handle:
            return dict(json.load(handle))
    except (OSError, ValueError) as exc:
        logger.info("no usable manifest at %s: %s", path, exc)
        return None


def check_run_against_manifest(manifest: Optional[Dict[str, Any]], records: Sequence[Dict[str, Any]]) -> List[str]:
    """Return the notes a report must carry about how the run compares with what was declared.

    An empty list means the run matched its declaration on every checked axis. The absence of a manifest is
    itself a note: an undeclared run is not a clean run.
    """
    if manifest is None:
        return ["no manifest accompanies these results, so nothing about the run was declared in advance"]

    notes: List[str] = []

    declared_seeds = {int(s) for s in manifest.get("dataset_seeds", [])}
    present_seeds = {int(r["dataset_seed"]) for r in records if "dataset_seed" in r}
    extra = sorted(present_seeds - declared_seeds)
    if extra:
        notes.append(f"OPTIONAL STOPPING: {len(extra)} dataset seed(s) beyond the declared set were run ({extra[:8]}{'...' if len(extra) > 8 else ''})")

    declared_arms = set(manifest.get("arms", []))
    declared_families = set(manifest.get("arm_families", [])) or {arm_family(a) for a in declared_arms}
    present_arms = {str(r["arm"]) for r in records if "arm" in r}
    undeclared = sorted(arm for arm in present_arms - declared_arms if arm_family(arm) not in declared_families)
    if declared_arms and undeclared:
        notes.append(f"UNDECLARED ARMS: {undeclared}")

    declared_scenarios = set(manifest.get("scenarios", []))
    present_scenarios = {str(r["scenario"]) for r in records if "scenario" in r}
    extra_scenarios = sorted(present_scenarios - declared_scenarios)
    if declared_scenarios and extra_scenarios:
        notes.append(f"UNDECLARED SCENARIOS: {extra_scenarios}")

    recorded_hash = manifest.get("preregistration_sha256")
    current_hash = _file_sha256(os.path.join(_repo_root(), _PREREG_RELATIVE))
    if recorded_hash and current_hash and recorded_hash != current_hash:
        notes.append("PRE-REGISTRATION CHANGED since this run: the document as it stands today did not bind it")
    elif not recorded_hash:
        notes.append("the pre-registration was not hashed at run time, so this run is not tied to a version of it")

    current_env = environment_tuple()
    recorded_env = manifest.get("environment") or {}
    if recorded_env.get("platform") and recorded_env["platform"] != current_env["platform"]:
        notes.append("ENVIRONMENT DIFFERS from the run's: quality metrics still compare, timings do not")

    return notes
