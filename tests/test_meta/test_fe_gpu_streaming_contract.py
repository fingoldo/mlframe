"""Pins the FE GPU / streaming / profiling contract for the feature-engineering modules the MRMR layer imports.

The contract (``audits/2026-10-22/fe_gpu_coverage.md``): a numba CPU kernel and a fused GPU kernel, column-block streaming with a RAM and a VRAM bound,
the crit dtype, dispatch through the kernel-tuning cache, and a recorded profile by cProfile AND a GPU profiler (nsys, ncu or nvprof) that covers the
whole-GPU picture (occupancy, throughput / utilization), not only the hotspot kernel.
The GPU kernels are fused (the profile states the launches per call) and type-generic: ``{REAL}`` for values, ``{ACC}`` for accumulators, no bare double / float.

``_fe_gpu_contract.json`` is a ratchet: modules that meet the contract are listed as compliant and checked in full; modules that do not yet are listed as
pending and the list only shrinks; a module that is neither compliant nor exempt (a new FE module) fails the test, so a new module must meet the contract
before it lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlframe.feature_selection._benchmarks.fe_gpu_coverage_inventory import PROFILES_DIR, contract_violations, fe_modules

REGISTRY = Path(__file__).with_name("_fe_gpu_contract.json")


@pytest.fixture(scope="module")
def registry() -> dict:
    """The compliant / exempt / pending registry."""
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_every_compliant_module_meets_the_whole_contract(registry):
    """A compliant module has the CPU and GPU kernels, the streaming, the crit dtype, the KTC registration and both profiles on record."""
    failures = {name: contract_violations(name) for name in registry["compliant"]}
    failures = {name: why for name, why in failures.items() if why}
    assert not failures, f"modules listed as compliant that miss contract points: {failures}"


def test_every_fe_module_is_classified(registry):
    """No FE module the MRMR layer imports is left outside the registry; a new one must meet the contract and be added to ``compliant``."""
    known = set(registry["compliant"]) | set(registry["exempt"]) | set(registry["pending"])
    unclassified = sorted(set(fe_modules()) - known)
    assert not unclassified, f"FE modules outside the contract registry (make them meet the contract, then list them as compliant): {unclassified}"


def test_a_pending_module_that_meets_the_contract_is_promoted(registry):
    """The ratchet only shrinks: a pending module that now meets every point must move to ``compliant``."""
    promoted = [name for name in registry["pending"] if not contract_violations(name)]
    assert not promoted, f"these pending modules meet the contract now; move them to 'compliant': {promoted}"


def test_the_pending_list_never_grows(registry):
    """The pending list is at most what it was when the ratchet was created, and no module is both pending and compliant."""
    assert len(registry["pending"]) <= registry["n_pending_at_creation"]
    assert not set(registry["pending"]) & set(registry["compliant"])


def test_every_exemption_states_its_reason(registry):
    """An exempt module says why it has no numeric kernel."""
    assert all(isinstance(why, str) and len(why) > 10 for why in registry["exempt"].values())


def test_profile_records_live_in_the_audit_directory():
    """Profile records are kept next to the audit, so the meta-test can read them."""
    assert PROFILES_DIR.parent.is_dir(), PROFILES_DIR
