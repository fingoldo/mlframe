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

import orjson
from pathlib import Path

import pytest

from mlframe.feature_selection._benchmarks.fe_gpu_coverage_inventory import (
    CPU_TWINS,
    KTC_DISPATCHERS,
    PROFILES_DIR,
    contract_violations,
    fe_modules,
    cpu_twin_problems,
    ktc_dispatch_problems,
)

REGISTRY = Path(__file__).with_name("_fe_gpu_contract.json")


@pytest.fixture(scope="module")
def registry() -> dict:
    """The compliant / exempt / pending registry."""
    return orjson.loads(REGISTRY.read_bytes())


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


def test_every_tuner_dispatched_backend_names_a_real_dispatcher():
    """A backend that meets the kernel-tuning point through its dispatcher: the dispatcher still registers a tuner entry and imports it, and the reason is stated."""
    broken = {name: ktc_dispatch_problems(name) for name in KTC_DISPATCHERS if ktc_dispatch_problems(name)}
    assert not broken, f"KTC_DISPATCHERS entries that no longer hold: {broken}"
    assert all(len(why) > 30 for _dispatcher, why in KTC_DISPATCHERS.values())
    assert all(name in set(fe_modules()) for name in KTC_DISPATCHERS), "an entry names a module that is not an FE module"


def test_every_cpu_twin_names_a_real_module_with_a_numba_kernel():
    """A device module that meets the CPU-kernel point through its host twin: the twin still carries an ``@njit`` kernel and the reason is stated."""
    broken = {name: cpu_twin_problems(name) for name in CPU_TWINS if cpu_twin_problems(name)}
    assert not broken, f"CPU_TWINS entries that no longer hold: {broken}"
    assert all(len(why) > 30 for _twin, why in CPU_TWINS.values())
    assert all(name in set(fe_modules()) for name in CPU_TWINS), "an entry names a module that is not an FE module"


def test_integer_code_kernels_meet_the_type_point_through_their_code_type_placeholder():
    """A kernel over integer codes of a caller-chosen width is typed by {DT} / {XT}; the bit-set packer module (typed by {DT}) has no violation about type placeholders."""
    from mlframe.feature_selection._benchmarks.fe_gpu_coverage_inventory import _TYPE_PLACEHOLDERS

    assert {"{REAL}", "{ACC}", "{DT}", "{XT}"} <= set(_TYPE_PLACEHOLDERS)
    assert not [v for v in contract_violations("_batch_mi_noise_gate_bits") if "placeholder" in v]


def test_every_carved_module_is_still_in_its_parent_and_the_family_has_one_verdict():
    """A carved module is judged with its parent only while the parent imports it (or contains it, for a package); a child's verdict is the parent's and the family has its record under the
    parent's name. A child may keep a record of its own with deeper per-module measurements, but is never required to have one."""
    from mlframe.feature_selection._benchmarks.fe_gpu_coverage_inventory import CARVED_FAMILIES, PROFILES_DIR, carved_family_problems

    for parent, children in CARVED_FAMILIES.items():
        assert (PROFILES_DIR / f"{parent}.md").is_file(), f"the family record of {parent} is missing"
        for child in children:
            assert not carved_family_problems(child), (parent, child, carved_family_problems(child))
    assert carved_family_problems("not_a_carved_module") == []
