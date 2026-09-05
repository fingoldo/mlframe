"""A broad `except` must not cache a process-lifetime "unavailable" verdict.

An availability probe wrapped in `except Exception` and memoised into a module global turns the first
exception of the run into the answer for the rest of it. What such probes actually see is a moment, not a
fact about the machine: another process holding the device, an allocation failing at that instant, a driver
reset, a fault raised out of a device-count call under contention. `ImportError` is the one genuine
absence, and caching that is correct.

Three instances were fixed in this repository, each silent and each paid for the whole run: the metrics
argsort probe, giving back the ~10% end-to-end win its own A/B header records at 200k rows; the ShapProxied
cluster-SU probe, putting the whole pair loop on the CPU for every later fit; and the transformer probe,
which review had missed and this check found.

The flags below are the ones it still reports. All five are deliberate and are recorded with the reason,
because a latch that a caller re-arms, or one reached only after the retries are exhausted, is doing what
it should.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# flag -> why latching it is correct here.
DELIBERATE_LATCHES = {
    "_KSG_GPU_FAILED": "circuit breaker, re-armed per fit by _rearm_gpu_circuit_breakers",
    "_CMI_GPU_FAILED": "circuit breaker, re-armed per fit by _rearm_gpu_circuit_breakers",
    "_MI_DIRECT_GPU_FAILED": "circuit breaker, re-armed per fit by _rearm_gpu_circuit_breakers",
    "_CUDA_USABLE_CACHE": "retries inside the call and only pins the verdict once they are exhausted; a mixed answer across callers would split one fit between device and CPU kernels",
    "_CB_GPU_USABLE_CACHE": "same shape, plus a genuine-absence signature for a wheel built without GPU support; the probe fit costs ~4s so re-probing per caller is not an option",
}


def test_no_new_latched_availability_flags():
    """Fail on any availability flag pinned inside a broad `except` beyond the deliberate ones above."""
    from py_ci_shared.latched_availability_flags import assert_no_latched_availability_flags

    assert_no_latched_availability_flags(
        [REPO_ROOT / "src"],
        exclude=("_benchmarks", "_cpx36_baseline"),
        allow=DELIBERATE_LATCHES,
    )


def test_the_recorded_latches_still_exist():
    """A flag that no longer latches must leave the list, or the list stops describing the code."""
    from py_ci_shared.latched_availability_flags import find_latched_availability_flags

    reported = {f.flag for f in find_latched_availability_flags([REPO_ROOT / "src"], exclude=("_benchmarks", "_cpx36_baseline"))}
    stale = sorted(set(DELIBERATE_LATCHES) - reported)
    assert not stale, f"these no longer latch and should be dropped from DELIBERATE_LATCHES: {stale}"


def test_every_recorded_latch_carries_a_reason():
    """An entry without a reason is a baseline entry, and a baseline is how this class survived."""
    missing = sorted(flag for flag, reason in DELIBERATE_LATCHES.items() if not reason.strip())
    assert not missing, f"recorded without a reason: {missing}"
