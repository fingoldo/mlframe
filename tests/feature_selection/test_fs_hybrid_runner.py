"""Tiers, dry-run pricing, and a worker pool whose timeout actually stops a cell.

The two failure modes these guard against are both quiet. A tier that silently ran a different grid from
the one it names would publish the wrong experiment under the right label; a timeout that does not fire
would let one hung cell hold a machine for a day while the run looked healthy.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._runner_pool import CellPool, prewarm_kernels
from mlframe.feature_selection._benchmarks.fs_hybrid._tiers import SOURCES, TIERS, estimate, format_estimate, get_tier, median_cell_seconds, scenarios_for


def _record(arm: str, wall: float, status: str = "ok", drained: bool = True) -> Dict[str, Any]:
    """Build one stored cell carrying a timing."""
    return {"arm": arm, "status": status, "wall_time_s": wall, "memo_drained": drained}


def test_every_tier_names_a_source_that_exists() -> None:
    """A tier resolving its beds from an environment variable is a run nobody can reproduce from its manifest."""
    assert TIERS.values()
    for tier in TIERS.values():
        assert tier.source in SOURCES, f"tier {tier.name!r} names source {tier.source!r}, which is not one of {SOURCES}"


def test_unknown_tier_is_refused_rather_than_defaulted() -> None:
    """A typo'd tier that fell back to a default would run one experiment and label it another."""
    with pytest.raises(KeyError, match="unknown tier"):
        get_tier("night1y")


def test_unknown_source_is_refused() -> None:
    """Falling through to the smallest bed library would make a wide run silently narrow."""
    with pytest.raises(ValueError, match="unknown scenario source"):
        scenarios_for("scmm")


def test_the_smoke_tier_is_small_enough_to_be_worth_having() -> None:
    """Its whole role is being affordable on a pull request; a smoke tier nobody can afford is not one."""
    tier = get_tier("smoke")

    assert tier.arms is not None and len(tier.arms) <= 4
    assert tier.scenarios is not None and len(tier.scenarios) <= 3
    assert tier.cell_count(available_scenarios=99, available_arms=99) <= 12


def test_the_weekly_tier_meets_the_pre_registered_seed_floor() -> None:
    """Twenty paired seeds is the floor the headline test was powered against; fewer is a different design."""
    assert len(get_tier("weekly").dataset_seeds) >= 20


def test_no_tier_reaches_into_the_report_only_seed_range() -> None:
    """Seeds 1000-1099 are reserved for the report, and tuning anything against them is the violation."""
    assert TIERS.values()
    for tier in TIERS.values():
        assert all(seed < 1000 for seed in tier.dataset_seeds), f"tier {tier.name!r} uses a report-only seed"


def test_cell_count_multiplies_every_axis() -> None:
    """A miscounted grid makes the dry-run estimate wrong by the same factor."""
    tier = get_tier("nightly")

    assert tier.cell_count(available_scenarios=10, available_arms=16) == 10 * 16 * len(tier.dataset_seeds) * len(tier.cv_seeds)


def test_median_cell_seconds_is_per_arm_not_pooled() -> None:
    """The roster spans four orders of magnitude, so a pooled median predicts nothing about a new arm mix."""
    history = [_record("cheap", 0.1), _record("cheap", 0.3), _record("expensive", 50.0), _record("expensive", 70.0)]

    medians = median_cell_seconds(history)

    assert medians["cheap"] == pytest.approx(0.2)
    assert medians["expensive"] == pytest.approx(60.0)


def test_median_ignores_cells_whose_timing_was_invalidated() -> None:
    """A cell that hit the memo measured a dictionary lookup, and including it makes the estimate optimistic."""
    history = [_record("arm", 50.0), _record("arm", 0.01, drained=False)]

    assert median_cell_seconds(history)["arm"] == pytest.approx(50.0)


def test_median_ignores_failed_cells() -> None:
    """A crash is fast, and pricing a grid on crashes predicts a run that never happens."""
    history = [_record("arm", 30.0), _record("arm", 0.2, status="crashed")]

    assert median_cell_seconds(history)["arm"] == pytest.approx(30.0)


def test_estimate_refuses_to_guess_without_history() -> None:
    """A prediction from no data is worse than none, because only one of the two gets believed."""
    value = estimate(get_tier("nightly"), scenarios=["a", "b"], arms=["x", "y"], history=[])

    assert value.predicted_seconds is None
    assert any("UNKNOWN" in line for line in format_estimate(value))


def test_estimate_scales_with_the_grid() -> None:
    """Twice the seeds is twice the work, and an estimate that did not say so would be decorative."""
    history = [_record("x", 10.0)]
    smaller = estimate(get_tier("nightly"), scenarios=["a"], arms=["x"], history=history)
    larger = estimate(get_tier("weekly"), scenarios=["a"], arms=["x"], history=history)

    assert smaller.predicted_seconds is not None and larger.predicted_seconds is not None
    ratio = len(get_tier("weekly").dataset_seeds) / len(get_tier("nightly").dataset_seeds)
    assert larger.predicted_seconds == pytest.approx(smaller.predicted_seconds * ratio)


def test_estimate_names_the_arms_it_could_not_price() -> None:
    """An estimate covering half the grid is a FLOOR, and reading it as a total is how a run overruns."""
    value = estimate(get_tier("nightly"), scenarios=["a"], arms=["priced", "never_run"], history=[_record("priced", 5.0)])

    assert value.unpriced_arms == ("never_run",)
    assert any("FLOOR" in line for line in format_estimate(value))


def test_estimate_divides_by_the_worker_count() -> None:
    """Otherwise the prediction is for a serial run nobody intends to do."""
    history = [_record("x", 8.0)]
    serial = estimate(get_tier("nightly"), scenarios=["a"], arms=["x"], history=history, workers=1)
    parallel = estimate(get_tier("nightly"), scenarios=["a"], arms=["x"], history=history, workers=4)

    assert serial.predicted_seconds is not None and parallel.predicted_seconds is not None
    assert parallel.predicted_seconds == pytest.approx(serial.predicted_seconds / 4.0)


def test_worker_initializer_pins_threads_before_numpy_would_read_them() -> None:
    """Every BLAS reads these once at import, so setting them afterwards is silently ineffective.

    Run in a fresh interpreter, which is where a worker initializer runs anyway. Calling it inside the pytest worker
    left ``NUMBA_NUM_THREADS=3`` in that process long enough for numba to launch its pool at three threads; the env
    var was restored afterwards, numba re-read its config at the next compile, and refused to resize a launched pool.
    Every later numba test on that xdist worker then failed with the same RuntimeError - 36 of them in one run.
    """
    import os
    import subprocess
    import sys

    script = "; ".join([
        "import os",
        "from mlframe.feature_selection._benchmarks.fs_hybrid._runner_pool import worker_initializer",
        "worker_initializer(threads=3, cpu_only=True, prewarm=False)",
        "print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'], os.environ['NUMBA_NUM_THREADS'], len(os.environ['CUDA_VISIBLE_DEVICES']))",
    ])
    pinned = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "CUDA_VISIBLE_DEVICES")
    env = {k: v for k, v in os.environ.items() if k not in pinned}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=600, check=False)  # nosec B603 - fixed interpreter, literal script
    assert out.returncode == 0, out.stderr[-2000:]
    omp, mkl, nb, cuda_len = out.stdout.strip().splitlines()[-1].split(" ")
    assert (omp, mkl, nb) == ("3", "3", "3")
    assert cuda_len == "0", "a CPU worker must not contend for the one GPU"


def test_prewarm_reports_whether_it_completed() -> None:
    """A silent prewarm failure puts a first compile inside the first timed cell of every worker."""
    assert prewarm_kernels() is True


def _sleep_forever(seconds: float) -> Dict[str, Any]:
    """A cell that does not come back, for the timeout to stop."""
    time.sleep(seconds)
    return {"status": "ok"}


def _quick(value: int) -> Dict[str, Any]:
    """A cell that returns immediately."""
    return {"status": "ok", "value": value}


@pytest.mark.slow
def test_pool_runs_every_job_and_returns_each_outcome() -> None:
    """The basic contract: nothing is dropped, and each result comes back attached to its own key."""
    with CellPool(workers=2, threads=1, timeout_s=120.0, heartbeat_s=1e6) as pool:
        outcomes = list(pool.map([(index, _quick, (index,)) for index in range(6)]))

    assert len(outcomes) == 6
    assert {outcome.key for outcome in outcomes} == set(range(6))
    assert all(outcome.status == "ok" for outcome in outcomes)
    assert all(outcome.record is not None and outcome.record["value"] == outcome.key for outcome in outcomes)


@pytest.mark.slow
def test_pool_stops_a_cell_that_exceeds_its_budget_and_records_it() -> None:
    """`Future.cancel()` does nothing to a started task, so a timeout has to kill the process to mean anything.

    The surviving jobs matter as much as the killed one: terminating the whole pool would discard the work
    every other worker had in flight, which is why each slot owns its own process.
    """
    jobs: List[Any] = [("hung", _sleep_forever, (600.0,))] + [(index, _quick, (index,)) for index in range(3)]

    started = time.perf_counter()
    with CellPool(workers=2, threads=1, timeout_s=5.0, heartbeat_s=1e6) as pool:
        outcomes = {outcome.key: outcome for outcome in pool.map(jobs)}
    elapsed = time.perf_counter() - started

    assert outcomes["hung"].status == "timeout"
    assert outcomes["hung"].record is None
    assert {0, 1, 2} <= set(outcomes), "the surviving jobs were discarded along with the hung one"
    assert all(outcomes[index].status == "ok" for index in (0, 1, 2))
    assert elapsed < 120.0, f"the timeout did not actually stop the cell (took {elapsed:.0f}s)"
