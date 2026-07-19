"""Decision-equivalence + wall-time validation for the permutation-fallback prefilter (2026-07-19).

``_permutation_prefilter_reject`` (in ``_mdlp_validated_split.py``) is a reject-only O(1) shortcut
in front of the permutation-fallback branch's ``n_permutations``-cost shuffle loop -- the confirmed
cost driver of validated-MDLP (20-80x per column). It can only ever turn a would-be reject into an
early reject (never an accept), which makes it safe by construction, but "safe by construction"
still gets an empirical check per this project's convention: this module confirms zero decision
flips across the existing adversarial + robustness scenario suite, and measures the wall-time win.

Two entry points:
  * ``run_fast_subset()`` -- a handful of scenarios/seeds (seconds), backing the real pytest in
    ``tests/feature_selection/discretization/test_mdlp_prefilter_hybrid_fast.py``.
  * ``run_full_sweep()`` -- broader scenario x seed sweep with wall-time reporting, run standalone:
    ``python -m mlframe.feature_selection.filters._benchmarks.bench_mdlp_prefilter_hybrid --full``
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from unittest import mock

import numpy as np

from mlframe.feature_selection.filters import _mdlp_validated_split as _mvs
from mlframe.feature_selection.filters._benchmarks.bench_mdlp_validated_split_suite import (
    scen_interaction_only,
    scen_multimodal_target,
    scen_non_monotonic_sine,
    scen_pure_noise,
    scen_step_k_breakpoints,
    scen_with_nan,
)


@dataclass
class PrefilterCheckResult:
    scenario: str
    seed: int
    edges_match: bool
    n_bins_with: int
    n_bins_without: int
    wall_with: float
    wall_without: float


def _warm_jit(n: int = 300) -> None:
    """Trigger numba compilation of every njit kernel on BOTH the prefiltered and forced-off code
    paths before any timed run -- per this project's A/B convention, an untimed cold first call
    would otherwise attribute pure JIT-compile/cache-load cost to whichever variant happens to run
    first, swamping the actual (much smaller) per-node prefilter savings."""
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(n), rng.standard_normal(n) * 1000.0
    _mvs.mdlp_bin_edges_validated(x, y, seed=0, n_permutations=30)
    with mock.patch.object(_mvs, "_permutation_prefilter_reject", return_value=False):
        _mvs.mdlp_bin_edges_validated(x, y, seed=0, n_permutations=30)


def _timed_call(x: np.ndarray, y: np.ndarray, kwargs: dict, forced_off: bool, n_repeats: int) -> tuple[np.ndarray, float]:
    """Best-of-``n_repeats`` (median) timing, per this project's A/B convention (never one-shot)."""
    edges = None
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        if forced_off:
            with mock.patch.object(_mvs, "_permutation_prefilter_reject", return_value=False):
                edges = _mvs.mdlp_bin_edges_validated(x, y, **kwargs)
        else:
            edges = _mvs.mdlp_bin_edges_validated(x, y, **kwargs)
        times.append(time.perf_counter() - t0)
    assert edges is not None
    return edges, float(np.median(times))


def _run_one(scenario: str, x: np.ndarray, y: np.ndarray, seed: int, n_repeats: int = 3) -> PrefilterCheckResult:
    kwargs = dict(seed=seed, n_permutations=30)

    # Interleave order per scenario (even seed -> with-first, odd seed -> without-first) so no
    # single variant systematically absorbs any residual warm-up/cache-eviction asymmetry.
    if seed % 2 == 0:
        edges_with, wall_with = _timed_call(x, y, kwargs, forced_off=False, n_repeats=n_repeats)
        edges_without, wall_without = _timed_call(x, y, kwargs, forced_off=True, n_repeats=n_repeats)
    else:
        edges_without, wall_without = _timed_call(x, y, kwargs, forced_off=True, n_repeats=n_repeats)
        edges_with, wall_with = _timed_call(x, y, kwargs, forced_off=False, n_repeats=n_repeats)

    edges_match = edges_with.shape == edges_without.shape and bool(np.allclose(edges_with, edges_without, equal_nan=True))
    return PrefilterCheckResult(
        scenario=scenario,
        seed=seed,
        edges_match=edges_match,
        n_bins_with=int(edges_with.size - 1),
        n_bins_without=int(edges_without.size - 1),
        wall_with=wall_with,
        wall_without=wall_without,
    )


_FAST_SCENARIOS = {
    "pure_noise": lambda n, seed: scen_pure_noise(n, seed),
    "step_3": lambda n, seed: scen_step_k_breakpoints(n, 3, seed),
    "non_monotonic_sine": lambda n, seed: scen_non_monotonic_sine(n, seed),
    "with_nan_10pct": lambda n, seed: scen_with_nan(n, 0.10, seed),
}

_FULL_SCENARIOS = {
    **_FAST_SCENARIOS,
    "multimodal_target": lambda n, seed: scen_multimodal_target(n, seed),
    "interaction_only": lambda n, seed: scen_interaction_only(n, seed),
}


def run_fast_subset(n: int = 1500, n_seeds: int = 3) -> list[PrefilterCheckResult]:
    """A few seconds total -- exercises every fast-subset scenario at a small n and few seeds."""
    _warm_jit()
    results = []
    for name, gen in _FAST_SCENARIOS.items():
        for seed in range(n_seeds):
            x, y = gen(n, seed)
            results.append(_run_one(name, x, y, seed))
    return results


def run_full_sweep(n: int = 5000, n_seeds: int = 15) -> list[PrefilterCheckResult]:
    """Broader scenario x seed sweep with wall-time totals -- run standalone, not from pytest."""
    _warm_jit()
    results = []
    for name, gen in _FULL_SCENARIOS.items():
        for seed in range(n_seeds):
            x, y = gen(n, seed)
            results.append(_run_one(name, x, y, seed))
    return results


def _print_report(results: list[PrefilterCheckResult]) -> None:
    n_mismatch = sum(1 for r in results if not r.edges_match)
    total_with = sum(r.wall_with for r in results)
    total_without = sum(r.wall_without for r in results)
    print(f"scenarios x seeds run: {len(results)}")
    print(f"decision mismatches (prefilter changed accepted edges): {n_mismatch}")
    if n_mismatch:
        for r in results:
            if not r.edges_match:
                print(f"  MISMATCH scenario={r.scenario} seed={r.seed} bins_with={r.n_bins_with} bins_without={r.n_bins_without}")
    print(f"total wall with prefilter:    {total_with:.3f}s")
    print(f"total wall without prefilter: {total_without:.3f}s")
    if total_with > 0:
        print(f"speedup: {total_without / total_with:.2f}x")
    by_scenario: dict[str, list[PrefilterCheckResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r)
    print("\nper-scenario:")
    for name, rs in by_scenario.items():
        w = sum(r.wall_with for r in rs)
        wo = sum(r.wall_without for r in rs)
        speedup = wo / w if w > 0 else float("nan")
        print(f"  {name:24s} with={w:7.3f}s without={wo:7.3f}s speedup={speedup:.2f}x")


if __name__ == "__main__":
    results = run_full_sweep() if "--full" in sys.argv else run_fast_subset()
    _print_report(results)
