"""Paired interleaved timing for two-arm speedup assertions.

A bare ``t_slow / t_fast >= 1.3`` reads one measurement of each arm and calls the ratio a fact. Under
``-n`` contention a single scheduling stall landing on one arm inverts it, so the test goes red for a
reason that has nothing to do with the code under test -- and the usual response, loosening the literal,
costs the assertion its power against a real regression.

Interleaving the arms round by round cancels drifting load: whatever slows one arm this round slowed the
other one microseconds earlier. What survives is a load-independent signal -- did the fast arm win the
strict majority of PAIRED trials, and did the median paired ratio clear its floor -- which the shape at
tests/feature_selection/shap_proxied/test_shap_proxy_cluster_su_bitmap.py:231 already uses by hand. The
residual floor still goes through ``perf_speedup_floor`` so the ratio relaxes (never below 1.0x) under
xdist rather than being abandoned.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Tuple


def paired_speedup(baseline: Callable[[], Any], candidate: Callable[[], Any], *, n_trials: int = 5, warmup: bool = True) -> Tuple[float, int, Any, Any]:
    """Interleave ``baseline`` and ``candidate`` for ``n_trials`` rounds.

    Returns ``(median_ratio, wins, baseline_result, candidate_result)`` -- the median of the per-round
    ``t_baseline / t_candidate`` ratios, how many rounds the candidate actually won, and the LAST result
    from each arm so the caller can still pin correctness on real output rather than on timing alone.

    ``warmup`` runs each arm once before timing starts: a JIT compile or a lazy import landing inside the
    first timed round is measured as the candidate being slower, which is the single most common way this
    class of test lies.
    """
    if warmup:
        baseline()
        candidate()
    ratios, wins = [], 0
    base_result = cand_result = None
    for _ in range(int(n_trials)):
        t0 = time.perf_counter()
        base_result = baseline()
        t_base = time.perf_counter() - t0
        t0 = time.perf_counter()
        cand_result = candidate()
        t_cand = time.perf_counter() - t0
        ratios.append(t_base / max(t_cand, 1e-9))
        wins += int(t_cand < t_base)
    ratios.sort()
    return ratios[len(ratios) // 2], wins, base_result, cand_result


def assert_paired_speedup(
    baseline: Callable[[], Any],
    candidate: Callable[[], Any],
    *,
    base_ratio: float,
    n_trials: int = 5,
    what: str = "the candidate arm",
    warmup: bool = True,
) -> Tuple[Any, Any]:
    """Assert ``candidate`` beats ``baseline`` on paired interleaved trials; return both arms' last results.

    Two gates, both load-independent: the candidate wins a strict majority of paired rounds, and the median
    paired ratio clears ``perf_speedup_floor(base_ratio)``. A genuine regression collapses both at once; a
    stall on one round moves neither.
    """
    from .conftest import perf_speedup_floor

    median_ratio, wins, base_result, cand_result = paired_speedup(baseline, candidate, n_trials=n_trials, warmup=warmup)
    floor = perf_speedup_floor(base_ratio)
    majority = n_trials // 2 + 1
    assert wins >= majority and median_ratio >= floor, (
        f"{what} did not beat its baseline: median paired ratio={median_ratio:.2f}x over {n_trials} interleaved "
        f"trials, won {wins}/{n_trials} (need >= {majority} wins and >= {floor:.2f}x; {base_ratio:.2f}x is the "
        f"uncontended target)"
    )
    return base_result, cand_result
