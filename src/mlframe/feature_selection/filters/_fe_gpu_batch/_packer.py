"""Heterogeneous multi-GPU block packer for the FE batcher (2026-06-26).

Assigns candidate-column BLOCKS to devices so the speed-weighted MAKESPAN is minimised -- a faster GPU
gets proportionally more work. Each block is pre-sized by the caller to fit the SMALLEST device's per-wave
VRAM budget, so device VRAM capacity is satisfied by construction and the assignment problem reduces to
makespan on uniformly-related machines (R||Cmax with time = work / speed).

PRIMARY solver: OR-Tools CP-SAT on the live path -- a trivial ILP (a few dozen booleans) solved to proven
optimality in low-single-digit ms, run ONCE per fit. Greedy weighted-LPT is the labelled-SUBOPTIMAL
fallback used only if ortools is unavailable; it is NOT equal to the solver (LPT makespan worst case
4/3-1/3m, e.g. m=2 works [3,3,2,2,2] -> OPT 6, LPT 7), but it never changes which columns get scored, only
how evenly they are spread -- the per-column MI is assignment-invariant, so the packer can never alter the
selected features.
"""
from __future__ import annotations

import os

_CP_SAT_TIME_LIMIT_S = float(os.environ.get("MLFRAME_FE_VRAM_CPSAT_TIME_LIMIT_S", "1.0") or 1.0)
_SCALE = 1000  # integer scale for the relative speed weights


def _greedy_lpt(works: list[int], speeds: list[float]) -> list[int]:
    """Longest-Processing-Time-first onto the device with the earliest projected finish (load/speed).
    Labelled-suboptimal fallback; deterministic (ties -> lowest device index)."""
    d = len(speeds)
    load = [0.0] * d
    assign = [0] * len(works)
    order = sorted(range(len(works)), key=lambda b: works[b], reverse=True)
    for b in order:
        best = min(range(d), key=lambda di: ((load[di] + works[b]) / speeds[di], di))
        assign[b] = best
        load[best] += works[b]
    return assign


def _cpsat_pack(works: list[int], speeds: list[float]) -> list[int] | None:
    """Exact speed-weighted makespan assignment via CP-SAT; None if ortools is unavailable / no solution."""
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None
    B, D = len(works), len(speeds)
    mn = min(speeds)
    sp = [max(1, int(round(s / mn * _SCALE))) for s in speeds]  # integer relative speeds, slowest == _SCALE
    total = sum(int(w) for w in works)
    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{b}_{d}") for d in range(D)] for b in range(B)]  # type: ignore[attr-defined]  # ortools CamelCase alias, still present at runtime
    for b in range(B):
        model.Add(sum(x[b][d] for d in range(D)) == 1)  # type: ignore[attr-defined]
    cmax_ub = max(1, total) * _SCALE
    cmax = model.NewIntVar(0, cmax_ub, "cmax")  # type: ignore[attr-defined]
    for d in range(D):
        load_d = sum(x[b][d] * int(works[b]) for b in range(B))
        # time_d = load_d / speed_d <= cmax  <=>  load_d * _SCALE <= cmax * sp[d]   (linear; sp[d] constant)
        model.Add(load_d * _SCALE <= cmax * sp[d])  # type: ignore[attr-defined]
    model.Minimize(cmax)  # type: ignore[attr-defined]
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = _CP_SAT_TIME_LIMIT_S
    solver.parameters.num_search_workers = 1  # deterministic, reproducible
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [next(d for d in range(D) if solver.Value(x[b][d]) == 1) for b in range(B)]


def pack_blocks_to_devices(works: list[int], speeds: list[float], *, prefer: str | None = None) -> list[int]:
    """Return the device index for each block, minimising the speed-weighted makespan.

    ``works[b]`` = block b's relative work (e.g. its column count; rows are constant across blocks of one
    matrix). ``speeds[d]`` = device d's relative throughput. ``prefer`` forces a backend
    ("cpsat" | "greedy", also via env ``MLFRAME_FE_VRAM_PACKER``); default tries CP-SAT then greedy.
    """
    if not works:
        return []
    if len(speeds) <= 1:
        return [0] * len(works)
    choice = (prefer or os.environ.get("MLFRAME_FE_VRAM_PACKER", "")).strip().lower()
    if choice == "greedy":
        return _greedy_lpt(works, speeds)
    if choice in ("cpsat", "ortools", ""):
        res = _cpsat_pack(works, speeds)
        if res is not None:
            return res
    return _greedy_lpt(works, speeds)
