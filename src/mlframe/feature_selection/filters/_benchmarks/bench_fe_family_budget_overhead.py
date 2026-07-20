"""cProfile evidence for gt_07: the FE-family budgeting bookkeeping itself is negligible vs FE wall.

Run: python -m mlframe.feature_selection.filters._benchmarks.bench_fe_family_budget_overhead

Compares total MRMR.fit wall with fe_budget_learning=False vs True on a fixture sized to exercise the
triplet/quadruplet FE stages, isolating the budgeting bookkeeping's own cost (load + quota-scale +
credit + reallocate + persist) via direct microbenchmarks of the engine functions themselves (cProfile
cannot cleanly attribute the FE stages' own compiled-kernel time, per ``_fe_family_timing.py``'s own
module docstring, so the budgeting overhead is isolated by microbench rather than by cProfile delta).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _make_fixture(n=1500, seed=0):
    """Synthetic bed with a genuine triplet interaction, sized to exercise the triplet/quadruplet FE stages."""
    rng = np.random.default_rng(seed)
    x1, x2, x3, x4 = (rng.standard_normal(n) for _ in range(4))
    y = pd.Series((np.sign(x1 * x2 * x3) > 0).astype(int))
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
    return X, y


def bench_fit_wall_with_vs_without_budget_learning():
    """Wall-clock a full MRMR.fit with fe_budget_learning off vs on, at the same config otherwise."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    with tempfile.TemporaryDirectory() as td:
        import mlframe.feature_selection.filters._fe_family_budget as fb

        fb._BUDGET_CACHE_DIR = Path(td)

        def _one_fit(budget_learning: bool, seed: int) -> float:
            """Time one MRMR.fit call with fe_budget_learning set as requested, on a NOVEL seed
            (MRMR memoizes fits by content-hash -- reusing the same (X, y, params) across calls
            would short-circuit every repeat after the first via the cross-instance identity cache,
            making the comparison meaningless; see CLAUDE.md's GPU-profiling-traps note)."""
            X_s, y_s = _make_fixture(seed=seed)
            t0 = time.perf_counter()
            MRMR(
                fe_hybrid_orth_triplet_enable=True, fe_hybrid_orth_quadruplet_enable=True,
                fe_budget_learning=budget_learning, verbose=0, random_state=0,
            ).fit(X_s, y_s)
            return time.perf_counter() - t0

        # Warm njit kernels for BOTH variants before timing (A/B convention: warm first, never
        # compare a cold call against a warm one -- njit compile cost otherwise swamps everything else).
        _one_fit(False, seed=100)
        _one_fit(True, seed=101)

        walls_off = [_one_fit(False, seed=200 + i) for i in range(3)]
        walls_on = [_one_fit(True, seed=300 + i) for i in range(3)]
        wall_off = float(np.median(walls_off))
        wall_on = float(np.median(walls_on))

        overhead_pct = 100.0 * (wall_on - wall_off) / wall_off if wall_off > 0 else float("nan")
        print(f"fe_budget_learning=False: median {wall_off:.4f}s (all: {[round(w, 4) for w in walls_off]})")
        print(f"fe_budget_learning=True:  median {wall_on:.4f}s (all: {[round(w, 4) for w in walls_on]})")
        print(f"overhead: {overhead_pct:.2f}% (target <=0.5% of FE wall)")


def bench_engine_functions_isolated():
    """Microbench the budgeting engine functions themselves (load/credit/roi/reallocate/persist), isolated from any FE stage cost."""
    from mlframe.feature_selection.filters._fe_family_budget import (
        dataset_fingerprint,
        family_credit,
        family_roi,
        load_budgets,
        persist_budgets,
        reallocate_budgets,
    )

    prov = pd.DataFrame(
        {
            "feature_name": [f"a{i}*b{i}__He1_He1" for i in range(50)],
            "origin": ["hybrid_orth"] * 50,
            "mechanism_details": ["{'kind': 'orth_pair_cross', 'src_names': ['a', 'b']}"] * 50,
            "mrmr_gain": np.random.default_rng(0).random(50).tolist(),
            "support_rank": list(range(50)),
        }
    )
    wall = {"orth_pair": (1.0, 5), "triplet": (0.5, 3), "quadruplet": (0.3, 2)}
    base_budget = {"triplet": 0.34, "quadruplet": 0.33, "adaptive_arity": 0.33}

    with tempfile.TemporaryDirectory() as td:
        import mlframe.feature_selection.filters._fe_family_budget as fb

        fb._BUDGET_CACHE_DIR = Path(td)
        fp = dataset_fingerprint(50, prov["feature_name"])

        n_iters = 200
        t0 = time.perf_counter()
        for _ in range(n_iters):
            credit = family_credit(prov)
            roi = family_roi(credit, wall)
            budgets = reallocate_budgets(roi, base_budget=base_budget)
            persist_budgets(budgets, fingerprint=fp)
            load_budgets(fingerprint=fp)
        per_call = (time.perf_counter() - t0) / n_iters
        print(f"engine round-trip (credit+roi+reallocate+persist+load): {per_call * 1000:.4f}ms/call over {n_iters} iterations")


if __name__ == "__main__":
    bench_engine_functions_isolated()
    bench_fit_wall_with_vs_without_budget_learning()
