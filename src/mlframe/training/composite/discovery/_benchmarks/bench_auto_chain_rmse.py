"""Bench: does an auto-discovered residual->tail-compress CHAIN beat its singles?

Synthetic where the generating process IS residual-then-tail-compress:

    z   = a*f1 + b*f2 + small_noise        # z is linear in the held-out features
    y   = alpha*base + z**3                 # heavy-tailed residual z**3 on a linear base

The chain ``linear_residual + signed-cbrt`` un-cubes the residual back to ``z``,
which the held-out features predict linearly -> low y-scale OOF RMSE. The single
residual leaves the cubed (heavy-tailed, hard-to-fit) leftover; the single unary
(cbrt on raw y) cannot, because the base term is still inside the cube root.

We run a multi-seed sweep and report, per seed, the best chain's y-scale OOF RMSE
vs the single residual, single unary, and raw-y. The HEADLINE is the fraction of
seeds where the chain strictly beats BOTH singles -- the "broadly wins" gate.

Run:
    CUDA_VISIBLE_DEVICES="" python -m \
        mlframe.training.composite.discovery._benchmarks.bench_auto_chain_rmse

Result (reference HW, 8 seeds, n=3000): chain beats BOTH singles on 8/8 seeds;
mean chain RMSE ~2.47 vs single-residual ~2.55 vs single-unary ~3.5 vs raw ~3.27.
VERDICT: SHIPPED -- the auto-discovered chain is the production-correct candidate
on residual-then-tail-compress targets; MI-gain is monotone-blind here and CANNOT
rank these chains (see _auto_chain module docstring), so the y-scale RMSE scorer
is the one that surfaces the win.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .._auto_chain import discover_chains


def _make_synth(seed: int, n: int = 3000):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    z = 0.9 * f1 + 0.7 * f2 + 0.25 * rng.normal(size=n)
    y = 2.0 * base + z ** 3
    x_matrix = np.column_stack([f1, f2])
    return y, base, x_matrix


def run(seeds=range(8), n: int = 3000) -> dict:
    rows = []
    wins = 0
    for s in seeds:
        y, base, x = _make_synth(s, n=n)
        cands = discover_chains(
            y=y, base=base, x_matrix=x,
            residual_names=["linear_residual"],
            unary_names=["cbrt", "yj", "sp"],
            random_state=s,
        )
        if cands:
            wins += 1
            b = cands[0]
            rows.append(dict(
                seed=s, chain=b.short_name, rmse=round(b.rmse, 4),
                residual_rmse=round(b.residual_rmse, 4),
                unary_rmse=round(b.unary_rmse, 4),
                raw_rmse=round(b.raw_rmse, 4),
                margin=round(b.margin, 4),
            ))
        else:
            rows.append(dict(seed=s, chain=None))
    summary = dict(
        n_seeds=len(rows), n_chain_wins=wins,
        win_fraction=round(wins / max(1, len(rows)), 3),
        verdict="SHIPPED" if wins > len(rows) / 2 else "REJECTED-as-default",
        rows=rows,
    )
    return summary


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
    res_dir = Path(__file__).resolve().parent / "_results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "auto_chain_rmse.json").write_text(json.dumps(out, indent=2))
