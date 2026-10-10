"""Operator B against an additive baseline made of two C warps, and the 2-D shrinkage default (m = 3 / 10 / 20) at n = 20000.

Per (case, seed) the fit half chooses the pair and the cell table (``stress.fit_B2``, one table per ``m``) and the per-column cross-fitted warps (C) of the two columns of that pair.
Feature sets, all frozen from the fit half and scored on the held-out half by 5-fold CV: ``raw``; ``cw`` = raw + the two C warps (additive baseline); ``cwB_m<m>`` = raw + warps + B;
``B_m<m>`` = raw + B alone. Held-out MI: B, the additive composite (sum of the two warps), the best single warp and the best raw column.
Derived keys ``<metric>|<model>|B_over_cw_m<m>`` = relative error reduction of ``cwB`` over ``cw``.

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.b_vs_additive run <n> --cases W H --seeds 6 [--ms 3 10 20]`` then ``... summary``.
Output: ``wave2/results/rows_Badd_<n>.jsonl`` and ``b_vs_additive.txt``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..brainstorm.h import MI, clean, ybins
from ..common._paths import results_dir
from ..common.downstream import make_models
from .harness import cv_relative, prep, top_cols
from .stress import fit_B2, ms, read_rows, write_rows
from .targets import CASES
from .warp_service import oof_fit, replay

__all__ = ["one_case", "summarize", "main"]


def one_case(case: str, n: int, seed: int, ms_: tuple = (3.0, 10.0, 20.0), hgb_iter: int = 150) -> dict:
    """One (case, seed) row: held-out MI and downstream errors of the four feature sets for every shrinkage ``m`` in ``ms_``."""
    gen, cols0 = CASES["B"][case]
    rng = np.random.default_rng(10_000 + seed)
    X, y, truth = gen(rng, n)
    h = n // 2
    XA, XB, ya, yB = X[:h], X[h:], y[:h], y[h:]
    yb = ybins(ya, yB)
    cols = top_cols(XA, yb, cols0)
    row = {"case": case, "n": n, "seed": seed, "n_fit": h}
    sets = {"raw": XB}
    bests = {}
    for m in ms_:
        t0 = time.perf_counter()
        fa, rec, info = fit_B2(XA, ya, cols, yb, seed, m=m)
        row[f"t_B_m{m:g}"] = time.perf_counter() - t0
        fb = replay(rec, XB)
        bests[m] = (fa, fb, info)
        row[f"pair_m{m:g}"] = info["pair"]
        row[f"K_m{m:g}"] = info["K"]
        row[f"MI_B_m{m:g}"] = MI(fa, fb, yb)[1]
    i, j = bests[ms_[0]][2]["pair"]
    wa, wb = [], []
    for c in (i, j):
        f, rc = oof_fit("warp1d", XA, ya, [c], seed=seed, nb=20)
        wa.append(f)
        wb.append(replay(rc, XB))
    cwA = np.column_stack([prep(a, a) for a in wa])
    cwB = np.column_stack([prep(a, b) for a, b in zip(wa, wb)])
    sets["cw"] = np.column_stack([XB, cwB])
    addA, addB = wa[0] + wa[1], wb[0] + wb[1]
    row["MI_add"] = MI(addA, addB, yb)[1]
    row["MI_warp_best"] = max(MI(a, b, yb)[1] for a, b in zip(wa, wb))
    row["MI_raw_best"] = max(MI(clean(XA[:, c]), clean(XB[:, c]), yb)[1] for c in cols)
    if truth is not None:
        row["MI_truth"] = MI(truth[:h], truth[h:], yb)[1]
    del cwA
    for m in ms_:
        fa, fb, _ = bests[m]
        pb = prep(fa, fb)[:, None]
        sets[f"cwB_m{m:g}"] = np.column_stack([sets["cw"], pb])
        sets[f"B_m{m:g}"] = np.column_stack([XB, pb])
    row.update(cv_relative(sets, yB, seed=seed, models=make_models(hgb_iter)))
    for m in ms_:
        for met in ("mae", "rmse"):
            for mod in ("ridge", "hgb"):
                a, b = row[f"{met}|{mod}|cw"], row[f"{met}|{mod}|cwB_m{m:g}"]
                row[f"{met}|{mod}|B_over_cw_m{m:g}"] = 1.0 - (1.0 - b) / (1.0 - a)
    return row


def summarize(n: int) -> str:
    """Text table of the rows of one ``n``: per case and ``m``, mean+-sd over seeds."""
    rows = [r for r in read_rows(f"rows_Badd_{n}.jsonl")]
    lines = [f"# B vs additive C-warp baseline, n={n} (relative error reduction, positive = better; seeds per case in brackets)"]
    for case in ("W", "H", "N", "HN", "0"):
        rs = [r for r in rows if r["case"] == case]
        if not rs:
            continue
        lines.append(
            f"[{case}] seeds={len(rs)} MI raw_best={ms([r['MI_raw_best'] for r in rs])} warp_best={ms([r['MI_warp_best'] for r in rs])} "
            f"add={ms([r['MI_add'] for r in rs])} truth={ms([r.get('MI_truth') for r in rs])}"
        )
        lines.append("    vs raw     : " + " ".join(f"{k}={ms([r[k] for r in rs])}" for k in ("mae|ridge|cw", "rmse|ridge|cw", "mae|hgb|cw", "rmse|hgb|cw")))
        for m in sorted({k.split("_m")[-1] for k in rs[0] if k.startswith("MI_B_m")}, key=float):
            lines.append(
                f"    m={m:>3}: MI_B={ms([r[f'MI_B_m{m}'] for r in rs])} (B-add {ms([r[f'MI_B_m{m}'] - r['MI_add'] for r in rs])}); "
                + " ".join(
                    f"{met}|{mod}|cwB={ms([r[f'{met}|{mod}|cwB_m{m}'] for r in rs])} B_only={ms([r[f'{met}|{mod}|B_m{m}'] for r in rs])} B/cw={ms([r[f'{met}|{mod}|B_over_cw_m{m}'] for r in rs])};"
                    for met, mod in (("mae", "ridge"), ("rmse", "ridge"), ("mae", "hgb"), ("rmse", "hgb"))
                )
            )
    return "\n".join(lines)


def main(argv=None) -> None:
    """CLI: ``run`` appends rows, ``summary`` writes ``b_vs_additive.txt`` for the given ``n`` values."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("n", type=int, nargs="+")
    ap.add_argument("--cases", nargs="+", default=["W", "H", "N", "HN", "0"])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--ms", type=float, nargs="+", default=[3.0, 10.0, 20.0])
    a = ap.parse_args(argv)
    if a.mode == "summary":
        txt = "\n\n".join(summarize(n) for n in a.n)
        (results_dir("wave2") / "b_vs_additive.txt").write_text(txt)
        print(txt)
        return
    n = a.n[0]
    for case in a.cases:
        for s in range(a.seed0, a.seed0 + a.seeds):
            t0 = time.time()
            r = one_case(case, n, s, tuple(a.ms))
            write_rows(f"rows_Badd_{n}.jsonl", [r])
            print(f"B_add {case} n={n} seed={s} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
