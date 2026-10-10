"""Operators B, C and G at n = 20000 / 100000 / 300000: acceptance, decision stability across n, seconds per operator and per column / pair, and a cProfile of one run.

Per (operator, case, n, seed): the fit half builds the engineered column, the held-out half scores it against ``stress.cheap_existing`` (best raw column plus the preset table on the best
two columns, because the full all-pairs table costs 4.6 s per pair at 100000 rows). Timings are single-process CPU wall seconds on a possibly loaded box. Cases are those of ``targets.CASES``
(W should win, H hard win, N / HN / 0 should not). No downstream models are fitted here (the decisions are MI based; downstream errors at 20000 are in ``summary.txt``).

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.large_n run <n> --ops B C G --cases W H N HN 0 --seeds 3`` /
``... profile <op> [--n 100000]`` / ``... summary 20000 100000 300000``. Output: ``wave2/results/rows_large_<n>.jsonl``, ``large_n.txt`` and ``profile_<op>_<n>.txt``.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from ..brainstorm.h import clean, ybins
from ..common._paths import results_dir
from .harness import heldout_mi, top_cols
from .stress import accept, cheap_existing, fit_op, read_rows, replay_op, write_rows
from .targets import CASES

__all__ = ["one_run", "summarize", "profile_op", "main"]

B_KW = {"m": 3.0}


def one_run(op: str, case: str, n: int, seed: int) -> dict:
    """One row: held-out MI of the engineered column and of the best existing candidate, the gain, acceptance verdicts and seconds for baseline, fit, replay."""
    gen, cols0 = CASES[op][case]
    rng = np.random.default_rng(30_000 + seed)
    X, y, truth = gen(rng, n)
    h = n // 2
    XA, XB, ya, yB = X[:h], X[h:], y[:h], y[h:]
    yb = ybins(ya, yB)
    XcA, XcB = clean(XA), clean(XB)
    cols = top_cols(XcA, yb, cols0)
    t0 = time.perf_counter()
    ex = cheap_existing(XcA, XcB, yb, cols, 2)
    t_ex = time.perf_counter() - t0
    t0 = time.perf_counter()
    fa, rec, info = fit_op(op, XA, ya, cols0 if op == "G" else cols, yb, seed, **(B_KW if op == "B" else {}))
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    fb = replay_op(op, rec, XB)
    t_rep = time.perf_counter() - t0
    tr, te = heldout_mi(fa, fb, yb)
    gain = te - ex[1]
    k = len(cols0 if op == "G" else cols)
    units = k * (k - 1) // 2 * 2 if op == "B" else k if op == "C" else 1
    row = {
        "op": op,
        "case": case,
        "n": n,
        "seed": seed,
        "n_fit": h,
        "ex_name": ex[4],
        "ex_te": ex[1],
        "new_te": te,
        "new_tr": tr,
        "gain": gain,
        "gain_n": gain * h,
        "info": info,
    }
    row.update(rel_gain=gain / ex[1] if ex[1] > 0 else None, t_baseline=t_ex, t_fit=t_fit, t_replay=t_rep, units=units, t_per_unit=t_fit / units)
    if truth is not None:
        row["truth_te"] = heldout_mi(truth[:h], truth[h:], yb)[1]
    row.update(accept(gain, ex[1], h))
    return row


def summarize(ns: list) -> str:
    """Per (op, case, n): mean+-sd of held-out MI and gain * n_fit, acceptance counts under c = 40 / floor / both, seconds of fit, per unit (column, pair x K, or 1 for G) and of replay."""
    out = ["# large-n study; unit = column (C), pair x K in (6, 10) (B), 1 (G); gain_n = held-out MI gain over best existing x n_fit"]
    for op in "BCG":
        for case in ("W", "H", "N", "HN", "0"):
            for n in ns:
                rs = list({(r["op"], r["case"], r["seed"]): r for r in read_rows(f"rows_large_{n}.jsonl") if r["op"] == op and r["case"] == case}.values())
                if not rs:
                    continue
                k = len(rs)

                def f(key, rs=rs):
                    """Mean+-sd text of field ``key`` over the rows."""
                    return f"{np.mean([r[key] for r in rs]):+.4f}+-{np.std([r[key] for r in rs]):.4f}"

                out.append(
                    f"{op}_{case:2s} n={n:6d} seeds={k} ex_te={f('ex_te')} new_te={f('new_te')} gain_n={np.mean([r['gain_n'] for r in rs]):8.1f} (min {min(r['gain_n'] for r in rs):8.1f} "
                    f"max {max(r['gain_n'] for r in rs):8.1f}) acc c40={sum(r['acc_c40'] for r in rs)}/{k} floor={sum(r['acc_floor'] for r in rs)}/{k} both={sum(r['acc_both'] for r in rs)}/{k} "
                    f"t_fit={np.mean([r['t_fit'] for r in rs]):6.2f}s per_unit={np.mean([r['t_per_unit'] for r in rs]):6.3f}s replay={np.mean([r['t_replay'] for r in rs]):6.3f}s "
                    f"baseline={np.mean([r['t_baseline'] for r in rs]):5.1f}s"
                )
    return "\n".join(out)


def profile_op(op: str, n: int, case: str = "H", top: int = 3) -> str:
    """cProfile one fit + replay of operator ``op`` at ``n`` rows; returns the ``top`` entries by self time (``tottime``) as text and writes the top 15 to ``profile_<op>_<n>.txt``."""
    gen, cols0 = CASES[op][case]
    X, y, _ = gen(np.random.default_rng(30_000), n)
    h = n // 2
    XA, XB, ya, yB = X[:h], X[h:], y[:h], y[h:]
    yb = ybins(ya, yB)
    cols = top_cols(clean(XA), yb, cols0)
    fit_op(op, XA[:2000], ya[:2000], cols0 if op == "G" else cols, ybins(ya[:2000], yB[:2000]), 0, **(B_KW if op == "B" else {}))
    pr = cProfile.Profile()
    pr.enable()
    fa, rec, _ = fit_op(op, XA, ya, cols0 if op == "G" else cols, yb, 0, **(B_KW if op == "B" else {}))
    replay_op(op, rec, XB)
    pr.disable()
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    st.print_stats(15)
    (results_dir("wave2") / f"profile_{op}_{n}.txt").write_text(buf.getvalue())
    total = st.total_tt
    items = sorted(st.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:top]
    return f"{op} n={n} case={case} total={total:.2f}s: " + "; ".join(
        f"{k[2]} ({k[0].split(chr(92))[-1].split('/')[-1]}:{k[1]}) self={v[2]:.2f}s calls={v[1]}" for k, v in items
    )


def main(argv=None) -> None:
    """CLI: ``run``, ``profile`` or ``summary`` (see the module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "profile", "summary"])
    ap.add_argument("arg", nargs="*")
    ap.add_argument("--ops", nargs="+", default=["B", "C", "G"])
    ap.add_argument("--cases", nargs="+", default=["W", "H", "N", "HN", "0"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n", type=int, default=100_000)
    a = ap.parse_args(argv)
    if a.mode == "summary":
        txt = summarize([int(x) for x in a.arg])
        (results_dir("wave2") / "large_n.txt").write_text(txt)
        print(txt)
    elif a.mode == "profile":
        txt = "\n".join(profile_op(op, a.n) for op in (a.arg or a.ops))
        with (results_dir("wave2") / "profile_top3.txt").open("a") as fh:
            fh.write(txt + "\n")
        print(txt)
    else:
        n = int(a.arg[0])
        for op in a.ops:
            for case in a.cases:
                for s in range(a.seed0, a.seed0 + a.seeds):
                    r = one_run(op, case, n, s)
                    write_rows(f"rows_large_{n}.jsonl", [r])
                print(f"{op}_{case} n={n} seeds={a.seeds} last gain_n={r['gain_n']:.1f} t_fit={r['t_fit']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
