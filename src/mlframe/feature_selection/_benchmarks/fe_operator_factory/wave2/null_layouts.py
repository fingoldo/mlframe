"""Hard pure-noise layouts for operators B, C and G: false-accept rates under the acceptance rules c = 40, the 3% relative-gain floor and both.

Layouts (``LAYOUTS``): ``a20`` / ``a40`` 20 / 40 irrelevant uniform columns; ``tails`` heavy-tailed and skewed columns (t with 1.5 df, lognormal, exponential, Pareto); ``weak`` ten columns and a
target that is noise plus a weak real linear term in one column; ``discrete`` binary / 3-level / 10-level / 50-level integer columns; ``discrete_sig`` the same with a step signal in the
10-level column; ``dupconst`` near-duplicate, exact-duplicate, constant and near-constant columns; ``dupconst_sig`` the same with a weak linear signal in column 0; ``missing`` 20% NaN per
column; ``missing_sig`` the same with a weak signal in column 0 (computed before masking). The target is pure noise unless the name ends in ``sig`` or is ``weak``.

Per (layout, operator, n, seed): the fit half chooses the engineered column, the held-out half scores it; the best existing candidate is ``stress.cheap_existing`` (best raw column plus the
preset table on the best three, at n >= 100000 two, columns). Operator input is capped as in the design: G sees the 16, B the ``--b-cap`` best columns (by univariate train MI).

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.null_layouts run <n> --seeds 12 [--layouts ...] [--ops B C G] [--b-cap 40]`` then ``... summary``.
Output: ``wave2/results/rows_null_<n>.jsonl`` and ``null_layouts.txt``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ..brainstorm.h import clean, ybins
from ..common._paths import results_dir
from .harness import heldout_mi
from .stress import accept, cap_columns, cheap_existing, fit_op, read_rows, replay_op, write_rows

__all__ = ["LAYOUTS", "one_run", "summarize", "main"]


def _std(v: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance copy of ``v``."""
    return (v - v.mean()) / v.std()


def _unif(p):
    """Generator of ``p`` uniform columns."""
    return lambda rng, n: rng.random((n, p))


def _tails(rng, n):
    """Heavy-tailed and skewed columns."""
    return np.column_stack(
        [rng.standard_t(1.5, n), rng.lognormal(0, 1.5, n), rng.exponential(1, n), 1 + rng.pareto(1.2, n), rng.standard_t(3, n), rng.lognormal(0, 0.5, n)]
    )


def _disc(rng, n):
    """Integer columns with 2, 3, 10 and 50 levels plus two continuous ones."""
    return np.column_stack([rng.integers(0, 2, n), rng.integers(0, 3, n), rng.integers(0, 10, n), rng.integers(0, 50, n), rng.random(n), rng.random(n)]).astype(
        float
    )


def _dup(rng, n):
    """Columns: 0-3 uniform, 4 = col 0 + 1e-6 noise, 5 = col 1 exactly, 6 constant 1, 7 constant 0, 8 = 1 except 0.1% of the rows."""
    X = rng.random((n, 9))
    X[:, 4] = X[:, 0] + 1e-6 * rng.standard_normal(n)
    X[:, 5] = X[:, 1]
    X[:, 6] = 1.0
    X[:, 7] = 0.0
    X[:, 8] = np.where(rng.random(n) < 0.001, 0.0, 1.0)
    return X


def _miss(rng, n):
    """Ten uniform columns; the NaN mask is applied by the layout wrapper after the target is drawn."""
    return rng.random((n, 10))


def _make(gen, sig_col=None, slope=0.25, step=False, mask=0.0):
    """Layout factory: columns from ``gen``; target = noise (+ slope * standardised signal column or its step function); ``mask`` fraction of NaN per column applied last."""

    def f(rng, n):
        """One data set ``(X, y)``."""
        X = gen(rng, n)
        y = rng.standard_normal(n)
        if sig_col is not None:
            s = _std(X[:, sig_col])
            if step:
                s = _std((X[:, sig_col] >= 5).astype(float) + 0.5 * (X[:, sig_col] % 2))
            y = y + slope * s
        if mask:
            X = np.where(rng.random(X.shape) < mask, np.nan, X)
        return X, y

    return f


LAYOUTS = {
    "a20": _make(_unif(20)),
    "a40": _make(_unif(40)),
    "tails": _make(_tails),
    "weak": _make(_unif(10), sig_col=7),
    "discrete": _make(_disc),
    "discrete_sig": _make(_disc, sig_col=2, slope=0.3, step=True),
    "dupconst": _make(_dup),
    "dupconst_sig": _make(_dup, sig_col=0),
    "missing": _make(_miss, mask=0.2),
    "missing_sig": _make(_miss, sig_col=0, mask=0.2),
}


def one_run(layout: str, op: str, n: int, seed: int, b_cap: int = 40, g_cap: int = 16) -> dict:
    """One (layout, operator, seed) row with held-out MI of the engineered column, of the best existing candidate, the gain and the acceptance verdicts."""
    rng = np.random.default_rng(20_000 + seed)
    X, y = LAYOUTS[layout](rng, n)
    h = n // 2
    XA, XB, ya, yB = X[:h], X[h:], y[:h], y[h:]
    yb = ybins(ya, yB)
    XcA, XcB = clean(XA), clean(XB)
    p = X.shape[1]
    allc = list(range(p))
    cap = {"B": b_cap, "C": p, "G": g_cap}[op]
    cols = allc if cap >= p else cap_columns(XcA, yb, allc, cap)
    t0 = time.perf_counter()
    ex = cheap_existing(XcA, XcB, yb, allc, 3 if n < 100_000 else 2)
    t_ex = time.perf_counter() - t0
    t0 = time.perf_counter()
    fa, rec, info = fit_op(op, XA, ya, cols, yb, seed, **({"m": 3.0} if op == "B" else {}))
    t_fit = time.perf_counter() - t0
    fb = replay_op(op, rec, XB)
    tr, te = heldout_mi(fa, fb, yb)
    gain = te - ex[1]
    row = {
        "layout": layout,
        "op": op,
        "n": n,
        "seed": seed,
        "n_fit": h,
        "p": p,
        "n_cols_used": len(cols),
        "ex_name": ex[4],
        "ex_te": ex[1],
        "new_te": te,
        "new_tr": tr,
    }
    row.update(gain=gain, gain_n=gain * h, rel_gain=gain / ex[1] if ex[1] > 0 else None, t_fit=t_fit, t_existing=t_ex, info=info)
    row.update(accept(gain, ex[1], h))
    return row


def summarize(ns: list) -> str:
    """Accept-rate table per (n, layout, operator): c = 40 alone, floor alone, both; plus the largest gain * n and the largest relative gain."""
    out = []
    for n in ns:
        rows = list({(r["layout"], r["op"], r["seed"]): r for r in read_rows(f"rows_null_{n}.jsonl")}.values())
        out.append(f"# false-accept rates, n={n} (n_fit={n // 2}); acc = accepted seeds / seeds; gain_n = held-out MI gain over best existing x n_fit")
        out.append(
            f"{'layout':13s} {'op':2s} {'seeds':>5s} {'c40':>6s} {'floor3%':>8s} {'both':>6s} {'max gain_n':>11s} {'max rel_gain':>13s} {'mean t_fit':>10s}"
        )
        for lay in LAYOUTS:
            for op in "BCG":
                rs = [r for r in rows if r["layout"] == lay and r["op"] == op]
                if not rs:
                    continue
                k = len(rs)
                rel = [r["rel_gain"] for r in rs if r["rel_gain"] is not None]
                out.append(
                    f"{lay:13s} {op:2s} {k:5d} {sum(r['acc_c40'] for r in rs):3d}/{k:<2d} {sum(r['acc_floor'] for r in rs):4d}/{k:<3d} {sum(r['acc_both'] for r in rs):3d}/{k:<2d} "
                    f"{max(r['gain_n'] for r in rs):11.1f} {max(rel) if rel else float('nan'):13.3f} {np.mean([r['t_fit'] for r in rs]):10.2f}"
                )
        out.append("")
    return "\n".join(out)


def main(argv=None) -> None:
    """CLI: ``run`` appends rows for the chosen layouts / operators, ``summary`` writes ``null_layouts.txt``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("n", type=int, nargs="+")
    ap.add_argument("--layouts", nargs="+", default=list(LAYOUTS))
    ap.add_argument("--ops", nargs="+", default=["B", "C", "G"])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--b-cap", type=int, default=40)
    a = ap.parse_args(argv)
    if a.mode == "summary":
        txt = summarize(a.n)
        (results_dir("wave2") / "null_layouts.txt").write_text(txt)
        print(txt)
        return
    n = a.n[0]
    for lay in a.layouts:
        for op in a.ops:
            for s in range(a.seed0, a.seed0 + a.seeds):
                r = one_run(lay, op, n, s, a.b_cap)
                write_rows(f"rows_null_{n}.jsonl", [r])
            print(f"null {lay} {op} n={n} seeds={a.seeds} last gain_n={r['gain_n']:.1f} t_fit={r['t_fit']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
