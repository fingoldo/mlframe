"""Shared pieces of the wave-2 follow-up studies (B against an additive baseline, hard null layouts, classification targets, large n).

* ``fit_B2``: operator B pair search with the shrinkage pseudo-count kept apart from the train MI (``run_ops.fit_B`` reuses the name ``m`` for both, so every pair after the first
  was fitted with ``m`` equal to the previous train MI, about 0.1 instead of 3 / 10 / 20).
* ``cheap_existing``: a cheaper stand-in for the best existing candidate (best raw column plus the 1734-combo preset on the top pairs only).
* ``fit_op`` / ``replay_op``: one entry point per operator on possibly NaN-containing columns.
* ``accept``: the acceptance rules compared in the studies (c alone, relative-gain floor, both).
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np

from ..brainstorm.h import MI, clean, existing_pair, mi_pair
from ..common._paths import results_dir
from .harness import top_cols
from .rowstats import fit_rowstat, replay_rowstat
from .warp_service import oof_fit, replay

__all__ = [
    "C_PROJECT",
    "REL_FLOOR",
    "fit_B2",
    "fit_C2",
    "fit_G2",
    "fit_op",
    "replay_op",
    "cheap_existing",
    "accept",
    "write_rows",
    "read_rows",
    "ms",
    "cap_columns",
]

C_PROJECT = 40.0
REL_FLOOR = 0.03


def _train_mi(f: np.ndarray, yb: tuple) -> float:
    """In-sample MI of a (out-of-fold) training feature against the fit-half target codes."""
    f = clean(f)
    return mi_pair(f, f, yb[0], yb[0], yb[2])[0]


def fit_B2(XA, ya, cols, yb, seed, m: float = 3.0, ks=(6, 10)):
    """Operator B: over all pairs of ``cols`` and the grid sizes ``ks`` keep the cross-fitted cell table with the best train MI; returns (train column, recipe, info)."""
    best = None
    for i, j in itertools.combinations(cols, 2):
        for K in ks:
            f, rec = oof_fit("cell2d", XA, ya, [i, j], seed=seed, K=K, m=m)
            score = _train_mi(f, yb)
            if best is None or score > best[0]:
                best = (score, f, rec, {"pair": [i, j], "K": K, "m": m})
    return best[1], best[2], best[3]


def fit_C2(XA, ya, cols, yb, seed, nb: int = 20):
    """Operator C: per column a cross-fitted ``nb``-bin warp; keep the column with the best train MI; returns (train column, recipe, info)."""
    best = None
    for j in cols:
        f, rec = oof_fit("warp1d", XA, ya, [j], seed=seed, nb=nb)
        score = _train_mi(f, yb)
        if best is None or score > best[0]:
            best = (score, f, rec, {"col": j})
    return best[1], best[2], best[3]


def fit_G2(XA, ya, cols, yb, seed):
    """Operator G: learned (statistic, subset) over ``cols``; returns (train column, recipe, info)."""
    rec = fit_rowstat(XA, ya, list(cols), seed=seed)
    return replay_rowstat(rec, XA), rec, {"stat": rec["stat"], "cols": rec["src"]}


def fit_op(op: str, XA, ya, cols, yb, seed, **kw):
    """Dispatch to the fit function of operator ``op`` (``B``, ``C`` or ``G``)."""
    return {"B": fit_B2, "C": fit_C2, "G": fit_G2}[op](XA, ya, cols, yb, seed, **kw)


def replay_op(op: str, rec: dict, X) -> np.ndarray:
    """Transform-time column of operator ``op`` from its recipe."""
    return replay_rowstat(rec, X) if op == "G" else replay(rec, X)


def cap_columns(Xc: np.ndarray, yb: tuple, cols: list, cap: int) -> list:
    """The ``cap`` columns with the largest univariate train MI (``Xc`` must be NaN-free)."""
    return top_cols(Xc, yb, list(cols), cap)


def cheap_existing(Xc_tr, Xc_te, yb, cols, npair_cols: int = 3):
    """Cheaper best existing candidate: best raw column by train MI, then the 1734-combo preset on every pair of the ``npair_cols`` best columns.

    Returns ``(train MI, held-out MI, train col, held-out col, name)``; inputs must be NaN-free (``clean``). A weaker baseline than the full all-pairs table, so gains measured against it are
    upper bounds of the gains against the production candidate pool.
    """
    sc = [(mi_pair(Xc_tr[:, c], Xc_te[:, c], yb[0], yb[1], yb[2])[0], c) for c in cols]
    sc.sort(reverse=True)
    a, c = sc[0]
    best = (a, MI(Xc_tr[:, c], Xc_te[:, c], yb)[1], Xc_tr[:, c], Xc_te[:, c], f"raw{c}")
    top = [c for _, c in sc[:npair_cols]]
    for i, j in itertools.combinations(top, 2):
        r = existing_pair(Xc_tr[:, i], Xc_tr[:, j], Xc_te[:, i], Xc_te[:, j], yb)
        if r[0] > best[0]:
            best = r
    return best


def accept(gain: float, ex_te: float, n_fit: int, c: float = C_PROJECT, rel: float = REL_FLOOR) -> dict:
    """Acceptance verdicts for a held-out MI ``gain`` over the best existing candidate (held-out MI ``ex_te``) at ``n_fit`` fit rows.

    ``c40``: ``gain * n_fit > c``; ``floor``: ``gain >= rel * ex_te``; ``both``: the two together.
    """
    a = bool(gain * n_fit > c)
    b = bool(gain >= rel * max(ex_te, 0.0))
    return {"acc_c40": a, "acc_floor": b, "acc_both": a and b}


def write_rows(name: str, rows: list) -> None:
    """Append JSON rows to ``wave2/results/<name>``."""
    out = results_dir("wave2") / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_rows(pattern: str) -> list:
    """All JSON rows of the files matching ``pattern`` in ``wave2/results``."""
    rows = []
    for f in sorted(results_dir("wave2").glob(pattern)):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return rows


def ms(v) -> str:
    """``mean+-sd`` text of a list (NaN and None ignored)."""
    a = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return f"{a.mean():+.3f}+-{a.std():.3f}" if len(a) else "nan"


def timer() -> float:
    """Monotonic seconds."""
    return time.perf_counter()
