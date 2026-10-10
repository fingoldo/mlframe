"""Operator M: does the number of bins of the additive model change the residual pair screen (lack-of-fit leakage on the additive N target, true-pair rank on W)?

Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.wave2.m_bins``. Output: ``wave2/results/m_bins.json``. Cell statistic: 6 x 6 ANOVA minus its null expectation, times n (see ``run_M.cell_screen``).
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from ..common._paths import results_dir
from .additive_residual import additive_oof_residual, bins_for
from .run_M import _qcodes, cell_screen
from .targets import gen_M


def main(seeds: int = 8) -> None:
    """Grid over n, hard, kind and bins; print and store the mean / max of the top residual cell score times n and the mean rank of the first true pair."""
    out = {}
    for n in (5000, 20000):
        for hard in (False, True):
            for kind in ("W", "N", "0"):
                for nb in (15, 30, 60, "auto"):
                    sc, rk, hit = [], [], []
                    for s in range(seeds):
                        X, y, tp = gen_M(np.random.default_rng(20_000 + s), n, kind, hard)
                        h = n // 2
                        XA, ya = X[:h], y[:h]
                        pairs = list(itertools.combinations(range(X.shape[1]), 2))
                        b = bins_for(h) if nb == "auto" else nb
                        r = additive_oof_residual(XA, ya, nb=b, seed=s)
                        Q = np.column_stack([_qcodes(XA[:, j], 6) for j in range(XA.shape[1])])
                        c = cell_screen(Q, r, pairs)
                        sc.append(float(c.max() * h))
                        if tp:
                            order = np.argsort(-c)
                            rk.append(int(np.where(order == pairs.index(tp[0]))[0][0]) + 1)
                            hit.append(tuple(pairs[int(order[0])]) in tp)
                    key = f"n={n}|{'hard' if hard else 'ideal'}|{kind}|nb={nb}"
                    out[key] = {
                        "top_score_x_n_mean": float(np.mean(sc)),
                        "top_score_x_n_max": float(np.max(sc)),
                        "mean_rank_first_true": float(np.mean(rk)) if rk else None,
                        "top1_true": float(np.mean(hit)) if hit else None,
                    }
                    print(key, out[key], flush=True)
    (results_dir("wave2") / "m_bins.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
