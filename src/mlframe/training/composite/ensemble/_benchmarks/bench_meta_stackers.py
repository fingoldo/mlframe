"""Bench: NNLS vs ridge vs GBM cross-target meta-stackers.

Two scenarios over multiple seeds (single-seed wins do not count):

- ``interaction`` -- the optimal blend is region-switching / non-convex (y = comp_A where gate>0 else comp_B). A linear
  stacker (NNLS / ridge) cannot represent it; the shallow GBM stacker can. GBM should win the holdout RMSE by a wide margin.
- ``convex`` -- the optimal blend is a plain convex linear combination. NNLS is already optimal here; GBM/ridge should TIE
  (within noise), and NNLS must NOT regress.

Run: ``python -m mlframe.training.composite.ensemble._benchmarks.bench_meta_stackers``
"""
from __future__ import annotations

import numpy as np

from mlframe.training.composite.ensemble import (
    CompositeCrossTargetEnsemble as E,
    build_meta_stack_ensemble,
)


class _Col:
    """Component model that emits the k-th column of its input matrix."""

    def __init__(self, col: int) -> None:
        self.col = col

    def predict(self, X):  # noqa: D401
        return np.asarray(X)[:, self.col]


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def gen_interaction(n: int, seed: int):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=n)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = np.where(g > 0, a, b) + 0.05 * rng.normal(size=n)
    return np.column_stack([a, b, g]), y


def gen_convex(n: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    y = 0.6 * a + 0.3 * b + 0.1 * c + 0.05 * rng.normal(size=n)
    return np.column_stack([a, b, c]), y


def run(n: int = 2000, n_seeds: int = 6) -> dict:
    out = {}
    for name, gen in (("interaction", gen_interaction), ("convex", gen_convex)):
        nn, ri, gb = [], [], []
        for seed in range(n_seeds):
            xtr, ytr = gen(n, seed)
            xte, yte = gen(n, seed + 100)
            k = xtr.shape[1]
            models = [_Col(i) for i in range(k)]
            names = ["c%d" % i for i in range(k)]
            m_nn = E.from_nnls_stack(models, names, xtr, ytr)
            m_ri = build_meta_stack_ensemble(E, models, names, xtr, ytr, stacker="ridge")
            m_gb = build_meta_stack_ensemble(E, models, names, xtr, ytr, stacker="gbm")
            nn.append(_rmse(m_nn.predict(xte), yte))
            ri.append(_rmse(m_ri.predict(xte), yte))
            gb.append(_rmse(m_gb.predict(xte), yte))
        out[name] = {
            "nnls": float(np.mean(nn)),
            "ridge": float(np.mean(ri)),
            "gbm": float(np.mean(gb)),
        }
    return out


if __name__ == "__main__":
    res = run()
    for scen, d in res.items():
        print(scen, "NNLS=%.4f RIDGE=%.4f GBM=%.4f" % (d["nnls"], d["ridge"], d["gbm"]))
