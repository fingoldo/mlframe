"""biz_value: richer cross-target meta-stackers (GBM / ridge) vs NNLS.

Two synthetics across multiple seeds (single-seed wins do not count):

- ``interaction`` -- the optimal blend is region-switching / non-convex (y = comp_A where gate>0 else comp_B). A linear
  stacker (NNLS / ridge) cannot represent it; the shallow GBM stacker can. Measured GBM holdout RMSE ~0.136 vs NNLS ~0.709
  (~5.2x better). Floor pinned at 3x to absorb seed noise while still tripping a real GBM-stacker regression.
- ``convex`` -- the optimal blend is a plain convex linear combination. NNLS is already optimal; the GBM stacker must NOT
  beat it (and ridge ties it). This LOCKS the "keep NNLS default" verdict: if GBM ever silently started winning the convex
  case too, the default-flip decision would need revisiting -- this test forces that re-evaluation.

Verdict (recorded here + in bench_meta_stackers.py): GBM wins decisively only on the non-convex blend; on the convex blend
NNLS ties ridge and beats GBM. So a richer stacker does NOT win on the MAJORITY of (convex) scenarios -> NNLS stays the
DEFAULT, GBM/ridge ship as opt-in (REJECTED-as-default != deleted).
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


class _Col:
    def __init__(self, col: int) -> None:
        self.col = col

    def predict(self, X):
        return np.asarray(X)[:, self.col]


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _gen_interaction(n, seed):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=n)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = np.where(g > 0, a, b) + 0.05 * rng.normal(size=n)
    return np.column_stack([a, b, g]), y


def _gen_convex(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.normal(size=n)
    y = 0.6 * a + 0.3 * b + 0.1 * c + 0.05 * rng.normal(size=n)
    return np.column_stack([a, b, c]), y


def _mean_rmse(gen, stacker, n_seeds=6, n=1500):
    from mlframe.training.composite.ensemble import (
        CompositeCrossTargetEnsemble as E, build_meta_stack_ensemble,
    )
    out = []
    for seed in range(n_seeds):
        xtr, ytr = gen(n, seed)
        xte, yte = gen(n, seed + 100)
        k = xtr.shape[1]
        models = [_Col(i) for i in range(k)]
        names = ["c%d" % i for i in range(k)]
        if stacker == "nnls":
            ens = E.from_nnls_stack(models, names, xtr, ytr)
        else:
            ens = build_meta_stack_ensemble(E, models, names, xtr, ytr, stacker=stacker)
        out.append(_rmse(ens.predict(xte), yte))
    return float(np.mean(out))


def test_biz_val_gbm_stacker_wins_on_interaction_blend():
    """GBM meta-stacker must beat NNLS by >= 3x holdout RMSE on the non-convex region-switch blend (measured ~5.2x)."""
    nnls = _mean_rmse(_gen_interaction, "nnls")
    gbm = _mean_rmse(_gen_interaction, "gbm")
    ratio = nnls / max(gbm, 1e-12)
    assert ratio >= 3.0, "GBM stacker should win >=3x on interaction blend (NNLS=%.4f GBM=%.4f ratio=%.2f)" % (nnls, gbm, ratio)


def test_biz_val_nnls_not_regressed_on_convex_blend():
    """On a plain convex blend NNLS is optimal: ridge must TIE it (within 10%) and GBM must NOT beat it -> keeps NNLS default."""
    nnls = _mean_rmse(_gen_convex, "nnls")
    ridge = _mean_rmse(_gen_convex, "ridge")
    gbm = _mean_rmse(_gen_convex, "gbm")
    assert ridge <= nnls * 1.10, "ridge should tie NNLS on convex (NNLS=%.4f RIDGE=%.4f)" % (nnls, ridge)
    assert gbm >= nnls * 0.98, "GBM must NOT beat NNLS on a trivially-convex blend (NNLS=%.4f GBM=%.4f)" % (nnls, gbm)
