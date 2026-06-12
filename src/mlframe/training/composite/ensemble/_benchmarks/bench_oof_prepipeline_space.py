"""Bench: OOF pre_pipeline projection -- raw-X fallback (old) vs leak-free clone-fit (new).

Context. The deployed component (``PrePipelinePredictShim.predict``) routes
every predict through ``pre_pipeline.transform`` and refuses to feed raw X to
the inner. The honest-OOF refit loop must mirror that. The old ``_transform_via``
called ``pp.transform`` per OOF slice inside the hot loop; on an UNFITTED
pre_pipeline this raised ``NotFittedError``, logged a warning, and fell back to
RAW X -- a per-slice raise+log+fallback AND an OOF scored in a DIFFERENT feature
space than deployed (biasing the NNLS weights). The new ``_transform_pair_via``
fits a leak-free clone once per (component, slice-pair) and reuses it.

What this bench shows (measured 2026-06-12, 600x2, 5 components, kfold=5,
best-of-N wall per call, full prod-style WARNING logging handler attached):

  case            path                         wall ms/call   NotFitted warns/call
  FITTED pp       new == old (pp.transform)        ~61              0
  UNFITTED pp     OLD raise+log+rawX               ~53             50
  UNFITTED pp     NEW clone-fit+reuse              ~113             0

Read this correctly: on the UNFITTED case the new path is ~2.1x SLOWER because
it now does the CORRECT (previously-skipped) work -- fitting a clone of the
pre_pipeline on the train slice so the OOF lives in the deployed space. The old
path was "fast" only because it raised cheaply and scored the WRONG (raw)
space. This is a CORRECTNESS fix (val/OOF must mirror deployment), not a perf
win; the extra wall is the price of an honest OOF. On the FITTED pp case (the
suite-normal path: the entry's pre_pipeline was fit-transformed on full train
during the main pass) new == old -- both call ``pp.transform`` once per slice,
0 NotFitted warnings, bit-identical projection. So the realistic production
path pays nothing.

REJECTED != DELETED note: a "cache the fitted clone across components that share
the same pp on the same fold-train slice" memo was considered to claw back the
unfitted-case cost. Not implemented: (a) the suite-normal path is FITTED pp
(zero clone fits), so the win applies only to the already-degenerate unfitted
case, and (b) kfold requires a per-fold fit (different train slice each fold),
so cross-fold reuse is leak-unsafe. Re-open only if a real workload is shown to
land many UNFITTED-pp components sharing one pp on one slice.

Run: ``python -m mlframe.training.composite.ensemble._benchmarks.bench_oof_prepipeline_space``
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import mlframe.training.composite.ensemble as oof_mod
from mlframe.training.composite.ensemble import compute_oof_holdout_predictions
from mlframe.training.composite.post_shim import PrePipelinePredictShim


def _old_pair(pp, X_train, X_holdout, *, y_train=None):
    """The pre-fix raise+log+rawX per-slice behaviour, for A/B timing only."""
    def _tv(p, X):
        if p is None:
            return X
        try:
            return p.transform(X)
        except Exception as exc:  # noqa: BLE001
            oof_mod.logger.warning(
                "[ensemble] pre_pipeline.transform failed (%s: %s); falling back "
                "to RAW X for this slice -- OOF may evaluate a different space "
                "than deployed.",
                type(exc).__name__, exc,
            )
            return X
    return _tv(pp, X_train), _tv(pp, X_holdout)


def _bench(comps, names, X, y, *, iters=20, warmup=3):
    specs = [None] * len(comps)

    def _run():
        return compute_oof_holdout_predictions(
            comps, names, specs, X, y, {}, 0.3, 0, kfold=5,
        )

    for _ in range(warmup):
        _run()
    best = min(
        (lambda t0: (_run(), time.perf_counter() - t0)[1])(time.perf_counter())
        for _ in range(iters)
    )
    return best


def main() -> None:
    # Prod-style logging handler so the per-slice warning formatting cost (the
    # thing that dominated the old unfitted path) is included in the wall.
    n_warns = {"n": 0}

    class _H(logging.Handler):
        def emit(self, r):
            if "pre_pipeline.transform failed" in r.getMessage():
                n_warns["n"] += 1

    h = _H()
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    oof_mod.logger.addHandler(h)
    oof_mod.logger.setLevel(logging.WARNING)
    oof_mod.logger.propagate = False

    rng = np.random.default_rng(3)
    n = 600
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    y = (X["f0"].values * 2.0 + rng.normal(size=n) * 0.1).astype(np.float64)

    pp = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
    inner = Ridge().fit(pp.fit_transform(X), y)

    names = [f"raw#{i}" for i in range(5)]

    # 1) FITTED pp (suite-normal): new == old, 0 warnings.
    shim_fit = PrePipelinePredictShim(inner, pp, "raw#0")
    n_warns["n"] = 0
    t_fit = _bench([shim_fit] * 5, names, X, y)
    print(f"FITTED pp   NEW           : {t_fit * 1e3:8.2f} ms/call  warns/call={n_warns['n'] / 20:.0f}")

    # 2) UNFITTED pp -- NEW clone-fit.
    pp_unfit = Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())])
    shim_unfit = PrePipelinePredictShim(inner, pp_unfit, "raw#0")
    n_warns["n"] = 0
    t_new = _bench([shim_unfit] * 5, names, X, y)
    print(f"UNFITTED pp NEW clone-fit : {t_new * 1e3:8.2f} ms/call  warns/call={n_warns['n'] / 20:.0f}")

    # 3) UNFITTED pp -- OLD raise+log+rawX (monkeypatched).
    _orig = oof_mod._transform_pair_via
    oof_mod._transform_pair_via = _old_pair
    try:
        n_warns["n"] = 0
        t_old = _bench([shim_unfit] * 5, names, X, y)
    finally:
        oof_mod._transform_pair_via = _orig
    print(f"UNFITTED pp OLD raise+raw : {t_old * 1e3:8.2f} ms/call  warns/call={n_warns['n'] / 20:.0f}")
    print(
        f"\nNEW vs OLD on the bugged (unfitted) case: {t_old / t_new:.2f}x "
        f"({'faster' if t_new < t_old else 'SLOWER -- correctness cost, see module docstring'})"
    )


if __name__ == "__main__":
    main()
