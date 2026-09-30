"""Bench the levers for reusing CatBoost Pool / quantization work across RFECV fold fits.

Every RFECV fold fit on a new column subset rebuilds the train and val Pools and re-quantizes them. Three reuse levers:

1. ``ignored_features``: build and quantize the fold's full-width Pool once, then fit each subset with the dropped
   columns ignored. REJECTED: not bit-identical to fitting on the column subset (pandas input, 100k x 30, 60 its:
   max |delta p| = 0.287; a numpy 20k x 20 run happened to match, so it is data-dependent, not a safe gate).
2. Pre-quantized full-width Pool + ``ignored_features``: same non-identity (max |delta p| = 0.38 at 375k x 88).
3. Cached per-fold quantization borders (``save_quantization_borders`` once, re-indexed ``input_borders`` per subset):
   bit-identical on float data (max |delta p| = 0.0), quantize 2.54s -> 1.34s at 375k x 88 on a 4-core host.
   Against a raw fit whose fixed per-fit overhead is ~2.2s (its=1) and whose trees cost ~0.1s each at this size,
   the saving is a few percent of a realistic fold fit at 100k x 30. UPDATE: re-measured end to end per subset fit (bench_cb_border_cache_fold.py) it is
   7.6% (100k x 30), 11.3% (100k x 88), 13.7% (375k x 88) of CPU time, and it is now wired as ``rfecv/_cb_border_cache.py`` (CatBoost CPU, default on,
   env opt-out MLFRAME_RFECV_CB_CACHED_BORDERS=0), identity-gated by bench_cb_border_cache_identity.py. Levers 1-2 stay REJECTED.

Run: python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.bench_cb_fold_pool_reuse [n_rows] [n_cols]
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd


def _best_of(fn, k: int = 3) -> float:
    best = float("inf")
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(n: int = 100_000, p: int = 30, its: int = 60) -> None:
    from catboost import CatBoostClassifier, Pool

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, p)).astype(np.float32), columns=[f"f{i}" for i in range(p)])
    y = (X.f0 + X.f1 * X.f2 + rng.normal(size=n) > 0).astype(int)
    Xv, yv = X.iloc[: n // 10], y[: n // 10]
    keep = list(range(0, p, 2))
    ign = [i for i in range(p) if i not in keep]

    def _proba(m, data):
        return m.predict_proba(data)[:, 1]

    ref = _proba(CatBoostClassifier(iterations=its, verbose=0, random_seed=0).fit(X.iloc[:, keep], y, eval_set=(Xv.iloc[:, keep], yv)), Xv.iloc[:, keep])

    m_ign = CatBoostClassifier(iterations=its, verbose=0, random_seed=0, ignored_features=ign).fit(X, y, eval_set=(Xv, yv))
    print(f"ignored_features vs subset fit: max|dp|={np.abs(_proba(m_ign, Xv) - ref).max():.3g}")

    with tempfile.TemporaryDirectory() as tmp:
        full_path, sub_path = os.path.join(tmp, "full.tsv"), os.path.join(tmp, "sub.tsv")
        full = Pool(X, y)
        full.quantize()
        full.save_quantization_borders(full_path)
        remap = {old: new for new, old in enumerate(keep)}
        with open(full_path) as src, open(sub_path, "w") as dst:
            for line in src:
                parts = line.rstrip("\n").split("\t")
                if int(parts[0]) in remap:
                    dst.write("\t".join([str(remap[int(parts[0])])] + parts[1:]) + "\n")
        sub = Pool(X.iloc[:, keep], y)
        sub.quantize(input_borders=sub_path)
        m_b = CatBoostClassifier(iterations=its, verbose=0, random_seed=0).fit(sub, eval_set=Pool(Xv.iloc[:, keep], yv))
        print(f"cached-borders vs subset fit: max|dp|={np.abs(_proba(m_b, Xv.iloc[:, keep]) - ref).max():.3g}")

        t_q = _best_of(lambda: Pool(X, y).quantize())
        t_qb = _best_of(lambda: Pool(X, y).quantize(input_borders=full_path))
        print(f"quantize full width: computed borders {t_q:.2f}s, cached borders {t_qb:.2f}s")
    t_fit1 = _best_of(lambda: CatBoostClassifier(iterations=1, verbose=0, random_seed=0).fit(X, y, eval_set=(Xv, yv)), k=2)
    print(f"raw fit fixed overhead (iterations=1): {t_fit1:.2f}s")


if __name__ == "__main__":
    _args = [int(a) for a in sys.argv[1:3]]
    main(*_args)
