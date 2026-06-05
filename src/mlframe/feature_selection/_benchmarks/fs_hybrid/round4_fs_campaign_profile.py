"""Per-FS cProfile campaign on the wide real dataset (scene, the one that ran 3-4h with idle CPU).

Profiles ONE Feature Selector's .fit on a scene subsample (warm-JIT, so the dump is free of one-time numba compile),
and prints the top mlframe-side hotspots by tottime + cumtime. Drives the optimize-each-hotspot loop: MRMR -> RFECV
-> BorutaShap -> ShapProxiedFS -> Hybrid. Reuses the benchmark Sel wrappers (valid per-FS construction).

  FS=rfecv|boruta|shap|mrmr|mrmr_fe  SCENE_N=700  python round4_fs_campaign_profile.py
"""
from __future__ import annotations
import os, sys, time, cProfile, pstats, io
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FS = os.environ.get("FS", "rfecv").lower()
N_ROWS = int(os.environ.get("SCENE_N", "700"))


def load_scene(n_rows):
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name="scene", version=1, as_frame=True, parser="auto")
    X = d.data.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    y = pd.Series(pd.factorize(d.target)[0]); y = (y == y.value_counts().idxmax()).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if n_rows < len(X):
        idx = np.random.default_rng(0).choice(len(X), size=n_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    return X, y


def make_selector(fs):
    import fs_selectors as S
    if fs == "rfecv":
        return S.RFECVSel("lgbm_perm")  # permutation-importance path (~4-5x cost; the RFECV hotspot)
    if fs == "rfecv_impurity":
        return S.RFECVSel("lgbm")
    if fs == "boruta":
        return S.BorutaSel()
    if fs == "shap":
        return S.ShapSel()
    if fs == "mrmr":
        return S.MRMRSel(fe=False)
    if fs == "mrmr_fe":
        return S.MRMRSel(fe=True)
    if fs == "hybrid":
        from mlframe.feature_selection import HybridSelector
        return HybridSelector(random_state=0)
    raise SystemExit(f"unknown FS={fs}")


def main():
    X, y = load_scene(N_ROWS)
    print(f"[FS={FS}] scene subsample: shape={X.shape} pos={float(y.mean()):.3f}", flush=True)
    # warm JIT/imports on a tiny fit (discarded), so the profiled fit shows the TRUE per-fit hotspot.
    tw = time.time()
    try:
        make_selector(FS).fit(X.iloc[:120], y.iloc[:120])
    except Exception as e:
        print(f"[warm-up note] {type(e).__name__}: {e}", flush=True)
    print(f"[warm-up {time.time()-tw:.1f}s] profiling the warm fit...", flush=True)

    sel = make_selector(FS)
    pr = cProfile.Profile(); t0 = time.time(); pr.enable()
    sel.fit(X, y)
    pr.disable(); dt = time.time() - t0
    try:
        nsel = int(np.asarray(sel.transform(X.iloc[:5])).shape[1])
    except Exception:
        nsel = -1
    print(f"[FS={FS}] fit {dt:.1f}s; selected n={nsel}", flush=True)

    s = io.StringIO(); ps = pstats.Stats(pr, stream=s)
    print("\n========== TOP 30 by TOTTIME ==========")
    ps.sort_stats("tottime").print_stats(30); print(s.getvalue())
    s2 = io.StringIO(); ps2 = pstats.Stats(pr, stream=s2)
    print("========== TOP 20 by CUMTIME ==========")
    ps2.sort_stats("cumulative").print_stats(20); print(s2.getvalue())


if __name__ == "__main__":
    main()
