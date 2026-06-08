"""Baseline + golden + cProfile harness for the HybridSelector orchestration-overhead optimization campaign.

Targets the MLFRAME-SIDE glue (combine/aggregate/vote/dedup/augment/cluster/shared-FI), NOT the sub-selector
compute (already optimized: MRMR 2.10x, RFECV 3.09x, BorutaShap+ShapProxiedFS ~0 mlframe-side). Captures ONE golden:
the final selected feature set + combined ranking, so every later opt is gated bit-identical against it.

Representative config: hard_synth (220 raw cols at moderate n) -> exercises the REAL combination path: all four
members run (MRMR+FE, ShapProxiedFS, BorutaShap-premerge, tree-GBM co-occurrence FE), corr clustering + dedup,
shared-FI prescreen, cluster-aware vote. use_fe=True (the shipped default) so the augment/share plumbing is live.

  MODE=baseline   -> wall + golden (selected set + ranking) dumped to JSON, + per-stage breakdown (glue vs members)
  MODE=profile    -> cProfile top hotspots (warm-JIT) by tottime + cumtime, mlframe-side filtered
  MODE=verify     -> re-fit, assert selected set + ranking BIT-IDENTICAL to the saved golden (the HARD GATE)

  HYB_N=1500  HYB_SEED=0  MODE=baseline  python hybrid_opt_baseline.py
"""
from __future__ import annotations
import os, sys, time, json, cProfile, pstats, io
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODE = os.environ.get("MODE", "baseline").lower()
N_ROWS = int(os.environ.get("HYB_N", "1500"))
SEED = int(os.environ.get("HYB_SEED", "0"))
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results", "hybrid_opt_golden.json")


def _disable_kernel_tuning_sweep():
    """Force kernel-tuning fallback + in-memory cache so the harness is not gated on a cross-process tuning lock /
    postgres-disk cache load (the scene-profile lesson). The FE/MI kernels still run; only the sweep+disk are bypassed."""
    try:
        import pyutilz.performance.kernel_tuning.cache as _M

        def _no_sweep(self, kernel_name, *, dims, tuner, axes, fallback, **kw):
            return fallback() if callable(fallback) else fallback
        _M.KernelTuningCache.get_or_tune = _no_sweep
        _inmem = _M.KernelTuningCache(in_memory=True)
        _M.KernelTuningCache.load_or_create = classmethod(lambda cls: _inmem)
        print("[kernel-tuning sweep+disk DISABLED -> in-memory fallback]", flush=True)
    except Exception as e:
        print(f"[no-sweep patch failed: {e}]", flush=True)


_disable_kernel_tuning_sweep()


def load_data():
    from hard_synth import make_hard_dataset
    X, y, _ = make_hard_dataset(n_samples=N_ROWS, seed=SEED)
    return X, y


def make_hybrid():
    from mlframe.feature_selection import HybridSelector
    # the shipped default config (use_fe=True, tree member ON, vote=1) -> exercises the full combination path
    return HybridSelector(random_state=SEED)


def _fingerprint(h):
    """The golden fingerprint: the exact selected set + ORDER (raw_selected_) and the combined ranking the glue
    produces. Numeric FI scores may differ <=1e-9 under reordered/parallel reduction, so we do NOT pin raw floats;
    we pin the selected SET+ORDER (the hard gate) plus the FI-derived RANK ORDER of the selected columns."""
    sel = list(h.raw_selected_)
    fi = h.fi_
    # combined ranking = selected cols sorted by shared FI desc (ties broken by name) -- the glue's emitted ranking
    ranking = sorted(sel, key=lambda c: (-float(fi.get(c, 0.0)), c))
    return {
        "selected_set_ordered": sel,
        "combined_ranking": ranking,
        "n_selected": len(sel),
        "n_engineered": int(h.n_engineered_),
        "member_selections": {k: list(v) for k, v in h.member_selections_.items()},
        "n_clusters": len(h.members_),
    }


def run_baseline():
    X, y = load_data()
    print(f"hard_synth: shape={X.shape} pos={float(y.mean()):.3f} seed={SEED}", flush=True)
    h = make_hybrid()
    t0 = time.time(); h.fit(X, y); dt = time.time() - t0
    fp = _fingerprint(h)
    print(f"FIT {dt:.2f}s; n_selected={fp['n_selected']} n_eng={fp['n_engineered']} "
          f"n_clusters={fp['n_clusters']} members={ {k: len(v) for k, v in fp['member_selections'].items()} }", flush=True)
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    payload = dict(wall=dt, n_rows=N_ROWS, seed=SEED, shape=list(X.shape), **fp)
    with open(GOLDEN, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[golden written -> {GOLDEN}]", flush=True)
    print(f"selected_set_ordered = {fp['selected_set_ordered']}", flush=True)
    return dt


def run_verify():
    if not os.path.exists(GOLDEN):
        raise SystemExit("no golden; run MODE=baseline first")
    with open(GOLDEN) as f:
        g = json.load(f)
    X, y = load_data()
    h = make_hybrid()
    t0 = time.time(); h.fit(X, y); dt = time.time() - t0
    fp = _fingerprint(h)
    ok_set = fp["selected_set_ordered"] == g["selected_set_ordered"]
    ok_rank = fp["combined_ranking"] == g["combined_ranking"]
    ok_members = fp["member_selections"] == g["member_selections"]
    print(f"FIT {dt:.2f}s (golden {g['wall']:.2f}s -> speedup {g['wall']/dt:.2f}x)", flush=True)
    print(f"selected_set BIT-IDENTICAL: {ok_set}", flush=True)
    print(f"combined_ranking BIT-IDENTICAL: {ok_rank}", flush=True)
    print(f"member_selections BIT-IDENTICAL: {ok_members}", flush=True)
    if not ok_set:
        a, b = fp["selected_set_ordered"], g["selected_set_ordered"]
        print(f"  golden : {b}", flush=True)
        print(f"  current: {a}", flush=True)
        print(f"  set-diff added={sorted(set(a)-set(b))} dropped={sorted(set(b)-set(a))}", flush=True)
    if not ok_members:
        for k in set(fp["member_selections"]) | set(g["member_selections"]):
            ga, ca = g["member_selections"].get(k, []), fp["member_selections"].get(k, [])
            if ga != ca:
                print(f"  member[{k}] golden={ga}\n           current={ca}", flush=True)
    hard = ok_set and ok_rank and ok_members
    print(f"HARD GATE (set+rank+members bit-identical): {'PASS' if hard else 'FAIL'}", flush=True)
    if not hard:
        raise SystemExit(1)
    return dt


def run_profile():
    X, y = load_data()
    print(f"hard_synth: shape={X.shape} pos={float(y.mean()):.3f}", flush=True)
    tw = time.time()
    try:
        make_hybrid().fit(X.iloc[:200], y.iloc[:200])
    except Exception as e:
        print(f"[warm-up note] {type(e).__name__}: {e}", flush=True)
    print(f"[warm-up {time.time()-tw:.1f}s] profiling warm fit...", flush=True)
    h = make_hybrid()
    pr = cProfile.Profile(); t0 = time.time(); pr.enable()
    h.fit(X, y)
    pr.disable(); dt = time.time() - t0
    print(f"FIT {dt:.2f}s; n_selected={len(h.raw_selected_)}", flush=True)
    s = io.StringIO(); ps = pstats.Stats(pr, stream=s)
    print("\n========== TOP 35 by TOTTIME ==========")
    ps.sort_stats("tottime").print_stats(35); print(s.getvalue())
    s2 = io.StringIO(); ps2 = pstats.Stats(pr, stream=s2)
    print("========== TOP 25 by CUMTIME ==========")
    ps2.sort_stats("cumulative").print_stats(25); print(s2.getvalue())
    # mlframe-side glue filter: the hybrid_selector module's own functions
    s3 = io.StringIO(); ps3 = pstats.Stats(pr, stream=s3)
    print("========== hybrid_selector.py GLUE (tottime) ==========")
    ps3.sort_stats("tottime").print_stats("hybrid_selector"); print(s3.getvalue())


if __name__ == "__main__":
    if MODE == "baseline":
        run_baseline()
    elif MODE == "verify":
        run_verify()
    elif MODE == "profile":
        run_profile()
    else:
        raise SystemExit(f"unknown MODE={MODE}")
