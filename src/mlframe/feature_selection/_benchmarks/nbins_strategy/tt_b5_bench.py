"""B5 / question 2 benchmark driver: one (criterion, n, p, variant) cell per process (so the peak working set is clean), one JSON line out.

    python tt_b5_bench.py target  N P VARIANT [K] [B]       VARIANT: full | cap64
    python tt_b5_bench.py feature N P VARIANT [K] [B]       VARIANT: null_all | null_sub50k | obs_only
    python tt_b5_bench.py feature_np N P VARIANT [K] [B]    the numpy-SIMD-block version of the feature criterion (tt_b5_cost_np), same variants
    python tt_b5_bench.py ridge   N P                       sklearn Ridge (float32, 'cholesky') on the same data, in memory
    python tt_b5_bench.py hgb     N P [max_iter]            sklearn HistGradientBoostingRegressor on y (log y is the target of the fixture: same data)

Reported: ``compute`` = wall seconds of the criterion MINUS the seconds spent generating data chunks; peak working set (bytes, from the OS) of the whole process and the input
size; the stage timings.  Data: standard normal float32 columns, y = exp(0.8 x0 + 0.5 x1 + 0.3 x2 + 0.3 e).  Cells whose X would exceed ~1.2 GB are streamed
(chunks generated on the fly, outside the timer); smaller cells hold X in memory (``ArrayProvider``), so the pass reads RAM, not a freshly written chunk.
"""
import json
import sys
import time

import numpy as np
import psutil

import tt_b5_cost as C
import tt_b5_cost_np as CN

CHUNK_ROWS = 5000
STREAM_ABOVE_BYTES = 1.2e9


def make_provider(n, p, seed=0):
    if n * p * 4 > STREAM_ABOVE_BYTES:
        return C.GenProvider(n, p, seed, CHUNK_ROWS), "stream"
    g = C.GenProvider(n, p, seed, CHUNK_ROWS)
    Xt = np.empty((p, n), dtype=np.float32)
    y = np.empty(n)
    for i in range(g.nchunks):
        a, b = g.bounds(i)
        xc, yc = g.chunk(i)
        Xt[:, a:b] = xc
        y[a:b] = yc
    return C.ArrayProvider(Xt, y, CHUNK_ROWS), "memory"


def warmup(K):
    prov, _ = make_provider(8000, 12, 99)
    prov.chunk_rows = 2000
    small = C.ArrayProvider(np.ascontiguousarray(np.concatenate([prov.chunk(i)[0] for i in range(prov.nchunks)], 1)), prov.target(), 2000)
    C.target_criterion(small, K=K, B=2, cap=None)
    C.feature_criterion(small, K=K, B=2, null_rows=None, row_block=1000)
    C.feature_criterion(small, K=K, B=2, null_rows=200, row_block=200)


def main():
    kind, n, p = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    variant = sys.argv[4] if len(sys.argv) > 4 else ""
    K = int(sys.argv[5]) if len(sys.argv) > 5 else 9
    B = int(sys.argv[6]) if len(sys.argv) > 6 else 40
    proc = psutil.Process()
    if kind in ("target", "feature", "feature_np"):
        warmup(K)
    prov, mode = make_provider(n, p)
    prov.target()
    gen0 = prov.gen_seconds
    rss0 = proc.memory_info().rss
    cpu0 = proc.cpu_times()
    tm = {}
    t0 = time.perf_counter()
    if kind == "target":
        res = C.target_criterion(prov, K=K, B=B, cap=64 if variant == "cap64" else None, timing=tm)
        extra = dict(best=res["best"], gain=round(res["best_gain"], 4), tau=round(res["tau"], 4), accepted=res["accepted"], m_design=res["m_design"])
    elif kind == "feature":
        nr = None
        if variant == "null_sub50k":
            nr = max(1, int(np.ceil(50000 / prov.nchunks)))
            nr = None if nr >= CHUNK_ROWS else nr
        res = C.feature_criterion(prov, K=K, B=0 if variant == "obs_only" else B, null_rows=nr, timing=tm)
        extra = dict(null_rows_per_chunk=nr, n_accept=int(res["accepted"].sum()) if "accepted" in res else None)
    elif kind == "feature_np":
        nr = None
        if variant == "null_sub50k":
            nr = max(1, int(np.ceil(50000 / prov.nchunks)))
            nr = None if nr >= CHUNK_ROWS else nr
        if nr is not None:
            res = CN.feature_criterion_np_sub(prov, K=K, B=B, null_rows=nr, timing=tm, workers=1)
        else:
            res = CN.feature_criterion_np(prov, K=K, B=0 if variant == "obs_only" else B, timing=tm, workers=1)
        extra = dict(null_rows_per_chunk=nr, n_accept=int(res["accepted"].sum()) if "accepted" in res else None)
    elif kind == "ridge":
        from sklearn.linear_model import Ridge
        X = np.ascontiguousarray(np.concatenate([prov.chunk(i)[0] for i in range(prov.nchunks)], 1).T) if mode == "stream" else np.ascontiguousarray(prov.Xt.T)
        y = np.log(prov.target())
        t0 = time.perf_counter(); cpu0 = proc.cpu_times()
        Ridge(alpha=1.0, solver="cholesky").fit(X, y)
        extra = {}
        gen0 = prov.gen_seconds
    elif kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = np.ascontiguousarray(np.concatenate([prov.chunk(i)[0] for i in range(prov.nchunks)], 1).T) if mode == "stream" else np.ascontiguousarray(prov.Xt.T)
        y = np.log(prov.target())
        mi = int(variant) if variant else 100
        t0 = time.perf_counter(); cpu0 = proc.cpu_times()
        HistGradientBoostingRegressor(max_iter=mi, early_stopping=False, random_state=0).fit(X, y)
        extra = dict(max_iter=mi)
        gen0 = prov.gen_seconds
    wall = time.perf_counter() - t0
    gen = prov.gen_seconds - gen0
    cpu1 = proc.cpu_times()
    peak = proc.memory_info().peak_wset
    inp = n * p * 4 + n * 8 if mode == "memory" else 0
    rec = dict(kind=kind, n=n, p=p, variant=variant, K=K, B=B, mode=mode, compute_s=round(wall - gen, 3), wall_s=round(wall, 3), gen_s=round(gen, 3),
               cpu_user_s=round(cpu1.user - cpu0.user, 2), peak_wset_MB=round(peak / 2 ** 20, 1), input_MB=round(inp / 2 ** 20, 1), rss_before_MB=round(rss0 / 2 ** 20, 1),
               peak_minus_input_MB=round((peak - inp) / 2 ** 20, 1), stages={k: round(v, 3) for k, v in tm.items()}, load=psutil.cpu_percent(interval=None), **extra)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
