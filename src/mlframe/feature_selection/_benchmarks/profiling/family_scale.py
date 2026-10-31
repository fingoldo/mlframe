"""One MRMR fit at a given scale with ONE FE family switched on (or none / the defaults), CPU or GPU path, in a child process with a time limit and memory sampling.

Parent:  python -m mlframe.feature_selection._benchmarks.profiling.family_scale run  --n 1000000 --p 1000 --mode gpu|cpu --family <flag stem>|BASE|DEFAULTS [--timeout 1500] [--out results.jsonl]
Child:   python -m mlframe.feature_selection._benchmarks.profiling.family_scale child ...      (run by the parent as a script; prints one JSON line)

BASE switches every fe_*_enable off. DEFAULTS leaves the library defaults. A family name is the flag without ``fe_`` / ``_enable``.
Modes: cpu hides the GPU (CUDA_VISIBLE_DEVICES=-1); gpu is the strict-resident path the benchmarks use.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import threading
import time
import warnings

import numpy as np

from mlframe.feature_selection._benchmarks.profiling._paths import OUT_DIR

HERE = os.path.dirname(os.path.abspath(__file__))


def _make(n: int, p: int, seed: int):
    """float32 frame: five informative numeric columns (the F2-like signal), p - 5 noise columns, four small-integer columns, one group column."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p), dtype=np.float32)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)], copy=False)
    for k in range(4):
        df[f"i{k}"] = rng.integers(0, 20, n).astype(np.int8)
    df["g"] = rng.integers(0, 12, n).astype(np.int8)
    df["c0"] = pd.Categorical(rng.integers(0, 15, n).astype(str))  # a categorical column for the cat_* / count / frequency / rare-category families
    for k in range(2):
        col = rng.standard_normal(n).astype(np.float32)
        col[rng.random(n) < 0.1] = np.nan  # a column with missing values for the missingness_* families
        df[f"m{k}"] = col
    x0, x1, x2, x3, x4 = (df[f"x{i}"].to_numpy() for i in range(5))
    logit = 1.2 * x0 * x1 + 0.8 * x2 / (1.5 + np.abs(x3)) + 0.9 * np.sin(1.7 * x4) + 1.0 * (((df["i0"] + df["i1"]) % 3) == 0) + 0.2 * (df["g"] - 6)
    y = (logit + rng.normal(0, 0.8, n) > np.median(logit)).astype(np.int8)
    return df, y


def _flag_names() -> list[str]:
    """Every ``fe_*_enable`` parameter of MRMR."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    return sorted(p for p in inspect.signature(MRMR.__init__).parameters if p.startswith("fe_") and p.endswith("_enable"))


def child(a: argparse.Namespace) -> None:
    """Fit once and print one JSON line with the wall, the family walls, the peak resident memory and the selected features."""
    warnings.simplefilter("ignore")
    if a.mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
            os.environ.setdefault(k, v)
    import psutil

    from mlframe.feature_selection.filters import _fe_family_timing as FT
    from mlframe.feature_selection.filters.mrmr import MRMR

    peak = {"rss": 0, "gpu_mb": 0, "other_sum": 0.0, "other_n": 0}
    stop = threading.Event()
    proc = psutil.Process()
    ncpu = psutil.cpu_count() or 1
    psutil.cpu_percent(None)
    proc.cpu_percent(None)

    def sample() -> None:
        """Track the peak resident set, the share of the CPU used by other processes and, in gpu mode, the peak device memory in use."""
        while not stop.is_set():
            peak["rss"] = max(peak["rss"], proc.memory_info().rss)
            peak["other_sum"] += max(0.0, psutil.cpu_percent(None) - proc.cpu_percent(None) / ncpu)
            peak["other_n"] += 1
            if a.mode == "gpu":
                try:
                    import cupy as cp

                    pool = cp.get_default_memory_pool()
                    peak["gpu_mb"] = max(peak["gpu_mb"], int(pool.total_bytes() / 2**20))
                except Exception:  # nosec B110 - best-effort sampling
                    pass
            stop.wait(2.0)

    threading.Thread(target=sample, daemon=True).start()
    rec = {"n": a.n, "p": a.p, "mode": a.mode, "family": a.family}
    try:
        t0 = time.perf_counter()
        df, y = _make(a.n, a.p, a.seed)
        rec["make_s"] = round(time.perf_counter() - t0, 1)
        kw: dict = {}
        if a.family != "DEFAULTS":
            kw = {f: False for f in _flag_names()}
            if a.family != "BASE":
                kw[f"fe_{a.family}_enable"] = True
        if a.warmup:
            wdf, wy = _make(20_000, 30, a.seed + 1)  # same flags on a small frame: JIT and kernel compilation stay out of the timed fit
            MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(wdf, wy)
            del wdf, wy
        FT.reset_fe_family_wall()
        t0 = time.perf_counter()
        m = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(df, y)
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        rec["selected"] = [str(s) for s in m.get_feature_names_out()][:30]
        rec["n_selected"] = len(m.get_feature_names_out())
        rec["family_walls"] = {k: round(v[0], 2) for k, v in sorted(FT.get_fe_family_wall().items(), key=lambda kv: -kv[1][0])[:25]}
        rec["status"] = "ok"
    except MemoryError as e:
        rec["status"] = "MEMORY_ERROR"
        rec["error"] = str(e)[:200]
    except Exception as e:
        rec["status"] = "ERROR"
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    stop.set()
    rec["peak_rss_gb"] = round(peak["rss"] / 2**30, 2)
    rec["peak_gpu_mb"] = peak["gpu_mb"]
    rec["other_cpu_pct"] = round(peak["other_sum"] / max(1, peak["other_n"]), 1)
    rec["contaminated"] = rec["other_cpu_pct"] > 10.0  # other processes used more than a tenth of the machine during the fit: timings are not comparable
    print("RESULT " + json.dumps(rec), flush=True)


def _free_virtual_gb() -> float:
    """Free virtual memory of the machine in GB (Windows commit headroom; physical free memory elsewhere)."""
    import psutil

    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            """MEMORYSTATUSEX."""

            _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong), ("av2", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]

        s = MS()
        s.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return min(s.ap, s.av) / 2**30
    return psutil.virtual_memory().available / 2**30


def run(a: argparse.Namespace) -> None:
    """Start the child with a memory guard and a time limit, append its record to the results file."""
    import psutil

    quiet_wait = 0
    while psutil.cpu_percent(interval=5) > 10.0 and quiet_wait < 20 * 60:  # do not start a timed fit on a busy machine
        quiet_wait += 5
    need = a.n * a.p * 4 * a.mem_factor / 2**30
    waited = 0
    while _free_virtual_gb() < need and waited < a.wait_mem_min * 60:
        time.sleep(30)
        waited += 30
    rec = None
    if _free_virtual_gb() < need:
        rec = {"n": a.n, "p": a.p, "mode": a.mode, "family": a.family, "status": "SKIPPED_MEMORY", "need_gb": round(need, 1), "free_gb": round(_free_virtual_gb(), 1)}
    else:
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "child", "--n", str(a.n), "--p", str(a.p), "--mode", a.mode, "--family", a.family, "--seed", str(a.seed), "--warmup", str(a.warmup)]
        t0 = time.perf_counter()
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
        try:
            out, _ = pr.communicate(timeout=a.timeout)
            for line in out.splitlines():
                if line.startswith("RESULT "):
                    rec = json.loads(line[7:])
            if rec is None:
                rec = {"n": a.n, "p": a.p, "mode": a.mode, "family": a.family, "status": "NO_RESULT", "tail": out[-300:]}
        except subprocess.TimeoutExpired:
            import psutil

            try:
                parent = psutil.Process(pr.pid)
                for ch in parent.children(recursive=True):
                    ch.kill()
                parent.kill()
            except psutil.Error:
                pass
            rec = {"n": a.n, "p": a.p, "mode": a.mode, "family": a.family, "status": "TIMEOUT", "timeout_s": a.timeout, "elapsed_s": round(time.perf_counter() - t0)}
    with open(a.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in rec if k not in ("family_walls", "selected")}), flush=True)


def main() -> None:
    """Dispatch to ``run`` or ``child``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "child"])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--family", default="BASE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--out", default=str(OUT_DIR / "family_scale_results.jsonl"))
    ap.add_argument("--mem-factor", type=float, default=2.5, help="free memory required, in multiples of the float32 frame")
    ap.add_argument("--wait-mem-min", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=1, help="1: fit a 20000 x 30 frame with the same flags first")
    a = ap.parse_args()
    child(a) if a.cmd == "child" else run(a)


if __name__ == "__main__":
    main()
