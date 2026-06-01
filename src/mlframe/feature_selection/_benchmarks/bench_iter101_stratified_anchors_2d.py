"""Iter101 2D (n_rows, width) calibration sweep for ``trust_guard_stratified_anchors``.

Iter100 (n_rows=5000 fixed) showed the win/lose direction flips between iter99 and iter100 at SAME
width: W2K-dense lost at n_rows=2000 (iter99) but won at n_rows=5000 (iter100); W6K-dense won at
n_rows=10000 (iter99) but lost at n_rows=5000 (iter100). The samples-to-features ratio is the
likely missing axis. This sweep fits the 2D (n_rows, width) contour on dense-redundant only.

Grid (dense-only -- n_red=20, rho=0.85, n_inf=20, snr=8.0, seed=0):
- width in {2000, 4000, 6000, 10000}
- n_rows in {2000, 5000, 10000}
- modes in {stratified=True, stratified=False}

12 cells x 2 modes = 24 fits. Per-fit timeout 600s (n_rows=10000 at width=10000 dominates).

Re-measures iter99 W2K (width=2000, n_rows=2000) and W6K (width=6000, n_rows=10000) sanity cells.

Output: ``D:/Temp/iter101_2d_contour.json`` + per-cell ``iter101_cell_<w>_<n>_<strat>.json``.

Note: n_rows=5000 cells overlap iter100 (already on disk under ``iter100_cell_*_dense_*.json``);
``main()`` reuses cached results when ``--reuse-iter100`` is passed (default), otherwise re-runs all.

Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter101_stratified_anchors_2d
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import subprocess
import sys
import time


WIDTHS = (2000, 4000, 6000, 10000)
N_ROWS_GRID = (2000, 5000, 10000)
MODES = (True, False)

# Dense-only (rho=0.85, n_red=20). The iter99/iter100 flip was on the dense axis; sparse was
# already explored at n_rows=5000 in iter100 and showed no clean separator either.
N_RED = 20
RHO = 0.85
N_INF = 20
SNR = 8.0
SEED = 0
PER_FIT_TIMEOUT_S = 600


WORKER_SRC = '''\
from __future__ import annotations
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

width = int(sys.argv[1])
n_red = int(sys.argv[2])
rho = float(sys.argv[3])
n_rows = int(sys.argv[4])
n_inf = int(sys.argv[5])
snr = float(sys.argv[6])
seed = int(sys.argv[7])
stratified = sys.argv[8] == "1"
out_path = sys.argv[9]

from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

n_noise = max(0, width - n_inf - n_red)
X, y, _roles = make_regime_dataset(
    n_samples=n_rows, n_informative=n_inf, n_redundant=n_red, redundancy_rho=rho,
    n_noise=n_noise, snr=snr, task="binary", seed=seed)

informative = {f"inf{i}" for i in range(n_inf)}

sel = ShapProxiedFS(
    classification=True, metric="brier", optimizer="auto",
    prefilter_top=500, prefilter_method="two_stage",
    cluster_features=True, cluster_corr_threshold=0.7,
    top_n=20, n_splits=4, n_revalidation_models=3,
    n_anchors=30, trust_guard=True,
    trust_guard_stratified_anchors=stratified,
    random_state=seed, verbose=False, n_jobs=1)

t0 = time.perf_counter()
sel.fit(X, y)
total = time.perf_counter() - t0

rep = sel.shap_proxy_report_
trust = rep.get("trust", {}) or {}
chosen = sorted(sel.selected_features_)
recovery = len(informative & set(chosen))

with open(out_path, "w") as f:
    json.dump(dict(
        width=width, n_red=n_red, rho=rho, n_rows=n_rows, n_inf=n_inf, snr=snr,
        seed=seed, stratified=bool(stratified), total_s=total,
        n_chosen=len(chosen), recovery=recovery,
        anchor_sampling=trust.get("anchor_sampling"),
        trustworthy=trust.get("trustworthy"),
        spearman=trust.get("spearman"), kendall=trust.get("kendall"),
        recall_at_k=trust.get("recall_at_k"),
        proxy_fidelity_score=trust.get("proxy_fidelity_score"),
    ), f)
'''


def _write_worker(worker_path: str) -> None:
    with open(worker_path, "w") as f:
        f.write(WORKER_SRC)


def run_cell(worker_path, width, n_rows, stratified, out_root):
    fit_out = os.path.join(out_root, f"iter101_cell_w{width}_n{n_rows}_s{int(stratified)}.json")
    cmd = [sys.executable, worker_path,
           str(width), str(N_RED), str(RHO),
           str(n_rows), str(N_INF), str(SNR), str(SEED),
           "1" if stratified else "0", fit_out]
    label = f"w={width} n_rows={n_rows} stratified={stratified}"
    print(f"[iter101][{label}] starting...", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=PER_FIT_TIMEOUT_S, check=False)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            return dict(width=width, n_red=N_RED, rho=RHO, n_rows=n_rows,
                        stratified=bool(stratified), error=f"rc={proc.returncode}",
                        elapsed_outer=elapsed, stderr_tail=(proc.stderr or "")[-1000:])
        if os.path.exists(fit_out):
            with open(fit_out) as f:
                r = json.load(f)
            r["elapsed_outer"] = elapsed
            print(f"[iter101][{label}] OK total={r['total_s']:.1f}s sp={r['spearman']:.4f} "
                  f"fid={r['proxy_fidelity_score']:.4f} rec={r['recovery']}/{r['n_inf']}", flush=True)
            return r
        return dict(width=width, n_red=N_RED, rho=RHO, n_rows=n_rows,
                    stratified=bool(stratified), error="no_output", elapsed_outer=elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return dict(width=width, n_red=N_RED, rho=RHO, n_rows=n_rows,
                    stratified=bool(stratified), error="timeout", elapsed_outer=elapsed)


def _maybe_reuse_iter100(width, n_rows, stratified, out_root):
    """For n_rows=5000 dense, iter100 already measured the same cell. Reuse to save ~5 min."""
    if n_rows != 5000:
        return None
    cached = os.path.join(out_root, f"iter100_cell_{width}_dense_{int(stratified)}.json")
    if not os.path.exists(cached):
        return None
    with open(cached) as f:
        r = json.load(f)
    r["reused_from"] = "iter100"
    print(f"[iter101][w={width} n_rows={n_rows} stratified={stratified}] REUSED iter100 "
          f"sp={r['spearman']:.4f} fid={r['proxy_fidelity_score']:.4f}", flush=True)
    return r


def main():
    out_root = os.environ.get("ITER101_OUT_DIR", "D:/Temp")
    worker_path = os.path.join(out_root, "iter101_worker.py")
    _write_worker(worker_path)
    print(f"[iter101] grid: widths={WIDTHS} n_rows={N_ROWS_GRID} modes={MODES}", flush=True)
    t_start = time.perf_counter()
    results = []
    for w in WIDTHS:
        for n_rows in N_ROWS_GRID:
            for strat in MODES:
                # Heartbeat ping per cell so the harness can see liveness.
                with open(os.path.join(out_root, "HEARTBEAT_iter101.txt"), "w") as hb:
                    hb.write(f"iter101 cell w={w} n={n_rows} strat={strat} t={int(time.time())}\n")
                cached = _maybe_reuse_iter100(w, n_rows, strat, out_root)
                if cached is not None:
                    results.append(cached)
                else:
                    r = run_cell(worker_path, w, n_rows, strat, out_root)
                    results.append(r)
                with open(os.path.join(out_root, "iter101_2d_contour.json"), "w") as f:
                    json.dump(dict(
                        meta=dict(n_inf=N_INF, snr=SNR, seed=SEED, n_red=N_RED, rho=RHO,
                                  widths=list(WIDTHS), n_rows_grid=list(N_ROWS_GRID),
                                  modes=list(MODES)),
                        results=results), f, indent=2)
    print(f"[iter101] done wall={(time.perf_counter()-t_start)/60:.1f} min cells={len(results)}",
          flush=True)


if __name__ == "__main__":
    main()
