"""Iter100 calibration sweep for ``trust_guard_stratified_anchors`` auto-dispatcher.

Iter99 measured the lever at three regimes (W6K/W10K wins, W2K-dense regresses) and kept default
OFF; the open question was whether a width-aware ``'auto'`` mode could route to True at wide
cohorts and False at dense-narrow cohorts. This sweep tests that hypothesis at n_rows=5000 with
4 widths x 2 redundancy conditions x {stratified=True, False} = 16 fits.

Outcome (this sweep, 2026-06-01): the win/loss contour is NOT separable by a single-axis width
threshold. Stratified wins at (width<=4000, dense) but loses at (width>=6000, dense) and at most
sparse cells -- W6K-dense even regresses fidelity by -0.14 vs uniform. Per the iter100 calibration
gate ("if contour can't be cleanly fit, do not ship auto"), the lever remains opt-in. See
``shap_proxied_fs.py`` ``trust_guard_stratified_anchors`` docstring for the full table.

Run::

    $env:PYTHONPATH = '<worktree>/src'
    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_iter100_stratified_anchors_contour
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
COND = (
    # name, n_redundant, rho
    ("sparse", 0, 0.0),
    ("dense", 20, 0.85),
)
MODES = (True, False)

N_ROWS = 5000
N_INF = 20
SNR = 8.0
SEED = 0
PER_FIT_TIMEOUT_S = 300


WORKER_SRC = """\
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
"""


def _write_worker(worker_path: str) -> None:
    with open(worker_path, "w") as f:
        f.write(WORKER_SRC)


def run_cell(worker_path, width, cond_name, n_red, rho, stratified, out_root):
    fit_out = os.path.join(out_root, f"iter100_cell_{width}_{cond_name}_{int(stratified)}.json")
    cmd = [sys.executable, worker_path, str(width), str(n_red), str(rho), str(N_ROWS), str(N_INF), str(SNR), str(SEED), "1" if stratified else "0", fit_out]
    label = f"w={width} cond={cond_name} stratified={stratified}"
    print(f"[iter100][{label}] starting...", flush=True)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=PER_FIT_TIMEOUT_S, check=False)
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            return dict(width=width, cond=cond_name, n_red=n_red, rho=rho,
                        stratified=bool(stratified), error=f"rc={proc.returncode}",
                        elapsed_outer=elapsed, stderr_tail=(proc.stderr or "")[-1000:])
        if os.path.exists(fit_out):
            with open(fit_out) as f:
                r = json.load(f)
            r["cond"] = cond_name
            r["elapsed_outer"] = elapsed
            print(f"[iter100][{label}] OK total={r['total_s']:.1f}s sp={r['spearman']:.4f} "
                  f"fid={r['proxy_fidelity_score']:.4f} rec={r['recovery']}/{r['n_inf']}", flush=True)
            return r
        return dict(width=width, cond=cond_name, n_red=n_red, rho=rho, stratified=bool(stratified), error="no_output", elapsed_outer=elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return dict(width=width, cond=cond_name, n_red=n_red, rho=rho, stratified=bool(stratified), error="timeout", elapsed_outer=elapsed)


def main():
    out_root = os.environ.get("ITER100_OUT_DIR", "D:/Temp")
    worker_path = os.path.join(out_root, "iter100_worker.py")
    _write_worker(worker_path)
    print(f"[iter100] grid: widths={WIDTHS} cond={[c[0] for c in COND]} modes={MODES}", flush=True)
    t_start = time.perf_counter()
    results = []
    for w in WIDTHS:
        for cond_name, n_red, rho in COND:
            for strat in MODES:
                r = run_cell(worker_path, w, cond_name, n_red, rho, strat, out_root)
                results.append(r)
                with open(os.path.join(out_root, "iter100_calib_results.json"), "w") as f:
                    json.dump(dict(
                        meta=dict(n_rows=N_ROWS, n_inf=N_INF, snr=SNR, seed=SEED,
                                  widths=list(WIDTHS), cond=list(COND), modes=list(MODES)),
                        results=results), f, indent=2)
    print(f"[iter100] done wall={(time.perf_counter()-t_start)/60:.1f} min cells={len(results)}",
          flush=True)


if __name__ == "__main__":
    main()
