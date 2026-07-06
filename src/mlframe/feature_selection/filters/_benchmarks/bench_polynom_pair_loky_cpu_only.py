"""Repro / validation bench for the loky CPU-only pair-search fix (2026-07-05).

Runs ``run_polynom_pair_fe`` on n=30k with >=16 pairs and n_jobs=16 (the parallel
branch), while sampling ``nvidia-smi`` VRAM in a background thread, and prints:
  - wall time,
  - VRAM baseline vs peak-during-run delta (per-worker cupy context => ~250 MB/worker;
    the fix makes workers CPU-only => VRAM stays flat),
  - the SELECTED engineered pairs, compared against the serial (n_jobs=1) branch for
    selection-equivalence.

Run:  CUDA_VISIBLE_DEVICES not set (parent keeps GPU), from repo root:
  PYTHONPATH=src python src/mlframe/feature_selection/filters/_benchmarks/bench_polynom_pair_loky_cpu_only.py
"""

import subprocess  # nosec B404 - subprocess used below with fixed list args, no shell=True
import threading
import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import polynom_pair_fe as ppf


def _vram_used_mib():
    try:
        out = subprocess.check_output(  # nosec B603, B607 - fixed/trusted executable (git) with list args, no untrusted input, resolved via PATH intentionally
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


class _VramSampler(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self.samples.append(_vram_used_mib())
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


def _make_data(n=30000, n_cols=8, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"c{i}" for i in range(n_cols)]
    data = {c: rng.standard_normal(n) for c in cols}
    X = pd.DataFrame(data)
    # target with a non-monotone pair signal so the optimiser has real work.
    a, b = X["c0"].values, X["c1"].values
    logit = 1.5 * (a * a - b * b) + 0.5 * X["c2"].values
    y = (1.0 / (1.0 + np.exp(-logit)) > rng.random(n)).astype(np.int64)
    return X, cols, y


def _run(n_jobs, sample_vram):
    X, cols, y = _make_data()
    n_cols = len(cols)
    pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols)]
    prospective_pairs = {((i, j), 0.5): None for (i, j) in pairs}
    eng_feats, eng_recipes, herm = set(), {}, []

    sampler = None
    if sample_vram:
        sampler = _VramSampler()
        sampler.start()
    t0 = time.perf_counter()
    ppf.run_polynom_pair_fe(
        X=X, is_polars_input=False, prospective_pairs=prospective_pairs,
        classes_y=y, cols=list(cols), nbins=np.full(n_cols, 8, dtype=np.int64),
        data=X.values.copy(), engineered_features=eng_feats,
        engineered_recipes=eng_recipes, hermite_features_list=herm,
        feature_names_in=list(cols),
        fe_smart_polynom_iters=1, fe_smart_polynom_optimization_steps=30,
        fe_min_polynom_degree=1, fe_max_polynom_degree=4,
        fe_min_polynom_coeff=-3.0, fe_max_polynom_coeff=3.0,
        fe_min_engineered_mi_prevalence=0.1, fe_hermite_l2_penalty=0.0,
        fe_polynomial_basis="hermite", fe_mi_estimator="plugin",
        fe_optimizer="cma", fe_warm_start=False, fe_multi_fidelity=False,
        quantization_nbins=8, quantization_method="quantile",
        quantization_dtype=np.int16, n_jobs=n_jobs, verbose=0,
        subsample_n=0,
    )
    wall = time.perf_counter() - t0
    vram = None
    if sampler is not None:
        sampler.stop()
        vals = [v for v in sampler.samples if v >= 0]
        vram = (min(vals), max(vals)) if vals else (-1, -1)
    selected = sorted(eng_recipes.keys())
    return wall, vram, selected


def main():
    print("baseline VRAM MiB:", _vram_used_mib())
    print("=== PARALLEL (n_jobs=16, CPU-only workers) ===")
    w_par, vram, sel_par = _run(16, sample_vram=True)
    print(f"wall={w_par:.1f}s  VRAM min/max during run={vram} MiB (delta={vram[1]-vram[0]})")
    print(f"selected {len(sel_par)} engineered pairs")
    print("=== SERIAL (n_jobs=1, reference for selection) ===")
    w_ser, _, sel_ser = _run(1, sample_vram=False)
    print(f"wall={w_ser:.1f}s")
    print(f"selected {len(sel_ser)} engineered pairs")
    print("=== SELECTION EQUIVALENCE ===")
    print("parallel==serial selection:", sel_par == sel_ser)
    if sel_par != sel_ser:
        print("  parallel:", sel_par)
        print("  serial  :", sel_ser)


if __name__ == "__main__":
    main()
