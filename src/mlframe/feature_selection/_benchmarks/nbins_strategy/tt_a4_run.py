"""Driver (agent 4): safe judge on all fixtures x seeds x models, scores the equal-width MI proposer of target_transform_mi_ext. Writes tt_a4_results.json.

Usage: ``python tt_a4_run.py [--seeds 7] [--rows 12000] [--procs 2]`` (set CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1).
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

import target_transform_mi_ext as ext
import tt_a4_oracle as O

PROPOSER_SET = ("identity", "cbrt", "signed_sqrt", "log", "yeo_johnson")


def one(args):
    fx, seed, n = args
    X, y = O.make(fx, n, 8, seed)
    tr = np.arange(n) < int(0.7 * n)
    cand = ext.candidates(y[tr])
    mi = {k: sum(ext.mutual_information(ext.equal_width_codes(X[:, j], 10), ext.equal_width_codes(f(y), 10)) for j in range(X.shape[1])) for k, (f, _i, _o) in cand.items()}
    picked = max(mi, key=lambda k: mi[k])
    per_model = {m: O.judge(X, y, m, seed=seed) for m in O.MODELS}
    return {"fixture": fx, "seed": seed, "picked": picked, "mi": mi, "judge": per_model}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--rows", type=int, default=12000)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    jobs = [(fx, s, a.rows) for fx in O.FIXTURES for s in range(a.seeds)]
    with Pool(a.procs) as p:
        out = []
        for r in p.imap_unordered(one, jobs):
            out.append(r)
            print("done", r["fixture"], r["seed"], r["picked"], flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "tt_a4_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
