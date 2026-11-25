"""Control experiment of agent B3 (2026-11-24): the OLD pipeline (equal-width-MI shortlist of 3 + ridge held-out gate, ``filters/_target_transform_search.py``) given the SAME candidate set as the reference.

Why: the default pool of the old code (identity, cbrt, signed_sqrt, log) cannot express reciprocal, logit, Yeo-Johnson, asinh, signed_square, log1p, so a head-to-head with the reference mixes two effects (candidate
coverage and the selection criterion). Here the old code runs on the pool of the reference (domain-checked on y: identity, cbrt, signed_sqrt, signed_square, asinh, log, log1p, reciprocal, logit,
Yeo-Johnson with the sklearn marginal-likelihood lambda); the pick is judged with the stored oracle scores (every candidate name is in the judge's set).

Usage: ``python tt_b3_oldext.py RESULT_DIR [--procs 2]`` writes ``RESULT_DIR/oldext_<block>.jsonl`` for every e*.jsonl block (one line per case: kind, n, p, seed, old_ext_mi, old_ext_search).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tt_b3_fixtures as F
import tt_b3_judge as J
import tt_b3_reference as R

warnings.simplefilter("ignore")


def ext_pool(M, y: np.ndarray):
    """Old-code TargetTransform pool = the reference candidates valid for the range of y (box_cox excluded)."""
    from sklearn.preprocessing import PowerTransformer

    names, _ = R._valid_names(y, list(R.DEFAULT_CANDIDATES))
    s = float(np.std(y)) + 1e-12
    pool = []
    for n in names:
        if n == "yeo_johnson":
            pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(y.reshape(-1, 1))
            pool.append(M.TargetTransform(n, lambda t, pt=pt: pt.transform(np.asarray(t).reshape(-1, 1)).ravel(), lambda z, pt=pt: pt.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()))
        else:
            pool.append(M.TargetTransform(n, lambda t, n=n: R._fixed_forward(n, np.asarray(t, dtype=float), s), lambda z, n=n: R._fixed_inverse(n, np.asarray(z, dtype=float), s)))
    return pool


def one(args):
    kind, n, p, seed = args
    X, y = F.make(kind, n, p, seed)
    rows = J.search_rows(n, seed)
    Xs, ys = X[rows], y[rows]
    M = J.old_module()
    pool = ext_pool(M, ys)
    top = M.propose_target_transforms(Xs, ys, pool)[0][0]
    chosen, _ = M.search_target_transform(Xs, ys, pool, seed=seed)
    return {"kind": kind, "n": n, "p": p, "seed": seed, "old_ext_mi": top, "old_ext_search": chosen.name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    for f in sorted(glob.glob(os.path.join(a.dir, "e*.jsonl"))):
        if "v0_buggy" in f:
            continue
        out = os.path.join(a.dir, "oldext_" + os.path.basename(f))
        if os.path.exists(out):
            continue
        keys = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                keys.append((r["kind"], r["n"], r["p"], r["seed"]))
        with Pool(a.procs) as pool, open(out, "w", encoding="utf-8") as fo:
            for rec in pool.imap_unordered(one, keys):
                fo.write(json.dumps(rec) + "\n")
        print("done", out, len(keys), flush=True)


if __name__ == "__main__":
    main()
