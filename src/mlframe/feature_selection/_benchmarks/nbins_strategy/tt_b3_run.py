"""Driver of the B3 validation (agent B3, 2026-11-24): fresh fixtures x n x p x seeds; the reference recommender, the OLD proposers and the oracle judge on the untouched test block.

Per case one JSON line: the picks of the reference (head linear / spline, metric rmse / mae, 1 and 2 folds optional), the picks of the three old proposers, the wall time of every call, the
oracle scores of every candidate (a4 set + extras + the reference picks as fitted) for every requested model.

Usage (``CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1``): ``python tt_b3_run.py --n 2000 --p 5 --seeds 12 --models ridge,ridge_spline,hgb --out FILE.jsonl [--fixtures a,b] [--folds2 1] [--procs 2]``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tt_b3_fixtures as F
import tt_b3_ext as X3
import tt_b3_judge as J
import tt_b3_reference as R

warnings.simplefilter("ignore")


def _pick_record(ch) -> dict:
    return {
        "name": ch.name, "params": ch.params, "abstained": ch.abstained, "gain": ch.ll_gain, "tau": ch.null_threshold, "p": ch.p_value, "err_id": ch.err_identity, "err_ch": ch.err_chosen,
        "best_candidate": ch.best_candidate, "reason": ch.reason, "gains": {k: (v if np.isfinite(v) else None) for k, v in ch.gains.items()},
    }


def _extra_for_pick(ch):
    if ch.transform is None:
        return None
    t = ch.transform
    key = f"ref:{t.name}" + (f":{t.params['lambda']:.3f}" if "lambda" in t.params else "")
    lam = t.params.get("lambda")
    bound = (lambda zt, m: R._safe_bounds(t.name, zt, m, lam)) if (t.name == "reciprocal" or lam is not None) else None
    return key, (t.forward, t._inv, bound)


def one(args):
    kind, n, p, seed, models, heads, folds2, hgb_only, polish = args
    X, y = F.make(kind, n, p, seed)
    rows = J.search_rows(n, seed)
    Xs, ys = X[rows], y[rows]
    rec = {"kind": kind, "n": n, "p": p, "seed": seed, "ref": {}, "old": {}, "time": {}, "judge": {}}
    picks = {}
    for head in heads:
        for folds in ([1, 2] if folds2 else [1]):
            t0 = time.perf_counter()
            an = R.analyse_target_transforms(Xs, ys, seed=seed, head=head, folds=folds)
            rec["time"][f"{head}_f{folds}"] = time.perf_counter() - t0
            variants = {"": {}, "_strict": {"fallthrough": False}, "_notie": {"tie_se": 0.0}, "_byerr": {"select": "err"}} if folds == 1 else {"": {}}
            for metric in ("rmse", "mae"):
                for suffix, kw in variants.items():
                    ch = R.decide(an, metric=metric, **kw)
                    rec["ref"][f"{head}_f{folds}_{metric}{suffix}"] = _pick_record(ch)
                    picks[f"{head}_f{folds}_{metric}{suffix}"] = ch
                if polish and folds == 1:
                    chp = X3.polish_lambda(an, picks[f"{head}_f{folds}_{metric}"])
                    rec["ref"][f"{head}_f{folds}_{metric}_polish"] = _pick_record(chp)
                    picks[f"{head}_f{folds}_{metric}_polish"] = chp
            rec.setdefault("an", {})[f"{head}_f{folds}"] = {"tau": an.tau, "p": an.p_value, "passing": an.passing, "err": an.err, "lam_full": an.lam_full, "notes": an.notes}
            if head == "linear" and folds == 1:
                rec["lam_yj_reg"] = an.lam_full.get("yeo_johnson")
                rec["valid"] = an.names
                rec["null_max"] = [float(v) for v in an.null_max[:5]]
    t0 = time.perf_counter()
    rec["old"] = J.old_picks(Xs, ys, seed)
    rec["time"]["old"] = time.perf_counter() - t0
    extra = J.extra_candidates(ys, rec.get("lam_yj_reg"))
    ref_keys = {}
    for k, ch in picks.items():
        e = _extra_for_pick(ch)
        if e is not None:
            extra[e[0]] = e[1]
            ref_keys[k] = e[0]
        else:
            ref_keys[k] = "identity"
    rec["ref_keys"] = ref_keys
    for model in models:
        only = None
        if model == "hgb" and hgb_only:
            # trees are slow: only identity, every pick (reference and old) and the best candidate of the ridge oracle (rmse and mae) are scored, so the HGB "oracle" is a restricted one
            only = {"identity"} | {v for k, v in ref_keys.items() if not k.endswith(("_strict", "_notie", "_byerr", "_polish"))} | set(rec["old"].values())
            for m_ in ("ridge", "ridge_spline"):
                r_ = rec["judge"].get(m_)
                if r_:
                    only |= {min(r_, key=lambda k: r_[k]["rmse_smear"]), min(r_, key=lambda k: r_[k]["mae_clip"])}
        t0 = time.perf_counter()
        rec["judge"][model] = J.judge(X, y, model, seed, extra, only=only)
        rec["time"][f"judge_{model}"] = time.perf_counter() - t0
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--models", default="ridge,ridge_spline")
    ap.add_argument("--heads", default="linear,spline")
    ap.add_argument("--fixtures", default=",".join(F.FIXTURES))
    ap.add_argument("--folds2", type=int, default=0)
    ap.add_argument("--hgb_only", type=int, default=1)
    ap.add_argument("--polish", type=int, default=0)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(k, a.n, a.p, s, a.models.split(","), a.heads.split(","), bool(a.folds2), bool(a.hgb_only), bool(a.polish)) for k in a.fixtures.split(",") for s in range(a.seed0, a.seed0 + a.seeds)]
    if os.path.exists(a.out):  # resume: skip the cases already written (a worker may have died of memory pressure)
        with open(a.out, encoding="utf-8") as fh:
            done = {(d["kind"], d["seed"]) for d in (json.loads(line) for line in fh if line.strip())}
        jobs = [j for j in jobs if (j[0], j[3]) not in done]
    t0 = time.perf_counter()
    with Pool(a.procs) as pool, open(a.out, "a", encoding="utf-8") as fh:
        for i, rec in enumerate(pool.imap_unordered(one, jobs)):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{i + 1}/{len(jobs)} {rec['kind']} s{rec['seed']} ref={rec['ref'].get('linear_f1_rmse', {}).get('name')} old={rec['old']} t={time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
