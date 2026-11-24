"""Extra tables of the B3 report (agent B3, 2026-11-24): null calibration of the LL test, variants of the decision rule, Yeo-Johnson boundary picks, failure counts.

``python tt_b3_misc.py RESULT_DIR`` (reads every e*.jsonl except the *_v0_buggy files).
"""

from __future__ import annotations

import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import tt_b3_analyze as A
import tt_b3_fixtures as F
import tt_b3_tables as T


def main(d: str) -> None:
    files = sorted(f for f in glob.glob(os.path.join(d, "e*.jsonl")) if "v0_buggy" not in f)
    recs = A.load(files)
    print(f"records: {len(recs)} from {len(files)} files\n")

    print("## Null calibration of the LL test (head linear, alpha 0.05): share of cases with p <= alpha on fixtures whose null holds exactly (linear mean, symmetric noise)\n")
    print("| block | fixture group | cases | LL test fires | final non-identity |\n|---|---|---|---|---|")
    for blk, names in T.BLOCKS.items():
        rs = T.load_block(d, names)
        for label, kinds, head in (("core controls (lin, lin_hetero, lin_mixed, lin_t3, lin_pos), linear head", T.CORE, "linear"), ("lin_quad, spline head", ("lin_quad",), "spline"), ("lin_quad, linear head", ("lin_quad",), "linear"), ("lin_skewnoise (skewed noise), linear head", ("lin_skewnoise",), "linear")):
            sub = [r for r in rs if r["kind"] in kinds and f"{head}_f1_rmse" in r["ref"]]
            if not sub:
                continue
            ds = [r["ref"][f"{head}_f1_rmse"] for r in sub]
            fire = np.mean([x["p"] <= 0.05 and x["gain"] is not None for x in ds])
            fin = np.mean([not x["abstained"] for x in ds])
            print(f"| {blk} | {label} | {len(sub)} | {100 * fire:.1f}% ({sum(x['p'] <= 0.05 for x in ds)}/{len(sub)}) | {100 * fin:.1f}% ({sum(not x['abstained'] for x in ds)}/{len(sub)}) |")
    core = [r for r in recs if r["kind"] in T.CORE and "linear_f1_rmse" in r["ref"]]
    fires = sum(r["ref"]["linear_f1_rmse"]["p"] <= 0.05 for r in core)
    print(f"\nall blocks, core controls, linear head: LL test fires {fires}/{len(core)} = {100 * fires / len(core):.1f}% (nominal 5%); final non-identity {sum(not r['ref']['linear_f1_rmse']['abstained'] for r in core)}/{len(core)}")

    print("\n## Decision-rule variants (ridge, smeared RMSE, full oracle; all blocks pooled): good / mean regret / worst ratio / FP on core controls / miss on gain\n")
    rows = []
    for _blk, names in T.BLOCKS.items():
        rr = T.load_block(d, names)
        if rr:
            rows += A.case_rows(rr, "ridge", "rmse_smear", ["ref", "ref2", "ref_strict", "ref_notie", "ref_byerr", "ref_polish"], "full")
    for m in ("ref", "ref2", "ref_strict", "ref_notie", "ref_byerr", "ref_polish"):
        sub = [r for r in rows if m in r]
        if not sub:
            continue
        s = T.stats(sub, m)
        core_rows = [r for r in sub if r["kind"] in T.CORE]
        fpcore = np.mean([r[m]["name"] != "identity" for r in core_rows]) if core_rows else float("nan")
        print(f"- {m}: cases {s[0]}, good {100 * s[1]:.1f}%, mean regret {s[2]:.4f}, median {s[3]:.4f}, worst ratio {s[5]:.2f}, miss on gain {100 * s[6]:.1f}%, FP on core controls {100 * fpcore:.1f}%")

    print("\n## Yeo-Johnson picks: lambda at a grid edge (-1 or 3) vs inside (ridge RMSE; ratio = pick / identity on the test block)\n")
    edge, inner = [], []
    for r in recs:
        pk = r["ref"].get("linear_f1_rmse")
        j = r["judge"].get("ridge")
        if not pk or not j or pk["name"] != "yeo_johnson":
            continue
        lam = pk["params"]["lambda"]
        ratio = j[r["ref_keys"]["linear_f1_rmse"]]["rmse_smear"] / j["identity"]["rmse_smear"]
        (edge if lam in (-1.0, 3.0) else inner).append((r["kind"], ratio))
    for lab, v in (("edge", edge), ("inner", inner)):
        if v:
            rat = np.array([x[1] for x in v])
            print(f"- {lab}: {len(v)} picks, mean ratio {rat.mean():.3f}, share worse than identity {100 * np.mean(rat > 1.0):.0f}%, by fixture {dict(Counter(x[0] for x in v))}")

    print("\n## Pick frequencies of the reference (ridge RMSE head linear) by fixture, all blocks\n")
    byfx = defaultdict(Counter)
    for r in recs:
        pk = r["ref"].get("linear_f1_rmse")
        if pk:
            byfx[r["kind"]][pk["name"]] += 1
    for fx in F.FIXTURES:
        print(f"- {fx}: {dict(byfx[fx])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
