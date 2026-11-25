"""Aggregate the JSON outputs of tt_b1_cls_feature_transform.py into the tables of research_tt_b1.md.

Case tiers by the LR judge (held-out log loss, nats per row, id minus best candidate): gain tier  > 0.010 (a transform exists), none tier < 0.003 (nothing to find), ambiguous in between.
A decision is correct on a gain case if the criterion accepted and the pick costs at most 0.005 nats more than the judge best; on a none case if it did not accept; on an ambiguous case if it is not harmful
(an accepted pick that costs more than 0.005 nats over identity is harmful on every tier).

Usage: python tt_b1_report.py GROUP_NAME file1.json [file2.json ...]
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np

import tt_b1_cls_feature_transform as T

GAIN, NONE, TOL, HARM = 0.010, 0.003, 0.005, 0.005


def load(files):
    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(json.load(fh))
    return rows


def tier(r):
    jl = r["judge_lr"]
    g = jl["id"][0] - min(v[0] for v in jl.values())
    return "gain" if g > GAIN else ("none" if g < NONE else "ambig")


def verdict(r, nk, c):
    d = T.decisions(r)[(nk, c)]
    pick, acc = d[0], d[1]
    jl = r["judge_lr"]
    best = min(v[0] for v in jl.values())
    harm = acc and jl[pick][0] - jl["id"][0] > HARM
    t = tier(r)
    if t == "gain":
        ok = bool(acc and jl[pick][0] <= best + TOL)
    elif t == "none":
        ok = not acc
    else:
        ok = not harm
    return ok, bool(acc), bool(harm), pick


def main():
    group, files = sys.argv[1], sys.argv[2:]
    rows = load(files)
    seeds = len({r["seed"] for r in rows})
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    nks = [nk for nk in ("null_perm", "null_bs") if nk in rows[0]]
    print(f"\n##### group {group}: {len(rows)} cases, {seeds} seeds, {len(kinds)} fixtures")
    for nk in nks:
        print(f"\n--- correct decisions per fixture ({nk}); tier counts gain/ambig/none; judge-best LR candidate; mean LR gain in nats")
        print(f"{'fixture':16s} {'g/a/n':>7s} {'judge best':>14s} {'gain':>6s} " + " ".join(f"{c:>8s}" for c in T.CRIT))
        for k in kinds:
            rs = [r for r in rows if r["kind"] == k]
            tc = collections.Counter(tier(r) for r in rs)
            best = collections.Counter(min(r["judge_lr"], key=lambda q: r["judge_lr"][q][0]) for r in rs).most_common(1)[0]
            gain = np.mean([r["judge_lr"]["id"][0] - min(v[0] for v in r["judge_lr"].values()) for r in rs])
            line = f"{k:16s} {tc['gain']}/{tc['ambig']}/{tc['none']:<3d} {best[0] + ' ' + str(best[1]):>14s} {gain:6.3f} "
            for c in T.CRIT:
                line += f"{sum(verdict(r, nk, c)[0] for r in rs):>8d} "
            print(line)
        print(f"-- pooled ({nk}): hits on gain cases | false positives on none cases | harmful accepts on all cases")
        ng = sum(tier(r) == "gain" for r in rows)
        nn = sum(tier(r) == "none" for r in rows)
        na = len(rows) - ng - nn
        print(f"   cases: gain {ng}, ambiguous {na}, none {nn}, total {len(rows)}")
        for c in T.CRIT:
            hit = sum(verdict(r, nk, c)[0] for r in rows if tier(r) == "gain")
            fp = sum(verdict(r, nk, c)[1] for r in rows if tier(r) == "none")
            harm = sum(verdict(r, nk, c)[2] for r in rows)
            print(f"   {c:8s} hits {hit}/{ng}  FP {fp}/{nn}  harmful {harm}/{len(rows)}")
        print(f"-- FP on control fixtures only ({nk}): accepted although the fixture is a control (ctl_*)")
        ctl = [r for r in rows if KIND_CTL(r["kind"])]
        for c in T.CRIT:
            print(f"   {c:8s} FP {sum(verdict(r, nk, c)[1] for r in ctl)}/{len(ctl)}")
    # pick distribution
    nk = nks[0]
    print(f"\n--- accepted picks ({nk}) per fixture: lrt | resid | eta | mi_ew")
    for k in kinds:
        rs = [r for r in rows if r["kind"] == k]
        parts = []
        for c in ("lrt", "resid", "eta", "mi_ew"):
            cnt = collections.Counter(verdict(r, nk, c)[3] if verdict(r, nk, c)[1] else "-" for r in rs)
            parts.append(c + ":" + ",".join(f"{a}x{b}" for a, b in cnt.most_common(3)))
        print(f"   {k:16s} " + " | ".join(parts))
    if len(nks) == 2:
        print("\n--- uncalibrated argmax on none-tier cases: non-identity pick rate and harmful pick rate (what the null removes)")
        nn = sum(tier(r) == "none" for r in rows)
        for c in T.CRIT:
            u = sum(T.decisions(r)[("null_perm", c)][0] != "id" for r in rows if tier(r) == "none")
            h = sum(r["judge_lr"][T.decisions(r)[("null_perm", c)][0]][0] - r["judge_lr"]["id"][0] > HARM for r in rows if tier(r) == "none")
            print(f"   {c:8s} non-id {u}/{nn}  harmful {h}/{nn}")
    print("\n--- judge agreement across model families (gain cases): does the LDA / GNB judge-best cost <= 0.005 nats more than the LR judge-best under the LR judge")
    for k in kinds:
        rs = [r for r in rows if r["kind"] == k and tier(r) == "gain"]
        if not rs:
            continue
        a = np.mean([r["judge_lr"][min(r["judge_lda"], key=lambda q: r["judge_lda"][q][0])][0] <= min(v[0] for v in r["judge_lr"].values()) + TOL for r in rs])
        b = np.mean([r["judge_lr"][min(r["judge_gnb"], key=lambda q: r["judge_gnb"][q][0])][0] <= min(v[0] for v in r["judge_lr"].values()) + TOL for r in rs])
        print(f"   {k:16s} LDA-best agrees {a:.2f}  GNB-best agrees {b:.2f}  (n={len(rs)})")
    print("\n--- replace vs append, GAM ceiling, trees (mean nats gain over LR on raw x [z, x]; AUC gain of the best replace candidate)")
    for k in kinds:
        rs = [r for r in rows if r["kind"] == k]
        rep = np.mean([r["judge_lr"]["id"][0] - min(v[0] for v in r["judge_lr"].values()) for r in rs])
        app = np.mean([r["judge_lr"]["id"][0] - min(v[0] for v in r["judge_app"].values()) for r in rs])
        auc = np.mean([max(v[1] for v in r["judge_lr"].values()) - r["judge_lr"]["id"][1] for r in rs])
        gam = np.mean([r["judge_lr"]["id"][0] - r["gam"][0] for r in rs])
        rh = [r for r in rs if "judge_hgb" in r]
        hg = np.mean([r["judge_hgb"]["id"][0] - min(v[0] for v in r["judge_hgb"].values()) for r in rh]) if rh else float("nan")
        hgl = np.mean([r["judge_lr"]["id"][0] - r["judge_hgb"]["id"][0] for r in rh]) if rh else float("nan")
        print(f"   {k:16s} replace {rep:+.4f} (AUC {auc:+.4f})  append {app:+.4f}  GAM {gam:+.4f}  | HGB: gain from any fitted transform {hg:+.4f}, LR(raw x) minus HGB(raw x) {hgl:+.4f}")
    print("\n--- cost per candidate (ms, mean) : " + ", ".join(f"{k} {1e3 * np.mean([r['t_per_cand'][k] for r in rows]):.2f}" for k in rows[0]["t_per_cand"]))


def KIND_CTL(kind):
    return kind.startswith("ctl_")


if __name__ == "__main__":
    main()
