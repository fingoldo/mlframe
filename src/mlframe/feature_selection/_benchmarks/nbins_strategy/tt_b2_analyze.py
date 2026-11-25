"""B2 analysis: tables from the JSONL files written by tt_b2_run_target.py / tt_b2_run_feat.py / tt_b2_run_p.py.

python tt_b2_analyze.py target FILE[,FILE...] [--group fam,var] [--crit LL,LLnull,W1] [--metric rmse|mae]
python tt_b2_analyze.py feat FILE[,FILE...] [--group dist,link]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from tt_b2_core import outcome


EXT_CRITS = ("E_LLnull", "E_W1", "EG", "SEG")


def load(files):
    rows = []
    for f in files.split(","):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    for r in rows:
        pk = r.get("picks")
        if pk and "CV" in pk:  # hybrids: the cross-fitted CV decides when it accepts, otherwise the LL pipeline (W1, or N_W1 with the unclipped null when available)
            pk["H"] = pk["CV"] if pk["CV"] != "identity" else pk["W1"]
            if "N_W1" in pk:
                pk["HN"] = pk["CV"] if pk["CV"] != "identity" else pk["N_W1"]
    return rows


def _ratios(score: dict, id_key: str):
    idv = score[id_key]
    return {k: (v / idv if np.isfinite(v) and np.isfinite(idv) and idv > 0 else np.inf) for k, v in score.items()}


def classify(row, crit: str, metric: str, id_key: str = "identity", pick_key: str = "picks"):
    sc = row[metric]
    if "base_names" in row and crit not in EXT_CRITS:
        sc = {k: v for k, v in sc.items() if k in row["base_names"]}
    r = _ratios(sc, id_key)
    nonid = {k: v for k, v in r.items() if k != id_key}
    best = min(nonid, key=nonid.get) if nonid else id_key
    rb = nonid[best] if nonid else 1.0
    pk = row[pick_key][crit] if pick_key == "picks" else row[crit]
    rp = r.get(pk, 1.0) if pk != id_key else 1.0
    return outcome(rp, rb, picked_id=(pk == id_key)), rp, rb, best, pk


def summarize(rows, crit, metric="rmse", id_key="identity", pick_key="picks"):
    oc = Counter()
    reg, rps, rbs = [], [], []
    for r in rows:
        o, rp, rb, best, pk = classify(r, crit, metric, id_key, pick_key)
        oc[o] += 1
        rps.append(rp)
        rbs.append(rb)
        reg.append(rp / min(rb, 1.0))
    nh = oc["HIT"] + oc["WRONG"] + oc["MISS"]
    nn = oc["TN"] + oc["FP"] + oc["HARM"]
    return dict(n=len(rows), helps=nh, nohelp=nn, hit=oc["HIT"], wrong=oc["WRONG"], miss=oc["MISS"], fp=oc["FP"] + oc["HARM"], harm=oc["HARM"], regret=float(np.mean(reg)),
                ratio=float(np.mean(rps)), worst=float(np.max(rps)), oracle=float(np.mean([min(x, 1.0) for x in rbs])))


def table(rows, group, crits, metric="rmse", id_key="identity", pick_key="picks", title=""):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in group)].append(r)
    print(f"\n== {title} metric={metric}; per criterion: hit/helps | fp(harm)/nohelp | regret(mean pick/best) | pick ratio mean(worst)")
    hdr = f"{'/'.join(group):26s} {'N':>3s} {'help':>4s} {'oracle':>6s} " + " ".join(f"{c:>34s}" for c in crits)
    print(hdr)
    for key in sorted(g):
        rr = g[key]
        ss = {c: summarize(rr, c, metric, id_key, pick_key) for c in crits}
        s0 = ss[crits[0]]
        line = f"{'/'.join(map(str, key)):26s} {s0['n']:3d} {s0['helps']:4d} {s0['oracle']:6.3f} "
        for c in crits:
            s = ss[c]
            line += f"  {s['hit']}/{s['helps']} {s['fp']}({s['harm']})/{s['nohelp']} {s['regret']:5.3f} {s['ratio']:6.3f}({s['worst']:6.2f})".rjust(36)
        print(line)
    # totals
    print(f"{'ALL':26s}", end=" ")
    ss = {c: summarize(rows, c, metric, id_key, pick_key) for c in crits}
    s0 = ss[crits[0]]
    print(f"{s0['n']:3d} {s0['helps']:4d} {s0['oracle']:6.3f} " + " ".join(f"  {ss[c]['hit']}/{ss[c]['helps']} {ss[c]['fp']}({ss[c]['harm']})/{ss[c]['nohelp']} {ss[c]['regret']:5.3f} {ss[c]['ratio']:6.3f}({ss[c]['worst']:6.2f})".rjust(36) for c in crits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["target", "feat", "multi", "md", "multipool"])
    ap.add_argument("files")
    ap.add_argument("--group", default=None)
    ap.add_argument("--crit", default=None)
    ap.add_argument("--metric", default="rmse")
    ap.add_argument("--filter", default="", help="key=value,key=value row filter")
    a = ap.parse_args()
    rows = load(a.files)
    for kv in [x for x in a.filter.split(",") if x]:
        k, v = kv.split("=")
        rows = [r for r in rows if str(r[k]) == v]
    if a.kind == "multipool":
        multipool(rows)
        return
    if a.kind == "md":
        md_table(rows, (a.group or "fam,var").split(","), (a.crit or "W1,CV").split(","), a.metric)
        return
    if a.kind == "multi":
        multi_table(rows, a.metric)
        return
    if a.kind == "target":
        crits = (a.crit or "LL,LLnull,W1").split(",")
        table(rows, (a.group or "fam,var").split(","), crits, a.metric, title="TARGET")
    else:
        for r in rows:
            r["pk"] = {"c": r["pick"]}
        feat_table(rows, (a.group or "dist,link").split(","), a.metric)


def feat_table(rows, group, metric):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in group)].append(r)
    print(f"\n== FEATURE metric={metric}: hit/helps | fp(harm)/nohelp | regret | pick ratio mean(worst) | judge-best counts")
    for key in sorted(g):
        rr = g[key]
        oc = Counter()
        reg, rps, rbs, bests = [], [], [], Counter()
        for r in rr:
            ra = _ratios(r[metric], "id")
            nonid = {k: v for k, v in ra.items() if k != "id"}
            best = min(nonid, key=nonid.get)
            rb = nonid[best]
            pk = r["pick"]
            rp = ra[pk] if pk != "id" else 1.0
            oc[outcome(rp, rb, picked_id=(pk == "id"))] += 1
            reg.append(rp / min(rb, 1.0))
            rps.append(rp)
            rbs.append(min(rb, 1.0))
            bests[best if rb <= 0.98 else "-"] += 1
        nh = oc["HIT"] + oc["WRONG"] + oc["MISS"]
        nn = oc["TN"] + oc["FP"] + oc["HARM"]
        print(f"{'/'.join(map(str, key)):28s} N{len(rr):3d} hit {oc['HIT']}/{nh} wrong {oc['WRONG']} miss {oc['MISS']} | fp {oc['FP'] + oc['HARM']}({oc['HARM']})/{nn} | regret {np.mean(reg):.3f} "
              f"| ratio {np.mean(rps):.3f}({np.max(rps):.2f}) oracle {np.mean(rbs):.3f} | {dict(bests.most_common(3))}")




def multi_table(rows, metric="rmse", rules=("perfeat", "maxT", "maxT_stud", "bonf")):
    """p-axis: noise-feature false accept rate, family-wise error rate (any noise feature accepted in a run) and informative-feature outcomes per decision rule."""
    g = defaultdict(list)
    for r in rows:
        g[(r["n"], r["p"], r["amp"])].append(r)
    print(f"\n== MULTI-FEATURE metric={metric}: per rule: noise-FA rate | FWER | informative outcomes HIT/WRONG/MISS (x0 log, x1 square) and FP on x2 (linear control)")
    for key in sorted(g):
        rr = g[key]
        print(f"n={key[0]} p={key[1]} amp={key[2]} N={len(rr)}  t_crit={np.mean([r['t_crit'] for r in rr]):.1f}s")
        for rule in rules:
            fa, fw = [], []
            oc = {j: Counter() for j in range(3)}
            for r in rr:
                acc = np.array(r[rule][3:]) if r["p"] > 3 else np.zeros(0)
                if len(acc):
                    fa.append(acc.mean())
                    fw.append(acc.any())
                for j in range(3):
                    sc = r["judge"][str(j)][metric]
                    ra = _ratios(sc, "id")
                    nonid = {k: v for k, v in ra.items() if k != "id"}
                    bk = min(nonid, key=nonid.get)
                    pk = r["best"][j] if r[rule][j] else "id"
                    oc[j][outcome(ra[pk] if pk != "id" else 1.0, nonid[bk], picked_id=(pk == "id"))] += 1
            def fmt(c):
                return "/".join(str(c[k]) for k in ("HIT", "WRONG", "MISS", "TN", "FP", "HARM"))
            print(f"   {rule:10s} noiseFA {np.mean(fa) if fa else float('nan'):.4f} FWER {np.mean(fw) if fw else float('nan'):.3f} | x0 H/W/M/TN/FP/HARM {fmt(oc[0])} | x1 {fmt(oc[1])} | x2 {fmt(oc[2])}")


def md_table(rows, group, crits, metric="rmse"):
    """Compact markdown table: per cell the oracle gain and, per criterion, hit/helps, false picks/no-help, harm, mean regret."""
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in group)].append(r)
    head = "| " + " | ".join(group) + " | N | helps | oracle | " + " | ".join(f"{c} hit/helps, FP(harm)/nohelp, regret" for c in crits) + " |"
    print(head)
    print("|" + "---|" * (len(group) + 3 + len(crits)))
    for key in sorted(g):
        rr = g[key]
        ss = [summarize(rr, c, metric) for c in crits]
        cells = [f"{s['hit']}/{s['helps']}, {s['fp']}({s['harm']})/{s['nohelp']}, {s['regret']:.3f}" for s in ss]
        print("| " + " | ".join(map(str, key)) + f" | {ss[0]['n']} | {ss[0]['helps']} | {ss[0]['oracle']:.3f} | " + " | ".join(cells) + " |")
    ss = [summarize(rows, c, metric) for c in crits]
    print("| **ALL** " + "| " * (len(group) - 1) + f"| {ss[0]['n']} | {ss[0]['helps']} | {ss[0]['oracle']:.3f} | " + " | ".join(f"{s['hit']}/{s['helps']}, {s['fp']}({s['harm']})/{s['nohelp']}, {s['regret']:.3f}" for s in ss) + " |")


def wilson(k, n, z=1.96):
    import math
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def multipool(rows, rules=("perfeat", "maxT", "maxT_stud")):
    """p-axis pooled over seeds: noise-feature false-accept rate, FWER with Wilson CI, power on the two informative features and false picks on the linear control."""
    g = defaultdict(list)
    for r in rows:
        g[(r["n"], r["p"])].append(r)
    print("p-axis: rule | noise-feature false-accept rate | FWER (any false accept among the p-3 noise features) [Wilson 95%] | HIT x0(log) x1(square) | false picks on x2 (linear control)")
    for key in sorted(g):
        rr = g[key]
        print(f"n={key[0]} p={key[1]} runs={len(rr)} t_crit={np.mean([r['t_crit'] for r in rr]):.1f}s")
        for rule in rules:
            fa = np.mean([np.mean(r[rule][3:]) for r in rr]) if key[1] > 3 else float("nan")
            k = sum(any(r[rule][3:]) for r in rr)
            lo, hi = wilson(k, len(rr))
            hit = [0, 0]
            fp2 = 0
            for r in rr:
                for j in (0, 1):
                    ra = _ratios(r["judge"][str(j)]["rmse"], "id")
                    nonid = {q: v for q, v in ra.items() if q != "id"}
                    bk = min(nonid, key=nonid.get)
                    pk = r["best"][j] if r[rule][j] else "id"
                    hit[j] += outcome(ra[pk] if pk != "id" else 1.0, nonid[bk], picked_id=(pk == "id")) == "HIT"
                fp2 += bool(r[rule][2])
            print(f"   {rule:10s} noiseFA {fa:.4f} FWER {k}/{len(rr)}={k / len(rr):.3f} [{lo:.3f},{hi:.3f}] | HIT x0 {hit[0]}/{len(rr)} x1 {hit[1]}/{len(rr)} | x2 {fp2}/{len(rr)}")


if __name__ == "__main__":
    main()
