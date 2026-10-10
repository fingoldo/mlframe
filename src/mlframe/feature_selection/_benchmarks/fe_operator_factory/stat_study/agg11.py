"""Aggregate the exp11 pre-filter runs (``pf_<n>_*.json``): near-miss window membership, recall of the planted pair and pairs kept by each pre-filter.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.agg11``."""

import json

import numpy as np

from ..common._paths import data_dir

recs = []
for f in sorted(data_dir("stat_study").glob("pf_*.json")):
    nn = int(f.name.split("pf_")[1].split("_")[0])
    for r in json.loads(f.read_text()):
        r["n"] = nn
        recs.append(r)
for nn in (5000, 30000):
    print(f"\n##### n={nn}: per dataset, 10 pairs; ratio0 = MM preset-best / MM joint; ratio1 = MM max(preset, ols1, ols2) / MM joint")
    for ds in dict.fromkeys(r["ds"] for r in recs):
        rs = [r for r in recs if r["ds"] == ds and r["n"] == nn]
        if not rs:
            continue
        seeds = len(rs) // 10
        pl = [r for r in rs if r["planted"]]

        def win(r):
            """Whether the pair is in the near-miss window: preset-best / joint MI in [0.6, 0.9) and joint MI above 0.02."""
            return 0.6 <= r["preset"] / max(r["joint"], 1e-9) < 0.9 and r["joint"] > 0.02

        jr = []
        for s in range(seeds):
            blk = rs[s * 10 : (s + 1) * 10]
            order = sorted(range(10), key=lambda i: -blk[i]["joint"])
            for i, r in enumerate(blk):
                if r["planted"]:
                    jr.append(order.index(i) + 1)
        nwin = np.mean([sum(win(r) for r in rs[s * 10 : (s + 1) * 10]) for s in range(seeds)])
        line = f"{ds[:44].ljust(46)} seeds={seeds} "
        if pl:
            line += f"planted: joint {np.mean([r['joint'] for r in pl]):.3f} ratio0 {np.mean([r['preset'] / r['joint'] for r in pl]):.2f} ratio1 {np.mean([max(r['preset'], r['ols1'], r['ols2']) / r['joint'] for r in pl]):.2f} joint-rank {np.mean(jr):.1f}; in near-miss window {np.mean([win(r) for r in pl]):.2f}; "
        line += f"pairs in window / fit: {nwin:.1f} of 10"
        print(line)
        # non-planted pairs: ratio spread
        npl = [r for r in rs if not r["planted"] and r["joint"] > 0.02]
        if npl:
            print(
                "     non-planted pairs with joint>.02: ratio0 mean %.2f min %.2f"
                % (np.mean([r["preset"] / r["joint"] for r in npl]), np.min([r["preset"] / r["joint"] for r in npl]))
            )

print(
    "\n##### filter comparison (MM values): F_near = ratio0 in [.3,.9) & joint>.02 ; F_syn = F_near & (joint - max marginal) >= .02 ; recall of planted pair, mean #pairs kept of 10"
)
for nn in (5000, 30000):
    for ds in dict.fromkeys(r["ds"] for r in recs):
        rs = [r for r in recs if r["ds"] == ds and r["n"] == nn]
        if not rs or not any(r["planted"] for r in rs):
            continue
        seeds = len(rs) // 10

        def fnear(r):
            """Filter F_near: preset-best / joint MI ratio in [0.3, 0.9) and joint MI above 0.02."""
            return 0.3 <= r["preset"] / max(r["joint"], 1e-9) < 0.9 and r["joint"] > 0.02

        def fsyn(r):
            """Filter F_syn: F_near and a synergy of at least 0.02 over the best marginal."""
            return fnear(r) and r["joint"] - r["marg"] >= 0.02

        ftop3 = None
        res = []
        for nm, f in (("F_near", fnear), ("F_syn", fsyn)):
            rec = np.mean([f(r) for r in rs if r["planted"]])
            kept = np.mean([sum(f(r) for r in rs[s * 10 : (s + 1) * 10]) for s in range(seeds)])
            res.append(f"{nm}: recall {rec:.2f} kept {kept:.1f}")
        top3 = []
        for s in range(seeds):
            blk = rs[s * 10 : (s + 1) * 10]
            order = sorted(range(10), key=lambda i: -(blk[i]["joint"] - blk[i]["marg"]))[:3]
            top3.append(np.mean([blk[i]["planted"] for i in order]) * 3 > 0)
        res.append(f"top3 by synergy(joint-marg): recall {np.mean(top3):.2f}")
        print(f"n={nn} {ds[:40].ljust(42)}" + " | ".join(res))
