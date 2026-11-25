"""One-line-per-setting summary of detection JSONL files (research B4). Usage: python tt_b4_summary_detect.py NAME=FILE.jsonl [NAME=FILE.jsonl ...]"""
from __future__ import annotations

import collections
import sys

import numpy as np

sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from tt_b4_analyze_detect import load, score


def main(args):
    print("setting | datasets | strong cols (oracle gain >= 0.01) | stepwise hit/partial/miss | weak cols accepted by stepwise | linear cols given a transform (stepwise) | "
          "partner+noise cols given a transform per dataset (M1 gated / M2 gated / stepwise) | useless adds per dataset (M1 / M2 / stepwise) | seconds per dataset (M1 / M2 / stepwise)")
    for a in args:
        name, f = a.split("=", 1)
        rows = load([f])
        out = []
        for m, r in (("M1", "gated"), ("M2", "gated"), ("M3", "stepwise")):
            st = collections.Counter(); wk_acc = wk_n = 0; wk_real = 0.0; lin = linn = pn = us = 0
            for o in rows:
                s = score(o, m, r)
                for x in s["nl"]:
                    st[x["status"]] += 1
                    if x["status"] == "weak":
                        wk_n += 1; wk_acc += x["pick"] is not None; wk_real += x["real"]
                lin += sum(z["accepted"] for z in s["lin"]); linn += len(s["lin"])
                pn += s["partner_acc"] + s["noise_acc"]; us += s["useless"]
            out.append((st, wk_acc, wk_n, wk_real / max(wk_n, 1), lin, linn, pn / len(rows), us / len(rows), np.mean([o["methods"][m]["seconds"] for o in rows])))
        st1, st2, st3 = out[0][0], out[1][0], out[2][0]
        strong = st3["hit"] + st3["partial"] + st3["miss"]
        print(f"{name} | {len(rows)} | {strong} | {st3['hit']}/{st3['partial']}/{st3['miss']} (M1 gated {st1['hit']}/{st1['partial']}/{st1['miss']}, M2 gated {st2['hit']}/{st2['partial']}/{st2['miss']}) | "
              f"{out[2][1]}/{out[2][2]} (mean realized gain {out[2][3]:+.4f}) | {out[2][4]}/{out[2][5]} | {out[0][6]:.2f} / {out[1][6]:.2f} / {out[2][6]:.2f} | "
              f"{out[0][7]:.2f} / {out[1][7]:.2f} / {out[2][7]:.2f} | {out[0][8]:.1f} / {out[1][8]:.1f} / {out[2][8]:.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
