"""Non-destructive driver for round4_broad_realdata_bench WITH ReliefF added.

Adds zero behaviour to the shipped bench except:
  * writable output paths (the bench's D:/Temp defaults don't exist on this host),
  * a trimmed dataset subset so a full multi-selector sweep (now incl. relieff) finishes
    in a tractable wall-clock while still spanning interaction-heavy (madelon) + others.

The ReliefF strategy itself lives in the shipped files (fs_selectors.ReliefFSel +
STRATEGIES['relieff']); this only chooses datasets and prints a compact AUC+time table.
"""
from __future__ import annotations
import os, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.normpath(os.path.join(HERE, "..", "..", "src", "mlframe",
                                      "feature_selection", "_benchmarks", "fs_hybrid"))
sys.path.insert(0, BENCH)

import round4_broad_realdata_bench as B  # noqa: E402

_out = os.path.join(tempfile.gettempdir(), "relieff_bench")
B.PROGRESS = _out + "_progress.txt"
B.RESULTS = _out + "_results.md"

# subset: madelon (interaction anchor) + gina_agnostic + scene + breast_cancer fallback
SUBSET = [
    ("madelon",       dict(name="madelon", version=1)),
    ("gina_agnostic", dict(name="gina_agnostic", version=1)),
    ("scene",         dict(name="scene", version=1)),
]


def main():
    allrows = []
    for name, kw in SUBSET:
        try:
            rows = B.load_with_retry(name, kw)
        except Exception as e:
            print(f"[skip {name}: {type(e).__name__}: {e}]", flush=True)
            rows = []
        allrows += rows
    # guaranteed-offline real dataset
    try:
        X, y, note = B.fallback_breast_cancer()
        allrows += B.run_dataset("breast_cancer", X, y, note)
    except Exception as e:
        print(f"[skip breast_cancer: {e}]", flush=True)

    # compact table
    print("\n" + "=" * 96)
    print(f"{'dataset':<16}{'strategy':<11}{'n':>5}{'fit_s':>9}{'auc_mean':>10}{'lgbm':>9}{'logit':>9}{'knn':>9}")
    print("=" * 96)
    for r in allrows:
        if r.get("n", -1) == -1:
            print(f"{r['dataset']:<16}{r['strategy']:<11}  FAILED: {r.get('error','')[:50]}")
            continue
        print(f"{r['dataset']:<16}{r['strategy']:<11}{r['n']:>5}{r['fit_s']:>9.1f}"
              f"{r['auc_mean']:>10.4f}{r.get('lgbm',float('nan')):>9.4f}"
              f"{r.get('logit',float('nan')):>9.4f}{r.get('knn',float('nan')):>9.4f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n[total {time.time()-t0:.0f}s]")
