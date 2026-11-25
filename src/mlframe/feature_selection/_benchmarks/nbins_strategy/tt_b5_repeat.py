"""Reproducibility of fe_max_steps=1: the SAME fit (fixture, data seed, random_state=0, quantile) repeated in one process.   python tt_b5_repeat.py FIXTURE [repeats]"""
import sys, warnings
warnings.simplefilter("ignore")
import tt_b5_invariance as I
fx = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
X, y, inf, kind = I.make(fx, 3000, 0)
for i in range(reps):
    names, wall, core = I.fit_names(X, y, "quantile", 1, 0, kind)
    print(f"{fx} repeat {i}: {wall:.0f} s {names}", flush=True)
