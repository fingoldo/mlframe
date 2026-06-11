import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from mlframe.feature_selection.filters.mrmr import MRMR


def build():
    rng = np.random.default_rng(88)
    n = 500
    sig = rng.standard_normal(n)
    X = pd.DataFrame({"signal": sig, "n0": rng.standard_normal(n), "n1": rng.standard_normal(n)})
    y = pd.Series((sig > 0).astype(np.int64))
    return X, y


def one():
    X, y = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_a = MRMR(verbose=0, random_seed=7).fit_transform(X, y)
        sel = MRMR(verbose=0, random_seed=7).fit(X, y)
        out_b = sel.transform(X)
    return list(out_a.columns), list(out_b.columns)


def main():
    baseline = None
    fails = 0
    for i in range(40):
        a, b = one()
        if a != b:
            fails += 1
            print(f"[iter {i}] INTERNAL MISMATCH A vs B")
            print("  A:", a)
            print("  B:", b)
        if baseline is None:
            baseline = a
        elif a != baseline:
            print(f"[iter {i}] CROSS-ITER DRIFT vs baseline")
            print("  base:", baseline)
            print("  now :", a)
    print("FAILS:", fails)


if __name__ == "__main__":
    main()
