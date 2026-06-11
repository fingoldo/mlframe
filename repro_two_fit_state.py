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


def gstate():
    s = np.random.get_state()
    return (int(s[2]), hash(s[1].tobytes()))


def main():
    X, y = build()
    # pytest-randomly reseeds numpy.random to a per-test fixed seed before the test.
    np.random.seed(99999)
    s0 = gstate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_a = MRMR(verbose=0, random_seed=7).fit_transform(X, y)
        s1 = gstate()
        sel = MRMR(verbose=0, random_seed=7).fit(X, y)
        s2 = gstate()
        out_b = sel.transform(X)
    a, b = list(out_a.columns), list(out_b.columns)
    print("global RNG s0:", s0)
    print("global RNG s1 (after fit_transform):", s1, " changed:", s1 != s0)
    print("global RNG s2 (after 2nd fit):", s2, " changed:", s2 != s1)
    print("A == B:", a == b)
    if a != b:
        print("A:", a)
        print("B:", b)

    # FIT_CACHE inspection
    fc = getattr(MRMR, "_FIT_CACHE", None)
    print("MRMR._FIT_CACHE type:", type(fc), "len:", (len(fc) if hasattr(fc, "__len__") else "n/a"))


if __name__ == "__main__":
    main()
