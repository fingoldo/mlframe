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


def gpos():
    return int(np.random.get_state()[2]), hash(np.random.get_state()[1].tobytes())


def run(disable_cache):
    X, y = build()
    MRMR.clear_fit_cache()
    np.random.seed(42)  # pytest-randomly-style fixed reseed before the test
    p0 = gpos()
    kw = dict(verbose=0, random_seed=7)
    if disable_cache:
        kw["skip_retraining_on_same_content"] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_a = MRMR(**kw).fit_transform(X, y)
        p1 = gpos()
        sel = MRMR(**kw).fit(X, y)
        p2 = gpos()
        out_b = sel.transform(X)
    a, b = list(out_a.columns), list(out_b.columns)
    print(f"disable_cache={disable_cache}: global RNG p0={p0[0]} p1={p1[0]} p2={p2[0]}  consumed_by_fit1={p1!=p0}  A==B={a==b}")
    if a != b:
        print("  A:", a)
        print("  B:", b)


def main():
    run(False)
    run(True)


if __name__ == "__main__":
    main()
