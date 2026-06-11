import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from mlframe.feature_selection.filters.mrmr import MRMR

N = 20_000
SEED = 42
SECOND_SIGNAL_SCALE = 3.0


def make():
    rng = np.random.default_rng(SEED)
    a = rng.uniform(1.0, 5.0, N); b = rng.uniform(1.0, 5.0, N)
    c = rng.uniform(1.0, 5.0, N); d = rng.uniform(0.0, 2.0*np.pi, N)
    e = rng.normal(0.0, 1.0, N); f = rng.normal(0.0, 1.0, N)
    y = a**2/b + f/5.0 + SECOND_SIGNAL_SCALE*np.log(c)*np.sin(d)
    return pd.DataFrame({"a":a,"b":b,"c":c,"d":d,"e":e}), pd.Series(y, name="y")


def gpos():
    s = np.random.get_state()
    return int(s[2])


def names_with_global_seed(gs):
    df, y = make()
    MRMR.clear_fit_cache()
    np.random.seed(gs)
    p0 = gpos()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=7).fit(df, y)
    p1 = gpos()
    eng = [n for n in fs.get_feature_names_out() if n not in {"a","b","c","d","e"}]
    return eng, (p0 != p1)


def main():
    base, consumed = names_with_global_seed(1)
    print("fit consumed global RNG:", consumed)
    print("base eng:", base)
    drift = False
    for gs in (2, 7, 13, 101, 999, 31337):
        eng, _ = names_with_global_seed(gs)
        if eng != base:
            drift = True
            print(f"[global-seed {gs}] DRIFT: {eng}")
    print("DRIFT DETECTED:", drift)


if __name__ == "__main__":
    main()
