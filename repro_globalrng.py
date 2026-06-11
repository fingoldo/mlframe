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


def state_tuple():
    s = np.random.get_state()
    # s = ('MT19937', key_array, pos, has_gauss, cached_gaussian)
    return (s[0], hash(s[1].tobytes()), int(s[2]), int(s[3]), float(s[4]))


def main():
    X, y = build()
    np.random.seed(12345)
    before = state_tuple()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MRMR(verbose=0, random_seed=7).fit(X, y)
    after = state_tuple()
    print("GLOBAL RNG before:", before)
    print("GLOBAL RNG after :", after)
    print("GLOBAL RNG CONSUMED DURING FIT:", before != after)


if __name__ == "__main__":
    main()
