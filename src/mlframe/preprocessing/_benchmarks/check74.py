import scipy.stats, numba, numpy as np, pandas as pd  # noqa
import mlframe.preprocessing.cleaning as C

NEW = C._get_nunique


def OLD(vals, skip_nan=True, skip_vals=None):
    u = np.unique(vals)
    if skip_nan:
        if u.dtype.kind in ("f", "c"):
            u = u[~np.isnan(u)]
        else:
            u = u[~pd.isna(u)]
    if skip_vals:
        for v in skip_vals:
            u = u[u != v]
    return len(u)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 5, 1000)
    a[::7] = np.nan
    cases = [
        ("plain", rng.uniform(0, 5, 1000), (0.0, 1.0)),
        ("withnan", a, (0.0, 1.0)),
        ("falsy_skip", np.modf(rng.uniform(-10, 10, 1000))[1], (0.0)),
        ("none_skip", rng.uniform(0, 1, 500), None),
        ("all_same", np.zeros(100), (0.0, 1.0)),
        ("allnan", np.full(50, np.nan), (0.0, 1.0)),
        ("int_dtype", rng.integers(0, 7, 200), (0.0, 1.0)),  # non-float -> np.unique path
    ]
    ok = True
    for name, arr, sk in cases:
        o = OLD(arr, skip_vals=sk)
        n = NEW(arr, skip_vals=sk)
        print(name, o, n, o == n)
        ok = ok and (o == n)
    print("ALL_OK", ok)
