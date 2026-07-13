"""iter75 cProfile harness: K-fold target encoding fit + apply at n=200k.

Import-order workaround for py3.14: scipy.stats + numba imported before mlframe
(cold mlframe.feature_selection imports native-segfault otherwise).
"""
import cProfile
import io
import pstats

import scipy.stats  # noqa: F401  -- ABI prewarm before mlframe
import numba  # noqa: F401
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._target_encoding_fe import (
    kfold_target_encode_fit,
    apply_target_encoding,
)

N = 200_000
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Mixed-cardinality categorical columns + binary y. Object column = the common
    # raw-categorical prod case; int column exercises the np.unique fast path.
    card_obj = 200
    card_int = 1500
    obj_vals = np.array([f"cat_{k}" for k in range(card_obj)], dtype=object)
    X = pd.DataFrame({
        "c_obj": obj_vals[rng.integers(0, card_obj, N)],
        "c_int": rng.integers(0, card_int, N),
    })
    y = rng.integers(0, 2, N).astype(np.float64)

    # Warm: build recipes once (also warms any njit).
    te_df, recipes = kfold_target_encode_fit(X, y, ["c_obj", "c_int"])

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        kfold_target_encode_fit(X, y, ["c_obj", "c_int"])
        apply_target_encoding(X, "c_obj", recipes["c_obj"])
        apply_target_encoding(X, "c_int", recipes["c_int"])
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(30)
    print(s.getvalue())
