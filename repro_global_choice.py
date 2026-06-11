import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from mlframe.feature_selection.filters.mrmr import MRMR

# Wide input that should exercise the pair-FE external-validation path
# (many numeric columns -> external_factors list > fe_max_external_validation_factors).
def build():
    rng = np.random.default_rng(88)
    n = 1200
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    cols = {"a": a, "b": b}
    # an interaction target so pair FE engages
    for k in range(20):
        cols[f"x{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    # target depends on a*b (pair interaction) so pair-FE generates+validates
    y = pd.Series(((a * b + 0.3 * a) > 0).astype(np.int64))
    return X, y


def cols_for(seed_global):
    X, y = build()
    np.random.seed(seed_global)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0, random_seed=7, skip_retraining_on_same_content=False,
                   fe_max_external_validation_factors=5).fit(X, y)
        out = sel.transform(X)
    return list(out.columns)


def main():
    MRMR.clear_fit_cache()
    base = cols_for(1)
    drift = False
    for gs in (2, 7, 13, 101, 999):
        MRMR.clear_fit_cache()
        now = cols_for(gs)
        if now != base:
            drift = True
            print(f"[global-seed {gs}] DRIFT vs base:")
            print("  base:", base)
            print("  now :", now)
            print("  diff (in now not base):", [c for c in now if c not in base])
            print("  diff (in base not now):", [c for c in base if c not in now])
    print("DRIFT DETECTED:", drift)


if __name__ == "__main__":
    main()
