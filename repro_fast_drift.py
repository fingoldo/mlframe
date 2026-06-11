import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Count how often the external-validation choice path is taken (proves the path is hit).
import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core as pc
_orig_choice = np.random.choice
_hits = {"global": 0}
def _counting_global_choice(*a, **k):
    _hits["global"] += 1
    return _orig_choice(*a, **k)


def make(n=2500, seed=11):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n); b = rng.uniform(1.0, 5.0, n)
    c = rng.uniform(1.0, 5.0, n); d = rng.uniform(0.0, 2*np.pi, n)
    cols = {"a": a, "b": b, "c": c, "d": d}
    for k in range(8):
        cols[f"x{k}"] = rng.uniform(1.0, 5.0, n)
    X = pd.DataFrame(cols)
    y = pd.Series(a**2 / b + 3.0 * np.log(c) * np.sin(d), name="y")
    return X, y


np.random.choice = _counting_global_choice


def eng_names(global_seed):
    from mlframe.feature_selection.filters.mrmr import MRMR
    X, y = make()
    MRMR.clear_fit_cache()
    np.random.seed(global_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=7, fe_max_external_validation_factors=3).fit(X, y)
    return [n for n in fs.get_feature_names_out() if n not in {"a", "b", "c", "d"} and not n.startswith("x")]


def main():
    base = eng_names(1)
    print("base eng:", base)
    drift = False
    for gs in (2, 13, 101, 9999):
        e = eng_names(gs)
        if e != base:
            drift = True
            print(f"[global-seed {gs}] DRIFT: {e}")
    print("global np.random.choice hits during fits:", _hits["global"])
    print("DRIFT DETECTED:", drift)


if __name__ == "__main__":
    main()
