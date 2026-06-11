import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core as pc

# Wrap default_rng to detect construction of the extval generator + count its .choice calls.
_hits = {"extval_choice": 0, "global_choice": 0}
_orig_global_choice = np.random.choice
def _cg(*a, **k):
    _hits["global_choice"] += 1
    return _orig_global_choice(*a, **k)
np.random.choice = _cg

import numpy.random as _nr
_orig_default_rng = _nr.default_rng
class _CountingGen:
    def __init__(self, g): self._g = g
    def choice(self, *a, **k):
        _hits["extval_choice"] += 1
        return self._g.choice(*a, **k)
    def __getattr__(self, n): return getattr(self._g, n)

# Only wrap the generator created at the extval seed site is hard to isolate; instead count
# ALL .choice on generators by patching default_rng to return a counting proxy.
def _patched_default_rng(*a, **k):
    return _CountingGen(_orig_default_rng(*a, **k))
pc.np.random.default_rng = _patched_default_rng

N = 20000
def make(seed=42):
    rng = _orig_default_rng(seed)
    a = rng.uniform(1.0, 5.0, N); b = rng.uniform(1.0, 5.0, N)
    c = rng.uniform(1.0, 5.0, N); d = rng.uniform(0.0, 2*np.pi, N)
    e = rng.normal(0.0, 1.0, N); f = rng.normal(0.0, 1.0, N)
    y = a**2/b + f/5.0 + 3.0*np.log(c)*np.sin(d)
    return pd.DataFrame({"a":a,"b":b,"c":c,"d":d,"e":e}), pd.Series(y, name="y")

def main():
    from mlframe.feature_selection.filters.mrmr import MRMR
    X, y = make()
    MRMR.clear_fit_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(verbose=0, random_seed=7).fit(X, y)
    eng = [n for n in fs.get_feature_names_out() if n not in {"a","b","c","d","e"}]
    print("eng:", eng)
    print("extval generator .choice calls:", _hits["extval_choice"])
    print("global np.random.choice calls:", _hits["global_choice"])

if __name__ == "__main__":
    main()
