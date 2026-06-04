"""Round-4: cProfile the tree-member overhead in HybridSelector.fit (per the per-feature cProfile rule).

Confirms the tree member is not a hidden hotspot: prints the cumulative time of _tree_signals (the shallow GBM fit +
trees_to_dataframe co-occurrence count) and _admit_tree_products as a share of total fit. The product columns also
feed the single shared FI pass, so the only NEW cost is the one shallow GBM fit + the pair-count loop -- expected
to be a small fraction of the member fits (MRMR/shap/boruta dominate).
"""
from __future__ import annotations
import os, sys, cProfile, pstats, io
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_selector import HybridSelector


def make(n=4000, p=200, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    noise = rng.normal(size=(n, p))
    y = ((a * b + 0.6 * c + 0.3 * rng.normal(size=n)) > 0).astype(int)
    X = pd.DataFrame(np.column_stack([a, b, c, noise]), columns=["a", "b", "c"] + [f"n{i}" for i in range(p)])
    return X, pd.Series(y)


def main():
    X, y = make()
    pr = cProfile.Profile(); pr.enable()
    HybridSelector(vote=1, use_fe=True, use_tree_member=True, random_state=0).fit(X, y)
    pr.disable()
    s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    out = s.getvalue()
    print(out)
    # extract the tree-specific lines
    print("\n=== tree-member specific frames ===")
    for line in out.splitlines():
        if any(k in line for k in ("_tree_signals", "_admit_tree_products", "trees_to_dataframe", "_augment")):
            print(line)


if __name__ == "__main__":
    main()
