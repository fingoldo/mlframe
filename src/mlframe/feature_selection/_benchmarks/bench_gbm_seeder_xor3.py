"""Does the surrogate-GBM interaction seeder find the 3-way XOR triple of the four-pillars xor3 frame?

Run from the repository root: ``python src/mlframe/feature_selection/_benchmarks/bench_gbm_seeder_xor3.py``. Prints, per variant (binary parity, continuous sign product) and seed, the rank of the
true triple among ``seeded_triplets`` (``None`` = not in the top 8). Result of 2026-10-26: binary 3 of 5 seeds (ranks 0, 0, 1), continuous 0 of 5 (nothing emitted); the other
top triples are noise columns, which is why ``fe_gbm_seeder_enable`` stays OFF by default.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
sys.path.insert(0, os.getcwd())
from tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_xor3_noise import _make_frame
from mlframe.feature_selection.filters._surrogate_interaction_seeder import surrogate_gbm_interaction_seeds
for kind in ("binary", "continuous"):
    for seed in range(5):
        out = _make_frame(kind, seed=seed)
        X, y = out[0], out[1]
        names = list(X.columns)
        # ordinal-encode: 10 quantile bins per column (the screening matrix the seeder receives)
        disc = np.column_stack([pd.qcut(X[c].rank(method="first"), 10, labels=False).to_numpy() for c in names]).astype(np.int64)
        yy = np.asarray(y).astype(np.int64)
        pairs, trips, info = surrogate_gbm_interaction_seeds(disc, yy, list(range(len(names))), is_classification=True, top_k_pairs=12, top_k_triples=8, random_seed=seed)
        truth = tuple(sorted(i for i, c in enumerate(names) if c in ("x1", "x2", "x3")))
        rank = next((r for r, t in enumerate(trips) if tuple(sorted(t)) == truth), None)
        print(f"{kind} seed {seed}: truth={truth} gated={info.get('gated')} n_triples={len(trips)} true-triple rank={rank} top3={[tuple(names[i] for i in t) for t in trips[:3]]}", flush=True)
