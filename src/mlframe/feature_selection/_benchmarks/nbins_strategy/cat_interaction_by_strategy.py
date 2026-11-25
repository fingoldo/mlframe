"""Does each ``nbins_strategy`` keep a categorical XOR interaction in a wide frame? (6000 rows, 120 noise numeric columns, ``cat_a == cat_b`` drives the target.)

Found 2026-11-23 after the default had been changed to ``quantile_spacing``: its bin count is ``clamp(round(0.3 n), 3, 64)``, i.e. 64 at every n from 214, so every noise column got 64 bins, the 64 x 64 pair tables
raised the maxT floor and the interaction was lost in 3 of 3 seeds. ``sturges`` and ``uniform`` kept it in 3 of 3, ``freedman_diaconis`` lost it in 3 of 3, ``mdlp`` and ``optimal_joint`` kept it.

Usage: ``python cat_interaction_by_strategy.py --strategies sturges,quantile --seeds 7,11,23`` (the frame is ``_combined_frame`` of ``tests/feature_selection/mrmr/core/test_mrmr_cov_sweep_four_paths.py``).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from mlframe.feature_selection._benchmarks.binning_label import binning_params

warnings.simplefilter("ignore")


def main() -> None:
    """One MRMR fit per (seed, strategy); prints whether an output feature uses both categorical columns."""
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(root, "tests", "feature_selection", "mrmr", "core"))
    import test_mrmr_cov_sweep_four_paths as frames

    from mlframe.feature_selection.filters.mrmr import MRMR

    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="sturges,quantile,uniform,freedman_diaconis,mdlp")
    ap.add_argument("--seeds", default="7,11,23")
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--numeric", type=int, default=120)
    a = ap.parse_args()
    for seed in (int(s) for s in a.seeds.split(",")):
        df, y = frames._combined_frame(seed, a.rows, 2, p_numeric=a.numeric)
        for strategy in a.strategies.split(","):
            sel = MRMR(verbose=0, random_state=42, fe_max_steps=1, **binning_params(strategy)).fit(df, y)
            names = [str(c) for c in sel.get_feature_names_out()]
            recovered = any("cat_a" in n and "cat_b" in n for n in names)
            print(f"seed {seed} {strategy}: interaction recovered {recovered}, {len(names)} output columns: {names[:6]}", flush=True)


if __name__ == "__main__":
    main()
