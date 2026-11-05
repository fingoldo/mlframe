"""Selection equivalence of the FE auto-escalation target rank: row-position tie split (current) vs average ranks.

``_stash_fe_targets`` stores the rank of y that the escalation proposers fit against. Today tied target values are ordered by row position, so
a discrete target gets an arbitrary within-tie order; average ranks give tied rows one shared value. This script fits MRMR on seeded synthetic
targets with many ties under both rank forms and prints, per dataset, the selected columns and the escalation-admitted recipe names, so a
change of the default can be judged on decisions rather than on rank values.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_escalation_rank_ties``
"""

from __future__ import annotations

import sys
import warnings
from unittest import mock

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _average_rank_stash(original):
    """Wrap ``_stash_fe_targets`` so the stashed rank uses average ranks for ties."""
    from scipy.stats import rankdata

    def stash(self, y_np, X):
        original(self, y_np, X)
        rank = getattr(self, "_fe_escalation_y_rank_", None)
        if rank is not None:
            y = np.asarray(y_np)
            r = rankdata(y, method="average") - 1.0
            self._fe_escalation_y_rank_ = (r / max(len(r) - 1, 1)).astype(np.asarray(rank).dtype)

    return stash


def _targets(seed: int):
    """Seeded datasets whose targets carry many ties: rounded to a few levels, and with a zero spike."""
    rng = np.random.default_rng(seed)
    n, p = 3000, 6
    X = rng.normal(size=(n, p))
    base = np.sin(3.7 * X[:, 0]) * X[:, 1] + 0.1 * rng.normal(size=n)
    levels = np.round(base * 3.0) / 3.0
    spike = base.copy()
    spike[rng.random(n) < 0.4] = 0.0
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    return frame, {"rounded_levels": levels, "zero_spike": spike}


def _fit(frame, y):
    from mlframe.feature_selection.filters.mrmr import MRMR

    sel = MRMR(verbose=0, random_state=42)
    sel.fit(frame, pd.Series(y, name="y"))
    recipes = sorted(r.name for r in (getattr(sel, "_engineered_recipes_", None) or []))
    return sorted(getattr(sel, "selected_features_", None) or []), recipes


def main(seeds=(0, 1, 2)) -> int:
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages import _inputs

    original = _inputs._stash_fe_targets
    same = total = 0
    for seed in seeds:
        frame, targets = _targets(seed)
        for name, y in targets.items():
            old = _fit(frame, y)
            with mock.patch.object(_inputs, "_stash_fe_targets", _average_rank_stash(original)):
                new = _fit(frame, y)
            equal = old == new
            same += equal
            total += 1
            print(f"seed={seed} target={name} equal={equal}\n  position-split: {old}\n  average-rank:   {new}", flush=True)
    print(f"equivalent decisions: {same}/{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
