"""The real-data leg: the cached OpenML beds exposed as Phase-0 scenario generators.

A real bed carries no ground truth, so its `truth` deliberately omits `base` -- that is what selects the
absolute K grid over the multiplier grid (pre-registration section 3a). Nothing here may invent a target-set
size; a guess would let the K grid be chosen after seeing results.

`dataset_seed` drives the row subsample rather than a regeneration. That is the closest honest analogue of an
i.i.d. redraw available on real data: the population is fixed, so the replications are subsamples of it, and
the paired-across-seeds statistic therefore speaks about this bed rather than about tabular data in general.
Every figure built from this leg has to say so.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Excluded from the kill-criterion denominator by pre-registration section 2: 26 roughly balanced classes
# binarised majority-vs-rest give a ~3.8% positive rate, about 115 positives after the row budget -- far
# below the minority count this repository requires before a minority-class metric is stable.
INELIGIBLE_BEDS: Tuple[str, ...] = ("isolet",)

ROW_BUDGET = 3000
FEAT_BUDGET = 1200


def eligible_bed_names() -> List[str]:
    """Cached bed names that count toward the kill criterion, in a stable order."""
    from ._realdata_cache import available

    names = sorted(str(meta["name"]) for meta in available())
    return [n for n in names if n not in INELIGIBLE_BEDS]


def all_bed_names() -> List[str]:
    """Every cached bed, including the ineligible ones -- reported, but carrying no weight in the verdict."""
    from ._realdata_cache import available

    return sorted(str(meta["name"]) for meta in available())


def _binarise(y_raw: np.ndarray) -> np.ndarray:
    """Collapse a target to 0/1 as majority-vs-rest, matching the existing bench's semantics."""
    values, counts = np.unique(y_raw, return_counts=True)
    majority = values[int(np.argmax(counts))]
    return np.asarray(y_raw == majority, dtype=np.int64)


def _prepare(frame: pd.DataFrame, y_raw: np.ndarray, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Coerce to numeric, drop constant columns, then subsample rows and cap width under the shared budgets.

    Column selection is by variance rank rather than at random: a random cap would make the bed itself a
    function of the seed in a way that changes which signal is present, whereas the variance rank is fixed
    for the bed and only the ROWS move with the seed.
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    nunique = numeric.nunique()
    numeric = numeric.loc[:, nunique[nunique > 1].index]
    if numeric.shape[1] > FEAT_BUDGET:
        keep = numeric.var(axis=0).sort_values(ascending=False).index[:FEAT_BUDGET]
        numeric = numeric.loc[:, sorted(keep, key=list(numeric.columns).index)]
    numeric.columns = [f"f{i}" for i in range(numeric.shape[1])]

    y = _binarise(y_raw)
    if numeric.shape[0] > ROW_BUDGET:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(numeric.shape[0], size=ROW_BUDGET, replace=False)
        idx.sort()
        numeric = numeric.iloc[idx].reset_index(drop=True)
        y = y[idx]
    return numeric, y


def make_real_bed(name: str, seed: int = 0) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Load one cached bed and return `(X, y, truth)` with no declared target set."""
    from ._realdata_cache import load_cached

    frame, y_raw, meta = load_cached(name)
    x, y = _prepare(frame, np.asarray(y_raw), seed)
    truth: Dict[str, Any] = {
        "leg": "real",
        "bed": name,
        "eligible": name not in INELIGIBLE_BEDS,
        "source": {k: meta.get(k) for k in ("openml_name", "version", "sha256")},
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "positive_rate": float(y.mean()),
        # No `base` and no `declared_target_size`: a real bed has no ground truth, which is precisely what
        # routes it to the absolute K grid. Do not add either key here.
    }
    return x, y, truth


def real_bed_scenarios(include_ineligible: bool = False) -> List[Tuple[str, Callable[[int], Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]]]]:
    """Return `[(bed_name, generator)]` for the real leg, eligible beds only unless asked otherwise."""
    names = all_bed_names() if include_ineligible else eligible_bed_names()
    if not names:
        raise RuntimeError("the real-data cache is empty; run `python -m ...fs_hybrid._realdata_cache` to fill it")
    out: List[Tuple[str, Callable[[int], Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]]]] = []
    for bed in names:
        out.append((bed, (lambda nm: (lambda seed: make_real_bed(nm, seed)))(bed)))
    return out
