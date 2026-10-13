"""Null-MI significance floor for a feature search, looked up from the offline table built by ``_benchmarks/build_null_mi_table``.

The threshold is the 95th percentile of the largest plug-in MI that pure noise reaches among the N candidates a search scores, for
the same row count, bin count and target class prior. Quantile binning makes the null independent of the marginal shape, so the
fingerprint is (n, bins, target entropy, N); the table is read once and cached, and the lookup is a nearest-neighbour match on
log n, bins, entropy and log N.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

_TABLE_PATH = Path(__file__).resolve().parent / "data" / "null_mi_table.json"


@lru_cache(maxsize=1)
def _entries() -> tuple:
    """The table's entries as a tuple of dicts, read once per process; empty when the table file is absent."""
    if not _TABLE_PATH.exists():
        return ()
    data = json.loads(_TABLE_PATH.read_text(encoding="utf-8"))
    return tuple(data.get("entries", {}).values())


def target_entropy(y: np.ndarray) -> float:
    """Shannon entropy of the target's class frequencies, in nats (the unit of the plug-in MI the table is built in)."""
    _, counts = np.unique(np.asarray(y).ravel(), return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def null_mi_threshold(n_rows: int, n_bins: int, entropy: float, n_candidates: int) -> float | None:
    """The table's null-MI floor for this fingerprint (nearest entry), or None when the table is missing or empty.

    Distance is taken on log n, bins, entropy and log N, so a fingerprint between two table points gets the nearer one's floor.
    """
    entries = _entries()
    if not entries:
        return None
    ln = math.log(max(int(n_rows), 1))
    lc = math.log(max(int(n_candidates), 1))
    best, best_d = None, math.inf
    for e in entries:
        d = (math.log(e["n"]) - ln) ** 2 + (e["bins"] - n_bins) ** 2 / 16.0 + (e["entropy"] - entropy) ** 2 * 4.0 + (math.log(e["candidates"]) - lc) ** 2
        if d < best_d:
            best, best_d = e, d
    return float(best["threshold"]) if best is not None else None


def pair_search_floor(n_rows: int, classes_y, n_bins: int, n_candidates: int) -> float | None:
    """The null floor for a pair search that scored ``n_candidates`` engineered forms against ``classes_y``, or None without a table."""
    return null_mi_threshold(int(n_rows), int(n_bins), target_entropy(classes_y), int(n_candidates))


__all__ = ["null_mi_threshold", "pair_search_floor", "target_entropy"]
