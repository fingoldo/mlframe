"""Profile per_group_rank @10M with many groups, including the tiebreak_values path."""
from __future__ import annotations

import sys
import types

sys.modules["cupy"] = types.ModuleType("cupy")
import scipy.stats  # noqa
import numba  # noqa
import cProfile, pstats, io, time
from typing import Any, Optional

import numpy as np


def _profile_one(fn: Any, vals: np.ndarray, gids: np.ndarray, method: str, tiebreak_values: Optional[np.ndarray], name: str, n_groups: int) -> None:
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    fn(vals, gids, method=method, tiebreak_values=tiebreak_values)
    pr.disable()
    wall = time.perf_counter() - t0
    print(f"=== {name} wall={wall:.3f}s (n_groups={n_groups}) ===")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8)
    print("\n".join(s.getvalue().splitlines()[:16]))


def main() -> None:
    from mlframe.feature_engineering.grouped import per_group_rank

    n = 10_000_000
    rng = np.random.default_rng(0)
    n_groups = 200_000  # ~50 rows/group
    gids = rng.integers(0, n_groups, size=n).astype(np.int64)
    # Deliberately low-cardinality values so groups carry many tied rows -- this is the
    # exact shape where plain method="ordinal" gives an uninformative arbitrary tie split
    # and tiebreak_values earns its keep.
    vals = rng.integers(0, 5, size=n).astype(np.float64)
    tiebreak = rng.standard_normal(n)
    # warm
    per_group_rank(vals[:10000], gids[:10000], method="average")
    per_group_rank(vals[:10000], gids[:10000], method="ordinal", tiebreak_values=tiebreak[:10000])

    _profile_one(per_group_rank, vals, gids, "average", None, "per_group_rank(method=average)", n_groups)
    _profile_one(per_group_rank, vals, gids, "ordinal", None, "per_group_rank(method=ordinal)", n_groups)
    _profile_one(per_group_rank, vals, gids, "ordinal", tiebreak, "per_group_rank(method=ordinal, tiebreak_values=...)", n_groups)


if __name__ == "__main__":
    main()
