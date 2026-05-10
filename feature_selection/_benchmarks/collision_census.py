"""arr2str collision census for the B12 silent correctness bug.

The pre-B12 ``arr2str`` (filters.py:70-78) does naive string concatenation
``"".join(str(el) for el in arr)``. Multiset collisions are routine:

    sorted([1, 11])    -> "111"
    sorted([1, 1, 1])  -> "111"   <-- collision

Cache keys derived this way silently return wrong cached entropies. This
script enumerates the multiset universe for each Phase-0 scenario size
(n_features x interactions_max_order) and counts colliding string buckets.

Run::

    python -m mlframe.feature_selection._benchmarks.collision_census \\
        [--n-features 100 --max-order 2] [--out _results/collision_census_pre_refactor.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "_results"


def _legacy_arr2str(arr: list[int]) -> str:
    """Faithful pure-Python copy of pre-B12 ``arr2str`` (filters.py:70-78)."""
    out = ""
    for el in arr:
        out += str(el)
    return out


def census_for(n_features: int, max_order: int) -> dict:
    """Enumerate all ``sorted`` multisets of size 1..max_order over [0, n_features)
    and report collision statistics."""
    bucket_to_multisets: defaultdict[str, list[tuple[int, ...]]] = defaultdict(list)
    total = 0
    for order in range(1, max_order + 1):
        for combo in combinations_with_replacement(range(n_features), order):
            key = _legacy_arr2str(list(combo))
            bucket_to_multisets[key].append(combo)
            total += 1
    n_unique_keys = len(bucket_to_multisets)
    colliding_buckets = {k: v for k, v in bucket_to_multisets.items() if len(v) > 1}
    n_colliding_keys = len(colliding_buckets)
    n_colliding_multisets = sum(len(v) for v in colliding_buckets.values())
    bucket_size_dist = Counter(len(v) for v in bucket_to_multisets.values())
    examples = []
    for k, v in list(colliding_buckets.items())[:5]:
        examples.append({"key": k, "multisets": [list(m) for m in v]})
    return {
        "n_features": n_features,
        "max_order": max_order,
        "n_total_multisets": total,
        "n_unique_keys": n_unique_keys,
        "n_colliding_keys": n_colliding_keys,
        "n_colliding_multisets": n_colliding_multisets,
        "collision_rate_per_multiset": n_colliding_multisets / total if total else 0.0,
        "bucket_size_distribution": dict(bucket_size_dist),
        "examples": examples,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "collision_census_pre_refactor.json")
    args = p.parse_args()

    # Mirror the bench scenarios' (p, max_order) shapes.
    grid = [
        (50, 1), (50, 2), (50, 3),
        (100, 1), (100, 2),
        (200, 1), (200, 2),
        (500, 1), (500, 2),
        (1000, 1),
    ]
    results: dict[str, dict] = {}
    for n_features, max_order in grid:
        key = f"p{n_features}_o{max_order}"
        results[key] = census_for(n_features, max_order)
        c = results[key]
        print(
            f"{key}: total={c['n_total_multisets']:>10,}  unique_keys={c['n_unique_keys']:>10,}  "
            f"colliding_keys={c['n_colliding_keys']:>8,}  colliding_multisets={c['n_colliding_multisets']:>8,}  "
            f"rate={c['collision_rate_per_multiset']:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
