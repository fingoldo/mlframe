"""Shared top-N-by-support group coding for the per-group chart builders.

Two builders carried near-identical copies of this: map raw group labels to integer codes, keep the largest
``max_groups`` groups and fold the rest into one "other" bucket. They differed only in what they returned and in
whether the labels came back support-ordered, which is a real difference in what each chart wants to show -- so the
ordering is a parameter here rather than a second implementation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

OTHER_LABEL = "other"


def group_codes_capped(
    groups: np.ndarray,
    max_groups: int,
    *,
    sort_by_support: bool = True,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """``(codes, labels, supports)`` for ``groups``, capped at ``max_groups`` plus one folded "other" bucket.

    ``codes`` indexes into ``labels`` and is parallel to ``groups``. When the group count exceeds the cap, the rare
    tail is remapped to a single trailing "other" code whose support is the sum of what it absorbed.

    ``sort_by_support`` orders the kept labels by descending row count. A per-group comparison chart wants that (the
    biggest groups lead); a group-by-time heatmap wants the natural label order so its rows stay in a stable,
    readable sequence across runs. Groups BEYOND the cap are always chosen by support either way -- only the order
    of the kept ones changes.
    """
    raw = np.asarray(groups).ravel()
    encodable = raw if raw.dtype.kind in "iuf" else raw.astype(str)
    uniq, inv, counts = np.unique(encodable, return_inverse=True, return_counts=True)
    inv = inv.astype(np.int64)
    n_unique = int(uniq.shape[0])

    if n_unique <= max_groups:
        if not sort_by_support:
            return inv, [str(v) for v in uniq], [int(c) for c in counts]
        order = np.argsort(counts)[::-1]
        remap = np.empty(n_unique, dtype=np.int64)
        remap[order] = np.arange(n_unique, dtype=np.int64)
        return remap[inv], [str(uniq[i]) for i in order], [int(counts[i]) for i in order]

    keep = np.argsort(counts)[::-1][:max_groups]
    if not sort_by_support:
        keep = np.sort(keep)  # kept groups chosen by support, then restored to natural label order
    remap = np.full(n_unique, max_groups, dtype=np.int64)  # default code = the "other" bucket
    remap[keep] = np.arange(max_groups, dtype=np.int64)
    labels = [str(uniq[i]) for i in keep] + [OTHER_LABEL]
    other_support = int(counts.sum() - counts[keep].sum())
    return remap[inv], labels, [int(counts[i]) for i in keep] + [other_support]


__all__ = ["OTHER_LABEL", "group_codes_capped"]
