"""Group a flat value array by a per-row label in one pass instead of one pass per label.

``values[labels == lab]`` inside a loop over the labels is O(n * groups) for an O(n) answer, and it is the
same idiom in three places. The subtlety, which is the same one the multiclass subsample hit, is that the
obvious replacement is SLOWER: a stable argsort over the labels themselves costs more than the masks it
replaces when the labels are strings or wide integers. Factorising to small integer codes first is what
makes it a win -- numpy radix-sorts narrow integers.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

# Codes above this need a wider dtype; below it numpy can radix-sort them, which is the whole point.
_MAX_NARROW_CODE = np.iinfo(np.int16).max


def _codes_and_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Integer codes for ``labels`` plus the distinct labels in FIRST-APPEARANCE order."""
    try:
        import pandas as pd

        codes, uniq = pd.factorize(labels)
        return np.asarray(codes), np.asarray(uniq)
    except (ImportError, TypeError):
        # Unhashable or unorderable labels: fall back to the ordering numpy can manage.
        uniq, codes = np.unique(labels, return_inverse=True)
        return np.asarray(codes), np.asarray(uniq)


def group_indices_by_label(labels: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(order, bounds, distinct_labels)``: ``order[bounds[i]:bounds[i + 1]]`` are the rows of label ``i``.

    Row order WITHIN a group is ascending, matching what a boolean mask returned, and the groups come back
    in first-appearance order.
    """
    labels_arr = np.asarray(labels)
    codes, uniq = _codes_and_labels(labels_arr)
    if codes.size and int(codes.max()) <= _MAX_NARROW_CODE:
        codes = codes.astype(np.int16, copy=False)
    order = np.argsort(codes, kind="stable")
    bounds = np.append(np.searchsorted(codes[order], np.arange(len(uniq))), len(order))
    return order, bounds, uniq


def group_values_by_label(values: Any, labels: Sequence[Any]) -> Dict[str, np.ndarray]:
    """``{str(label) -> values of that label}``, in first-appearance order of the labels."""
    vals = np.asarray(values)
    order, bounds, uniq = group_indices_by_label(labels)
    return {str(u): vals[order[bounds[i] : bounds[i + 1]]] for i, u in enumerate(uniq)}
