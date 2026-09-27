"""The rows each target trains and is scored on, when some of its labels are missing.

Split positions index the ORIGINAL frame, as outlier detection already relies on: every full-length array (target,
weights, group ids, timestamps) is sliced by them. A target with missing labels keeps the suite's single split and drops
its unlabelled rows from every split, so its train never overlaps another target's val or test.

:class:`TargetRows` carries the narrowed indices of every split and, per split, the positions to keep INSIDE that
split: the frames aligned to a split (``train_df``, ``filtered_train_df``, ...) are sliced by those. Targets whose masks
are equal share one :class:`TargetRows`, keyed by the mask's signature.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ._target_labels import label_mask

# The split index attributes of ``TrainingContext`` and the frames aligned to each: train and val frames exist before
# and after outlier detection, test is never filtered, calib is its own slice.
SPLIT_INDEX_FIELDS: tuple[str, ...] = ("train_idx", "val_idx", "test_idx", "calib_idx", "filtered_train_idx", "filtered_val_idx")


def mask_signature(mask: np.ndarray) -> str:
    """A short stable id of a label mask: equal masks share it, so their targets share one narrowing."""
    mask = np.asarray(mask, dtype=bool)
    digest = hashlib.blake2b(np.packbits(mask).tobytes(), digest_size=8)
    digest.update(int(mask.size).to_bytes(8, "little"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetRows:
    """One label mask applied to the suite's split.

    ``idx[name]`` is the split's index array narrowed to labelled rows (global positions, order kept); ``pos[name]``
    the positions inside that split's array that survive, so ``split_idx[pos[name]] == idx[name]``. A split the suite
    does not have is absent from both.
    """

    signature: str
    mask: np.ndarray = field(repr=False)
    idx: Mapping[str, np.ndarray] = field(repr=False)
    pos: Mapping[str, np.ndarray] = field(repr=False)
    n_labelled: Mapping[str, int]
    n_total: Mapping[str, int]

    def labelled_share(self, split: str) -> Optional[float]:
        """Share of a split's rows that carry a label, or None when the split is absent or empty."""
        total = self.n_total.get(split)
        return None if not total else self.n_labelled[split] / total


def build_target_rows(mask: np.ndarray, splits: Mapping[str, Any]) -> TargetRows:
    """Narrow every split index in ``splits`` (``SPLIT_INDEX_FIELDS`` names, None for an absent split) to ``mask``."""
    mask = np.asarray(mask, dtype=bool)
    idx: dict[str, np.ndarray] = {}
    pos: dict[str, np.ndarray] = {}
    n_labelled: dict[str, int] = {}
    n_total: dict[str, int] = {}
    for name in SPLIT_INDEX_FIELDS:
        split_idx = splits.get(name)
        if split_idx is None:
            continue
        split_idx = np.asarray(split_idx)
        keep = mask[split_idx]
        pos[name] = np.flatnonzero(keep)
        idx[name] = split_idx[pos[name]]
        n_labelled[name] = int(pos[name].size)
        n_total[name] = int(split_idx.size)
    return TargetRows(signature=mask_signature(mask), mask=mask, idx=idx, pos=pos, n_labelled=n_labelled, n_total=n_total)


def splits_of(ctx: Any) -> dict[str, Any]:
    """The split index arrays of a ``TrainingContext``, by ``SPLIT_INDEX_FIELDS`` name."""
    return {name: getattr(ctx, name, None) for name in SPLIT_INDEX_FIELDS}


def group_targets_by_rows(
    target_by_type: Mapping[Any, Mapping[str, Any]], splits: Mapping[str, Any]
) -> "tuple[dict[str, TargetRows], dict[tuple[Any, str], str]]":
    """``({signature: TargetRows}, {(target type, name): signature})`` for every target with missing labels.

    A fully labelled target is in neither map: it trains on the split as it is, with no copy of anything.
    """
    rows_by_sig: dict[str, TargetRows] = {}
    sig_by_target: dict[tuple[Any, str], str] = {}
    for target_type, named in (target_by_type or {}).items():
        if not isinstance(named, Mapping):
            continue
        for name, values in named.items():
            mask = label_mask(values)
            if mask is None:
                continue
            sig = mask_signature(mask)
            if sig not in rows_by_sig:
                rows_by_sig[sig] = build_target_rows(mask, splits)
            sig_by_target[(target_type, str(name))] = sig
    return rows_by_sig, sig_by_target
