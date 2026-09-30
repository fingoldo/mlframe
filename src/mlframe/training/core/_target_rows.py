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

# The split index attributes of ``TrainingContext`` and the frames aligned to each: train and val frames exist before
# and after outlier detection, test is never filtered, calib is its own slice.
SPLIT_INDEX_FIELDS: tuple[str, ...] = ("train_idx", "val_idx", "test_idx", "calib_idx", "filtered_train_idx", "filtered_val_idx")


def mask_signature(mask: np.ndarray) -> str:
    """A short stable id of a label mask: equal masks share it, so their targets share one narrowing."""
    mask = np.asarray(mask, dtype=bool)
    digest = hashlib.blake2b(np.packbits(mask).data, digest_size=8)
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
    # Splits this target does without: too few labelled rows there. A dropped val trains with no val (no early stopping),
    # a dropped calib skips calibration.
    dropped: frozenset = frozenset()

    def labelled_share(self, split: str) -> Optional[float]:
        """Share of a split's rows that carry a label, or None when the split is absent or empty."""
        total = self.n_total.get(split)
        return None if not total else self.n_labelled[split] / total

    def without(self, splits: "set[str]") -> "TargetRows":
        """These rows with ``splits`` emptied (val and its outlier-filtered twin go together); the signature changes with them."""
        if "val_idx" in splits:
            splits = splits | {"filtered_val_idx"}
        splits = {s for s in splits if s in self.idx}
        if not splits:
            return self
        empty = np.empty(0, dtype=np.int64)
        idx = {k: (empty if k in splits else v) for k, v in self.idx.items()}
        pos = {k: (empty if k in splits else v) for k, v in self.pos.items()}
        n_labelled = {k: (0 if k in splits else v) for k, v in self.n_labelled.items()}
        dropped = self.dropped | frozenset(splits)
        signature = self.signature + "-" + mask_signature(np.array([s in dropped for s in SPLIT_INDEX_FIELDS]))[:4]
        return TargetRows(signature=signature, mask=self.mask, idx=idx, pos=pos, n_labelled=n_labelled, n_total=self.n_total, dropped=dropped)


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
