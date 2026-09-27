"""Per-target decisions for a target with missing labels: which rows it uses, which splits it does without, or a skip.

Split thresholds (``TrainingBehaviorConfig.min_labelled_*_rows``) each have their own action: too few labelled train rows
skips the target; too few val rows trains it with no val (an empty val, the suite's ``val_size=0`` path); too few calib
rows skips its calibration; too few test rows only marks its test metrics ``low_n``. A classification target also needs
two classes with at least two labelled train rows each, and its val/test/calib rows of a class train never saw are left
out (no model can predict that class; scoring them would only measure its absence).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..configs import TargetTypes
from ..extractors import intize_targets
from ._target_labels import label_mask, labelled_unique
from ._target_rows import TargetRows, build_target_rows, mask_signature, splits_of

logger = logging.getLogger(__name__)

TARGET_ROWS_METADATA_KEY = "target_rows"
_CLASSIFICATION = (TargetTypes.BINARY_CLASSIFICATION, TargetTypes.MULTICLASS_CLASSIFICATION, TargetTypes.MULTILABEL_CLASSIFICATION)
MIN_ROWS_PER_TRAIN_CLASS = 2


def _as_array(values: Any) -> np.ndarray:
    """``values`` as a numpy array (polars / pandas via ``to_numpy``)."""
    to_numpy = getattr(values, "to_numpy", None)
    return to_numpy() if callable(to_numpy) else np.asarray(values)


def _first(counts: dict, *splits: str) -> Optional[int]:
    """The count of the first split present: the outlier-filtered train/val when outlier detection ran, else the split."""
    for split in splits:
        if split in counts:
            return counts[split]
    return None


def working_target(values: Any, mask: np.ndarray, target_type: Any, name: str) -> Any:
    """The full-length target the models are handed while its rows are narrowed.

    Regression keeps NaN on unlabelled rows: every split index in scope excludes them. A classification target gets an
    integer dtype, so its unlabelled rows are filled with one of its own labels; no split index in scope reaches them.
    """
    if target_type not in _CLASSIFICATION:
        return values
    arr = _as_array(values)
    if arr.dtype.kind == "O":
        return values
    filled = np.array(arr, dtype=np.float64, copy=True)
    fill = 0.0 if filled.ndim > 1 else float(labelled_unique(arr)[0])
    filled[np.isnan(filled)] = fill
    cast = {name: filled}
    intize_targets(cast)  # the smallest integer dtype, as the extractor gives a fully labelled class column
    return cast[name]


def _known_class_mask(arr: np.ndarray, mask: np.ndarray, splits: dict) -> "tuple[np.ndarray, int]":
    """``mask`` without the rows whose class never occurs in labelled train rows, and how many that removes."""
    train_idx = splits.get("filtered_train_idx")
    if train_idx is None:
        train_idx = splits.get("train_idx")
    if train_idx is None or arr.ndim > 1:
        return mask, 0
    train_classes = labelled_unique(arr[np.asarray(train_idx)])
    known = np.isin(arr, train_classes)
    unknown = mask & ~known
    return mask & known, int(unknown.sum())


def _train_class_problem(arr: np.ndarray, rows: TargetRows) -> Optional[str]:
    """Why a classification target cannot train on these rows, or None when it can."""
    train_idx = rows.idx.get("filtered_train_idx", rows.idx.get("train_idx"))
    if train_idx is None or arr.ndim > 1:
        return None
    _, counts = np.unique(arr[train_idx], return_counts=True)
    if counts.size < 2:
        return f"its labelled train rows hold {counts.size} class(es)"
    if counts.min() < MIN_ROWS_PER_TRAIN_CLASS:
        return f"a class has {int(counts.min())} labelled train row(s), fewer than {MIN_ROWS_PER_TRAIN_CLASS}"
    return None


def _splits_to_drop(arr: np.ndarray, rows: TargetRows, cfg: Any, is_classification: bool) -> "tuple[set[str], list[str]]":
    """Splits this target does without, and the notes explaining each decision."""
    drop: set[str] = set()
    notes: list[str] = []
    n_val = _first(rows.n_labelled, "filtered_val_idx", "val_idx")
    if _first(rows.n_total, "filtered_val_idx", "val_idx"):
        val_idx = rows.idx.get("filtered_val_idx", rows.idx.get("val_idx"))
        if n_val < cfg.min_labelled_val_rows:
            drop.add("val_idx")
            notes.append(f"val has {n_val} labelled row(s) < min_labelled_val_rows={cfg.min_labelled_val_rows}: trained without val")
        elif is_classification and arr.ndim == 1 and np.unique(arr[val_idx]).size < 2:
            drop.add("val_idx")
            notes.append("val holds one class: trained without val")
    if rows.n_total.get("calib_idx") and rows.n_labelled["calib_idx"] < cfg.min_labelled_calib_rows:
        drop.add("calib_idx")
        notes.append(f"calib has {rows.n_labelled['calib_idx']} labelled row(s) < min_labelled_calib_rows={cfg.min_labelled_calib_rows}: no calibration")
    return drop, notes


def rows_for_target(ctx: Any, target_type: Any, name: str, values: Any, metadata: dict, cache: dict) -> "tuple[Optional[TargetRows], Any, bool]":
    """``(rows, working target, train it)`` for one target; ``rows`` is None for a fully labelled one (nothing changes).

    ``cache`` maps a mask signature to its :class:`TargetRows` across the targets of one suite call.
    """
    mask = label_mask(values)
    if mask is None:
        return None, values, True
    cfg = ctx.behavior_config
    splits = splits_of(ctx)
    arr = _as_array(values)
    is_classification = target_type in _CLASSIFICATION
    record: dict = {}
    if is_classification:
        mask, n_unknown = _known_class_mask(arr, mask, splits)
        if n_unknown:
            record["rows_of_classes_absent_from_train"] = n_unknown
            logger.warning("%s/%s: %s labelled val/test/calib row(s) carry a class its train rows never show; left out of scoring.", target_type, name, f"{n_unknown:_}")
    sig = mask_signature(mask)
    if sig not in cache:
        cache[sig] = build_target_rows(mask, splits)
    rows = cache[sig]
    record.update(signature=rows.signature, n_labelled=dict(rows.n_labelled), n_total=dict(rows.n_total))
    entry = metadata.setdefault(TARGET_ROWS_METADATA_KEY, {})
    key = f"{target_type}/{name}"
    n_train = _first(rows.n_labelled, "filtered_train_idx", "train_idx") or 0
    reason = None
    if n_train < cfg.min_labelled_train_rows:
        reason = f"{n_train} labelled train row(s) < min_labelled_train_rows={cfg.min_labelled_train_rows}"
    elif is_classification:
        reason = _train_class_problem(arr, rows)
    if reason is not None:
        entry[key] = {**record, "skipped": reason}
        logger.warning("%s/%s: not trained -- %s.", target_type, name, reason)
        return None, values, False
    drop, notes = _splits_to_drop(arr, rows, cfg, is_classification)
    rows = rows.without(drop)
    n_test = rows.n_labelled.get("test_idx")
    if n_test is not None and rows.n_total.get("test_idx") and n_test < cfg.min_labelled_test_rows:
        record["low_n"] = True
        notes.append(f"test has {n_test} labelled row(s) < min_labelled_test_rows={cfg.min_labelled_test_rows}: its test metrics are low_n")
    if notes:
        logger.warning("%s/%s: %s.", target_type, name, "; ".join(notes))
    entry[key] = {**record, "signature": rows.signature, "dropped_splits": sorted(rows.dropped), "notes": notes}
    return rows, working_target(values, mask, target_type, name), True
