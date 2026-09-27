"""Missing labels in the suite's targets: which rows each target has a label for.

A target with missing labels used to fail in two unrelated places: a classification column in the extractor, and a
regression column deep inside the first model fit of that target, after every earlier target had already trained.
:func:`raise_on_missing_labels` checks every target once, right after they are built, and names all of them in one
error.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_mask(values: Any) -> Optional[np.ndarray]:
    """Rows of ``values`` that carry a label, or ``None`` when every row does (the common case costs one pass).

    A missing label is a null, a NaN or ``None``. For a 2-D target (multilabel, multi-output regression) a row counts
    as labelled only when every column is.
    """
    if values is None:
        return None
    to_numpy = getattr(values, "to_numpy", None)
    arr = to_numpy() if callable(to_numpy) else np.asarray(values)
    if arr.dtype.kind in "iub":
        return None  # integer and boolean arrays cannot hold a missing value
    missing = pd.isna(arr)
    if missing.ndim > 1:
        missing = missing.reshape(missing.shape[0], -1).any(axis=1)
    if not missing.any():
        return None
    return np.asarray(~missing, dtype=bool)


def labelled_unique(values: Any) -> np.ndarray:
    """Distinct labels of a target, missing ones excluded. ``np.unique`` counts NaN as a class of its own, so a target
    with {0, NaN} looked like two classes."""
    arr = np.asarray(values)
    missing = pd.isna(arr)
    return np.unique(arr[~missing]) if missing.any() else np.unique(arr)


def missing_label_counts(target_by_type: dict) -> "dict[tuple[Any, str], tuple[int, int]]":
    """``{(target type, name): (rows without a label, rows)}`` for every target that has missing labels."""
    out: dict = {}
    for target_type, named in (target_by_type or {}).items():
        if not isinstance(named, dict):
            continue
        for name, values in named.items():
            mask = label_mask(values)
            if mask is not None:
                out[(target_type, str(name))] = (int((~mask).sum()), int(mask.size))
    return out


def raise_on_missing_labels(target_by_type: dict) -> None:
    """Fail before any training when a target has missing labels, naming every such target with its count."""
    counts = missing_label_counts(target_by_type)
    if not counts:
        return
    listing = "; ".join(f"{tt}/{name}: target contains {n:_} NaN/null label(s) of {total:_} ({n / total:.1%})" for (tt, name), (n, total) in counts.items())
    raise ValueError(f"{len(counts)} target(s) have missing labels -- {listing}. Drop or impute them upstream before training.")


def raise_on_infinite_labels(target_by_type: dict) -> None:
    """Refuse targets holding +-inf, under every policy: inf is not a missing label but almost always a broken upstream
    computation (a division by zero, log(0)), and dropping those rows would silently train on a biased subset."""
    found = []
    for target_type, named in (target_by_type or {}).items():
        if not isinstance(named, dict):
            continue
        for name, values in named.items():
            arr = np.asarray(values.to_numpy() if callable(getattr(values, "to_numpy", None)) else values)
            if arr.dtype.kind == "f":
                n_inf = int(np.isinf(arr).sum())
                if n_inf:
                    found.append(f"{target_type}/{name}: target contains {n_inf:_} infinity value(s)")
    if found:
        raise ValueError(
            f"{len(found)} target(s) hold infinite values -- {'; '.join(found)}. Fix the computation upstream; if they "
            "stand for a missing label, replace them with NaN (target_null_policy='drop_rows' then trains without those rows)."
        )


def apply_target_null_policy(target_by_type: dict, policy: str) -> "dict[tuple[Any, str], tuple[int, int]]":
    """Refuse targets with missing labels under ``"raise"``; under ``"drop_rows"`` log each one and return the counts."""
    raise_on_infinite_labels(target_by_type)
    if policy == "raise":
        raise_on_missing_labels(target_by_type)
        return {}
    counts = missing_label_counts(target_by_type)
    for (tt, name), (n, total) in counts.items():
        logger.info("%s/%s: %s of %s rows have no label (%.1f%%); it trains and is scored on the other rows only.", tt, name, f"{n:_}", f"{total:_}", 100 * n / total)
    return counts
