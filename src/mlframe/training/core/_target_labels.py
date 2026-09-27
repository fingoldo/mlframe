"""Missing labels in the suite's targets: which rows each target has a label for.

A target with missing labels used to fail in two unrelated places: a classification column in the extractor, and a
regression column deep inside the first model fit of that target, after every earlier target had already trained.
:func:`raise_on_missing_labels` checks every target once, right after they are built, and names all of them in one
error.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


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
