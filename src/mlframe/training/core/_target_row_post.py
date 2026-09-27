"""The labelled rows of each target for the steps that run after the target loop.

The cross-target ensemble, composite wrapping, the MoE gate and the recurrent models work per original target on
split frames and indices passed in as arguments. For a target with missing labels those must be its labelled rows, the
same ones its models trained and were scored on in the loop. :func:`split_args_by_target` narrows the arguments once per
such target, so every consumer sees the same frame objects (the wrap pass caches train predictions by frame identity and
the ensemble reads that cache).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ._target_row_decisions import TARGET_ROWS_METADATA_KEY, rows_for_target, working_target
from ._target_row_scope import _take, target_row_scope
from ._target_rows import TargetRows

# Argument name -> (the split its rows follow, the argument that is that split's index).
SPLIT_ARGS: dict[str, str] = {
    "filtered_train_df": "filtered_train_idx",
    "filtered_val_df": "filtered_val_idx",
    "test_df_pd": "test_idx",
    "train_df_pd": "train_idx",
    "val_df_pd": "val_idx",
    "train_df": "train_idx",
    "train_sequences": "train_idx",
    "val_sequences": "val_idx",
    "test_sequences": "test_idx",
}
_INDEX_FALLBACK = {"filtered_train_idx": "train_idx", "filtered_val_idx": "val_idx"}


def target_key(target_type: Any, name: Any) -> "tuple[str, str]":
    """The key a target is looked up by here: target types arrive both as TargetTypes members and as plain strings."""
    return str(target_type), str(name)


def rows_by_target(ctx: Any, target_by_type: Mapping, metadata: Mapping) -> "dict[tuple[str, str], TargetRows]":
    """``{(target type, name): TargetRows}`` for every trained target the loop narrowed; empty when none was."""
    recorded = (metadata or {}).get(TARGET_ROWS_METADATA_KEY) or {}
    if ctx is None or not recorded:
        return {}
    out: dict = {}
    cache: dict = {}
    for target_type, named in (target_by_type or {}).items():
        for name, values in (named or {}).items():
            record = recorded.get(f"{target_type}/{name}")
            if record is None or "skipped" in record:
                continue
            rows, _, _ = rows_for_target(ctx, target_type, name, values, {}, cache)
            if rows is not None:
                out[target_key(target_type, name)] = rows
    return out


def _pos(rows: TargetRows, index_arg: str) -> Optional[np.ndarray]:
    """Positions to keep inside the split ``index_arg`` names, with the unfiltered split standing in for a missing OD twin."""
    if index_arg in rows.pos:
        return rows.pos[index_arg]
    fallback = _INDEX_FALLBACK.get(index_arg)
    return rows.pos.get(fallback) if fallback else None


def narrow_split_args(rows: Optional[TargetRows], args: Mapping[str, Any]) -> dict:
    """``args`` with every split frame and index narrowed to ``rows``; a copy of ``args`` when ``rows`` is None.

    A split left with no labelled row gets an empty index and no frame, as the suite has with val_size / test_size 0.
    """
    out = dict(args)
    if rows is None:
        return out
    for index_arg in {*SPLIT_ARGS.values(), "filtered_train_idx", "filtered_val_idx"}:
        pos = _pos(rows, index_arg)
        if pos is not None and args.get(index_arg) is not None:
            out[index_arg] = np.asarray(args[index_arg])[pos]
    for frame_arg, index_arg in SPLIT_ARGS.items():
        pos = _pos(rows, index_arg)
        if pos is None:
            continue
        frame = args.get(frame_arg)
        if frame is not None:
            out[frame_arg] = _take(frame, pos) if pos.size else None
    return out


def split_args_by_target(ctx: Any, target_by_type: Mapping, metadata: Mapping, args: Mapping[str, Any]) -> "dict[tuple[str, str], dict]":
    """Narrowed ``args`` for every target with missing labels; a target not in the result uses ``args`` unchanged."""
    return {key: narrow_split_args(rows, args) for key, rows in rows_by_target(ctx, target_by_type, metadata).items()}


def train_recurrent_by_rows(suite_ctx: Any, train_fn: Any, target_by_type: Mapping, metadata: Mapping, /, **kwargs: Any) -> Any:
    """``train_fn`` (``train_recurrent_models``) over every target, each target with missing labels on its labelled rows.

    Fully labelled targets train in one call as before; each target with missing labels trains in its own call inside
    its row scope with its split arguments narrowed; a target the loop skipped (too few labelled rows) trains nowhere.
    Returns the models dict of the last call.
    """
    narrowed = rows_by_target(suite_ctx, target_by_type, metadata)
    recorded = (metadata or {}).get(TARGET_ROWS_METADATA_KEY) or {}
    if not recorded:
        return train_fn(target_by_type=target_by_type, **kwargs)
    skipped = {key for key, record in recorded.items() if "skipped" in record}
    plain = {
        tt: {n: v for n, v in named.items() if target_key(tt, n) not in narrowed and f"{tt}/{n}" not in skipped}
        for tt, named in (target_by_type or {}).items()
    }
    plain = {tt: named for tt, named in plain.items() if named}
    models = kwargs.pop("models")
    if plain:
        models = train_fn(target_by_type=plain, models=models, **kwargs)
    for tt, named in (target_by_type or {}).items():
        for name, values in named.items():
            rows = narrowed.get(target_key(tt, name))
            if rows is None:
                continue
            with target_row_scope(suite_ctx, rows):
                working = working_target(values, rows.mask, tt, name)  # the integer labels a classifier trained on in the loop
                models = train_fn(target_by_type={tt: {name: working}}, models=models, **narrow_split_args(rows, kwargs))
    return models
