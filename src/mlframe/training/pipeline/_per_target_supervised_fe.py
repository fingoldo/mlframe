"""Supervised composite-FE steps fitted once per target, not once per suite.

Two suite-level steps learn from labels: categorical auto-group concat (groups picked by MI with y) and two-step target
encoding (a per-entity lookup of y). Fitted once, they saw the first target only, so in a multi-target suite every other
target's models consumed features built from another target's labels. With more than one target each step is fitted
per target instead:

* fit: every target gets its own scratch metadata; the columns its fit adds are renamed with a ``__target_<type>_<name>``
  suffix so targets never collide, and the per-target state plus the rename map land in
  ``metadata["supervised_composite_fe_per_target"]``;
* train: ``target_scoped_frames`` hides the other targets' columns from one target's models (frames and cat_features on
  the context), restoring the full frames afterwards;
* predict: ``replay_per_target_supervised_fe`` replays each target's state and applies its renames; each model then keeps
  the columns it was fitted on.

A single-target suite keeps the one-fit path and its column names unchanged.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PER_TARGET_KEY = "supervised_composite_fe_per_target"
_CROSS_TARGET_ARTIFACTS = ("feature_side_cache", "dataset_reuse_cache")
_ID_KEYED_CACHES = ("_pandas_view_cache", "_recurrent_numpy_cache")
_FRAME_ATTRS = (
    "train_df_polars", "val_df_polars", "test_df_polars", "train_df_pd", "val_df_pd", "test_df_pd", "filtered_train_df", "filtered_val_df",
)


def target_key(target_type: Any, target_name: Any) -> str:
    """The key a target's state is stored under."""
    return f"{target_type}/{target_name}"


def _suffix(target_type: Any, target_name: Any) -> str:
    """The column-name suffix that marks a column as learned from one target's labels."""
    from pyutilz.strings import slugify

    return f"__target_{slugify(str(target_type).lower())}_{slugify(str(target_name))}"


def supervised_steps_enabled(config: Any) -> bool:
    """Whether any label-supervised composite step is configured."""
    return bool(
        config is not None and (getattr(config, "categorical_group_concat_auto_enabled", False) or getattr(config, "two_step_target_encode_columns", None))
    )


def iter_targets(target_by_type: Any) -> List[Tuple[Any, Any, np.ndarray]]:
    """``(target_type, target_name, y)`` for every target that carries values."""
    out = []
    for t, d in (target_by_type or {}).items():
        items = d.items() if hasattr(d, "items") else [(t, d)]
        for name, y in items:
            if y is not None:
                out.append((t, name, y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)))
    return out


def _rename(df: Any, mapping: Dict[str, str]) -> Any:
    """``df`` with the columns in ``mapping`` renamed (pandas or polars); absent columns are ignored."""
    if df is None or not mapping:
        return df
    present = {k: v for k, v in mapping.items() if k in df.columns}
    if not present:
        return df
    return df.rename(present) if hasattr(df, "with_columns") else df.rename(columns=present)


def _drop(df: Any, cols: List[str]) -> Any:
    """``df`` without ``cols`` (pandas or polars); absent columns are ignored."""
    if df is None or not cols:
        return df
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    return df.drop(present) if hasattr(df, "with_columns") else df.drop(columns=present)


def apply_per_target_supervised_fe(
    train_df: Any, val_df: Any, test_df: Any, config: Any, targets: List[Tuple[Any, Any, np.ndarray]], train_idx: Optional[np.ndarray],
    group_ids: Optional[np.ndarray], timestamps: Any, val_idx: Optional[np.ndarray], test_idx: Optional[np.ndarray], metadata: dict,
    verbose: int = 0,
) -> tuple:
    """Fit the supervised composite steps once per target (see the module docstring); returns the three frames."""
    from ._categorical_composite_fe import apply_categorical_composite_fe
    from ._target_encoding_composite_fe import apply_target_encoding_composite_fe

    # Only the label-supervised parts run per target; the unsupervised powerset concat stays one shared fit (see caller).
    supervised_only = config.model_copy(update={"categorical_powerset_concat_enabled": False}) if hasattr(config, "model_copy") else config
    per_target: Dict[str, dict] = {}
    base = (train_df, val_df, test_df)
    added: Tuple[List[Any], List[Any], List[Any]] = ([], [], [])
    for t, name, y in targets:
        y_train = y
        if train_idx is not None and train_df is not None and len(y) != train_df.shape[0]:
            idx = np.asarray(train_idx)
            if len(idx) != train_df.shape[0] or int(idx.max()) >= len(y):
                logger.warning("Per-target composite FE: target %s/%s does not align with the train rows; skipped.", t, name)
                continue
            y_train = y[idx]
        # Every target is fitted on the ORIGINAL frames: fitted on top of an earlier target's output, the auto-grouping would
        # take that target's composite columns as categorical sources of its own.
        state: dict = {}
        tr, va, te = base
        if getattr(config, "categorical_group_concat_auto_enabled", False):
            tr, va, te = apply_categorical_composite_fe(tr, va, te, supervised_only, y_train, state, verbose=verbose)
        if getattr(config, "two_step_target_encode_columns", None):
            tr, va, te = apply_target_encoding_composite_fe(
                tr, va, te, supervised_only, group_ids, timestamps, y_train, train_idx, val_idx, test_idx, metadata=state, verbose=verbose,
            )
        base_cols = set(base[0].columns)
        renames = {c: c + _suffix(t, name) for c in tr.columns if c not in base_cols}
        for i, frame in enumerate((tr, va, te)):
            if frame is not None and renames:
                added[i].append(_rename(frame[list(renames)] if not hasattr(frame, "with_columns") else frame.select(list(renames)), renames))
        per_target[target_key(t, name)] = {"state": state, "renames": renames}
        if verbose:
            logger.info("Per-target composite FE for %s/%s: %d column(s).", t, name, len(renames))
    metadata[PER_TARGET_KEY] = per_target
    metadata["composite_fe_supervised_target"] = "per_target"
    return tuple(_attach(frame, parts) for frame, parts in zip(base, added))


def _attach(frame: Any, parts: List[Any]) -> Any:
    """``frame`` with the per-target column blocks appended (same row order; pandas index preserved)."""
    if frame is None or not parts:
        return frame
    if hasattr(frame, "with_columns"):
        import polars as pl

        return pl.concat([frame, *parts], how="horizontal")
    import pandas as pd

    return pd.concat([frame, *[p.set_axis(frame.index) for p in parts]], axis=1)


def replay_per_target_supervised_fe(df: Any, metadata: dict, group_ids: Optional[np.ndarray], verbose: int = 0) -> Any:
    """Predict-time replay of every target's fitted state, each followed by its renames."""
    per_target = (metadata or {}).get(PER_TARGET_KEY) or {}
    if df is None or not per_target:
        return df
    from ._categorical_composite_fe import replay_categorical_composite_fe
    from ._target_encoding_composite_fe import replay_target_encoding_composite_fe

    for entry in per_target.values():
        state = entry.get("state") or {}
        df = replay_categorical_composite_fe(df, state, verbose=verbose)
        df = replay_target_encoding_composite_fe(df, state, group_ids, verbose=verbose)
        df = _rename(df, entry.get("renames") or {})
    return df


def foreign_columns(metadata: dict, target_type: Any, target_name: Any) -> List[str]:
    """The per-target supervised columns that belong to OTHER targets than this one."""
    per_target = (metadata or {}).get(PER_TARGET_KEY) or {}
    own = target_key(target_type, target_name)
    return [c for key, entry in per_target.items() if key != own for c in (entry.get("renames") or {}).values()]


@contextmanager
def target_scoped_frames(ctx: Any, target_type: Any, target_name: Any) -> Iterator[None]:
    """For one target's training, hide the other targets' supervised columns from the context's frames and cat_features."""
    foreign = foreign_columns(getattr(ctx, "metadata", None) or {}, target_type, target_name)
    if not foreign:
        yield
        return
    saved = {a: getattr(ctx, a, None) for a in (*_FRAME_ATTRS, "cat_features")}
    # Caches that assume every target has the same features: prepared frames and binned datasets carried across targets
    # (ctx.artifacts), and conversions keyed by id() of a frame, which a trimmed frame freed after this target could
    # hand to an unrelated frame at the same address. Each target gets fresh ones; the suite's are restored after.
    artifacts = getattr(ctx, "artifacts", None)
    saved_artifacts = {k: artifacts.pop(k) for k in _CROSS_TARGET_ARTIFACTS if isinstance(artifacts, dict) and k in artifacts}
    saved_id_caches = {a: getattr(ctx, a) for a in _ID_KEYED_CACHES if isinstance(getattr(ctx, a, None), dict)}
    for a in saved_id_caches:
        setattr(ctx, a, {})
    for a in _FRAME_ATTRS:
        setattr(ctx, a, _drop(saved[a], foreign))
    if saved["cat_features"] is not None:
        drop = set(foreign)
        ctx.cat_features = [c for c in saved["cat_features"] if c not in drop]
    try:
        yield
    finally:
        for a, v in saved.items():
            if a in _FRAME_ATTRS and getattr(ctx, a, None) is None and v is not None:
                continue  # the body released this frame to free RAM; restoring it would pin it again
            setattr(ctx, a, v)
        if isinstance(artifacts, dict):
            for k in _CROSS_TARGET_ARTIFACTS:
                artifacts.pop(k, None)
            artifacts.update(saved_artifacts)
        for a, v in saved_id_caches.items():
            setattr(ctx, a, v)
