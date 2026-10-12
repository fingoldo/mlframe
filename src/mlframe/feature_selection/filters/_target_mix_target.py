"""Build the mix-offset target ``T = y - beta * u(X)`` on all rows from a fit on the training split only.

The training suite needs the target before any model is trained, but the mix base is an output of the MRMR fit. This
function runs that fit on the training rows, applies the held-out gate there (``TargetOffsetRegressor``), and, when the gate
accepts, returns the target on every row. ``u`` is replayed from the raw frame; ``beta`` is the slope fitted on the even
training rows and never refitted. No row outside the training split reaches the selection, the slope or the decision.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["add_back_mix_offset", "assemble_full_frame", "build_mix_offset_target", "mix_offset_record"]


def build_mix_offset_target(
    X: pd.DataFrame,
    y: Any,
    train_idx: Any,
    *,
    selector: Any,
    gate_regressor: Any,
    offset_alpha: float = 0.05,
) -> Optional[dict]:
    """``{"name_suffix", "values", "recipe", "beta", "base"}`` when the gate accepts a base on the training split, else None.

    ``values`` is ``y - beta * u(X)`` on every row of ``X`` (length ``len(y)``), so it slots into ``target_by_type`` beside the
    other targets. ``selector`` is an MRMR configured with ``fe_target_mix_enable=True``; ``gate_regressor`` is the model the
    decision is made for (the held-out comparison is done with it)."""
    from ._target_offset_regressor import TargetOffsetRegressor, _replay_base

    y = np.asarray(y).ravel()
    tr = np.asarray(train_idx, dtype=np.int64)
    Xtr = X.iloc[tr].reset_index(drop=True)
    # The decision and the slope only: the final regressor is not fitted here, so the target build costs two gate fits, not three.
    wrapper = TargetOffsetRegressor(selector=selector, regressor=gate_regressor, offset_alpha=offset_alpha).decide(Xtr, y[tr])
    if wrapper.offset_base_ is None:
        return None
    u_all = _replay_base(wrapper.offset_base_, X)
    beta = float(wrapper.offset_beta_)
    logger.info("mix-offset target: base %r accepted on the training split (beta %+.4f).", wrapper.offset_base_, beta)
    return {"base": wrapper.offset_base_, "beta": beta, "values": y - beta * u_all}


def add_back_mix_offset(predictions: Any, X: pd.DataFrame, record: dict) -> np.ndarray:
    """Predictions of a ``__mixoff`` model returned to the y scale: ``predictions + beta * u(X)``, with ``u`` replayed from the raw frame.

    ``record`` is the metadata entry ``{"beta", "base"}`` written by the discovery hook; the base recipe travels with the
    fitted object (``record["recipe"]``), because the metadata holds only its repr. A non-finite base raises (no silent zero)."""
    from ._target_offset_regressor import _replay_base

    u = _replay_base(record["recipe"], X)
    if not np.all(np.isfinite(u)):
        raise ValueError("the mix base replays to non-finite values on this frame; refusing to fill them silently.")
    out: np.ndarray = np.asarray(predictions).ravel() + float(record["beta"]) * u
    return out


def mix_offset_record(metadata: Any, target_type: Any, target_name: Any) -> Optional[dict]:
    """The feature-base mix record of a ``<name>__mixoff`` companion target (``metadata["mix_offset_targets"]``), or None for any other target."""
    if not isinstance(metadata, dict):
        return None
    record: Optional[dict] = metadata.get("mix_offset_targets", {}).get(str(target_type), {}).get(target_name)
    return record


def assemble_full_frame(train_df: Any, val_df: Any, test_df: Any, train_idx: Any, val_idx: Any, test_idx: Any, n_rows: int) -> Optional[pd.DataFrame]:
    """The full-row pandas frame rebuilt from the three split frames, ordered by the full-table row positions.

    Returns None unless the three index arrays are a partition of ``range(n_rows)`` (no row missing, none repeated) and each
    frame has as many rows as its index: a mix-offset target built on a misaligned frame would be silently wrong, so the
    caller then skips the hook. Any of the split frames may be None only when its index is empty."""
    parts, idx = [], []
    for frame, index in ((train_df, train_idx), (val_df, val_idx), (test_df, test_idx)):
        index = np.asarray(index, dtype=np.int64).ravel()
        if index.size == 0:
            continue
        if frame is None or len(frame) != index.size:
            return None
        parts.append(frame.reset_index(drop=True))
        idx.append(index)
    if not parts:
        return None
    all_idx = np.concatenate(idx)
    if all_idx.size != n_rows or not np.array_equal(np.sort(all_idx), np.arange(n_rows)):
        return None
    X = pd.concat(parts, ignore_index=True)
    order = np.argsort(all_idx, kind="stable")
    return X.iloc[order].reset_index(drop=True)
