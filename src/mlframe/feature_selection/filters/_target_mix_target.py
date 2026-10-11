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

__all__ = ["build_mix_offset_target"]


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

    y = np.asarray(y, dtype=np.float64).ravel()
    tr = np.asarray(train_idx, dtype=np.int64)
    Xtr = X.iloc[tr].reset_index(drop=True)
    wrapper = TargetOffsetRegressor(selector=selector, regressor=gate_regressor, offset_alpha=offset_alpha).fit(Xtr, y[tr])
    if wrapper.offset_base_ is None:
        return None
    u_all = _replay_base(wrapper.offset_base_, X)
    beta = float(wrapper.offset_beta_)
    logger.info("mix-offset target: base %r accepted on the training split (beta %+.4f).", wrapper.offset_base_, beta)
    return {"base": wrapper.offset_base_, "beta": beta, "values": y - beta * u_all}
