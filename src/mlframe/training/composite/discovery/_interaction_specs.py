"""Interaction bases enter the spec screen: ``a * b`` (or ``a + b``, ``a - b``) whose MI with y beats both parents is added
to the base candidates before screening, so a composite on it is selected, gated, trained and served like any other.

The step used to run at the end of ``fit`` and only stash the surfaced columns on ``interaction_bases_``: on a pure
interaction target (y = f0 * f1) it found ``f1__mul__f0`` and discovery still shipped a unary spec, because nothing read the
column. Now the qualifying columns are computed over every row of the discovery frame and screened as bases; outside
discovery a base with such a name is recomputed from its parents (``composite._synthetic_bases``), so no frame needs the
column. ``div`` candidates stay reported only: their divisor floor is fitted on train rows, so they are not pure functions of
the parents.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from .._synthetic_bases import SPECABLE_OPS, synthetic_column
from ._interaction_bases import discover_interaction_bases
from .screening import _extract_column_array, _is_numeric_column, _sample_indices

logger = logging.getLogger(__name__)


def _with_column(df: Any, name: str, values: np.ndarray) -> Any:
    """``df`` with one more float column (pandas or polars), leaving the caller's frame untouched."""
    if hasattr(df, "with_columns"):
        import polars as pl

        return df.with_columns(pl.Series(name, values))
    out = df.copy(deep=False)
    out[name] = values
    return out


def add_interaction_bases(self: Any, df: Any, usable_features: Sequence[str], base_candidates: Sequence[str],
                          train_idx: np.ndarray, y_train: np.ndarray) -> tuple[Any, list, list]:
    """``(df, usable_features, base_candidates)`` with the qualifying interaction bases added as base candidates.

    Scored on the train rows' screening sample (the same sampler as the main screen), from the pairs of the leading base
    candidates. Surfaced candidates of every op are recorded on ``interaction_bases_`` / ``interaction_base_records_``; the
    ``mul`` / ``add`` / ``sub`` ones become base candidates. The features are unchanged: a synthetic column is a base, not
    an input of the inner model.
    """
    cfg = self.config
    self.interaction_bases_, self.interaction_base_records_ = {}, []
    usable, bases = list(usable_features), list(base_candidates)
    if not bool(getattr(cfg, "interaction_base_discovery_enabled", True)):
        return df, usable, bases
    parents = [c for c in bases if c and c in getattr(df, "columns", ()) and _is_numeric_column(df, c)]
    if len(parents) < 2:
        return df, usable, bases
    try:
        sample = _sample_indices(np.asarray(train_idx).size, cfg.mi_sample_n, cfg.random_state,
                                 strategy=getattr(cfg, "mi_sample_strategy", "stratified_quantile"), y=y_train,
                                 n_strata=getattr(cfg, "mi_n_strata", 10))
        rows = np.asarray(train_idx)[sample]
        y_screen = np.asarray(y_train, dtype=np.float64)[sample]
        candidates = {c: _extract_column_array(df, c, rows=rows).astype(np.float64) for c in parents}
        synth, records = discover_interaction_bases(
            candidates, y_screen, top_k=int(getattr(cfg, "interaction_base_top_k", 4)),
            max_pairs=int(getattr(cfg, "interaction_base_max_pairs", 3)), nbins=int(cfg.mi_nbins),
            train_mask=np.ones(y_screen.shape[0], dtype=bool),  # every screen row is a train row
        )
    except Exception as exc:  # the step adds candidates; failing it must not cost the fit
        logger.warning("[CompositeTargetDiscovery] interaction-base discovery failed (%s); continuing without it.", exc)
        return df, usable, bases
    self.interaction_bases_, self.interaction_base_records_ = synth, records
    added = []
    for rec in records:
        name = rec["synth_name"]
        if rec.get("op") not in SPECABLE_OPS or name in bases or name in getattr(df, "columns", ()):
            continue
        values = synthetic_column(df, name)
        if values is None or not np.isfinite(values[np.asarray(train_idx)]).any():
            continue
        df = _with_column(df, name, values)
        bases.append(name)
        added.append(name)
    if added:
        logger.info("[CompositeTargetDiscovery] interaction bases screened as base candidates: %s", added)
    return df, usable, bases


__all__ = ["add_interaction_bases"]
