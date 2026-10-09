"""Candidate-column sources for the binned-aggregate redundancy gate.

The gate scores ``CMI(candidate; y | group, agg)`` for every Tier-1 survivor. Building each survivor's out-of-fold column on the HOST (a per-fold gather over
all n rows) and then uploading it again to bin it was the dominant cost of the stage at 1M rows, although most survivors are rejected and their host
columns are never used. :class:`DeviceOofCandidates` rebuilds the out-of-fold columns on the device in small chunks (the same builder the Tier-1 device
gate uses), bins them there and hands resident codes to the CMI scorers; the host columns are then built only for the pairs the gate KEEPS.
:class:`HostCandidates` is the unchanged host route over an already-built frame.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Out-of-fold columns held on the device at once: 16 columns of 1M float64 rows is 128 MB.
_CHUNK_COLS = 16


class HostCandidates:
    """Candidates read from an already-built host frame. Construction wraps *feat_df*; *quantile_bin* is the host binner, *candidate_codes* picks the
    resident or host binning for one column."""

    def __init__(self, feat_df: pd.DataFrame, quantile_bin: Any, candidate_codes: Any) -> None:
        self._df = feat_df
        self._qbin = quantile_bin
        self._codes = candidate_codes
        self.names = list(feat_df.columns)

    def codes(self, name: str, nbins: int) -> Any:
        """Equi-frequency codes of the candidate *name* (resident under the strict-resident path)."""
        return self._codes(self._df[name].to_numpy(dtype=np.float64), nbins, self._qbin)


class DeviceOofCandidates:
    """Candidates rebuilt on the device from their recipes, chunk by chunk, never materialised on the host. Construction remembers the frame, the
    per-candidate recipe dicts and the OOF layout; nothing is built until a candidate is asked for."""

    def __init__(self, X: pd.DataFrame, raw: dict, names: Sequence[str], n_folds: int, random_state: int) -> None:
        self._X = X
        self._raw = raw
        self.names = list(names)
        self._n_folds = int(n_folds)
        self._random_state = int(random_state)
        self._pos = {nm: i for i, nm in enumerate(self.names)}
        self._chunk_start = -1
        self._chunk: Any = None

    @classmethod
    def create(cls, X: pd.DataFrame, raw: dict, names: Sequence[str], n_folds: int, random_state: int) -> Optional["DeviceOofCandidates"]:
        """The device source, or ``None`` when cupy is unavailable or a recipe lacks the fields the device builder needs (the caller then keeps the host route)."""
        try:
            import cupy  # noqa: F401
        except ImportError:
            return None
        for nm in names:
            r = raw.get(nm)
            if not r or any(k not in r for k in ("group_col", "agg_col", "edges", "global")):
                return None
        return cls(X, raw, names, n_folds, random_state)

    def _load_chunk(self, start: int) -> None:
        """Build the out-of-fold columns ``names[start:start + _CHUNK_COLS]`` on the device."""
        import cupy as cp

        from ._binned_numeric_agg_resident import binagg_fold_ids, build_binagg_oof_matrix_gpu

        specs = []
        for nm in self.names[start : start + _CHUNK_COLS]:
            r = self._raw[nm]
            specs.append({"name": nm, "group_col": r["group_col"], "agg_col": r["agg_col"], "stat": r.get("stat"), "edges": r["edges"], "global": r["global"]})
        fold_ids = binagg_fold_ids(len(self._X), self._n_folds, self._random_state)
        self._chunk = build_binagg_oof_matrix_gpu(cp, self._X, specs, fold_ids, self._n_folds)
        self._chunk_start = start

    def codes(self, name: str, nbins: int) -> Any:
        """Resident equi-frequency codes of the candidate *name*, binned on the device (an out-of-fold column is finite by construction)."""
        from ._mi_greedy_cmi_fe_binning import _quantile_bin_device

        i = self._pos[name]
        if self._chunk is None or not (self._chunk_start <= i < self._chunk_start + self._chunk.shape[1]):
            self._load_chunk((i // _CHUNK_COLS) * _CHUNK_COLS)
        return _quantile_bin_device(self._chunk[:, i - self._chunk_start], nbins)


def device_born_candidates(fit_fn: Any, X: pd.DataFrame, y: Any, gsel: Sequence[str], asel: Sequence[str], stats: Sequence[str], nbins_base: int, n_folds: int, random_state: int, precap_pairs: Any, cap_names: Any, redundancy_gate: bool, reject_sink: Any) -> tuple:
    """Device-born candidate selection for the binned-aggregate family: fit RECIPES-ONLY (no host out-of-fold loop), run the Tier-1 MI gate on the device from those
    recipes, then hand back either a lazy device source for the redundancy gate or, when that is unavailable or switched off, the host columns of the survivors.

    Returns ``(state, raw, feat_df, device_candidates)``: ``state`` is ``"empty"`` (nothing survived: the caller returns the input unchanged), ``"unavailable"``
    (the device gate could not run: the caller takes the exact host path) or ``"ok"``."""
    _rec_df, raw = fit_fn(
        X, y, group_num_cols=gsel, agg_num_cols=asel, stats=stats, nbins_base=nbins_base, n_folds=n_folds, random_state=random_state,
        pairs=precap_pairs, recipe_only=True,
    )
    cand_names = cap_names(list(raw.keys()), raw) if raw else []
    if not cand_names:
        return "empty", raw, None, None
    survivor_list = None
    try:
        from ._binned_numeric_agg_resident import local_mi_gate_binagg_resident

        survivor_list = local_mi_gate_binagg_resident(
            None, y, raw_X=X, recipes={nm: raw[nm] for nm in cand_names}, n_folds=n_folds, random_state=random_state, reject_sink=reject_sink,
            cand_cols=cand_names, n_rows=len(X),
        )
    except Exception as e:
        logger.debug("device candidate-survivor pass failed, falling back to the host path: %s", e)
    if survivor_list is None:
        return "unavailable", None, None, None
    keep = set(survivor_list)
    survivors = [c for c in cand_names if c in keep]  # capped order, filtered to survivors
    if not survivors:
        return "empty", raw, None, None
    # The redundancy gate scores device-rebuilt out-of-fold columns and rejects most survivors, so their host columns are never needed: the caller builds
    # them only for the pairs the gate keeps. Without the gate, or without a device source, build them for every survivor now.
    dev = DeviceOofCandidates.create(X, raw, survivors, n_folds, random_state) if redundancy_gate else None
    if dev is not None:
        return "ok", raw, None, dev
    # Host OOF for the survivor pairs only (bit-identical values). The pair-restricted fit may emit sibling stats of a survivor's pair, so keep exactly the
    # survivor columns, in survivor order.
    surv_pairs = {(raw[nm]["group_col"], raw[nm]["agg_col"]) for nm in survivors}
    feat_df, _ = fit_fn(
        X, y, group_num_cols=gsel, agg_num_cols=asel, stats=stats, nbins_base=nbins_base, n_folds=n_folds, random_state=random_state, pairs=surv_pairs,
    )
    have = set(feat_df.columns)
    feat_df = feat_df[[c for c in survivors if c in have]]
    if feat_df.shape[1] == 0:
        return "empty", raw, None, None
    return "ok", raw, feat_df, None
