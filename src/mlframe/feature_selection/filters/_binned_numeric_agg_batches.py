"""Memory-bounded batching of the binned-aggregate host path.

The host path builds the out-of-fold columns of every (group, aggregate) pair and scores them with the local MI gate. With many pairs the stored block is ``pairs x stats`` rows of ``n``
values, and the gate holds a few more copies of it (the candidate matrix, its rank codes, a device upload), which does not fit at a million rows. The pairs are therefore taken in batches
sized from the free memory of the device that does the work (host RAM, and the free VRAM when the gate runs through the STRICT GPU path); every pair is still scored.

The local gate is calibrated on the whole candidate family, so it is evaluated batch by batch without changing its result: the MI of a column does not depend on the other columns, and the
maximum of the family's null MI under one permutation is the maximum of the batches' maxima under that same permutation. :class:`StreamingLocalGate` accumulates the observed MIs and the
running null maxima, keeps only the columns that can still clear the floor, and applies the one shared selection rule at the end.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ._fe_usability_signal import _crit_np_dtype

logger = logging.getLogger(__name__)

# A batch may use this share of the free memory; the block, its DataFrame view, the gate's candidate matrix and the rank codes (or the device upload) coexist, so each stored row costs this many copies.
_BATCH_FREE_FRACTION = 0.5
_BLOCK_COPIES = 4
_DEVICE_COPIES = 3  # the double-precision upload of the candidate matrix, its edges / codes, and the kernel's working set


def _free_host_bytes() -> int:
    """Free host RAM in bytes (the streamed-MI probe, cached for a second)."""
    from ._orthogonal_univariate_fe._streamed_gpu_mi import _host_free_bytes

    return int(_host_free_bytes())


def _free_device_bytes(n_rows: int) -> Optional[int]:
    """Smallest free VRAM over the usable devices when the local gate would run through the STRICT GPU path for this ``n_rows``, else ``None`` (the device does not bound the batch)."""
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        from ._orthogonal_univariate_fe._streamed_gpu_mi import _devices

        if not fe_gpu_strict_enabled(n=int(n_rows), p=64):
            return None
        free = [int(f) for _d, f in _devices()]
        return min(free) if free else None
    except Exception as exc:
        logger.warning("device free-memory probe failed, the batch is bounded by host memory only: %s", exc)
        return None


def rows_per_batch(n_rows: int, itemsize: int, free_host: int, free_device: Optional[int]) -> int:
    """Stored columns one batch may hold: the host budget, and the device budget (double precision upload) when the gate runs on the device. At least one."""
    host_rows = int(_BATCH_FREE_FRACTION * free_host / (_BLOCK_COPIES * max(1, n_rows) * max(1, itemsize)))
    if free_device is None:
        return max(1, host_rows)
    device_rows = int(_BATCH_FREE_FRACTION * free_device / (_DEVICE_COPIES * max(1, n_rows) * 8))
    return max(1, min(host_rows, device_rows))


def pair_batches(ranked_pairs: Sequence[tuple], n_rows: int, n_stats: int) -> "list[list[tuple]]":
    """``ranked_pairs`` cut into consecutive batches whose stored columns fit the free memory (one batch when everything fits). Rank order is kept, so the concatenated batches are the original order."""
    if not ranked_pairs:
        return []
    rows = rows_per_batch(n_rows, int(np.dtype(_crit_np_dtype()).itemsize), _free_host_bytes(), _free_device_bytes(n_rows))
    per_batch = max(1, rows // max(1, int(n_stats)))
    return [list(ranked_pairs[i : i + per_batch]) for i in range(0, len(ranked_pairs), per_batch)]


def column_blocks(columns: Sequence[str], n_rows: int) -> "list[list[str]]":
    """``columns`` cut into consecutive blocks whose matrix (and the copies the local gate makes of it) fits the free memory; one block when everything fits."""
    if not columns:
        return []
    rows = rows_per_batch(n_rows, int(np.dtype(_crit_np_dtype()).itemsize), _free_host_bytes(), _free_device_bytes(n_rows))
    return [list(columns[i : i + rows]) for i in range(0, len(columns), rows)]


class StreamingLocalGate:
    """The Tier-1 local MI gate (``local_mi_gate``) over candidate columns that arrive in batches, with the same survivors in the same order as one call on the whole frame."""

    def __init__(
        self, y: Any, raw_X: Any, reject_sink: Optional[Callable[..., None]], nbins: int = 10, mad_mult: float = 3.5, floor0: Optional[float] = None, top_k: Optional[int] = None,
        keep_columns: bool = True,
    ) -> None:
        from ._unified_fe_gate import _coerce_y_classes, local_gate_permutations, raw_mi_noise_floor

        self._nbins = int(nbins)
        self._reject_sink = reject_sink
        self._top_k = top_k
        self._keep_columns = bool(keep_columns)
        self._y_bin = np.asarray(_coerce_y_classes(y))
        if floor0 is not None:
            self._floor0 = float(floor0)  # the caller's own floor (an explicit one, or the raw noise floor it already computed)
        else:
            self._floor0 = raw_mi_noise_floor(raw_X, y, nbins=nbins, mad_mult=mad_mult, columns=None) if raw_X is not None else 0.0
        self._perms = local_gate_permutations(len(self._y_bin))
        self._null_max = np.full(len(self._perms), np.nan, dtype=np.float64)  # Intentional float64: statistic vector (null max), exact permutation ties
        self._names: list[str] = []
        self._mi: list[float] = []
        self._kept_frames: list[pd.DataFrame] = []

    def score_batch(self, batch: pd.DataFrame) -> None:
        """Score one batch of candidate columns: observed MI per column, running family null maxima, and keep the columns that clear the raw floor (the final floor is never lower)."""
        from ._orthogonal_univariate_fe import _mi_classif_batch
        from ._orthogonal_univariate_fe._orth_mi_null_max import null_max_mi

        cols = [c for c in batch.columns if pd.api.types.is_numeric_dtype(batch[c])]
        if not cols:
            return
        arr = batch[cols].to_numpy(dtype=_crit_np_dtype())
        mi = np.asarray(_mi_classif_batch(arr, self._y_bin, nbins=self._nbins), dtype=np.float64)  # Intentional float64: statistic vector (MI), exact permutation ties
        if len(self._y_bin) > 0:
            self._null_max = np.fmax(self._null_max, null_max_mi(arr, self._y_bin, self._perms, self._nbins, _mi_classif_batch))
        self._names.extend(str(c) for c in cols)
        self._mi.extend(float(v) for v in mi)
        clear = [c for c, v in zip(cols, mi) if np.isfinite(v) and v >= self._floor0]
        if clear and self._keep_columns:
            self._kept_frames.append(batch[clear])

    def finalize(self) -> "tuple[list[str], pd.DataFrame]":
        """``(survivors, frame)``: the survivors of the gate on the whole family, and the frame of their columns in the order the batches delivered them."""
        from ._unified_fe_gate import select_above_floor

        floor = float(self._floor0)
        if len(self._names) >= 2 and len(self._y_bin) > 0:
            floor = max(floor, float(np.quantile(self._null_max, 0.95)))
        survivors = select_above_floor(self._names, np.asarray(self._mi, dtype=np.float64), floor, self._reject_sink, self._top_k)  # Intentional float64: statistic vector (MI)
        keep = set(survivors)
        frame = pd.concat(self._kept_frames, axis=1) if self._kept_frames else pd.DataFrame()
        return survivors, frame[[c for c in frame.columns if c in keep]]


def streamed_host_candidates(
    fit_fn: Any, X: pd.DataFrame, y: Any, gsel: Sequence[str], asel: Sequence[str], stats: Sequence[str], nbins_base: int, n_folds: int, random_state: int,
    batches: "list[list[tuple]]", cap_names: Callable[..., list], mi_gate: bool, reject_sink: Optional[Callable[..., None]],
) -> "tuple[pd.DataFrame, dict]":
    """``(feat_df, raw)`` of the host path built one batch of pairs at a time: each batch is fitted, capped and scored, and only the columns that can survive the local gate are kept.

    ``feat_df`` holds the gate's survivors (every column when ``mi_gate`` is off); ``raw`` the recipes of every fitted column."""
    gate = StreamingLocalGate(y, X, reject_sink) if mi_gate else None
    frames: list[pd.DataFrame] = []
    raw: dict = {}
    for pairs in batches:
        feat_b, raw_b = fit_fn(
            X, y, group_num_cols=gsel, agg_num_cols=asel, stats=stats, nbins_base=nbins_base, n_folds=n_folds, random_state=random_state, pairs=set(pairs),
        )
        if feat_b.shape[1] == 0:
            continue
        feat_b = feat_b[cap_names(list(feat_b.columns), raw_b)]
        raw.update(raw_b)
        if gate is not None:
            gate.score_batch(feat_b)
        else:
            frames.append(feat_b)
        del feat_b
    if gate is not None:
        return gate.finalize()[1], raw
    return (pd.concat(frames, axis=1) if frames else pd.DataFrame()), raw
