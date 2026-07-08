"""Layer 53 — incremental / streaming refit support for ``MRMR``.

The public ``MRMR.partial_fit(X_new, y_new)`` method is implemented as a
module-level function ``partial_fit`` taking ``self`` as the first argument
and bound onto the class at the bottom of ``mrmr.py``. Living in a sibling
module avoids pushing the 1700+ LOC parent module past the 1.8k line limit.

DESIGN
------
- First call: equivalent to ``self.fit(X_new, y_new)`` (initialise both the
  fitted state and the per-instance buffer).
- Subsequent calls:
    * append the new (X, y) to a per-instance buffer
    * apply rolling-window truncation when ``partial_fit_window`` is set
    * defer a full refit until at least ``partial_fit_min_recompute`` rows
      have been observed since the last refit
    * when a refit triggers, run ``self.fit`` over the buffered data with
      decay-weighted ``sample_weight`` reflecting recency

MEMORY: set ``partial_fit_window`` for long streams. With ``partial_fit_window=None`` the buffer grows UNBOUNDED -- every
call does ``pd.concat([buffer, new_batch])``, which copies the WHOLE accumulated buffer each time (O(total rows) per call,
O(total^2) over the stream). On an endless stream this eventually exhausts RAM (the 100GB-frame rule applies: an unbounded
in-memory buffer cannot fit a frame larger than host memory). A finite ``partial_fit_window`` caps the buffer to the most
recent ``window`` rows, bounding both peak memory and per-call copy cost.

DECAY SEMANTICS
---------------
``partial_fit_decay`` in ``[0, 1]``:

- ``0.0`` (default): no decay — every buffered row weighs equally. The
  effective behaviour is "extend train data, partial recompute".
- ``1.0``: full re-weight on the new batch — historical buffer rows are
  weighted to zero (clamped to a tiny floor for numerical safety), so the
  resulting fit reflects only the most-recent batch.
- intermediate values: per-batch geometric decay. Batch ``k`` rows back
  (``k=0`` is the most recent) receive weight ``(1 - decay) ** k``. The
  resulting per-row weight vector is forwarded to ``self.fit`` via the
  existing ``sample_weight`` path, which already handles the weighted
  resample inside ``_maybe_resample_for_sample_weight``.

API CONTRACT
------------
- ``partial_fit`` returns ``self`` (sklearn convention).
- Pickling: the buffer + counters are persisted via ``__setstate__``
  defaults (added to ``mrmr.py``) so clone()ing a fitted instance and
  resuming ``partial_fit`` on it Just Works.
- Existing ``fit()`` is untouched; users who never call ``partial_fit``
  see no behavioural drift.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Tiny floor on per-row weight to avoid zero-sum / all-zero degenerate
# sample_weight vectors when ``partial_fit_decay == 1.0`` after multiple
# batches; ``_maybe_resample_for_sample_weight`` rejects all-zero weights
# with ``"sample_weight sums to zero"``.
_WEIGHT_FLOOR = 1e-9


def _to_dataframe(X: Any) -> pd.DataFrame:
    """Normalise an X input to a pandas DataFrame so buffer concatenation
    is straightforward across mixed call sites (ndarray / DataFrame /
    polars). Keeps the original column names when available."""
    if isinstance(X, pd.DataFrame):
        return X
    try:
        import polars as _pl
        if isinstance(X, _pl.DataFrame):
            return X.to_pandas()
    except ImportError:
        pass
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _to_series(y: Any) -> pd.Series:
    """Normalise a y input to a pandas Series."""
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True)
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"MRMR.partial_fit: y as DataFrame must be single-column; got {y.shape[1]} cols")
        return y.iloc[:, 0].reset_index(drop=True)
    arr = np.asarray(y).ravel()
    return pd.Series(arr, name="y")


def _decay_weights(batch_sizes: list[int], decay: float) -> np.ndarray:
    """Per-row weight vector for the concatenated buffer.

    ``batch_sizes`` is in insertion order (oldest first); the newest batch
    is the last entry. Batch ``k`` from the end gets weight
    ``(1 - decay) ** k`` clamped to ``_WEIGHT_FLOOR``.
    """
    if not batch_sizes:
        return np.asarray([], dtype=np.float64)
    n_batches = len(batch_sizes)
    parts = []
    for idx, size in enumerate(batch_sizes):
        # idx=0 is the OLDEST batch, idx=n_batches-1 is the NEWEST.
        # Age (in batches, from the most recent) = n_batches - 1 - idx.
        age = n_batches - 1 - idx
        w = max((1.0 - float(decay)) ** age, _WEIGHT_FLOOR)
        parts.append(np.full(int(size), w, dtype=np.float64))
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.float64)


def _apply_rolling_window(
    X_buf: pd.DataFrame,
    y_buf: pd.Series,
    batch_sizes: list[int],
    window: int,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Drop the oldest rows when the buffer exceeds ``window``. Also rolls
    the ``batch_sizes`` registry so decay weights line up with what's
    actually in the buffer post-truncation. Batches at the front are
    dropped or partially trimmed; the newest batch is always preserved."""
    n = len(X_buf)
    if n <= window:
        return X_buf, y_buf, batch_sizes
    drop = n - window
    new_sizes: list[int] = []
    drop_remaining = drop
    for size in batch_sizes:
        if drop_remaining >= size:
            drop_remaining -= size
            # whole batch dropped
            continue
        # partial drop on this batch
        new_sizes.append(size - drop_remaining)
        drop_remaining = 0
    # any remaining batches (drop_remaining hit 0) keep their full sizes:
    # walk again to pick up everything past the consumed prefix.
    rebuilt: list[int] = []
    drop_remaining = drop
    consumed = False
    for size in batch_sizes:
        if not consumed:
            if drop_remaining >= size:
                drop_remaining -= size
                continue
            rebuilt.append(size - drop_remaining)
            drop_remaining = 0
            consumed = True
        else:
            rebuilt.append(size)
    X_trimmed = X_buf.iloc[drop:].reset_index(drop=True)
    y_trimmed = y_buf.iloc[drop:].reset_index(drop=True)
    return X_trimmed, y_trimmed, rebuilt


def partial_fit(
    self,
    X_new,
    y_new,
    *,
    sample_weight=None,
    classes=None,
    **fit_params,
):
    """sklearn-compatible incremental fit.

    Parameters
    ----------
    X_new, y_new
        Batch of new training rows. Same shape / dtype conventions as
        ``fit``.
    sample_weight
        Optional per-row weights for the new batch. Combined multiplicatively
        with the decay weights computed against the buffer history.
    classes
        Accepted for sklearn API symmetry; not consumed (MRMR derives
        target classes internally from y at fit time).
    fit_params
        Forwarded verbatim to ``self.fit`` on the recompute step.

    Returns
    -------
    self
        The fitted estimator. ``support_`` reflects the most recent
        recompute; on calls that buffer without recomputing, ``support_``
        carries the prior state unchanged.
    """
    X_df = _to_dataframe(X_new)
    y_ser = _to_series(y_new)
    if len(X_df) != len(y_ser):
        raise ValueError(f"MRMR.partial_fit: X_new has {len(X_df)} rows but y_new has " f"{len(y_ser)} rows; they must match.")
    if len(X_df) == 0:
        raise ValueError("MRMR.partial_fit: X_new / y_new must be non-empty.")

    decay = float(getattr(self, "partial_fit_decay", 0.0) or 0.0)
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"MRMR.partial_fit_decay must be in [0, 1]; got {decay!r}.")
    min_recompute = int(getattr(self, "partial_fit_min_recompute", 100) or 0)
    window = getattr(self, "partial_fit_window", None)
    if window is not None:
        window = int(window)
        if window <= 0:
            raise ValueError(f"MRMR.partial_fit_window must be positive when set; got {window!r}.")

    # First call -> initialise buffer + delegate to fit on the new batch.
    is_first = getattr(self, "_partial_fit_X_buffer_", None) is None or getattr(self, "_partial_fit_y_buffer_", None) is None
    if is_first:
        # Apply the rolling window even on the first call so callers that
        # ship a giant initial batch with a small window get the contracted
        # truncation (recency-first).
        batch_sizes = [len(X_df)]
        X_buf, y_buf = X_df.reset_index(drop=True), y_ser.reset_index(drop=True)
        if window is not None and len(X_buf) > window:
            X_buf, y_buf, batch_sizes = _apply_rolling_window(X_buf, y_buf, batch_sizes, window)
        self._partial_fit_X_buffer_ = X_buf
        self._partial_fit_y_buffer_ = y_buf
        self._partial_fit_batch_sizes_ = batch_sizes
        self._partial_fit_n_seen_ = len(X_buf)
        self._partial_fit_n_since_refit_ = 0
        # Run the full fit. Decay on a first call is moot (single batch),
        # so pass sample_weight straight through (caller-supplied or None).
        self.fit(X_buf, y_buf, sample_weight=sample_weight, **fit_params)
        return self

    # Subsequent call -> append, roll, possibly recompute.
    X_buf = self._partial_fit_X_buffer_
    y_buf = self._partial_fit_y_buffer_
    batch_sizes = list(getattr(self, "_partial_fit_batch_sizes_", [len(X_buf)]))

    # Reconcile column ordering: when the buffer has named columns and the
    # caller passes a positional ndarray, _to_dataframe will have assigned
    # generic ``f{i}`` names. Reuse the buffer columns when shapes match.
    if list(X_df.columns) != list(X_buf.columns) and X_df.shape[1] == X_buf.shape[1]:
        X_df = X_df.copy()
        X_df.columns = X_buf.columns

    X_buf = pd.concat([X_buf, X_df], axis=0, ignore_index=True)
    y_buf = pd.concat([y_buf, y_ser], axis=0, ignore_index=True)
    batch_sizes.append(len(X_df))

    if window is not None and len(X_buf) > window:
        X_buf, y_buf, batch_sizes = _apply_rolling_window(X_buf, y_buf, batch_sizes, window)

    self._partial_fit_X_buffer_ = X_buf
    self._partial_fit_y_buffer_ = y_buf
    self._partial_fit_batch_sizes_ = batch_sizes
    self._partial_fit_n_seen_ = int(getattr(self, "_partial_fit_n_seen_", 0)) + len(X_df)
    self._partial_fit_n_since_refit_ = int(getattr(self, "_partial_fit_n_since_refit_", 0)) + len(X_df)

    if self._partial_fit_n_since_refit_ < min_recompute:
        # Below the recompute threshold; buffer-only update keeps support_
        # from the prior fit (which is the documented sklearn partial_fit
        # contract: caller may issue many small batches before observing
        # an updated model).
        return self

    # Recompute. Build sample_weight from decay; multiply by caller-supplied
    # weights when given (forwarded to the most-recent batch only, since
    # historic weights are not retained across calls).
    if decay > 0.0:
        weights = _decay_weights(batch_sizes, decay)
    else:
        weights = np.ones(len(X_buf), dtype=np.float64)

    if sample_weight is not None:
        sw_new = np.asarray(sample_weight, dtype=np.float64).ravel()
        if sw_new.shape[0] != len(X_df) and is_first is False:
            # New batch may have been partially truncated by the rolling
            # window. Trim sw_new to the surviving suffix of the new batch.
            kept_new = batch_sizes[-1]
            if sw_new.shape[0] >= kept_new:
                sw_new = sw_new[-kept_new:]
            else:
                raise ValueError(f"MRMR.partial_fit: sample_weight length {sw_new.shape[0]} " f"does not align with the new batch length {len(X_df)}.")
        # Apply the caller weights to the trailing block (new batch survivors).
        n_new_kept = batch_sizes[-1]
        weights[-n_new_kept:] = weights[-n_new_kept:] * sw_new[-n_new_kept:]

    self._partial_fit_n_since_refit_ = 0
    self.fit(X_buf, y_buf, sample_weight=weights, **fit_params)
    return self
