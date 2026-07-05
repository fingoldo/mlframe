"""Purged + embargoed forward-walk time-series CV for temporal composite workflows.

The M6 time-aware discovery / OOF path needs an honest cross-validator for
autocorrelated, overlapping-label series. Plain ``KFold`` (or even sklearn's
``TimeSeriesSplit``) leaks signal across the train/test boundary: when a label
is built from a *window* of future rows (returns over the next ``h`` steps,
rolling aggregates, lead targets) the last train rows and the first test rows
share overlapping label windows, so the model "sees" the test answer through
adjacency. The Lopez de Prado purged-CV pattern fixes this with two gaps:

* **PURGE** -- drop the train rows whose label window overlaps the test window
  (``purge`` rows immediately *before* each test fold).
* **EMBARGO** -- skip a fraction of rows immediately *after* each test fold so
  serial-correlation leakage from test->next-train is removed.

This module is pure numpy index math: it never touches / copies the frame, so a
100+ GB carrier is fine. The splitter is sklearn-compatible (``.split(X)`` ->
``(train_idx, test_idx)``, ``.get_n_splits()``) and so can be passed straight
through as the discovery ``cv_splitter`` (see ``forward_stepwise``) or used to
carve the conformal/OOF holdout (held-out rows ONLY, never train rows).
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

__all__ = [
    "PurgedTimeSeriesSplit",
    "make_purged_cv",
    "purged_oof_holdout",
]


class PurgedTimeSeriesSplit:
    """Forward-walk CV with a purge gap and an embargo, sklearn-splitter shaped.

    Rows are assumed to be in chronological order (index ``i`` is earlier than
    ``i+1``). Each split is a contiguous expanding (or rolling) train block
    followed -- after a ``purge`` gap -- by a contiguous test block. Rows in the
    purge gap and in the embargo after the test fold are excluded from train, so
    no train sample shares a label window or short-range autocorrelation with the
    test fold it is scored against.

    Parameters
    ----------
    n_splits
        Number of forward-walk folds (>= 2).
    purge
        Number of rows immediately before each test fold dropped from the train
        block (the label-window overlap guard). ``0`` disables purging.
    embargo
        Embargo applied after each test fold, removed from any *later* train
        rows. ``float`` in ``[0, 1)`` -> fraction of ``n_samples``; ``int`` >= 1
        -> absolute row count. ``0`` disables the embargo.
    max_train_size
        If set, the train block is capped to the most-recent ``max_train_size``
        rows (rolling window instead of expanding). ``None`` = expanding.
    test_size
        Fixed test-fold size. ``None`` -> ``n_samples // (n_splits + 1)``
        (sklearn ``TimeSeriesSplit`` convention), folds walk forward by that
        amount.

    Notes
    -----
    No frame copy: ``split`` only reads ``len(X)`` (or ``n_samples``) and emits
    integer index arrays. Pass those to ``X.iloc[...]`` / ``X.filter(...)`` at
    the call site in the frame-native way.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        purge: int = 0,
        embargo: float | int = 0.0,
        max_train_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        if int(n_splits) < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if int(purge) < 0:
            raise ValueError(f"purge must be >= 0, got {purge}")
        if embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {embargo}")
        if max_train_size is not None and int(max_train_size) < 1:
            raise ValueError(f"max_train_size must be >= 1 or None, got {max_train_size}")
        if test_size is not None and int(test_size) < 1:
            raise ValueError(f"test_size must be >= 1 or None, got {test_size}")
        self.n_splits = int(n_splits)
        self.purge = int(purge)
        self.embargo = embargo
        self.max_train_size = None if max_train_size is None else int(max_train_size)
        self.test_size = None if test_size is None else int(test_size)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of folds (sklearn API). Ignores its arguments by design."""
        return self.n_splits

    def _embargo_rows(self, n_samples: int) -> int:
        """Resolve the embargo to an absolute, non-negative row count."""
        if self.embargo == 0:
            return 0
        if isinstance(self.embargo, (int, np.integer)) and self.embargo >= 1:
            return int(self.embargo)
        # fractional embargo in [0, 1)
        return int(np.floor(float(self.embargo) * n_samples))

    def _resolve_n_samples(self, X: Any, n_samples: int | None) -> int:
        if n_samples is not None:
            return int(n_samples)
        if X is None:
            raise ValueError("Provide X or n_samples to split().")
        try:
            return int(X.shape[0])
        except AttributeError:
            return int(len(X))

    def split(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
        *,
        n_samples: int | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` integer arrays, forward in time.

        ``X`` is read only for its row count; pass ``n_samples`` to split a known
        length without a frame. ``y`` / ``groups`` accepted for sklearn API
        compatibility and ignored.
        """
        n = self._resolve_n_samples(X, n_samples)
        n_folds = self.n_splits + 1
        if n < n_folds:
            raise ValueError(f"Too few samples ({n}) for n_splits={self.n_splits} (need >= {n_folds}).")

        test_size = self.test_size if self.test_size is not None else n // n_folds
        if test_size < 1:
            raise ValueError(f"Derived test_size < 1 for n={n}, n_splits={self.n_splits}.")

        emb = self._embargo_rows(n)
        indices = np.arange(n)
        test_size_eff = test_size

        # Forward-walk test starts: place n_splits test blocks ending at n.
        # Mirror TimeSeriesSplit when test_size is default: first test starts at
        # n - n_splits*test_size; otherwise evenly space within [test_size, n].
        if self.test_size is None:
            test_starts = range(test_size, n - test_size + 1, test_size)
            test_starts = list(test_starts)[-self.n_splits :]
        else:
            first = n - self.n_splits * test_size
            if first < 1:
                raise ValueError(f"test_size={test_size} too large for n={n}, n_splits={self.n_splits} " f"(would leave no train rows in the first fold).")
            test_starts = [first + i * test_size for i in range(self.n_splits)]

        for test_start in test_starts:
            test_stop = min(test_start + test_size_eff, n)
            test_idx = indices[test_start:test_stop]
            if test_idx.size == 0:
                continue

            # Train = everything strictly before the test fold, minus the purge
            # gap directly preceding it (label-window overlap guard), minus the
            # embargo shadow that trails the immediately-preceding test fold.
            # In a forward walk the train block is the prefix [:train_stop]; the
            # nearest earlier test fold ends at (test_start - test_size) and its
            # embargo extends emb rows beyond that. Both the purge and that
            # embargo sit at the tail of this prefix, so cutting the prefix at
            # (test_start - purge - embargo) removes both exactly.
            gap = self.purge + emb
            train_stop = max(test_start - gap, 0)
            train_idx = indices[:train_stop]

            if self.max_train_size is not None and train_idx.size > self.max_train_size:
                train_idx = train_idx[-self.max_train_size :]

            if train_idx.size == 0:
                continue
            yield train_idx, test_idx


def make_purged_cv(
    n_splits: int = 5,
    *,
    purge: int = 0,
    embargo: float | int = 0.0,
    max_train_size: int | None = None,
    test_size: int | None = None,
) -> PurgedTimeSeriesSplit:
    """Thin factory: build a :class:`PurgedTimeSeriesSplit` for the discovery
    ``cv_splitter`` slot (``forward_stepwise``/``_auto_chain``). Kept as a named
    helper so callers can wire a purged CV without importing the class directly
    and so default purge/embargo policy can evolve in one place.
    """
    return PurgedTimeSeriesSplit(
        n_splits=n_splits,
        purge=purge,
        embargo=embargo,
        max_train_size=max_train_size,
        test_size=test_size,
    )


def purged_oof_holdout(
    n_samples: int,
    *,
    holdout_frac: float = 0.2,
    purge: int = 0,
    embargo: float | int = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a single forward, leakage-guarded (train_idx, holdout_idx) split.

    The most-recent ``holdout_frac`` of rows become the honest held-out / OOF
    block (the model never sees them); the rows before it, minus the purge gap
    AND the embargo shadow, become train. Use for conformal calibration / the
    OOF holdout on a temporal series where a random split would leak via
    adjacency. Pure index math -- no frame copy.

    Returns ``(train_idx, holdout_idx)`` with **no overlap** and a guaranteed
    ``purge + embargo`` gap between the last train row and the first holdout row.
    """
    n = int(n_samples)
    if n < 2:
        raise ValueError(f"Need >= 2 samples, got {n}")
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")
    if int(purge) < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0")

    holdout_size = max(1, int(np.floor(holdout_frac * n)))
    holdout_start = n - holdout_size
    indices = np.arange(n)
    holdout_idx = indices[holdout_start:]

    if isinstance(embargo, (int, np.integer)) and embargo >= 1:
        emb = int(embargo)
    else:
        emb = int(np.floor(float(embargo) * n))

    train_stop = max(holdout_start - int(purge) - emb, 0)
    train_idx = indices[:train_stop]
    if train_idx.size == 0:
        raise ValueError(f"No train rows left: n={n}, holdout_frac={holdout_frac}, purge={purge}, embargo={embargo}.")
    return train_idx, holdout_idx
