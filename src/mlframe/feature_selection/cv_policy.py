"""One split policy for every feature selector that cross-validates or holds out rows internally.

The suite decides ONCE per target whether the rows are temporal, grouped or i.i.d. (``decide_cv_policy``) and every selector
then builds its folds / hold-out through this module, so RFECV, BorutaShap, ShapProxiedFS, ACE, ForwardSelect,
GreedyBackwardElimination, ZeroImportancePruning, CascadeSelect, MRMR's FE gates and the trainer's OOF pass cannot disagree
about what "the validation rows" are. An explicit ``cv`` the caller supplied always wins (see ``apply_cv_param``).

Decision table (first match wins):
    * ``hyperparams_config.has_time`` set explicitly -> that value (the same switch CatBoost's ``has_time`` reads);
    * no timestamps -> not temporal;
    * a caller-supplied splitter / ``cv_shuffle=True`` (``user_override``) -> not temporal, left to the caller;
    * ``split_config.cv_strategy`` in ``timeseries`` / ``purged`` -> temporal;
    * val and test both drawn entirely at random -> not temporal (the caller asked for an i.i.d. estimate);
    * otherwise temporal.
Not temporal + group labels present (and ``split_config.use_groups`` not False) -> grouped; else i.i.d.

The policy holds the train rows' timestamps / groups by reference (never copied) and drops them from pickles: a fitted selector
only needs the folds it already produced. A row-count mismatch at use time (a selector fit on a different row set than the
policy was built for) falls back to i.i.d. with a warning instead of reading row position as time.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold, TimeSeriesSplit

from mlframe.feature_selection._cv_splitters import TimestampOrderedSplit, chronological_order

logger = logging.getLogger(__name__)

__all__ = [
    "CVPolicy", "BoundGroupKFold", "TimestampOrderedSplit", "chronological_order", "cv_is_temporal", "decide_cv_policy", "get_cv_policy",
    "build_cv_splitter", "partition_folds", "holdout_indices", "apply_cv_param", "wire_selector_policy",
]

POLICY_ATTR = "_mlframe_cv_policy_"


def _has_sequential_part(size: Optional[float], shuffle: bool, sequential_fraction: Optional[float]) -> bool:
    """Whether a holdout of this size takes any rows from the time-sorted block (mirrors ``_calculate_split_sizes``)."""
    if size is None or size <= 0:
        return False
    if sequential_fraction is not None:
        return sequential_fraction > 0
    return not shuffle


def cv_is_temporal(timestamps: Any, split_config: Any, hyperparams_config: Any, user_override: Optional[str] = None) -> Tuple[bool, str]:
    """``(temporal, reason)`` per the module's decision table; ``reason`` is logged so the choice is never a mystery."""
    fields_set: set = set(getattr(hyperparams_config, "model_fields_set", None) or ())
    if "has_time" in fields_set:
        has_time = bool(getattr(hyperparams_config, "has_time", False))
        return has_time, f"hyperparams_config.has_time={has_time} was set explicitly"
    if timestamps is None:
        return False, "the suite has no timestamps"
    if user_override:
        return False, user_override
    if split_config is not None and str(getattr(split_config, "cv_strategy", "random")) in ("timeseries", "purged"):
        return True, f"split_config.cv_strategy={split_config.cv_strategy!r}"
    if split_config is not None and not (
        _has_sequential_part(split_config.test_size, split_config.shuffle_test, split_config.test_sequential_fraction)
        or _has_sequential_part(split_config.val_size, split_config.shuffle_val, split_config.val_sequential_fraction)
    ):
        return False, "timestamps are present but the val and test splits are both fully shuffled"
    return True, "timestamps are present and the val/test split takes the newest rows"


class CVPolicy:
    """The suite's split decision for one target: ``kind`` is ``"temporal"``, ``"grouped"`` or ``"iid"``.

    ``timestamps`` / ``groups`` are aligned to the rows the selectors are fit on (the suite's train rows).
    """

    def __init__(self, kind: str, reason: str, timestamps: Any = None, groups: Any = None) -> None:
        if kind not in ("temporal", "grouped", "iid"):
            raise ValueError(f"CVPolicy.kind must be temporal / grouped / iid; got {kind!r}.")
        self.kind = kind
        self.reason = reason
        self.timestamps = timestamps
        self.groups = groups

    @property
    def temporal(self) -> bool:
        """True when the policy splits by time."""
        return self.kind == "temporal"

    def subset(self, idx: np.ndarray) -> "CVPolicy":
        """The same policy restricted to the rows at positions ``idx`` (for a selector that fits on a sub-sample of its input)."""
        held_len = self._held_len()
        if self.kind == "iid" or held_len is None or (len(idx) and int(np.max(idx)) >= held_len):
            return self if self.kind == "iid" else CVPolicy("iid", f"{self.reason} (row arrays unavailable for the sub-sample)")
        return CVPolicy(self.kind, self.reason, _take(self.timestamps, idx), _take(self.groups, idx))

    def _held_len(self) -> Optional[int]:
        """Length of the timestamps or groups the policy holds, or None when it holds none."""
        held = self.timestamps if self.kind == "temporal" else self.groups
        return None if held is None else len(held)

    def __deepcopy__(self, memo: dict) -> "CVPolicy":
        """Share the row-aligned arrays: a per-clone copy of a train-length array is RAM waste since nothing mutates them."""
        return CVPolicy(self.kind, self.reason, self.timestamps, self.groups)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["timestamps"] = None
        state["groups"] = None
        state.pop("_warned", None)
        return state

    def __repr__(self) -> str:
        return f"CVPolicy(kind={self.kind!r}, reason={self.reason!r})"

    def matches(self, n: int) -> bool:
        """Whether the held arrays describe ``n`` rows; warns once per mismatch so a stale policy is visible."""
        held = self.timestamps if self.kind == "temporal" else self.groups
        if held is not None and len(held) == n:
            return True
        key = (None if held is None else len(held), n)
        warned = self.__dict__.setdefault("_warned", set())
        if key not in warned:
            warned.add(key)
            logger.warning(
                "CVPolicy(%s): holds %s rows but the selector was fit on %d (row arrays are not pickled); using an i.i.d. split.",
                self.kind, "no" if held is None else len(held), n,
            )
        return False


def _take(values: Any, idx: Optional[np.ndarray]) -> Any:
    """Rows ``idx`` of ``values``, keeping a pandas Series (a tz-aware one turns into objects through ``np.asarray``)."""
    if values is None or idx is None:
        return values
    if hasattr(values, "iloc"):
        return values.iloc[np.asarray(idx)]
    return np.asarray(values)[np.asarray(idx)]


def decide_cv_policy(
    *,
    timestamps: Any,
    train_idx: Optional[np.ndarray],
    groups: Any = None,
    split_config: Any = None,
    hyperparams_config: Any = None,
    user_override: Optional[str] = None,
) -> CVPolicy:
    """Build the suite's ``CVPolicy`` from the full-length ``timestamps`` / ``groups`` and the train row positions."""
    temporal, reason = cv_is_temporal(timestamps, split_config, hyperparams_config, user_override)
    if temporal and timestamps is not None:
        return CVPolicy("temporal", reason, _take(timestamps, train_idx), _take(groups, train_idx) if groups is not None else None)
    if groups is not None and bool(getattr(split_config, "use_groups", True)):
        g = _take(groups, train_idx)
        if len(np.unique(np.asarray(g))) >= 2:
            return CVPolicy("grouped", f"group labels are present ({reason})", None, g)
    return CVPolicy("iid", reason)


def get_cv_policy(selector: Any) -> Optional[CVPolicy]:
    """The policy stamped on ``selector`` by the suite when it is temporal or grouped, else None (keep the selector's own default)."""
    policy = getattr(selector, POLICY_ATTR, None)
    if isinstance(policy, CVPolicy) and policy.kind != "iid":
        return policy
    return None


class BoundGroupKFold:
    """``GroupKFold`` with the row groups bound at construction, for callers that never pass ``groups`` to ``split``."""

    deterministic_folds = True

    def __init__(self, n_splits: int, groups: Any) -> None:
        self.n_splits = int(n_splits)
        self.groups = groups

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Number of folds."""
        return self.n_splits

    def split(self, X: Any = None, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield GroupKFold folds over the groups bound at construction or passed here."""
        g = groups if groups is not None else self.groups
        if g is None:
            raise ValueError("BoundGroupKFold: no groups were bound and none were passed to split (they are not pickled).")
        yield from GroupKFold(n_splits=self.n_splits).split(np.empty(len(g)), groups=np.asarray(g))

    def __deepcopy__(self, memo: dict) -> "BoundGroupKFold":
        return BoundGroupKFold(self.n_splits, self.groups)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["groups"] = None
        return state

    def __repr__(self) -> str:
        return f"BoundGroupKFold(n_splits={self.n_splits})"


def build_cv_splitter(
    policy: Optional[CVPolicy], n_splits: int, *, classification: bool = False, random_state: Optional[int] = 0, gap: int = 0,
) -> Any:
    """Scoring-CV splitter for ``policy``: forward-chaining by timestamp, group-isolating, or (i.i.d.) shuffled stratified / plain KFold."""
    if policy is not None and policy.kind == "temporal" and policy.timestamps is not None:
        return TimestampOrderedSplit(n_splits=n_splits, timestamps=policy.timestamps, gap=gap, groups=policy.groups)
    if policy is not None and policy.kind == "grouped" and policy.groups is not None:
        n_groups = len(np.unique(np.asarray(policy.groups)))
        if n_groups >= 2:
            return BoundGroupKFold(min(n_splits, n_groups), policy.groups)
    if classification:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _chronological_positions(policy: CVPolicy, n: int) -> Optional[np.ndarray]:
    """Positions that put the policy's rows in time order, or None when the policy does not match ``n`` rows."""
    return chronological_order(policy.timestamps) if policy.matches(n) else None


def partition_folds(
    policy: Optional[CVPolicy], n: int, n_splits: int, *, y: Any = None, classification: bool = False,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    """``(train, test)`` folds whose test sets PARTITION all ``n`` rows, for consumers that need an out-of-fold value for every row.

    Temporal: contiguous blocks of the chronological order (forward chaining leaves the first block never held out, which such
    consumers cannot use). Grouped: ``GroupKFold``. Returns None for an i.i.d. / unusable policy so the caller keeps its own splitter.
    """
    if policy is None or policy.kind == "iid" or n < 2 * n_splits:
        return None
    if policy.kind == "temporal":
        order = _chronological_positions(policy, n)
        if order is None:
            return None
        blocks = np.array_split(order, n_splits)
        return [(np.sort(np.concatenate([b for j, b in enumerate(blocks) if j != i])), np.sort(blocks[i])) for i in range(n_splits)]
    if not policy.matches(n) or len(np.unique(np.asarray(policy.groups))) < n_splits:
        return None
    return [(tr, te) for tr, te in GroupKFold(n_splits=n_splits).split(np.empty(n), groups=np.asarray(policy.groups))]


def holdout_indices(policy: Optional[CVPolicy], n: int, test_size: float, *, random_state: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(search_idx, holdout_idx)`` positions, both ascending: the newest ``test_size`` rows (temporal) or whole groups (grouped).

    Returns None for an i.i.d. / unusable policy so the caller keeps its own (stratified, shuffled) split.
    """
    if policy is None or policy.kind == "iid":
        return None
    n_hold = round(n * test_size) if 0 < test_size < 1 else int(test_size)
    if n_hold < 1 or n - n_hold < 1:
        return None
    if policy.kind == "temporal":
        order = _chronological_positions(policy, n)
        if order is None:
            return None
        return np.sort(order[: n - n_hold]), np.sort(order[n - n_hold :])
    if not policy.matches(n) or len(np.unique(np.asarray(policy.groups))) < 2:
        return None
    search, hold = next(GroupShuffleSplit(n_splits=1, test_size=n_hold / n, random_state=random_state).split(np.empty(n), groups=np.asarray(policy.groups)))
    return np.sort(search), np.sort(hold)


def apply_cv_param(selector: Any, policy: Optional[CVPolicy], *, user_cv: Any = None, n_splits: int = 5, classification: bool = False) -> bool:
    """Replace ``selector.cv`` with the policy's splitter unless the caller supplied one (``user_cv``) or ``cv`` is a non-default splitter.

    Only an unset ``cv`` (None) or a plain fold count (int) counts as "the suite may choose"; a ``TimeSeriesSplit`` is upgraded to the
    timestamp-ordered one because on unsorted rows it chains in row order. Returns whether ``selector.cv`` changed.
    """
    if policy is None or policy.kind == "iid" or user_cv is not None or not hasattr(selector, "cv"):
        return False
    cv = selector.cv
    if cv is None:
        k = n_splits
    elif isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        k = int(cv)
    elif type(cv) is TimeSeriesSplit and policy.kind == "temporal":
        k = int(cv.n_splits)
    else:
        return False
    selector.cv = build_cv_splitter(policy, k, classification=classification)
    return True


def wire_selector_policy(selector: Any, policy: Optional[CVPolicy], *, classification: bool = False) -> bool:
    """Stamp ``policy`` on ``selector`` (read by holdout-based selectors via ``get_cv_policy``) and, for a ``cv``-parameter selector, swap in the
    policy's splitter through ``apply_cv_param`` (a caller-supplied non-default splitter is kept). No-op for a missing / i.i.d. policy."""
    if policy is None or policy.kind == "iid":
        return False
    setattr(selector, POLICY_ATTR, policy)
    apply_cv_param(selector, policy, classification=classification)
    return True
