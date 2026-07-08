"""Shard / slice construction for slice-stable early stopping.

``build_slice_eval_sets`` produces a list of ``SliceEvalSet`` objects that the caller registers
with the booster as additional eval-sets (in addition to the original full-val ``valid_0``).
The booster computes its native eval-metric on each shard at every iteration; the
``UniversalCallback`` aggregates the resulting per-shard scores into a single robust ES signal.

Supported shard sources:

- ``random``    classic K-fold partition of val rows; auto-stratified for classification
                via sklearn ``StratifiedKFold``. For ranker / group-aware paths we use
                ``GroupKFold`` instead so query boundaries are preserved (see ``group_ids``).
- ``temporal``  K consecutive time windows of val (``time_column``-sorted), analogous to
                ``TimeSeriesSplit``. Required when the upstream split is time-aware.
- ``fairness``  pre-computed ``indexed_subgroups`` from the fairness pipeline are reused
                directly as shards. Worst-group-robust ES; subgroups may overlap.
- ``both``      union of ``random`` and ``fairness`` shards. The caller chooses which subset
                drives the stop decision (typically random) and which is diagnostic-only.

Parallel-aligned per-row attributes (``sample_weight``, ``base_margin``, ``group_ids``) are
sharded via the same row index, so each ``SliceEvalSet`` is self-contained and ready to pass
straight to ``XGBClassifier.fit(eval_set=..., sample_weight_eval_set=..., base_margin_eval_set=...)``
or the LGB / CB equivalents.

Polars Enum / Categorical dtypes are preserved because we slice via ``.gather`` / ``.filter``
on polars frames and ``.iloc`` on pandas frames (never via ``.values`` which would coerce to
the underlying numpy representation and drop the categorical domain).
"""
from __future__ import annotations

import logging
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

try:
    import polars as _pl
except Exception:  # polars is a mlframe runtime dep, but keep the module importable just in case
    _pl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Collapse any run of non-word chars to a single underscore when sanitising a subgroup name.
_SLICE_NAME_SANITISE_RE = re.compile(r"[^0-9A-Za-z_]+")


_SliceSource = Literal["random", "temporal", "fairness", "both"]

# dataclass slots= keyword is 3.10+; keep the slots optimisation where available, fall back to a plain dataclass on 3.9.
_DC_SLOTS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DC_SLOTS)
class SliceEvalSet:
    """A single per-shard eval-set ready to register with a booster.

    Attributes
    ----------
    name
        Human-readable identifier (e.g. ``"valid_shard_r0"`` or ``"valid_shard_f_minority"``).
        Used for log messages only; the callback identifies shards positionally via the
        ``slice_dataset_indices`` list because XGB / CB sklearn APIs don't expose names.
    X, y
        Sliced feature frame and target. Same row order; same length.
    sample_weight, base_margin, group_ids
        Optional parallel row attributes, sliced with the same index. ``None`` when the
        caller didn't supply them.
    row_indices
        The integer positions of the shard's rows in the original (full-val) frame, kept
        around for downstream diagnostics (Pareto plot per-shard score table).
    """
    name: str
    X: Any
    y: np.ndarray
    sample_weight: np.ndarray | None = None
    base_margin: np.ndarray | None = None
    group_ids: np.ndarray | None = None
    row_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def _sanitize_name_component(s: str) -> str:
    """Make a subgroup name safe to splice into ``valid_shard_f_<name>``."""
    return _SLICE_NAME_SANITISE_RE.sub("_", str(s)).strip("_") or "_"


def _is_polars_df(obj: Any) -> bool:
    """Check whether ``obj`` is a polars DataFrame, tolerating environments where polars isn't installed (``_pl is None``)."""
    return _pl is not None and isinstance(obj, _pl.DataFrame)


def _row_select(frame: Any, row_idx: np.ndarray) -> Any:
    """Row-select preserving dtype (Polars Enum / pandas Categorical / numpy)."""
    try:
        import pandas as pd  # local import keeps the module importable in pandas-less envs
    except Exception:  # pragma: no cover - mlframe always has pandas in its deps
        pd = None

    if pd is not None and isinstance(frame, pd.DataFrame):
        return frame.iloc[row_idx].reset_index(drop=True)
    if _is_polars_df(frame):
        # Match the row-selection idiom used elsewhere in mlframe (composite_ensemble.py:616-620):
        # boolean mask via pl.Series keeps Enum / Categorical dtypes intact.
        assert _pl is not None  # nosec B101 - internal invariant check in src/mlframe/training/slicing, not reachable with untrusted input
        mask = np.zeros(frame.height, dtype=bool)
        mask[row_idx] = True
        return frame.filter(_pl.Series(mask))
    arr = np.asarray(frame)
    return arr[row_idx]


def _aligned_select(arr: Any, row_idx: np.ndarray) -> np.ndarray | None:
    """Row-select an auxiliary array (sample weights / base margin / group ids) aligned to ``row_idx``, passing through ``None`` unchanged for optional inputs."""
    if arr is None:
        return None
    return np.asarray(np.asarray(arr)[row_idx])


def _is_classification_target(y: np.ndarray) -> bool:
    """Heuristic mirroring ``_shap_proxy_compose.py:55-56``: integer dtype + small cardinality."""
    if y.dtype.kind in ("i", "u", "b"):
        return int(np.unique(y).size) <= max(20, int(0.05 * y.size))
    return False


def _random_shards(
    n: int, k: int, y: np.ndarray, *, random_state: int,
) -> list[np.ndarray]:
    """K-fold (Stratified for classification) row-index partition."""
    from sklearn.model_selection import KFold, StratifiedKFold

    classification = _is_classification_target(y)
    if classification:
        try:
            sk = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
            return [test_idx for _, test_idx in sk.split(np.arange(n), y)]
        except ValueError as exc:
            # Triggered when a class has fewer than k members. Fall through to plain KFold
            # with a warn; the caller will see single-class shards via downstream guards.
            logger.warning("StratifiedKFold(%d) failed (%s); falling back to plain KFold.", k, exc)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    return [test_idx for _, test_idx in kf.split(np.arange(n))]


def _temporal_shards(
    n: int, k: int, *, time_values: np.ndarray | None,
) -> list[np.ndarray]:
    """K consecutive windows along time_values (or row order when not provided)."""
    if time_values is None:
        order = np.arange(n)
    else:
        order = np.argsort(np.asarray(time_values), kind="stable")
    edges = np.linspace(0, n, k + 1, dtype=np.int64)
    return [order[edges[i] : edges[i + 1]] for i in range(k)]


def _fairness_shards(
    indexed_subgroups: dict[str, np.ndarray] | None,
) -> list[tuple[str, np.ndarray]]:
    """Subgroup-name -> row indices (may overlap; not necessarily a partition)."""
    if not indexed_subgroups:
        return []
    out: list[tuple[str, np.ndarray]] = []
    for sg_name, idx in indexed_subgroups.items():
        arr = np.asarray(idx, dtype=np.int64)
        if arr.size:
            out.append((str(sg_name), arr))
    return out


def build_slice_eval_sets(
    val_X: Any,
    val_y: np.ndarray | Any,
    *,
    source: _SliceSource = "random",
    k: int = 5,
    min_rows_per_shard: int = 100,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
    base_margin: np.ndarray | None = None,
    group_ids: np.ndarray | None = None,
    time_values: np.ndarray | None = None,
    indexed_subgroups: dict[str, np.ndarray] | None = None,
) -> list[SliceEvalSet]:
    """Return per-shard eval-sets for slice-stable early stopping.

    Returns an empty list (with a single ``logger.warning``) if the request is unfeasible:
    too few val rows per shard, or fairness-only request with no fairness subgroups.

    Parallel per-row attributes (``sample_weight`` / ``base_margin`` / ``group_ids``) are
    sliced with the same index and embedded into each returned ``SliceEvalSet``. The first
    full-val eval-set is NOT included here; the caller prepends it.
    """
    y_arr = np.asarray(val_y)
    n = y_arr.shape[0]
    if n == 0:
        logger.warning("build_slice_eval_sets: empty val target")
        return []

    if k < 2:
        logger.warning("build_slice_eval_sets: k=%d < 2; slice-stable ES requires K>=2", k)
        return []

    if source in ("random", "temporal", "both") and (n // k) < min_rows_per_shard:
        logger.warning(
            "build_slice_eval_sets: val n=%d / k=%d gives %d rows/shard < min_rows_per_shard=%d; "
            "slice-stable ES disabled for this run",
            n, k, n // k, min_rows_per_shard,
        )
        return []

    if group_ids is not None and source == "random":
        # Ranker / qid-aware path: partition by query group, not random rows. Falling back
        # silently is unsafe (NDCG on partial queries is meaningless); switch path + WARN.
        from sklearn.model_selection import GroupKFold
        logger.warning(
            "build_slice_eval_sets: source='random' with group_ids supplied; switching to " "GroupKFold so query boundaries are preserved (ranker-safe shards)",
        )
        gkf = GroupKFold(n_splits=k)
        row_idx_lists = [test_idx for _, test_idx in gkf.split(np.arange(n), groups=np.asarray(group_ids))]
        return _materialize(val_X, y_arr, row_idx_lists, prefix="valid_shard_r", sample_weight=sample_weight, base_margin=base_margin, group_ids=group_ids)

    out: list[SliceEvalSet] = []

    if source in ("random", "both"):
        row_idx_lists = _random_shards(n, k, y_arr, random_state=random_state)
        out.extend(_materialize(val_X, y_arr, row_idx_lists, prefix="valid_shard_r", sample_weight=sample_weight, base_margin=base_margin, group_ids=group_ids))

    if source == "temporal":
        row_idx_lists = _temporal_shards(n, k, time_values=time_values)
        out.extend(_materialize(val_X, y_arr, row_idx_lists, prefix="valid_shard_t", sample_weight=sample_weight, base_margin=base_margin, group_ids=group_ids))

    if source in ("fairness", "both"):
        sg_pairs = _fairness_shards(indexed_subgroups)
        if not sg_pairs and source == "fairness":
            logger.warning(
                "build_slice_eval_sets: source='fairness' but no indexed_subgroups supplied; " "returning empty shard list",
            )
            return []
        for sg_name, row_idx in sg_pairs:
            row_idx = row_idx[(row_idx >= 0) & (row_idx < n)]
            if row_idx.size < min_rows_per_shard:
                logger.info(
                    "build_slice_eval_sets: fairness subgroup %r has %d rows < min_rows_per_shard=%d; skipping",
                    sg_name, row_idx.size, min_rows_per_shard,
                )
                continue
            name = f"valid_shard_f_{_sanitize_name_component(sg_name)}"
            out.append(SliceEvalSet(
                name=name,
                X=_row_select(val_X, row_idx),
                y=y_arr[row_idx],
                sample_weight=_aligned_select(sample_weight, row_idx),
                base_margin=_aligned_select(base_margin, row_idx),
                group_ids=_aligned_select(group_ids, row_idx),
                row_indices=row_idx,
            ))

    return out


def _materialize(
    val_X: Any,
    y_arr: np.ndarray,
    row_idx_lists: Sequence[np.ndarray],
    *,
    prefix: str,
    sample_weight: np.ndarray | None,
    base_margin: np.ndarray | None,
    group_ids: np.ndarray | None,
) -> list[SliceEvalSet]:
    """Turn a list of per-shard row-index arrays into concrete ``SliceEvalSet`` objects, row-selecting X/y and the aligned auxiliary arrays for each shard; empty shards are skipped rather than emitted as degenerate eval sets."""
    out: list[SliceEvalSet] = []
    for i, row_idx in enumerate(row_idx_lists):
        row_idx = np.asarray(row_idx, dtype=np.int64)
        if row_idx.size == 0:
            continue
        out.append(SliceEvalSet(
            name=f"{prefix}{i}",
            X=_row_select(val_X, row_idx),
            y=y_arr[row_idx],
            sample_weight=_aligned_select(sample_weight, row_idx),
            base_margin=_aligned_select(base_margin, row_idx),
            group_ids=_aligned_select(group_ids, row_idx),
            row_indices=row_idx,
        ))
    return out


def effective_patience(patience: int, k: int) -> int:
    """Auto-bump ES patience to compensate for the extra noise of a K-shard aggregate.

    The aggregate adds a std-driven penalty whose own variance grows as ``2/(K-1)``; using the
    naive patience would cause spurious "no improvement" stops on the noisier signal. The
    closed-form factor ``1 + 1/sqrt(K-1)`` gives x1.5 at K=5 and x1.33 at K=10.
    """
    if k <= 1:
        return patience
    return math.ceil(patience * (1.0 + 1.0 / math.sqrt(k - 1)))
