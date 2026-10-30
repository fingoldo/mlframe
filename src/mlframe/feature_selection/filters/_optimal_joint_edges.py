"""Cross-validated bin-count search (``optimal_joint``) split into a target-independent plan and a target-dependent scoring.

The fold split depends only on the number of rows and the seed, the train-fold edges only on the training values, so everything except the validation-fold MI against the target can be computed once
per column. A null that permutes the target bins the same columns for every permutation and shares that work through :class:`OptimalJointPlanCache`; without a cache the search is the plain one.
The names are re-exported by ``_adaptive_nbins``.
"""

from __future__ import annotations

import threading
from typing import Any, Optional, cast

import numpy as np


def _a() -> Any:
    """The parent module, for its edge builders and the plug-in MI (a top-level import would be circular: the parent re-exports this module)."""
    from . import _adaptive_nbins

    return _adaptive_nbins


def _optimal_joint_plan(x: np.ndarray, candidates: tuple, n_splits: int, base: str, random_state: int) -> dict:
    """The part of :func:`edges_optimal_joint` that does not depend on the target: the fold split, and for every fold and candidate bin count the train-fold edges and the bins of the validation values.

    The fold of a row depends only on the number of rows, the edges only on the training values, so a null that permutes the target reuses one plan for every permutation. The bins are stored in
    the narrowest integer type that holds them (at most ``max(candidates) + 1`` distinct codes)."""
    n = x.size
    rng = np.random.default_rng(random_state)
    fold_idx = rng.permutation(n) % n_splits
    folds = []
    for k in range(n_splits):
        train_mask = fold_idx != k
        val_mask = ~train_mask
        if val_mask.sum() < 4:
            continue  # val-fold-too-small check does not depend on M -> applies to every M for this k
        train_x = x[train_mask]
        val_x = x[val_mask]
        n_train = int(train_mask.sum())
        cells = []
        for M in candidates:
            if M < 2 or n_train < M:
                continue
            edges = _a()._edges_from_quantiles(train_x, M) if base == "quantile" else _a()._edges_from_uniform(train_x, M)
            if edges.size == 0:
                continue
            x_b = np.searchsorted(edges, val_x, side="right")
            k_x = int(x_b.max()) + 1 if x_b.size else 1
            narrow = np.int8 if k_x <= 127 else np.int64
            cells.append((M, np.ascontiguousarray(x_b.astype(narrow)), k_x))
        folds.append((k, val_mask, cells))
    return {"folds": folds, "x": x, "base": base, "final": {}, "shape_key": (n, int(n_splits), int(random_state))}


def _optimal_joint_best_m(plan: dict, y: np.ndarray, candidates: tuple, max_y_classes: int, fold_target: Any = None) -> Optional[int]:
    """The candidate bin count with the highest mean fold MI against ``y`` (the first one on a tie), or ``None`` when no candidate was scored by any fold.

    ``fold_target(k, val_mask)`` may supply the binned validation target of fold ``k`` (shared between the columns that have the same folds); without it the target is binned here."""
    fold_scores_by_M: "dict[int, list]" = {M: [] for M in candidates if M >= 2}
    for k, val_mask, cells in plan["folds"]:
        # val_y quantization (_plug_in_mi's np.quantile-based 10-bin regression target) depends only on val_y, not on M -> computed once per fold and reused for every candidate M.
        val_y_b, K_y = fold_target(k, val_mask) if fold_target is not None else _a()._bin_y_for_mi(y[val_mask], max_y_classes=max_y_classes)
        for M, x_b, K_x in cells:
            if x_b.size == 0 or val_y_b.size == 0:
                mi = 0.0
            else:
                mi = 0.0 if (K_x < 1 or K_y < 1) else float(_a()._plug_in_mi_njit(x_b, val_y_b, K_x, K_y, False))
            fold_scores_by_M[M].append(mi)
    best_score = -np.inf
    best_M = None
    for M in candidates:
        if M < 2:
            continue
        fold_scores = fold_scores_by_M.get(M, [])
        if fold_scores:
            mean_mi = float(np.mean(fold_scores))
            if mean_mi > best_score:
                best_score = mean_mi
                best_M = M
    return best_M


def _optimal_joint_final_edges(plan: dict, M: int) -> np.ndarray:
    """Edges at bin count ``M`` built on the full finite data, computed once per plan."""
    edges = plan["final"].get(M)
    if edges is None:
        x = plan["x"]
        edges = cast(np.ndarray, _a()._edges_from_quantiles(x, M) if plan["base"] == "quantile" else _a()._edges_from_uniform(x, M))
        plan["final"][M] = edges
    return cast(np.ndarray, edges)


def edges_optimal_joint(
    x: np.ndarray, y: np.ndarray,
    *,
    candidates: tuple = (4, 8, 16, 32),
    n_splits: int = 3,
    base: str = "quantile",
    random_state: int = 0,
    max_y_classes: int = 64,
    plan_cache: "Optional[OptimalJointPlanCache]" = None,
) -> np.ndarray:
    """CV-folded MI maximisation across candidate nbins.

    For each n_bins in ``candidates``:
      * Split (x, y) into ``n_splits`` folds.
      * For each fold: compute quantile edges on TRAIN x, bin VAL x with those edges,
        estimate I(X_val; y_val) via the standard plug-in histogram MI.
      * Score = mean fold MI.
    Pick the n_bins with highest score, return its edges built on the FULL data.

    This is the OptimalJoint / "wrapper-style" method recommended for MRMR when
    compute is not a bottleneck. ~K_candidates * n_splits times more expensive
    than a single Freedman-Diaconis call.

    Args:
        x: the 1-D numeric column to bin.
        y: the target used to score each candidate bin count's fold MI.
        candidates: candidate bin counts to sweep.
        n_splits: number of CV folds used to score each candidate.
        base: ``'quantile'`` or ``'uniform'`` edge placement within a candidate.
        random_state: fold-split RNG seed.
        max_y_classes: Cardinality cap forwarded to :func:`_bin_y_for_mi` for the internal
            fold-scoring MI. Without this cap, an int/bool-dtype ``y`` with high cardinality (a
            continuous target mistakenly int-typed - e.g. a timestamp or counter column) was
            treated as one discrete class per distinct value: confirmed to SEGFAULT the process
            (oversized ``(K_x, K_y)`` dense joint-count allocation in ``_plug_in_mi_njit``) at
            n=50000 with ~50k unique int64 values. Same bug class as MDLP's ``max_y_classes``.
        plan_cache: an :class:`OptimalJointPlanCache` that keeps the target-independent plan of each column (folds, train edges, binned validation values) across calls with the
            same column and a different ``y``; the result is the same as without it.

    Returns:
        1-D double array of inner bin edges for the winning bin count, built on the full finite data; falls back to
        Freedman-Diaconis edges when there is too little data to cross-validate.
    """
    x_in = x
    x = np.asarray(x, dtype=np.float64)  # Intentional float64: bin-edge computation input; edges compared at full precision
    y = np.asarray(y)
    # A 1-D argument keeps its identity (``ravel`` would hand back a new view object each call, and a plan cache recognises the arrays it has hashed by identity).
    x = x if x.ndim == 1 else x.ravel()
    y = y if y.ndim == 1 else y.ravel()
    mask = np.isfinite(x)
    # The mask used to check only x's finiteness - a
    # NaN-y row with finite x survived and could propagate NaN into that fold's quantile edges via
    # _bin_y_for_mi's np.quantile call. Fold y's finiteness into the mask too (only meaningful when y is
    # float-dtype; an int/bool y is never NaN-capable).
    if y.dtype.kind == "f":
        mask &= np.isfinite(y)
    if not mask.all():  # nothing to drop: keep the caller's arrays, which a plan cache recognises by identity
        x = x[mask]
        y = y[mask]
    n = x.size
    if n < n_splits * 4:
        # Fall back to FD on small samples.
        return cast(np.ndarray, _a().edges_freedman_diaconis(x, base=base))
    # The plan depends on the column (after the mask) and the settings, not on y; it is kept per column when a cache is given. A y with a different finite pattern gives a different masked
    # column, hence a different key, so the cache can never serve a plan built for other rows.
    # The plan is keyed on the array the caller handed in (hashed once per call, shared with the column facts) when nothing was dropped; the float64 copy is only built on a miss.
    keyed = x_in if (plan_cache is not None and x is x_in) or (plan_cache is not None and mask.all() and np.asarray(x_in).ndim == 1) else x
    plan = plan_cache.get(keyed, candidates, n_splits, base, random_state) if plan_cache is not None else None
    if plan is None:
        plan = _optimal_joint_plan(x, candidates, n_splits, base, random_state)
        if plan_cache is not None:
            plan_cache.put(keyed, candidates, n_splits, base, random_state, plan)
    fold_target = plan_cache.fold_target_provider(y, plan["shape_key"], max_y_classes) if plan_cache is not None else None
    best_M = _optimal_joint_best_m(plan, y, candidates, max_y_classes, fold_target)
    if best_M is None:
        # No candidate M was ever scored by ANY fold - e.g. every candidate in ``candidates`` exceeds
        # every fold's train size, or every fold/candidate combination produced degenerate (empty) train
        # edges. The pre-fix code silently fell back to ``candidates[0]`` here UNVALIDATED by the CV
        # search - when ``candidates[0] < 2`` (a caller passing e.g. ``candidates=(1, 1000)`` to probe a
        # single-bin option) this returned EMPTY edges with no signal that the CV search never ran at
        # all, exactly the MDLP silent-empty-output bug class. Fall back to the same Freedman-Diaconis
        # path used for the too-small-n guard above - at least one principled, unconditional binning
        # instead of an untested/possibly-invalid M.
        return cast(np.ndarray, _a().edges_freedman_diaconis(x, base=base))
    # Return edges built on full data at the winning M.
    return _optimal_joint_final_edges(plan, best_M)


_MEMO_ENTRIES = 20000  # collapse refinements kept per cache: columns x the few edge sets each can take
_FOLD_TARGET_MEMO = 48  # binned validation targets kept per cache: a few permutations of a few fold shapes


class OptimalJointPlanCache:
    """Target-independent plans of :func:`edges_optimal_joint`, keyed by the content of the column and the settings; safe to share between threads.

    A null that permutes the target bins the same columns once per permutation; the plan (folds, train-fold edges, binned validation values) is the same every time. The cache is bounded by
    ``max_bytes`` (a plan that does not fit is simply not stored, and the call computes it as without a cache).

    Args:
        max_bytes: the most the cache holds, as an empty cache holding at most that many bytes of plans, column facts and memoised results.
    """

    def __init__(self, max_bytes: int = 512 * 1024 * 1024) -> None:
        self._plans: dict = {}
        self._local = threading.local()  # last (array, key) hashed by this thread: one call hashes the same column for the plan and for the column facts, and the same target for every fold
        self._facts: "dict[str, tuple]" = {}  # column hash -> (finite values, their unique values, the counts): the column facts the edge builders read
        self._memo: "dict[tuple, Any]" = {}  # small results that depend on a column and a few edge sets (the collapse refinement)
        self._targets: "dict[tuple, tuple]" = {}  # (target hash, folds, class cap) -> (binned validation target, classes): the target side of a fold, shared by every column with those folds
        self._bytes = 0
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()

    def content_key(self, arr: np.ndarray) -> str:
        """Content hash of ``arr``, reusing the hash of the array this thread hashed last when it is the same object (the cache holds a reference, so the object cannot be a recycled address)."""
        from mlframe.utils.disk_cache import hash_array_content

        last = getattr(self._local, "last", None)
        if last is not None and last[0] is arr:
            return cast(str, last[1])
        key = hash_array_content(arr)
        self._local.last = (arr, key)
        return key

    def _key(self, x: np.ndarray, candidates: tuple, n_splits: int, base: str, random_state: int) -> tuple:
        """Content hash of the column with the settings that shape the plan."""
        return (self.content_key(x), x.size, str(x.dtype), tuple(candidates), int(n_splits), str(base), int(random_state))

    def get(self, x: np.ndarray, candidates: tuple, n_splits: int, base: str, random_state: int) -> Optional[dict]:
        """The stored plan of this column and settings, or ``None``."""
        key = self._key(x, candidates, n_splits, base, random_state)
        with self._lock:
            return self._plans.get(key)

    def column_facts(self, col: np.ndarray) -> "tuple[str, np.ndarray, np.ndarray, np.ndarray]":
        """``(key, finite values, unique values, counts)`` of ``col``, computed once per distinct column content (the unique counts need a sort of the column)."""
        key = self.content_key(col)
        with self._lock:
            hit = self._facts.get(key)
        if hit is not None:
            return (key, *hit)
        finite = col[np.isfinite(col)]
        if finite.size > 0:
            uniq, counts = np.unique(finite, return_counts=True)
        else:
            uniq, counts = np.empty(0, dtype=col.dtype), np.empty(0, dtype=np.int64)
        size = int(finite.nbytes + uniq.nbytes + counts.nbytes)
        with self._lock:
            if key not in self._facts and self._bytes + size <= self._max_bytes:
                self._facts[key] = (finite, uniq, counts)
                self._bytes += size
        return key, finite, uniq, counts

    def memo(self, key: tuple, compute: Any) -> Any:
        """The value stored under ``key``, computed by ``compute()`` on the first request."""
        with self._lock:
            if key in self._memo:
                return self._memo[key]
        value = compute()
        with self._lock:
            if len(self._memo) < _MEMO_ENTRIES:
                self._memo[key] = value
        return value

    def _target_key(self, y: np.ndarray) -> str:
        """Content hash of the target, kept for the next call with the same array object (the columns of one permutation share it) apart from the column hash slot."""
        from mlframe.utils.disk_cache import hash_array_content

        last = getattr(self._local, "last_y", None)
        if last is not None and last[0] is y:
            return cast(str, last[1])
        key = hash_array_content(y)
        self._local.last_y = (y, key)
        return key

    def fold_target_provider(self, y: np.ndarray, shape_key: tuple, max_y_classes: int) -> Any:
        """``fold_target(k, val_mask)`` for the target ``y`` of one call: the binned validation target of fold ``k``, computed once for all columns whose folds are those of ``shape_key``
        (the folds depend only on the number of rows and the seed) and the same ``y``. The memo keeps the targets of the last few permutations only."""
        y_key = self._target_key(y)

        def fold_target(k: int, val_mask: np.ndarray) -> "tuple[np.ndarray, int]":
            """Binned validation target of fold ``k``."""
            key = (y_key, shape_key, int(k), int(max_y_classes))
            with self._lock:
                hit = self._targets.get(key)
            if hit is not None:
                return hit
            out = _a()._bin_y_for_mi(y[val_mask], max_y_classes=max_y_classes)
            with self._lock:
                while len(self._targets) >= _FOLD_TARGET_MEMO:
                    self._targets.pop(next(iter(self._targets)))
                self._targets[key] = out
            return cast("tuple[np.ndarray, int]", out)

        return fold_target

    def put(self, x: np.ndarray, candidates: tuple, n_splits: int, base: str, random_state: int, plan: dict) -> None:
        """Store ``plan`` unless it would exceed the byte budget."""
        size = int(plan["x"].nbytes) + sum(xb.nbytes + vm.nbytes for _k, vm, cells in plan["folds"] for _m, xb, _kx in cells)
        key = self._key(x, candidates, n_splits, base, random_state)
        with self._lock:
            if key not in self._plans and self._bytes + size <= self._max_bytes:
                self._plans[key] = plan
                self._bytes += size
