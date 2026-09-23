"""Post-selection-inference holdout re-scoring (winner's-curse de-bias).

The discovery driver selects the winner spec(s) on the SAME ``mi_gain`` statistic
it then reports. Because that number is the MAX over many candidates evaluated on
ONE screening sample, it is optimistically biased upward -- the classic winner's
curse / post-selection-inference gap. The ``mi_gain_lcb`` / FDR gate de-bias the
ADMISSION decision but not the reported point gain.

The cure is a fresh holdout the discovery never touched: ``_fit.py`` carves
``honest_holdout_frac`` of the train rows out BEFORE screening (so the screening
sample, FDR gate, tiny-rerank, multi-base promotion and opt-in steps all consume
only the disjoint screening pool), and this module RE-SCORES only the FINAL
selected spec(s) on the holdout -- recomputing the exact same
``MI(T, X_remaining) - MI(y, X_remaining)`` quantity that ``eval_one_transform``
computed in-screen, but on rows no selection decision ever saw. The result is an
honest, materially less biased generalisation gain, stamped onto each spec
alongside (NOT replacing) the in-screen ``mi_gain``.

100GB-frame rule: the holdout is referenced by INDEX (a row-index slice of
``train_idx``); only the narrow per-column gathers on those holdout rows are
materialised, never a frame copy.
"""
from __future__ import annotations

from ._spec_shared import spec_base_columns

import logging
import threading
from typing import Any, Sequence

import numpy as np
from pyutilz.parallel import cpu_count_physical

from ..transforms import UnknownTransformError, get_transform
from .screening import (
    _extract_column_array,
    _mi_to_target,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
)

logger = logging.getLogger(__name__)


_HOLDOUT_SIZE_TOL = 0.25
"""A group-disjoint holdout may deviate from the configured size by this fraction either way."""


def _group_disjoint_holdout(g_train: np.ndarray, n_holdout: int, rng: np.random.Generator, min_rows: int, max_rows: int) -> np.ndarray | None:
    """Positions of whole groups totalling ``n_holdout * (1 +/- 0.25)`` rows, or ``None`` when no such set turns up.

    Groups are visited in a seeded random order and one that would overshoot the upper bound is skipped rather than taken:
    the greedy carve took groups while under 50 rows whatever their size, so one group holding 90% of the rows became a
    90% holdout while ``honest_holdout_frac=0.2`` was reported.
    """
    uniq, inv, counts = np.unique(g_train, return_inverse=True, return_counts=True)
    lo = max(int(min_rows), int(np.ceil(n_holdout * (1.0 - _HOLDOUT_SIZE_TOL))))
    hi = min(int(max_rows), int(np.floor(n_holdout * (1.0 + _HOLDOUT_SIZE_TOL))))
    if lo > hi:
        return None
    take = np.zeros(uniq.size, dtype=bool)
    total = 0
    for k in rng.permutation(uniq.size):
        if total + counts[k] <= hi:
            take[k] = True
            total += int(counts[k])
            if total >= n_holdout:
                break
    if total < lo:
        return None
    return np.nonzero(take[inv])[0]


def split_screening_holdout(
    train_idx: np.ndarray,
    holdout_frac: float | None,
    random_state: int,
    *,
    min_screen_rows: int = 50,
    min_holdout_rows: int = 50,
    group_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Carve ``train_idx`` into (screening_pool, honest_holdout) by index.

    Returns ``(screen_idx, holdout_idx)`` where the two are DISJOINT and their
    union (as a set) is ``train_idx``. ``holdout_idx`` is ``None`` (no split)
    when the feature is disabled (``holdout_frac`` falsy / <= 0) or when either
    side would fall below its minimum row floor -- in that case every train row
    stays in the screening pool, preserving the pre-feature behaviour exactly.

    The split is a SEEDED permutation slice: a random subset is held out (not a
    tail slice) so the holdout is an i.i.d. draw rather than the latest rows
    (which on temporal data would be a distribution-shifted, non-representative
    estimate). No frame copy -- only the integer index array is partitioned.
    """
    train_idx = np.asarray(train_idx)
    if not holdout_frac or holdout_frac <= 0.0:
        return train_idx, None
    if holdout_frac >= 1.0:
        # Degenerate config: a full holdout leaves nothing to screen on. Treat
        # as disabled rather than starving the screening pass.
        logger.warning(
            "[CompositeTargetDiscovery] honest_holdout_frac=%.3f >= 1.0 leaves no " "screening rows; disabling the holdout split for this fit.",
            holdout_frac,
        )
        return train_idx, None
    n = train_idx.size
    n_holdout = round(n * float(holdout_frac))
    if n_holdout < min_holdout_rows or (n - n_holdout) < min_screen_rows:
        logger.info(
            "[CompositeTargetDiscovery] honest_holdout_frac=%.3f on %d train rows would "
            "leave screen=%d / holdout=%d (floor screen>=%d, holdout>=%d); keeping all "
            "rows in screening (honest holdout disabled this fit).",
            holdout_frac, n, n - n_holdout, n_holdout, min_screen_rows, min_holdout_rows,
        )
        return train_idx, None
    rng = np.random.default_rng(random_state)
    # GROUP-DISJOINT holdout when group ids align to ``train_idx``: hold out WHOLE groups so the
    # winner's-curse de-bias measures generalisation to UNSEEN groups -- the actual deployment regime
    # under a group-aware split. An i.i.d. row holdout keeps same-group rows on both sides, so its
    # "honest gain" is blind to the only failure mode that matters here (a residual whose inverse
    # extrapolates on unseen groups). Falls back to the i.i.d. row split when groups are absent /
    # mis-aligned / too few to hit the holdout fraction.
    if group_ids is not None:
        g = np.asarray(group_ids)
        # One contract, the one every other reader of the rerank group ids uses: frame-aligned, indexed by train_idx. The
        # carve used to treat a length-n array as aligned to train_idx instead, which is only the same thing when
        # train_idx is arange(n).
        if n and g.shape[0] <= int(np.max(train_idx)):
            raise ValueError(f"group_ids has {g.shape[0]} rows but train_idx reaches row {int(np.max(train_idx))}; group ids must be aligned to the frame")
        g_train = g[train_idx]
        holdout_pos = _group_disjoint_holdout(g_train, n_holdout, rng, min_holdout_rows, n - min_screen_rows)
        if holdout_pos is not None:
            hmask = np.zeros(n, dtype=bool)
            hmask[holdout_pos] = True
            return train_idx[np.nonzero(~hmask)[0]], train_idx[holdout_pos]
        logger.warning(
            "[CompositeTargetDiscovery] no set of whole groups gives a holdout within 25%% of %d rows; using an i.i.d. row "
            "holdout, which is not group-disjoint.", n_holdout,
        )
    perm = rng.permutation(n)
    holdout_pos = np.sort(perm[:n_holdout])
    screen_pos = np.sort(perm[n_holdout:])
    return train_idx[screen_pos], train_idx[holdout_pos]


def carve_screening_holdout(self, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split ``train_idx`` into the screening pool + honest holdout and stash the attrs.

    Returns ``(screen_idx, holdout_idx)``. Sets ``full_train_idx_`` (the union, for honest
    accounting), ``honest_holdout_idx_`` (the disjoint holdout for the post-selection
    re-score), and ``train_idx_`` (the screening pool -- the rows actually used to FIT
    transform params + score, so it stays aligned to ``_auto_base_pool`` which holds
    ``base[train_idx]``). ``holdout_idx`` is ``None`` when the split is disabled / too small.
    """
    # Stability-check sharing: ``fit_with_stability_check`` carves the honest holdout ONCE for the
    # whole replicate sweep and hands each replicate a subsample of the SCREEN pool only. Each
    # replicate must then reuse that shared holdout instead of carving its own -- per-replicate
    # carves draw holdouts that overlap other replicates' screening pools, so no row set stays
    # "never touched" across the sweep (plus each replicate pays a redundant carve). The attribute is
    # a (possibly empty) index array; empty means the holdout is disabled for the whole sweep.
    _shared = getattr(self, "_stability_shared_holdout_idx", None)
    if _shared is not None:
        shared = np.asarray(_shared)
        if shared.size and np.intersect1d(train_idx, shared).size:
            logger.warning(
                "[CompositeTargetDiscovery] stability-shared holdout overlaps the replicate's " "train subsample; falling back to a per-replicate carve."
            )
        else:
            self.full_train_idx_ = np.sort(np.concatenate([train_idx, shared])) if shared.size else train_idx
            self.honest_holdout_idx_ = shared if shared.size else None
            self.train_idx_ = train_idx
            return train_idx, (shared if shared.size else None)
    self.full_train_idx_ = train_idx
    screen_idx, holdout_idx = split_screening_holdout(
        train_idx,
        getattr(self.config, "honest_holdout_frac", 0.2),
        int(getattr(self.config, "random_state", 42)),
        group_ids=getattr(self, "_group_ids_for_rerank", None),
    )
    self.honest_holdout_idx_ = holdout_idx
    self.honest_holdout_select_idx_, self.honest_holdout_report_idx_ = split_holdout_select_report(
        holdout_idx, int(getattr(self.config, "random_state", 42)),
    )
    self.train_idx_ = screen_idx
    return screen_idx, holdout_idx


def split_holdout_select_report(holdout_idx: np.ndarray | None, random_state: int, *, min_side_rows: int = 50) -> tuple:
    """Halve the honest holdout into ``(selection_rows, report_rows)``.

    The carve promises rows no decision ever saw, but the drop gate, the honest-OOF rank key and the cross-target budget
    all read the holdout, and the number stamped on each surviving spec is then measured on those same rows. That number
    is a maximum over the survivors of a comparison made on the very rows it reports, so it carries the winner's curse
    the carve exists to remove, and it gets more optimistic the more candidates the gates reject.

    Splitting the holdout keeps both jobs honest: the selection half feeds every gate and ranking, the report half is
    read only by the final re-score. Below ``2 * min_side_rows`` there is nothing to split -- both halves would be too
    small to estimate anything -- so both roles keep the whole holdout and the reported number stays as it was.
    """
    if holdout_idx is None:
        return None, None
    idx = np.asarray(holdout_idx)
    if idx.size < 2 * min_side_rows:
        return idx, idx
    order = np.random.default_rng(random_state).permutation(idx.size)
    cut = idx.size // 2
    return np.sort(idx[order[:cut]]), np.sort(idx[order[cut:]])


def _build_x_remaining_holdout(
    df: Any,
    usable_features: Sequence[str],
    base_columns: Sequence[str],
    holdout_idx: np.ndarray,
) -> np.ndarray:
    """X-remaining (all usable features MINUS the spec's base column(s)) on holdout rows.

    Mirrors the in-screen ``x_remaining_matrix`` construction: the base column(s)
    are excluded so ``MI(., X_remaining)`` isolates the transform effect rather
    than re-measuring the dominant base feature. Unary specs (no base) keep the
    FULL feature matrix, exactly as the in-screen unary sentinel context does.
    """
    base_set = set(base_columns)
    cols = [c for c in usable_features if c not in base_set]
    if not cols:
        return np.zeros((holdout_idx.size, 0), dtype=np.float32)
    arrays = [_extract_column_array(df, c, rows=holdout_idx) for c in cols]
    return np.column_stack(arrays)


def _rescore_one_spec(spec, *, df, holdout_idx, y_holdout, x_remaining_for, prebinned_for, estimator, nbins, aggregation, n_neighbors, random_state, mi_y_memo, mi_y_memo_lock) -> None:
    """Recompute one spec's honest MI gain on the held-out rows and write it back onto the frozen spec in place via
    ``object.__setattr__``; any failure (unknown transform, empty remaining-feature matrix) leaves the spec's honest
    fields untouched."""
    try:
        transform = get_transform(spec.transform_name)
    except UnknownTransformError:
        return
    base_columns = spec_base_columns(spec)
    x_remaining = x_remaining_for(base_columns)
    if x_remaining.shape[1] == 0:
        return
    # Materialise the base argument shape the transform.forward expects:
    # a (n,) vector for single-base / a (n, k) matrix for multi-base /
    # a zeros placeholder for unary (forward ignores it).
    if not base_columns:
        base_arg = np.zeros(holdout_idx.size, dtype=np.float64)
    elif len(base_columns) == 1:
        base_arg = _extract_column_array(df, base_columns[0], rows=holdout_idx).astype(np.float64)
    else:
        base_arg = np.column_stack([_extract_column_array(df, c, rows=holdout_idx).astype(np.float64) for c in base_columns])
    y_h = y_holdout.astype(np.float64)
    # Domain filter on holdout, then the fitted-domain refinement -- the SAME
    # two-stage gate eval_one_transform applies, so T and y are scored on the
    # identical row population (else mi_t / mi_y compare different rows).
    try:
        valid = np.asarray(transform.domain_check(y_h, base_arg), dtype=bool)
    except Exception as exc:  # -- degenerate holdout for this spec
        logger.debug("honest-holdout domain_check failed for %s: %s", spec.name, exc)
        return
    if valid.shape != y_h.shape:
        return
    params = dict(spec.fitted_params)
    _dcf = getattr(transform, "domain_check_fitted", None)
    if _dcf is not None:
        try:
            valid_fitted = np.asarray(_dcf(y_h, base_arg, params), dtype=bool)
            if valid_fitted.shape == valid.shape:
                valid = valid & valid_fitted
        except Exception as e:  # -- treat as no refinement
            logger.debug("swallowed exception in _honest_holdout.py: %s", e)
            pass
    n_valid = int(valid.sum())
    if n_valid < 50:
        logger.debug(
            "honest-holdout: spec %s has only %d valid holdout rows (<50); "
            "leaving honest_holdout_gain=None.", spec.name, n_valid,
        )
        return
    base_valid = base_arg[valid] if base_arg.ndim == 1 else base_arg[valid, :]
    try:
        t_holdout = transform.forward(y_h[valid], base_valid, params)
    except Exception as exc:  # -- transform raised on holdout rows
        logger.debug("honest-holdout forward failed for %s: %s", spec.name, exc)
        return
    # The shared bin codes are only usable when this spec keeps every holdout row: the bin edges are quantiles of
    # the rows actually scored, so a spec whose domain filter drops rows must bin its own subset, as before.
    _codes = prebinned_for(base_columns) if (estimator == "bin" and bool(valid.all())) else None
    x_valid = x_remaining if _codes is not None else x_remaining[valid]
    _mi_kwargs: dict[str, Any] = dict(nbins=nbins, aggregation=aggregation)
    if _codes is not None:
        mi_t = _mi_to_target_prebinned(_codes, t_holdout, nbins=nbins, aggregation=aggregation)
    else:
        mi_t = _mi_to_target(
            x_valid, t_holdout,
            n_neighbors=n_neighbors, random_state=random_state,
            estimator=estimator, **_mi_kwargs,
        )
    _memo_key = (tuple(base_columns), hash(valid.tobytes()))
    with mi_y_memo_lock:
        _mi_y_cached = mi_y_memo.get(_memo_key)
    if _mi_y_cached is None:
        if _codes is not None:
            mi_y = _mi_to_target_prebinned(_codes, y_h[valid], nbins=nbins, aggregation=aggregation)
        else:
            mi_y = _mi_to_target(
                x_valid, y_h[valid],
                n_neighbors=n_neighbors, random_state=random_state,
                estimator=estimator, **_mi_kwargs,
            )
        with mi_y_memo_lock:
            mi_y_memo[_memo_key] = float(mi_y)
    else:
        mi_y = _mi_y_cached
    honest_gain = float(mi_t - mi_y)
    object.__setattr__(spec, "honest_holdout_gain", honest_gain)
    object.__setattr__(spec, "honest_holdout_mi_t", float(mi_t))
    object.__setattr__(spec, "honest_holdout_mi_y", float(mi_y))
    object.__setattr__(spec, "honest_holdout_n_rows", n_valid)


def apply_honest_holdout(
    self,
    df: Any,
    target_col: str,
    kept_specs: list,
    usable_features: Sequence[str],
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    y_full: np.ndarray,
) -> None:
    """Leakage-guarded driver for the post-selection re-score (called from ``_fit.py``).

    The winner set is FINAL here (eps/FDR gate, top-k trim, alpha-drift, linres->diff
    collapse, tiny-rerank, multi-base promotion + dedup, opt-in steps all done). Assert
    the holdout never entered the screening pool, then re-score the survivors on it. A
    re-score failure is non-fatal -- the specs keep ``honest_holdout_gain=None`` and the
    in-screen ``mi_gain`` is unaffected -- because the honest gain is a diagnostic
    overlay, never load-bearing for the spec itself.
    """
    from .._row_roles import note_rows

    note_rows("honest_holdout", "report", "honest_holdout_rescore", holdout_idx)
    if np.intersect1d(train_idx, holdout_idx).size:
        raise ValueError("[CompositeTargetDiscovery] honest-holdout indices overlap the screening " "pool -- post-selection estimate would leak.")
    try:
        rescore_specs_on_holdout(
            self, df, target_col, kept_specs, usable_features, holdout_idx, y_full,
        )
    except Exception as ho_err:  # -- diagnostic, never load-bearing
        logger.warning(
            "[CompositeTargetDiscovery] honest-holdout re-score failed (%s); specs keep " "honest_holdout_gain=None, in-screen mi_gain unaffected.",
            ho_err,
        )


def rescore_specs_on_holdout(
    self,
    df: Any,
    target_col: str,
    kept_specs: list,
    usable_features: Sequence[str],
    holdout_idx: np.ndarray | None,
    y_full: np.ndarray,
) -> None:
    """Stamp an honest holdout gain onto each FINAL spec (in place, frozen-safe).

    For every spec in ``kept_specs`` this recomputes
    ``honest_gain = MI(T_holdout, X_remaining_holdout) - MI(y_holdout, X_remaining_holdout)``
    on the never-touched holdout rows, using the spec's ALREADY-FITTED params (the
    transform is NOT re-fit -- re-fitting on holdout would defeat the point; we
    measure how the in-screen-fitted transform generalises). The two MIs use the
    SAME estimator / nbins / aggregation config the in-screen path used, so the
    honest gain is directly comparable to the in-screen ``mi_gain``.

    The spec dataclass is frozen, so the honest fields are written via
    ``object.__setattr__``. A degenerate spec (no holdout, too few valid rows,
    transform raises) leaves ``honest_holdout_gain=None`` -- the caller / report
    then falls back to the in-screen gain for that spec, clearly labelled.
    """
    if holdout_idx is None or holdout_idx.size == 0 or not kept_specs:
        return
    cfg = self.config
    # Cap the re-scored rows the same way the in-screen MI is capped. The holdout is a fraction of train with no cap of
    # its own, so on a multi-million-row frame every spec pulled the full feature block for all holdout rows, twice, in
    # parallel threads. The honest gain is an MI difference, which saturates at the same sample size the screen uses;
    # above the cap a seeded draw is taken, below it every row is kept and the stamped numbers are unchanged.
    _mi_cap = getattr(cfg, "mi_sample_n", 100000)
    holdout_idx = np.asarray(holdout_idx)
    if _mi_cap is not None and int(_mi_cap) > 0 and holdout_idx.size > int(_mi_cap):
        _draw = np.random.default_rng(int(getattr(cfg, "random_state", 42))).choice(holdout_idx.size, size=int(_mi_cap), replace=False)
        logger.info(
            "[CompositeTargetDiscovery.honest_holdout] re-scoring on a seeded %d-row draw of the %d-row holdout "
            "(mi_sample_n cap); the honest gain is an MI difference and saturates well below this size.",
            int(_mi_cap), holdout_idx.size,
        )
        holdout_idx = np.sort(holdout_idx[_draw])
    estimator = getattr(cfg, "mi_estimator", "bin")
    nbins = int(getattr(cfg, "mi_nbins", 16))
    aggregation = getattr(cfg, "mi_aggregation", "mean")
    n_neighbors = int(getattr(cfg, "mi_n_neighbors", 3))
    random_state = int(getattr(cfg, "random_state", 42))
    y_holdout = y_full[holdout_idx]
    # ``mi_y = MI(y_holdout, X_remaining)`` depends only on (the spec's base-column set, the valid
    # row mask) -- specs sharing both would recompute the identical scalar. Memoise across specs;
    # the biggest win is on the knn estimator where each per-column Kraskov call dominates cost.
    # Lock-guarded -- the per-spec re-scores run from a thread pool below.
    _mi_y_memo: dict[tuple, float] = {}
    _mi_y_memo_lock = threading.Lock()
    # X-remaining depends only on the spec's base-column set, so specs sharing a base rebuilt the identical matrix --
    # once per spec, concurrently, each a full holdout-rows x features copy. Build one per base set instead.
    _x_remaining_cache: dict[tuple, np.ndarray] = {}
    _x_remaining_lock = threading.Lock()

    _prebinned_cache: dict[tuple, np.ndarray] = {}
    _prebinned_lock = threading.Lock()

    def _x_remaining_for(base_columns: Sequence[str]) -> np.ndarray:
        """The holdout X-remaining matrix for this base set, built once and shared across the specs that need it."""
        key = tuple(sorted(base_columns))
        # The lock is held across the build, not just around the dict: the re-scores start together, so a
        # check-then-build would have every thread miss the empty cache and build its own copy, which is the
        # duplication this cache exists to remove.
        with _x_remaining_lock:
            cached = _x_remaining_cache.get(key)
            if cached is None:
                cached = _build_x_remaining_holdout(df, usable_features, base_columns, holdout_idx)
                _x_remaining_cache[key] = cached
        return cached

    def _prebinned_for(base_columns: Sequence[str]) -> np.ndarray:
        """The bin codes of this base set's X-remaining matrix, quantile-binned once and shared across specs.

        The bin estimator re-quantiles every feature column on each call, twice per spec. The codes depend only on the
        columns and the row set, so binning once per base set makes each later MI a pure histogram pass; the aggregated
        MI is the same number the per-call binning produced.
        """
        key = tuple(sorted(base_columns))
        # The matrix is fetched BEFORE the lock: taking it inside would re-enter the same non-reentrant lock that
        # ``_x_remaining_for`` holds across its own build, and every re-score thread would stop there.
        matrix = _x_remaining_for(base_columns)
        with _prebinned_lock:
            cached = _prebinned_cache.get(key)
            if cached is None:
                cached = _prebin_feature_columns(matrix, nbins=nbins)
                _prebinned_cache[key] = cached
        return cached

    def _rescore_one(spec) -> None:
        """Score one spec against the shared per-base caches."""
        _rescore_one_spec(
            spec, df=df, holdout_idx=holdout_idx, y_holdout=y_holdout,
            x_remaining_for=_x_remaining_for, prebinned_for=_prebinned_for,
            estimator=estimator, nbins=nbins, aggregation=aggregation,
            n_neighbors=n_neighbors, random_state=random_state,
            mi_y_memo=_mi_y_memo, mi_y_memo_lock=_mi_y_memo_lock,
        )

    # Per-spec re-scores are independent (each writes only its OWN frozen spec via object.__setattr__,
    # reads shared read-only arrays). The MI kernels release the GIL, so thread across physical cores.
    n_jobs = min(len(kept_specs), cpu_count_physical())
    if n_jobs > 1:
        from joblib import Parallel, delayed

        Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(delayed(_rescore_one)(s) for s in kept_specs)
    else:
        for s in kept_specs:
            _rescore_one(s)
