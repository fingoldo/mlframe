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

import logging
from typing import Any, Sequence

import numpy as np

from ..transforms import UnknownTransformError, get_transform
from .screening import (
    _extract_column_array,
    _mi_to_target,
)

logger = logging.getLogger(__name__)


def split_screening_holdout(
    train_idx: np.ndarray,
    holdout_frac: float | None,
    random_state: int,
    *,
    min_screen_rows: int = 50,
    min_holdout_rows: int = 50,
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
            "[CompositeTargetDiscovery] honest_holdout_frac=%.3f >= 1.0 leaves no "
            "screening rows; disabling the holdout split for this fit.", holdout_frac,
        )
        return train_idx, None
    n = train_idx.size
    n_holdout = int(round(n * float(holdout_frac)))
    if n_holdout < min_holdout_rows or (n - n_holdout) < min_screen_rows:
        logger.info(
            "[CompositeTargetDiscovery] honest_holdout_frac=%.3f on %d train rows would "
            "leave screen=%d / holdout=%d (floor screen>=%d, holdout>=%d); keeping all "
            "rows in screening (honest holdout disabled this fit).",
            holdout_frac, n, n - n_holdout, n_holdout, min_screen_rows, min_holdout_rows,
        )
        return train_idx, None
    rng = np.random.default_rng(random_state)
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
    self.full_train_idx_ = train_idx
    screen_idx, holdout_idx = split_screening_holdout(
        train_idx,
        getattr(self.config, "honest_holdout_frac", 0.2),
        int(getattr(self.config, "random_state", 0)),
    )
    self.honest_holdout_idx_ = holdout_idx
    self.train_idx_ = screen_idx
    return screen_idx, holdout_idx


def _spec_base_columns(spec) -> list[str]:
    """Full ordered base-column list for a spec: primary + any extra (multi-base)."""
    extra = tuple(getattr(spec, "extra_base_columns", ()) or ())
    if not spec.base_column:
        return []  # unary (base-free) spec.
    return [spec.base_column, *extra]


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
    if np.intersect1d(train_idx, holdout_idx).size:
        raise ValueError(
            "[CompositeTargetDiscovery] honest-holdout indices overlap the screening "
            "pool -- post-selection estimate would leak."
        )
    try:
        rescore_specs_on_holdout(
            self, df, target_col, kept_specs, usable_features, holdout_idx, y_full,
        )
    except Exception as ho_err:  # noqa: BLE001 -- diagnostic, never load-bearing
        logger.warning(
            "[CompositeTargetDiscovery] honest-holdout re-score failed (%s); specs keep "
            "honest_holdout_gain=None, in-screen mi_gain unaffected.", ho_err,
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
    estimator = getattr(cfg, "mi_estimator", "bin")
    nbins = int(getattr(cfg, "mi_nbins", 16))
    aggregation = getattr(cfg, "mi_aggregation", "mean")
    n_neighbors = int(getattr(cfg, "mi_n_neighbors", 3))
    random_state = int(getattr(cfg, "random_state", 0))
    y_holdout = y_full[holdout_idx]

    for spec in kept_specs:
        try:
            transform = get_transform(spec.transform_name)
        except UnknownTransformError:
            continue
        base_columns = _spec_base_columns(spec)
        x_remaining = _build_x_remaining_holdout(df, usable_features, base_columns, holdout_idx)
        if x_remaining.shape[1] == 0:
            continue
        # Materialise the base argument shape the transform.forward expects:
        # a (n,) vector for single-base / a (n, k) matrix for multi-base /
        # a zeros placeholder for unary (forward ignores it).
        if not base_columns:
            base_arg = np.zeros(holdout_idx.size, dtype=np.float64)
        elif len(base_columns) == 1:
            base_arg = _extract_column_array(df, base_columns[0], rows=holdout_idx).astype(np.float64)
        else:
            base_arg = np.column_stack(
                [_extract_column_array(df, c, rows=holdout_idx).astype(np.float64) for c in base_columns]
            )
        y_h = y_holdout.astype(np.float64)
        # Domain filter on holdout, then the fitted-domain refinement -- the SAME
        # two-stage gate eval_one_transform applies, so T and y are scored on the
        # identical row population (else mi_t / mi_y compare different rows).
        try:
            valid = np.asarray(transform.domain_check(y_h, base_arg), dtype=bool)
        except Exception as exc:  # noqa: BLE001 -- degenerate holdout for this spec
            logger.debug("honest-holdout domain_check failed for %s: %s", spec.name, exc)
            continue
        if valid.shape != y_h.shape:
            continue
        params = dict(spec.fitted_params)
        _dcf = getattr(transform, "domain_check_fitted", None)
        if _dcf is not None:
            try:
                valid_fitted = np.asarray(_dcf(y_h, base_arg, params), dtype=bool)
                if valid_fitted.shape == valid.shape:
                    valid = valid & valid_fitted
            except Exception:  # noqa: BLE001 -- treat as no refinement
                pass
        n_valid = int(valid.sum())
        if n_valid < 50:
            logger.debug(
                "honest-holdout: spec %s has only %d valid holdout rows (<50); "
                "leaving honest_holdout_gain=None.", spec.name, n_valid,
            )
            continue
        base_valid = base_arg[valid] if base_arg.ndim == 1 else base_arg[valid, :]
        try:
            t_holdout = transform.forward(y_h[valid], base_valid, params)
        except Exception as exc:  # noqa: BLE001 -- transform raised on holdout rows
            logger.debug("honest-holdout forward failed for %s: %s", spec.name, exc)
            continue
        x_valid = x_remaining[valid]
        _mi_kwargs = dict(nbins=nbins, aggregation=aggregation)
        mi_t = _mi_to_target(
            x_valid, t_holdout,
            n_neighbors=n_neighbors, random_state=random_state,
            estimator=estimator, **_mi_kwargs,
        )
        mi_y = _mi_to_target(
            x_valid, y_h[valid],
            n_neighbors=n_neighbors, random_state=random_state,
            estimator=estimator, **_mi_kwargs,
        )
        honest_gain = float(mi_t - mi_y)
        object.__setattr__(spec, "honest_holdout_gain", honest_gain)
        object.__setattr__(spec, "honest_holdout_mi_t", float(mi_t))
        object.__setattr__(spec, "honest_holdout_mi_y", float(mi_y))
        object.__setattr__(spec, "honest_holdout_n_rows", n_valid)
