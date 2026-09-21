"""knn-MI cost guard: auto-downgrade ``mi_estimator="knn"`` to ``"bin"`` when the sweep would blow a budget.

The Kraskov estimator is O(n log n) PER COLUMN and the discovery screen runs one full per-column
sweep per (base, transform) work item (``mi_t``, plus shrunk-domain ``mi_y_compare`` re-scores and
the upfront per-feature ``mi_y`` baseline). On a 100k-row screen with dozens of features and ~80
work items a knn fit silently takes HOURS where bin-MI takes seconds. This helper measures a tiny
per-column probe on a screen-sized sample, extrapolates the sweep cost
(``t_col * n_cols * (n_work_items + 1 + auto_base_sweeps)``), and swaps ``self.config`` for a bin-estimator copy (with
a WARNING) when the estimate exceeds ``knn_mi_budget_seconds``.

Why not cache the T-side neighbor structure across transforms instead: Kraskov MI is a JOINT-space
kNN query -- for each point the k-th neighbor radius is found in the (x_j, T) joint, then counted in
each marginal, and sklearn's ``mutual_info_regression`` additionally injects seeded jitter into the
features inside the call. The joint structure differs per (x_j, T) pair, T differs per transform,
and sklearn exposes no API to inject precomputed trees; a re-implementation would not be
bit-identical (the internal jitter draw) and could silently alter selection, which is forbidden.
The sound cross-transform reuse points are already wired: the per-column ``MI(y, x_j)`` vector is
computed ONCE per fit (``_per_feat_y_knn_full``), the shrunk-domain ``mi_y_compare`` is memoised per
valid-mask within each base context (``_mi_y_compare_memo``), and the honest-holdout re-score
memoises ``mi_y`` across specs sharing (base columns, valid mask) (``_honest_holdout``).
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Columns timed by the probe; the median damps a one-off cold-import / cache-miss outlier.
_PROBE_N_COLS: int = 2


def estimate_knn_sweep_seconds(
    per_column_seconds: float, n_cols: int, n_work_items: int, auto_base_sweeps: int = 0,
) -> float:
    """Extrapolated whole-fit knn-MI cost: one per-column sweep per work item, the upfront y baseline, and the auto-base
    ranking's sweeps (one MI(y, x) pass plus one per permutation-null draw)."""
    return float(per_column_seconds) * float(max(0, n_cols)) * float(max(0, n_work_items) + 1 + max(0, auto_base_sweeps))


def _planned_bases_and_auto_sweeps(cfg: Any) -> tuple[int, int]:
    """Bases the screen will evaluate and the per-column knn sweeps auto-base ranking runs before it (0 for a fixed list)."""
    cands = getattr(cfg, "base_candidates", "auto")
    if isinstance(cands, str) and cands == "auto":
        n_bases = int(getattr(cfg, "auto_base_top_k", 3) or 3)
        cap = getattr(cfg, "max_base_candidates", None)
        if cap is not None and int(cap) > 0:
            n_bases = min(n_bases, int(cap))
        return n_bases, 1 + int(getattr(cfg, "auto_base_null_perms", 0) or 0)
    return len(list(cands or [])), 0


def maybe_downgrade_knn_estimator(
    self: Any,
    df: Any,
    usable_features: Sequence[str],
    train_idx: np.ndarray,
    y_train: np.ndarray,
) -> None:
    """Swap ``self.config`` to a bin-estimator copy when the estimated knn sweep exceeds the budget.

    Runs before base resolution, because auto-base ranking with its permutation null is itself ~21 knn sweeps per column
    (the guard used to measure only after that work was done). The probe times columns on a random sample of
    ``mi_sample_n`` train rows, the size every later knn call sees. No-op unless ``mi_estimator == "knn"`` and ``knn_mi_auto_downgrade`` is on with a positive
    ``knn_mi_budget_seconds``. Mutates only a per-fit ``model_copy`` (the caller's shared config
    object is never touched -- same pattern as the heavy-tail ``mi_n_strata`` boost).
    """
    cfg = self.config
    if getattr(cfg, "mi_estimator", "bin") != "knn":
        return
    if not getattr(cfg, "knn_mi_auto_downgrade", True):
        return
    budget = float(getattr(cfg, "knn_mi_budget_seconds", 600.0) or 0.0)
    if budget <= 0.0:
        return
    feats = list(usable_features)
    if not feats or train_idx.size < 50:
        return
    rng = np.random.default_rng(int(getattr(cfg, "random_state", 0)))
    pos = np.sort(rng.choice(train_idx.size, size=min(train_idx.size, int(getattr(cfg, "mi_sample_n", 50_000))), replace=False))
    train_idx_screen = train_idx[pos]
    try:
        from sklearn.feature_selection import mutual_info_regression
        from .screening import _extract_column_array

        y = np.asarray(y_train, dtype=np.float64)[pos]
        y_fin = np.isfinite(y)
        times: list[float] = []
        for col in feats[:_PROBE_N_COLS]:
            x = _extract_column_array(df, col, rows=train_idx_screen)
            pair = y_fin & np.isfinite(x)
            if int(pair.sum()) < 50:
                continue
            t0 = timer()
            mutual_info_regression(
                x[pair].reshape(-1, 1), y[pair],
                n_neighbors=int(getattr(cfg, "mi_n_neighbors", 3)),
                random_state=int(getattr(cfg, "random_state", 0)),
            )
            times.append(timer() - t0)
        if not times:
            return
        t_col = float(np.median(times))
    except Exception as exc:  # -- the guard is best-effort; a probe failure keeps knn as configured
        logger.debug("[CompositeTargetDiscovery] knn budget probe failed: %s", exc)
        return
    n_bases, auto_sweeps = _planned_bases_and_auto_sweeps(cfg)
    n_work_items = max(1, n_bases) * max(1, len(getattr(cfg, "transforms", []) or []))
    est = estimate_knn_sweep_seconds(t_col, len(feats), n_work_items, auto_sweeps)
    if est <= budget:
        logger.info(
            "[CompositeTargetDiscovery] knn-MI cost estimate %.1fs (per-column %.3fs x %d cols x %d "
            "work items + %d auto-base sweeps) within budget %.0fs; keeping the knn estimator.",
            est, t_col, len(feats), n_work_items, auto_sweeps, budget,
        )
        return
    logger.warning(
        "[CompositeTargetDiscovery] knn-MI sweep estimated at %.1fs (per-column %.3fs x %d cols x "
        "%d work items + %d auto-base sweeps) exceeds knn_mi_budget_seconds=%.0f -- DOWNGRADING mi_estimator knn -> bin "
        "for this fit, auto-base ranking included (set knn_mi_auto_downgrade=False or raise the budget to keep knn).",
        est, t_col, len(feats), n_work_items, auto_sweeps, budget,
    )
    self.config = cfg.model_copy(update={"mi_estimator": "bin"})
