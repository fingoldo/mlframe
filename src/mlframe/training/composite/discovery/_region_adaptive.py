"""Region-adaptive composite-target transform selection (HONEST EXPERIMENT prototype).

Idea
----
The base-vs-y relation a composite transform must absorb is not always the same
shape across the whole base range. A target may depend *linearly* on ``base``
where ``base < 0`` and *quadratically* where ``base > 0``. A single global
transform (the best ``linear_residual`` OR the best ``monotonic_residual`` over
ALL rows) must compromise across both regimes and leaves structure in the
residual ``T`` in at least one region.

This prototype partitions the base range into ``K`` quantile regions (fitted
TRAIN-ONLY), lets discovery pick the *best transform per region* (e.g.
``linear_residual`` where the relation is linear, ``monotonic_residual`` where
it curves), and a :class:`RegionAdaptiveSpec` applies the matching inverse per
row by routing each row to its region via the frozen quantile edges. Region
fits use a held-out OOF split so the per-region transform choice is not
selected on the same rows it is scored on (no leakage / no self-selection).

Self-contained
--------------
This is a research prototype: it consumes the public ``Transform`` objects from
the registry (their ``fit`` / ``forward`` / ``inverse`` are reused verbatim) but
does NOT register a new ``Transform`` and does NOT touch ``__init__`` /
``registry``. Wiring into ``CompositeTargetDiscovery`` as an opt-in
``screening='region_adaptive'`` mode is gated on the verdict below.

VERDICT (measured 2026-06-11, ``_benchmarks/bench_region_adaptive.py``)
----------------------------------------------------------------------
On a synthetic with a region-DEPENDENT y-base relation (linear for ``base<0``,
quadratic for ``base>0``, plus Gaussian noise), region-adaptive transform
selection is compared against the single best global transform by OOS RMSE of
the reconstructed ``y`` (downstream GBM trained on ``T``, inverse-mapped back):

    global best transform (monotonic_residual)  OOS RMSE  : see bench output
    region-adaptive (K=4, per-region best)       OOS RMSE  : see bench output

The bench prints the exact numbers and the win ratio. ``test_region_adaptive.py``
pins the measured win as a biz_value floor. Outcome is recorded in the run
summary; if the win does not clear the floor on the wider grid this module
stays a committed-but-rejected prototype (REJECTED != DELETED): the code, the
bench and the verdict remain so a future re-test on different data/HW is one
command away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# Reuse the registry transforms verbatim (fit/forward/inverse), no new Transform.
from ..transforms.registry import _TRANSFORMS_REGISTRY

__all__ = [
    "RegionAdaptiveSpec",
    "assign_regions",
    "fit_region_adaptive",
    "DEFAULT_REGION_CANDIDATES",
]

# Per-region candidate transforms. Kept small + bivariate so each region's
# fit is cheap and the per-region winner is interpretable (linear vs curved
# vs robust-linear). All three share the (y, base) signature.
DEFAULT_REGION_CANDIDATES: tuple[str, ...] = (
    "linear_residual",
    "monotonic_residual",
    "polynomial_residual_deg2",
)


@dataclass(frozen=True, eq=False)
class RegionAdaptiveSpec:
    """Frozen region-adaptive composite spec.

    ``edges`` are the ``K-1`` interior quantile cut points of train ``base``;
    :func:`assign_regions` buckets any row's base into ``[0, K)`` via
    ``np.searchsorted`` so predict-time routing exactly mirrors fit-time.

    ``region_transforms[k]`` / ``region_params[k]`` are the winning transform
    name + its fitted params for region ``k``. ``forward`` / ``inverse`` route
    each row through its region's transform; the result is a single ``T`` array
    (resp. reconstructed ``y``) that the downstream learner consumes exactly
    like any other composite target.
    """

    base_column: str
    target_col: str
    edges: tuple[float, ...]
    region_transforms: tuple[str, ...]
    region_params: tuple[dict[str, Any], ...]
    region_oof_scores: tuple[float, ...] = field(default=())
    name: str = "region_adaptive"

    @property
    def k(self) -> int:
        return len(self.region_transforms)

    def forward(self, y: np.ndarray, base: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        base = np.asarray(base, dtype=np.float64)
        reg = assign_regions(base, self.edges)
        out = np.empty_like(y)
        for k in range(self.k):
            m = reg == k
            if not m.any():
                continue
            tr = _TRANSFORMS_REGISTRY[self.region_transforms[k]]
            out[m] = tr.forward(y[m], base[m], self.region_params[k])
        return out

    def inverse(self, t_hat: np.ndarray, base: np.ndarray) -> np.ndarray:
        t_hat = np.asarray(t_hat, dtype=np.float64)
        base = np.asarray(base, dtype=np.float64)
        reg = assign_regions(base, self.edges)
        out = np.empty_like(t_hat)
        for k in range(self.k):
            m = reg == k
            if not m.any():
                continue
            tr = _TRANSFORMS_REGISTRY[self.region_transforms[k]]
            out[m] = tr.inverse(t_hat[m], base[m], self.region_params[k])
        return out


def assign_regions(base: np.ndarray, edges: Sequence[float] | np.ndarray) -> np.ndarray:
    """Route each ``base`` value to a region index in ``[0, len(edges)]``.

    Uses the frozen interior quantile ``edges`` (train-fitted). ``searchsorted``
    with ``side='right'`` puts a value exactly on an edge into the upper region,
    matching the fit-time bucketing; out-of-range predict values clip to the
    edge regions (region 0 below the first edge, region K-1 above the last).
    """
    base = np.asarray(base, dtype=np.float64)
    if len(edges) == 0:
        return np.zeros(base.shape[0], dtype=np.int64)
    return np.searchsorted(np.asarray(edges, dtype=np.float64), base, side="right").astype(np.int64)


def _quantile_edges(base: np.ndarray, k: int) -> np.ndarray:
    """Interior quantile cut points splitting ``base`` into ``k`` regions.

    Deduped so a heavily-tied base does not create empty regions; if dedup
    collapses below ``k-1`` edges the effective region count shrinks (the spec
    just has fewer regions, still valid).
    """
    if k <= 1:
        return np.empty(0, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, k + 1)[1:-1]
    edges = np.quantile(np.asarray(base, dtype=np.float64), qs)
    return np.unique(edges)


def _oof_score_transform(
    tr_name: str, y: np.ndarray, base: np.ndarray, n_folds: int, rng: np.random.Generator,
) -> tuple[float, dict[str, Any]]:
    """Score one candidate transform on (y, base) by OOF residual-variance reduction.

    A transform is good in a region when, after fitting (alpha/beta/PCHIP) on the
    in-fold rows and forwarding the held-out rows, the resulting ``T`` has small
    variance relative to ``y`` (it has absorbed the base relation). Score =
    ``1 - var(T_oof) / var(y_oof)`` (higher better, capped at the fold mean). The
    OOF split prevents a flexible transform (deg2 / PCHIP) from winning purely by
    over-fitting in-region noise. Returns (score, full-region-fit-params).
    """
    n = len(y)
    tr = _TRANSFORMS_REGISTRY[tr_name]
    if n < max(2 * n_folds, 8):
        # Too few rows for OOF: fit + score in-sample (region degenerate).
        params = tr.fit(y, base)
        t = tr.forward(y, base, params)
        vy = float(np.var(y)) or 1.0
        return 1.0 - float(np.var(t)) / vy, params
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    scores = []
    for f in range(n_folds):
        te = folds[f]
        trn = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        if len(trn) < 2 or len(te) < 1:
            continue
        try:
            params = tr.fit(y[trn], base[trn])
            t_te = tr.forward(y[te], base[te], params)
        except Exception:  # nosec B112 - best-effort path
            continue
        if not np.all(np.isfinite(t_te)):
            continue
        vy = float(np.var(y[te])) or 1.0
        scores.append(1.0 - float(np.var(t_te)) / vy)
    if not scores:
        params = tr.fit(y, base)
        return -np.inf, params
    full_params = tr.fit(y, base)
    return float(np.mean(scores)), full_params


def fit_region_adaptive(
    y: np.ndarray,
    base: np.ndarray,
    *,
    base_column: str = "base",
    target_col: str = "y",
    k: int = 4,
    candidates: Sequence[str] = DEFAULT_REGION_CANDIDATES,
    n_folds: int = 3,
    random_state: int = 0,
) -> RegionAdaptiveSpec:
    """Fit a :class:`RegionAdaptiveSpec` on TRAIN ``(y, base)``.

    1. Cut ``base`` into ``k`` quantile regions (frozen edges).
    2. In each region, OOF-score every candidate transform and keep the winner
       + its full-region fitted params.
    The per-region OOF scoring makes the winner choice honest: a curved
    transform only wins a region when it generalises there, not when it
    over-fits the in-region noise.
    """
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(base)
    y, base = y[finite], base[finite]
    edges = _quantile_edges(base, k)
    reg = assign_regions(base, edges)
    k_eff = len(edges) + 1
    rng = np.random.default_rng(random_state)
    tr_names: list[str] = []
    tr_params: list[dict[str, Any]] = []
    tr_scores: list[float] = []
    for kk in range(k_eff):
        m = reg == kk
        yk, bk = y[m], base[m]
        if len(yk) < 2:
            # Empty/degenerate region: fall back to a no-op linear_residual fit.
            best_name = "linear_residual"
            best_params = _TRANSFORMS_REGISTRY[best_name].fit(yk if len(yk) else y[:2], bk if len(bk) else base[:2])
            tr_names.append(best_name)
            tr_params.append(best_params)
            tr_scores.append(-np.inf)
            continue
        # Seed best_params from a guaranteed full-region linear_residual fit so it is NEVER None: when every candidate scores -inf (degenerate region), `score > best_score` is `-inf > -inf == False`, leaving best_params None -> stored -> tr.forward/inverse hits None -> TypeError at predict.
        best_name = "linear_residual"
        best_params = _TRANSFORMS_REGISTRY[best_name].fit(yk, bk)
        best_score = -np.inf
        for cand in candidates:
            score, params = _oof_score_transform(cand, yk, bk, n_folds, rng)
            if score > best_score:
                best_score, best_name, best_params = score, cand, params
        tr_names.append(best_name)
        tr_params.append(best_params)
        tr_scores.append(best_score)
    return RegionAdaptiveSpec(
        base_column=base_column,
        target_col=target_col,
        edges=tuple(float(e) for e in edges),
        region_transforms=tuple(tr_names),
        region_params=tuple(tr_params),
        region_oof_scores=tuple(tr_scores),
    )
