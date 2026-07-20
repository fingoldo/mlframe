"""Engineered-pair separability scatter: the "corner-biting" cause-effect diagnostic.

A 2-D scatter of the top-2 features coloured by class, annotated with a 2-class Fisher discriminant ratio, so a
reviewer can judge visually whether an engineered feature pair actually separates the classes (the blobs pull to
opposite corners) or merely overlaps. The Fisher ratio is the squared Mahalanobis distance between the two class means
under the pooled within-class covariance: 0 when the class clouds are concentric, large when the means are far apart
relative to the spread.

The only length-n work is the ``separability_score`` accumulate (per-class means + pooled 2x2 within-class scatter in
one njit pass). Feature columns are pulled as narrow float64 ndarray views, never as a frame copy, and the scatter is
seeded-subsampled to a bounded cap before the (KDE-scale) per-point plotting cost.

cProfile verdict (profile_engineered_separability, best-of-3 walltime): the score kernel is a single memory-bound pass;
njit matches the numpy two-pass reduction numerically and wins at every profiled n. See the bench docstring for numbers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec

# Bounded scatter cap: matplotlib per-point primitives scale poorly past ~5k points and the Fisher ratio has converged
# on far fewer, so both the plot and the score run on the same seeded subsample.
DEFAULT_SAMPLE: int = 5_000

try:
    import numba

    @numba.njit(cache=True, fastmath=False)
    def _fisher_2d(z0: np.ndarray, z1: np.ndarray, y: np.ndarray) -> float:
        """njit single-pass-pair 2-class Fisher discriminant ratio: squared Mahalanobis distance between class means under the pooled within-class 2x2 covariance."""
        # Two length-n passes: per-class means, then the pooled within-class 2x2 scatter. J = (m1-m0)^T Sw^-1 (m1-m0),
        # the squared Mahalanobis distance between class means -- the maximal 2-class Fisher separation on this plane.
        n = y.shape[0]
        n0 = 0
        n1 = 0
        s0x = 0.0
        s0y = 0.0
        s1x = 0.0
        s1y = 0.0
        for i in range(n):
            if y[i] > 0.5:
                n1 += 1
                s1x += z0[i]
                s1y += z1[i]
            else:
                n0 += 1
                s0x += z0[i]
                s0y += z1[i]
        if n0 == 0 or n1 == 0:
            return 0.0
        m0x = s0x / n0
        m0y = s0y / n0
        m1x = s1x / n1
        m1y = s1y / n1
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        for i in range(n):
            if y[i] > 0.5:
                dx = z0[i] - m1x
                dy = z1[i] - m1y
            else:
                dx = z0[i] - m0x
                dy = z1[i] - m0y
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        denom = n0 + n1 - 2
        if denom <= 0:
            denom = 1
        cxx = sxx / denom
        cxy = sxy / denom
        cyy = syy / denom
        det = cxx * cyy - cxy * cxy
        if det < 1e-12:
            det = 1e-12
        ddx = m1x - m0x
        ddy = m1y - m0y
        inv_xx = cyy / det
        inv_xy = -cxy / det
        inv_yy = cxx / det
        return ddx * (inv_xx * ddx + inv_xy * ddy) + ddy * (inv_xy * ddx + inv_yy * ddy)

    _HAS_NUMBA = True
except Exception:  # numba unavailable: numpy two-pass reduction with the same pooled-covariance Mahalanobis formula.
    _HAS_NUMBA = False

    def _fisher_2d(z0, z1, y):
        """numpy two-pass-reduction fallback for ``_fisher_2d`` when numba is unavailable; identical pooled-covariance Mahalanobis formula as the njit variant."""
        mask1 = y > 0.5
        n1 = int(mask1.sum())
        n0 = int(y.shape[0] - n1)
        if n0 == 0 or n1 == 0:
            return 0.0
        m1 = np.array([z0[mask1].mean(), z1[mask1].mean()])
        mask0 = ~mask1
        m0 = np.array([z0[mask0].mean(), z1[mask0].mean()])
        d0 = np.column_stack([z0[mask0] - m0[0], z1[mask0] - m0[1]])
        d1 = np.column_stack([z0[mask1] - m1[0], z1[mask1] - m1[1]])
        denom = max(n0 + n1 - 2, 1)
        cov = (d0.T @ d0 + d1.T @ d1) / denom
        det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        if det < 1e-12:
            det = 1e-12
        inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]]) / det
        dd = m1 - m0
        return float(dd @ inv @ dd)


def separability_score(z2: np.ndarray, y: np.ndarray) -> float:
    """2-class Fisher discriminant ratio on the 2-D projection ``z2`` (shape ``(n, 2)``): between- over within-scatter.

    Returns the squared Mahalanobis distance between the two class means under the pooled within-class covariance -- 0
    for concentric class clouds, growing without bound as the means separate relative to the spread.
    """
    z2 = np.asarray(z2, dtype=np.float64)
    z0 = np.ascontiguousarray(z2[:, 0])
    z1 = np.ascontiguousarray(z2[:, 1])
    yv = np.ascontiguousarray(np.asarray(y), dtype=np.float64)
    return float(_fisher_2d(z0, z1, yv))


def _pull_feature(X: Any, feat: Any) -> np.ndarray:
    """Pull one feature column of ``X`` (pandas / polars / ndarray) as a contiguous float64 view, no frame copy."""
    if hasattr(X, "columns") and not isinstance(X, np.ndarray):
        c = X[feat]
        arr = c.to_numpy() if hasattr(c, "to_numpy") else np.asarray(c)
    else:
        arr = np.asarray(X)[:, feat]
    return np.ascontiguousarray(arr, dtype=np.float64)


def _column_names(X: Any) -> List[Any]:
    """Feature keys for ``X``: column names for a frame, integer indices for a bare ndarray."""
    if hasattr(X, "columns") and not isinstance(X, np.ndarray):
        return list(X.columns)
    return list(range(np.asarray(X).shape[1]))


def separability_panel(X: Any, y: np.ndarray, features: Sequence[Any], *, sample: int = DEFAULT_SAMPLE, seed: int = 0) -> ScatterPanelSpec:
    """ScatterPanelSpec of the two named ``features`` coloured by ``y``, titled with the 2-D Fisher separability score.

    Both features are pulled as narrow float64 views and seeded-subsampled to ``sample`` rows; the score is computed on
    the same subsample so the annotated number matches the points drawn.
    """
    f0, f1 = features[0], features[1]
    z0 = _pull_feature(X, f0)
    z1 = _pull_feature(X, f1)
    yv = np.ascontiguousarray(np.asarray(y), dtype=np.float64)
    if z0.shape[0] != yv.shape[0]:
        raise ValueError(f"separability_panel: length mismatch X={z0.shape[0]} y={yv.shape[0]}")
    n = z0.shape[0]
    if n > sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=sample, replace=False))
        z0, z1, yv = z0[idx], z1[idx], yv[idx]
    score = separability_score(np.column_stack([z0, z1]), yv)
    return ScatterPanelSpec(
        x=z0,
        y=z1,
        point_color=yv,
        colormap="coolwarm",
        title=f"Separability (Fisher J={score:.2f}): {f0} vs {f1}",
        xlabel=str(f0),
        ylabel=str(f1),
        point_alpha=0.4,
        colorbar_label="class",
        equal_aspect=False,
    )


def compose_separability_figure(X: Any, y: np.ndarray, features: Optional[Sequence[Any]] = None, *,
                                feature_importances: Optional[np.ndarray] = None, sample: int = DEFAULT_SAMPLE,
                                seed: int = 0, suptitle: str = "Engineered feature separability") -> FigureSpec:
    """One-panel FigureSpec wrapping :func:`separability_panel`, picking the top-2 features by importance when given."""
    if features is None:
        names = _column_names(X)
        if feature_importances is not None:
            order = np.argsort(np.asarray(feature_importances, dtype=np.float64))[::-1]
            features = [names[int(order[0])], names[int(order[1])]]
        else:
            features = [names[0], names[1]]
    panel = separability_panel(X, y, features, sample=sample, seed=seed)
    return FigureSpec(suptitle=suptitle, panels=((panel,),), figsize=(6.0, 5.5))


__all__ = [
    "DEFAULT_SAMPLE",
    "separability_score",
    "separability_panel",
    "compose_separability_figure",
]
