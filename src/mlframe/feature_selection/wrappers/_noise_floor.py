"""Permuted-y NOISE-FLOOR feature-count selection for RFECV-style over-selection.

On a noise-robust GBM the CV-vs-N curve is FLAT, so RFECV's band N-rules (one_se_max/min) land high and keep noise
(measured: RFECV keeps 251/500 on madelon ~ all-features). Knockoff-FDR degenerates on correlated probes. This module
implements a DIFFERENT mechanism that cuts the over-selection: compare the REAL per-N CV curve to a PERMUTED-y
reference curve (same top-N feature sets, y shuffled). The PLATEAU rule stops at the smallest N past which the real
curve's REMAINING climb is within the shuffled-y noise envelope - i.e. everything beyond N* is indistinguishable
from noise.

MEASURED (2026-06-04, post-hoc cut on a LightGBM-gain ranking): madelon 251 -> N*=8..16 (modal 12 over n_perm/seed),
downstream lgbm 0.9135 (N=8) / 0.940 (N=12) vs all-features 0.872 and RFECV-251 0.868 (knn 0.61->0.91 as the noise
probes uncurse the distance metric); synth guard N*=20 keeps base_recall 0.875 at AUC -0.007 (within noise) -> does
NOT over-cut a real-signal curve. The 95th-percentile envelope needs n_perm large enough that the quantile is an interior order statistic
(default 50; n_perm=3 makes it the MAX of 3 draws -> seed-unstable, anti-conservative floor). It is a POST-HOC cut on a feature
RANKING + the data (not wired into RFECV's fit-time N-rule, which has no access to X/y at the dispatch stage and whose
default config times out on wide frames); call it on a fitted model's importance ranking or any valid FI ordering.

The task (binary / multiclass / continuous) is inferred from y: binary -> roc_auc, multiclass -> one-vs-rest macro AUC,
continuous -> R^2; classification uses StratifiedKFold, regression KFold. A stratified class with fewer members than the
fold count still crashes the splitter (an intrinsic sklearn limit) - pass a smaller ``cv`` for such tiny rare classes.

The LITERAL "first N to clear the noise floor" rule is a signal-ONSET detector (fires at N~2, over-cuts); only the
PLATEAU rule is a valid stop - so only that one is exposed.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def noise_floor_plateau(n_grid: Sequence[int], real_curve: np.ndarray, perm_curves: np.ndarray, pct: float = 95.0) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Smallest N past which the real curve's REMAINING climb is within the permuted-y noise envelope.

    For each grid index i, ask whether ANY larger j adds real gain (real[j]-real[i]) exceeding the ``pct`` percentile
    of the permuted incremental gain (perm[:,j]-perm[:,i]) over that same span. N* is the smallest N for which NO
    larger j clears - the parsimonious analogue of RFECV's 'plateau' rule referenced to a permuted-y noise floor
    instead of a fixed 1-SE band, so on a flat (noise-robust-GBM) real curve it stops where signal genuinely flattens.

    Parameters
    ----------
    n_grid : the evaluated feature counts (ascending).
    real_curve : real per-N CV score (same length as n_grid).
    perm_curves : (n_perm, len(n_grid)) permuted-y CV scores.
    pct : noise-envelope percentile (95 = admit <=5% noise-driven climbs).

    Returns
    -------
    (n_star, idx_star, remaining_gain, remaining_env) - n_star is the chosen feature count; remaining_gain[i] /
    remaining_env[i] are the best real / noise incremental gain available beyond i (for diagnostics).
    """
    n_grid = list(n_grid)
    real_curve = np.asarray(real_curve, dtype=float)
    perm_curves = np.atleast_2d(np.asarray(perm_curves, dtype=float))
    G = len(n_grid)
    remaining_gain = np.full(G, -np.inf)
    remaining_env = np.zeros(G)
    star_idx = G - 1
    found = False
    for i in range(G):
        if i == G - 1:
            remaining_gain[i] = 0.0
            remaining_env[i] = 0.0
            break
        # Vectorize the inner j-loop: the noise envelope for ALL larger j is one column-wise percentile over the perm
        # axis (np.percentile along axis 0 == the scalar version applied per column), replacing ~G scalar percentile
        # dispatches per i with a single call. Bit-identical (same draws, same pct, same default linear interpolation).
        rg_all = real_curve[i + 1 :] - real_curve[i]
        env_all = np.percentile(perm_curves[:, i + 1 :] - perm_curves[:, i : i + 1], pct, axis=0)
        excess = rg_all - env_all
        k = int(np.argmax(excess))
        best_excess = float(excess[k])
        remaining_gain[i] = float(rg_all[k])
        remaining_env[i] = float(env_all[k])
        if best_excess <= 0 and not found:
            star_idx = i
            found = True
    return n_grid[star_idx], star_idx, remaining_gain, remaining_env


def noise_floor_band(n_grid: Sequence[int], real_curve: np.ndarray, perm_curves: np.ndarray, pct: float = 95.0) -> int:
    """Smallest N whose real score lies within the permuted-y null's own spread of the real best score.

    Let k be the grid index of the real curve's maximum. For each permutation the gap between the null's best score over
    the grid and its score at k is a draw of how far selection noise can move the curve at one fixed size. The tolerance is
    the ``pct`` percentile of those gaps, and N* is the smallest N whose real score is at least ``real_max - tolerance``.

    Unlike :func:`noise_floor_plateau` this compares levels, not increments. A null curve that falls as features are added
    (the re-ranked null) has large increments that make the plateau rule read noise as signal and keep every feature; the
    band rule reads the same curve as a spread of levels and stays where the real curve stops improving.
    """
    n_grid = list(n_grid)
    real_curve = np.asarray(real_curve, dtype=float)
    perm_curves = np.atleast_2d(np.asarray(perm_curves, dtype=float))
    k = int(np.argmax(real_curve))
    gap = perm_curves.max(axis=1) - perm_curves[:, k]
    tol = float(np.percentile(gap, pct))
    for i, n in enumerate(n_grid):
        if real_curve[i] >= real_curve[k] - tol:
            return int(n)
    return int(n_grid[-1])


def _default_grid(p: int) -> list:
    """Log-ish spaced default sweep of feature-count checkpoints to probe the noise-floor curve, capped and de-duplicated against the actual feature count ``p``."""
    grid = [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400, p]
    return sorted({n for n in grid if 1 <= n <= p})


def _infer_task_scoring(y_arr: np.ndarray):
    """Infer (task, scoring) from the target so the noise-floor curve works on binary, multiclass AND regression y.

    Binary -> roc_auc; multiclass -> roc_auc_ovr (one-vs-rest macro AUC); continuous -> r2. The continuous case also
    drives a KFold (vs StratifiedKFold) at the caller, since stratification on a continuous target is meaningless.
    """
    finite = y_arr[~_isnan_mask(y_arr)]
    n_unique = np.unique(finite).size
    looks_integer = np.allclose(finite, np.round(finite)) if finite.size else False
    if looks_integer and n_unique <= 2:
        return "binary", "roc_auc"
    # Integer-coded with a small alphabet -> multiclass; otherwise treat as a regression target.
    if looks_integer and n_unique <= max(20, int(0.05 * finite.size)):
        return "multiclass", "roc_auc_ovr"
    return "regression", "r2"


def _isnan_mask(arr: np.ndarray) -> np.ndarray:
    """NaN mask that is safe on integer / object dtype arrays (np.isnan raises on non-float)."""
    a = np.asarray(arr)
    if a.dtype.kind in "fc":
        return np.asarray(np.isnan(a))
    return np.zeros(a.shape, dtype=bool)


def _rerank_by_model(estimator_factory: Callable, X: Any, y: Any, ranking: Sequence, max_rows: int, seed: int) -> list:
    """The features of ``ranking`` re-ordered most-important-first by a fresh model fitted on ``y`` (feature_importances_ or |coef_|)."""
    is_df = hasattr(X, "columns")
    n = len(X)
    rows = np.sort(np.random.default_rng(seed).choice(n, size=min(n, int(max_rows)), replace=False))
    cols = list(ranking)
    Xs = X[cols].iloc[rows] if is_df else X[rows][:, cols]
    ys = y.iloc[rows] if hasattr(y, "iloc") else np.asarray(y)[rows]
    model = estimator_factory().fit(Xs, ys)
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64)  # Intentional float64: importances are summed in double
    else:
        imp = np.abs(np.asarray(model.coef_, dtype=np.float64)).reshape(-1, len(cols)).sum(axis=0)  # Intentional float64: importances are summed in double
    return [cols[i] for i in np.argsort(-imp, kind="stable")]


_RERANK_MAX_ROWS = 20_000


def cv_curve(estimator_factory: Callable, X: Any, y: Any, ranking: Sequence, n_grid: Sequence[int], cv: Any,
             permute: bool = False, n_perm: int = 1, base_seed: int = 100, scoring: str = "roc_auc",
             rerank_rows: Optional[int] = None) -> Any:
    """Per-N CV score (``scoring``) on the top-N features by ``ranking``. If permute, average over n_perm shuffles of y.

    ``rerank_rows``: when set and ``permute``, every shuffle re-ranks the features of ``ranking`` with a model fitted on the shuffled y
    (on at most ``rerank_rows`` random rows), so the null curve pays the same selection-on-the-labels optimism the real ranking did.

    estimator_factory() must return a FRESH unfitted estimator (a fresh model per N keeps the curve honest). Returns
    the real curve (permute=False) or (mean_curve, all_perm_curves) (permute=True).
    """
    from sklearn.model_selection import cross_val_score
    import pandas as pd

    is_df = hasattr(X, "columns")
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    curves = []
    for p in range(n_perm if permute else 1):
        if permute:
            rng = np.random.default_rng(base_seed + p)
            yy = rng.permutation(y_arr)
        else:
            yy = y_arr
        yy = pd.Series(yy, index=X.index) if is_df else yy
        ranking_p = _rerank_by_model(estimator_factory, X, yy, ranking, rerank_rows, base_seed + p) if permute and rerank_rows is not None else ranking
        row = []
        for n in n_grid:
            cols = list(ranking_p[:n])
            Xn = X[cols] if is_df else X[:, cols]
            row.append(float(np.mean(cross_val_score(estimator_factory(), Xn, yy, cv=cv, scoring=scoring, n_jobs=1))))
        curves.append(row)
    arr = np.asarray(curves)
    return (arr.mean(axis=0), arr) if permute else arr[0]


def select_features_noise_floor(estimator_factory: Callable, X: pd.DataFrame | np.ndarray, y: Any, ranking: Sequence,
                                n_grid: Optional[Sequence[int]] = None, cv: Any = 3, n_perm: int = 50,
                                pct: float = 95.0, random_state: int = 0, rerank: str = "none", stop_rule: str = "band") -> dict[str, Any]:
    """Post-hoc permuted-y noise-floor cut of an over-selected feature RANKING. Returns the top-N* features.

    Parameters
    ----------
    estimator_factory : callable -> fresh unfitted classifier (e.g. ``lambda: LGBMClassifier(...)``).
    X : DataFrame or ndarray; y : array-like target. Binary (roc_auc), multiclass (one-vs-rest macro AUC) and
        continuous (R^2) targets are all supported; the task is inferred from y and drives both the scorer and the
        splitter (StratifiedKFold for classification, KFold for regression).
    ranking : feature names (DataFrame) or integer indices (ndarray), MOST-IMPORTANT-FIRST (e.g. from a fitted
        model's importances, or RFECV's elimination order). The cut keeps a leading prefix of this ranking.
    n_grid : feature counts to evaluate; default a log-ish grid up to len(ranking).
    cv : int folds or a CV splitter; n_perm : permutations for the noise envelope. Default 50; must be large enough that
        the ``pct`` percentile is an interior order statistic (>=ceil(100/(100-pct)), e.g. >=20 at pct=95) - a tiny
        n_perm makes the envelope a high-variance sample maximum and the floor seed-unstable / anti-conservative.
    pct : noise-envelope percentile.
    stop_rule : ``"band"`` (default) cuts at the smallest N within the permuted-null spread of the real best score
        (:func:`noise_floor_band`). ``"plateau"`` is the earlier increment rule (:func:`noise_floor_plateau`), kept for callers
        that depend on it.
    rerank : ``"none"`` (default) reuses the real ranking for every shuffle. ``"permutation"`` re-ranks the features with a model fitted on each
        shuffled y (on at most 20,000 rows) so the null curve carries the same selection-on-the-labels optimism as the real ranking. Opt-in:
        the re-ranked null starts high and falls as noise features are added. The default ``stop_rule="band"`` reads that null as a spread of
        levels, so the re-ranked mode no longer keeps the noise features the plateau rule kept.

    Returns
    -------
    dict with keys: ``selected`` (top-N* feature ids), ``n_star``, ``n_grid``, ``real_curve``, ``perm_mean``.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    ranking = list(ranking)
    if not ranking:
        return dict(selected=[], n_star=0, n_grid=[], real_curve=np.array([]), perm_mean=np.array([]))
    if n_grid is None:
        n_grid = _default_grid(len(ranking))
    else:
        n_grid = sorted({int(n) for n in n_grid if 1 <= int(n) <= len(ranking)})
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    task, scoring = _infer_task_scoring(y_arr)
    if isinstance(cv, int):
        # Stratification is only meaningful for classification; a continuous target must use plain KFold.
        if task == "regression":
            splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        splitter = cv
    if stop_rule not in ("band", "plateau"):
        raise ValueError(f"select_features_noise_floor: stop_rule must be 'band' or 'plateau', got {stop_rule!r}")
    if rerank not in ("permutation", "none"):
        raise ValueError(f"select_features_noise_floor: rerank must be 'permutation' or 'none', got {rerank!r}")
    do_rerank = rerank == "permutation"
    real_curve = cv_curve(estimator_factory, X, y, ranking, n_grid, splitter, permute=False, scoring=scoring)
    perm_mean, perm_curves = cv_curve(estimator_factory, X, y, ranking, n_grid, splitter,
                                      permute=True, n_perm=n_perm, base_seed=100 + random_state, scoring=scoring,
                                      rerank_rows=_RERANK_MAX_ROWS if do_rerank else None)
    if stop_rule == "band":
        n_star = noise_floor_band(n_grid, real_curve, perm_curves, pct=pct)
    else:
        n_star, _, _, _ = noise_floor_plateau(n_grid, real_curve, perm_curves, pct=pct)
    # The envelope at each grid point is the ``pct`` percentile of ``n_perm`` permuted incremental gains. With a tiny
    # n_perm the percentile is a high-variance extreme order statistic (n_perm=3 makes the 95th pct literally the MAX of
    # 3 draws), so the floor jumps around with the seed and is anti-conservative. n_perm must be large enough that the
    # ``pct`` percentile is an interpolated interior order statistic: require at least ceil(100/(100-pct)) draws so the
    # requested quantile is not the sample maximum (e.g. >=20 for pct=95), and recommend >=50 for a stable estimate.
    min_perm = math.ceil(100.0 / max(100.0 - pct, 1e-9))
    if n_perm < min_perm:
        logger.warning(
            "select_features_noise_floor: n_perm=%d is too small for the %.1fth-percentile noise floor (the quantile "
            "degenerates to an extreme order statistic); use >=%d (>=50 recommended) for a low-variance envelope.",
            n_perm, pct, max(min_perm, 50),
        )
    return dict(selected=ranking[:n_star], n_star=int(n_star), n_grid=n_grid, real_curve=real_curve, perm_mean=perm_mean)
