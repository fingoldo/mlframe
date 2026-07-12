"""Layer 63 (2026-05-31): THREE-GATE + K-fold OOF MI ranking for hybrid orth-poly FE.

Why this layer
--------------

Layer 21 / Layer 62 select engineered orth-poly columns via a TWO-gate
rule on a PLUG-IN MI estimator that sees the SAME rows the column was
engineered from. Two failure modes follow:

  1. **Plug-in MI is biased UP on small n.** The estimator overestimates
     the true MI by roughly ``(K - 1) / (2n)`` per source / engineered
     column (Miller-Madow first-order correction). On rare-imbalance or
     small-fold settings the bias is large enough to push a noise-driven
     candidate past the absolute MAD floor: it's pure inflation, but the
     floor doesn't know that. K-fold OOF MI -- score on held-out folds and
     average -- naturally regularises the bias because the held-out rows
     never contributed to the column's bin edges.

  2. **Two gates miss "duplicate signal" candidates.** ``y = sign(x^2 -
     1)`` is already covered by ``x__He2``. A second basis column like
     ``x__T2`` or even another quadratic form has near-identical marginal
     MI (both monotone in ``|x|``) and clears both gates -- but it adds
     no new information conditional on the already-selected support.
     The third gate ``CMI(candidate; y | support) >= cmi_min`` kills
     exactly this case: once ``x__He2`` is in the support, ``CMI(x__T2;
     y | x__He2)`` collapses near zero, and the candidate is dropped.

Layer 63 combines both ideas into a single selection criterion:

  * **OOF MI baseline + engineered.** ``score_features_by_kfold_oof_mi``
    splits the rows into K folds, scores baseline_mi / engineered_mi on
    each held-out fold using bin edges fitted on the training fold, then
    averages. The returned uplift is OOF / OOF (both regularised).

  * **Three-gate selection.**
      Gate 1 (relative): ``uplift_oof >= min_uplift`` (default 1.05).
      Gate 2 (absolute): ``engineered_oof_mi >= max(legacy_floor,
                                                    MAD_noise_floor)``
                         -- same MAD construction as Layer 21 but on the
                         OOF distribution.
      Gate 3 (conditional): when ``current_support`` is non-empty,
                         ``CMI(candidate; y | support) >= cmi_min``.
                         When the support is empty Gate 3 is skipped
                         (CMI reduces to marginal MI which Gate 1 already
                         covers).

  * **Top-K by uplift_oof** among gate-survivors.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe (same kind
Layer 21 emits): the engineered VALUES are bit-equal to Layer 21 because
``generate_univariate_basis_features`` is shared. Only the SELECTION rule
changes. Replay therefore reuses the existing ``_apply_orth_univariate``
path -- no new recipe kind required.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_three_gate_enable=True``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._orthogonal_univariate_fe import (
    _mi_classif_batch,
    generate_univariate_basis_features,
)
from ._mi_greedy_cmi_fe import (
    _cmi_from_binned,
    _quantile_bin,
    _renumber_joint,
)

logger = logging.getLogger(__name__)

# CPX12b (2026-06-23): the batched-across-columns train-edge binner
# (_bin_with_train_edges_batched) shares one np.quantile partition per fold across
# all columns. Measured (bench_oof_three_gate_train_edge_binning.py): a clean win at
# small/mid per-fold train sizes (2.23x @ n=2k/p=50, 1.29x @ n=5k/p=100) that fades
# to neutral by ~16k train rows and turns into a slight regression (~0.93x @ 50k/100)
# as the batched quantile loses the per-column cache locality of the scalar path.
# We therefore GATE on per-fold TRAIN-row count: batch below the threshold, fall back
# to the bit-identical scalar per-column path above it. Override via env var for
# hardware retuning.
_OOF_BATCH_BINNING_MAX_TRAIN_ROWS = int(os.environ.get("MLFRAME_OOF_BATCH_BINNING_MAX_TRAIN_ROWS", "16000"))

__all__ = [
    "score_features_by_kfold_oof_mi",
    "hybrid_orth_mi_three_gate_fe",
    "hybrid_orth_mi_three_gate_fe_with_recipes",
]


def _coerce_y_int64(y) -> np.ndarray:
    """Dense int64 class labels. Non-integer y is densified via
    ``np.unique(return_inverse=...)`` rather than truncated with
    ``.astype(int64)`` -- plain truncation merges distinct labels and destroys
    continuous-y signal (everything in [0, 1) collapses to class 0)."""
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def _stratified_fold_indices(
    y_int: np.ndarray, n_folds: int, seed: int,
) -> list[np.ndarray]:
    """Build K stratified test-fold index arrays.

    Stratification preserves per-class proportions so each fold has at
    least one example of each class whenever possible. Falls back to
    plain shuffled K-fold when a class has fewer than ``n_folds``
    examples (avoids 0-size strata).
    """
    n = int(y_int.size)
    n_folds_eff = max(2, int(n_folds))
    rng = np.random.default_rng(int(seed))
    classes, inv = np.unique(y_int, return_inverse=True)
    # Fallback: plain K-fold if any class is too small to stratify.
    counts = np.bincount(inv)
    if classes.size < 2 or counts.min() < n_folds_eff:
        perm = rng.permutation(n)
        return [perm[i::n_folds_eff] for i in range(n_folds_eff)]
    # Stratified: distribute each class's indices round-robin into folds.
    fold_lists: list[list[int]] = [[] for _ in range(n_folds_eff)]
    for k in range(classes.size):
        idx_k = np.where(inv == k)[0]
        idx_k = rng.permutation(idx_k)
        for j, gi in enumerate(idx_k):
            fold_lists[j % n_folds_eff].append(int(gi))
    return [np.array(sorted(f), dtype=np.int64) for f in fold_lists]


def _bin_with_train_edges(
    train_vals: np.ndarray, test_vals: np.ndarray, nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-bin train_vals at ``nbins`` equi-frequency cuts; apply the
    SAME cut edges to test_vals (np.searchsorted with right=False).

    Returns ``(train_bins, test_bins)`` as int64 arrays. Edges fitted on
    train only -> no leakage from test rows into the binning step.
    Constant columns (only 1 unique edge) collapse to bin 0 everywhere.
    """
    train = np.ascontiguousarray(train_vals, dtype=np.float64)
    test = np.ascontiguousarray(test_vals, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, int(nbins) + 1)[1:-1]
    if qs.size == 0 or train.size == 0:
        return (
            np.zeros(train.size, dtype=np.int64),
            np.zeros(test.size, dtype=np.int64),
        )
    edges = np.quantile(train, qs)
    # Dedup edges so an all-constant column doesn't blow up.
    edges = np.unique(edges)
    if edges.size == 0:
        return (
            np.zeros(train.size, dtype=np.int64),
            np.zeros(test.size, dtype=np.int64),
        )
    train_bins = np.searchsorted(edges, train, side="right").astype(np.int64)
    test_bins = np.searchsorted(edges, test, side="right").astype(np.int64)
    return train_bins, test_bins


def _bin_with_train_edges_batched(
    train_arr: np.ndarray, test_arr: np.ndarray, nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched-across-columns equivalent of :func:`_bin_with_train_edges`.

    For a single fold, fit equi-frequency quantile edges on the TRAIN rows of
    ALL columns at once via one ``np.quantile(train_arr, qs, axis=0)`` call,
    then apply those train-fitted edges (per-column ``np.searchsorted``) to both
    the train and test rows of every column. The edges depend ONLY on the train
    rows -- test rows never influence a cut boundary -- so this is the SAME
    leakage-free K-fold OOF binning the scalar per-column path produces, just
    vectorised across columns (one quantile partition for the batch instead of
    ``p`` separate calls).

    Bit-identical to looping :func:`_bin_with_train_edges` over columns: the inner
    quantile cuts ``qs = linspace(0, 1, nbins+1)[1:-1]``, the per-column
    ``np.unique`` edge dedup, and the ``side='right'`` searchsorted convention all
    match exactly, including the degenerate empty-edges (constant column -> bin 0)
    and empty-quantile (nbins<=1) cases.

    Parameters
    ----------
    train_arr, test_arr : (n_train, p) / (n_test, p) float arrays
        Positionally column-aligned. Edges are fitted on ``train_arr`` only.
    """
    train = np.ascontiguousarray(train_arr, dtype=np.float64)
    test = np.ascontiguousarray(test_arr, dtype=np.float64)
    n_train = train.shape[0]
    p = train.shape[1] if train.ndim == 2 else 0
    n_test = test.shape[0]
    train_bins = np.zeros((n_train, p), dtype=np.int64)
    test_bins = np.zeros((n_test, p), dtype=np.int64)
    qs = np.linspace(0.0, 1.0, int(nbins) + 1)[1:-1]
    if qs.size == 0 or n_train == 0 or p == 0:
        return train_bins, test_bins
    # One batched quantile partition for all columns -> (qs.size, p). This is the
    # whole point: amortise the partition-based selector across columns instead of
    # p separate np.quantile calls, while keeping the edges train-row-only.
    edges_all = np.quantile(train, qs, axis=0)  # shape (qs.size, p)
    for j in range(p):
        edges = np.unique(edges_all[:, j])  # per-column dedup, matches scalar path
        if edges.size == 0:
            continue
        train_bins[:, j] = np.searchsorted(edges, train[:, j], side="right")
        test_bins[:, j] = np.searchsorted(edges, test[:, j], side="right")
    return train_bins, test_bins


def _fold_test_bins(
    arr: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_cols: int,
    nbins: int,
) -> np.ndarray:
    """Test-row bin codes for all columns of one fold, train-edges only.

    Dispatches between the batched-across-columns binner (small/mid folds, a
    measured win) and the scalar per-column path (large folds, where the batched
    quantile loses cache locality). Both paths fit edges on the fold's TRAIN rows
    only and are bit-identical -- the gate only affects performance, never the
    OOF MI values. See :data:`_OOF_BATCH_BINNING_MAX_TRAIN_ROWS`.
    """
    if train_idx.size <= _OOF_BATCH_BINNING_MAX_TRAIN_ROWS:
        _, test_bins = _bin_with_train_edges_batched(
            arr[train_idx, :], arr[test_idx, :], nbins=nbins,
        )
        return test_bins
    test_bins = np.zeros((test_idx.size, n_cols), dtype=np.int64)
    for j in range(n_cols):
        _, tb = _bin_with_train_edges(
            arr[train_idx, j], arr[test_idx, j], nbins=nbins,
        )
        test_bins[:, j] = tb
    return test_bins


def _mi_from_binned_xy(x_bin: np.ndarray, y_bin: np.ndarray, *, clip_zero: bool = True) -> float:
    """Miller-Madow-corrected plug-in MI ``H(X) + H(Y) - H(X, Y)`` from
    binned int arrays.

    The raw plug-in MI on a finite sample carries a positive bias from
    the second-order Miller-Madow expansion. On the small held-out folds
    K-fold OOF creates (n_test ~ n / K), that bias dominates a noise
    column's true MI -- without the correction OOF MI on noise reports
    VALUES STRICTLY LARGER than the full-frame plug-in MI, defeating
    the regularisation claim of Gate 2.

    We use the *maximum-support* form of Miller-Madow:
        bias ~= (K_X * K_Y - K_X - K_Y + 1) / (2 n)
    where K_X and K_Y are the MAXIMUM possible support sizes (==
    number of distinct values that COULD appear), not the observed
    sizes. On the small folds the observed K_XY is typically below
    K_X * K_Y due to under-sampling of joint cells, and using observed
    K_XY undershoots the true bias -- the very phenomenon that makes
    naive plug-in MI inflate on small samples. Using the maximum
    support gives the correct asymptotic correction and matches the
    Treves-Panzeri / Paninski analysis for small-n estimators.
    """
    if x_bin.size == 0 or y_bin.size == 0:
        return 0.0
    n = float(x_bin.size)
    xy, _ = _renumber_joint(x_bin, y_bin)

    def _h_and_k(arr: np.ndarray) -> tuple[float, int]:
        """Plug-in entropy (nats) and observed support size of a binned int array via bincount frequencies."""
        if arr.size == 0:
            return 0.0, 0
        counts = np.bincount(arr)
        counts = counts[counts > 0]
        if counts.size == 0:
            return 0.0, 0
        p = counts.astype(np.float64) / n
        return float(-np.sum(p * np.log(p))), int(counts.size)

    h_x, k_x = _h_and_k(x_bin)
    h_y, k_y = _h_and_k(y_bin)
    h_xy, _ = _h_and_k(xy)
    mi_raw = h_x + h_y - h_xy
    # Use the MAXIMUM joint support (k_x * k_y) rather than the OBSERVED
    # joint count -- on small folds observed joint count is depressed by
    # under-sampling, which under-corrects the bias and lets noise MI
    # inflate.
    if n > 0 and k_x > 0 and k_y > 0:
        k_xy_max = k_x * k_y
        bias = (k_xy_max - k_x - k_y + 1) / (2.0 * n)
        mi_corrected = mi_raw - bias
    else:
        mi_corrected = mi_raw
    # Per-fold clipping at zero would inject a positive bias when we
    # subsequently average folds (max(0, noisy_estimator) -> E > 0 even
    # when E[estimator] = 0). The caller clips the K-fold AVERAGE
    # instead by setting clip_zero=False on the per-fold call.
    if clip_zero:
        return max(0.0, mi_corrected)
    return mi_corrected


def score_features_by_kfold_oof_mi(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 0,
    nbins: int = 10,
) -> pd.DataFrame:
    """K-fold OOF MI uplift scorer.

    For each of K stratified folds:
      * Fit per-column equi-frequency quantile bin edges on the TRAIN
        rows of raw_X and engineered_X (concat helper internally).
      * Apply the train edges to the TEST rows; compute plug-in MI on
        the held-out fold for every raw column and every engineered
        column.

    Average the K per-fold MI estimates per column. The OOF MI is by
    construction smaller (or at most equal) to the plug-in MI on the
    full frame because the held-out rows did not influence the bin
    boundaries -- this regularises away the plug-in bias that drives
    the Layer 21 false-positive incidents.

    Returns
    -------
    DataFrame columns:
        engineered_col, source_col, baseline_mi_oof, engineered_mi_oof,
        uplift_oof
    sorted by ``uplift_oof`` descending.

    Notes
    -----
    * Joint subsampling at the index level: raw_X / engineered_X / y must
      share positional alignment (Layer 62 pattern). The function raises
      if row counts disagree -- silent coercion would hide real bugs.
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_kfold_oof_mi: raw_X has {len(raw_X)} rows "
            f"but engineered_X has {len(engineered_X)}; positional joint "
            f"K-fold splitting requires aligned indices."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(f"score_features_by_kfold_oof_mi: raw_X has {len(raw_X)} rows " f"but y has {len(np.asarray(y))}; aligned indices required.")

    y_arr = _coerce_y_int64(y)
    raw_cols = list(raw_X.columns)
    eng_cols = list(engineered_X.columns)
    n = len(raw_X)
    n_folds_eff = max(2, int(n_folds))
    if n < 2 * n_folds_eff:
        logger.warning(
            "score_features_by_kfold_oof_mi: n=%d < 2 * n_folds=%d; " "falling back to n_folds=2 to retain at least one held-out " "row per fold.",
            n,
            n_folds_eff,
        )
        n_folds_eff = 2

    src_map = {eng_name: (eng_name.split("__", 1)[0] if "__" in eng_name else eng_name) for eng_name in eng_cols}
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    raw_arr = raw_X.to_numpy(dtype=_dt)
    eng_arr = engineered_X.to_numpy(dtype=_dt)

    fold_test_idx = _stratified_fold_indices(y_arr, n_folds_eff, seed)
    raw_fold_mis: list[np.ndarray] = []
    eng_fold_mis: list[np.ndarray] = []

    for test_idx in fold_test_idx:
        if test_idx.size == 0:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        # Guard against degenerate single-class folds (rare-imbalance
        # frames). Skip the fold contribution to MI when y_test has < 2
        # classes -- can't compute joint entropy meaningfully.
        y_test = y_arr[test_idx]
        if np.unique(y_test).size < 2:
            continue
        # Bin y_test as a dense renumbering of classes (consistent with
        # _quantile_bin's int output for the support).
        _, y_test_bin = np.unique(y_test, return_inverse=True)
        y_test_bin = y_test_bin.astype(np.int64)

        # Raw columns. NOTE: clip_zero=False per fold -- clipping each
        # fold to >=0 before averaging would inject a positive bias
        # whenever the true MI is zero (E[max(0, noisy)] > 0). We clip
        # the K-fold AVERAGE below, which is the correct unbiased OOF
        # estimator.
        #
        # CPX12b (2026-06-23): bin every column for this fold via a single,
        # train-row-only quantile partition shared across COLUMNS (gated on
        # per-fold train size -- see _OOF_BATCH_BINNING_MAX_TRAIN_ROWS). The edges
        # are still fitted on TRAIN rows only (leakage-free K-fold OOF) and the
        # output is bit-identical to the prior per-column _bin_with_train_edges
        # loop; only the quantile partition is shared. Above the threshold the
        # batched quantile loses cache locality, so we keep the scalar fallback.
        raw_test_bins = _fold_test_bins(
            raw_arr, train_idx, test_idx, len(raw_cols), nbins,
        )
        raw_mi_k = np.array(
            [_mi_from_binned_xy(raw_test_bins[:, j], y_test_bin, clip_zero=False) for j in range(len(raw_cols))],
            dtype=np.float64,
        )
        raw_fold_mis.append(raw_mi_k)

        # Engineered columns (same per-fold non-clipping rule).
        eng_test_bins = _fold_test_bins(
            eng_arr, train_idx, test_idx, len(eng_cols), nbins,
        )
        eng_mi_k = np.array(
            [_mi_from_binned_xy(eng_test_bins[:, j], y_test_bin, clip_zero=False) for j in range(len(eng_cols))],
            dtype=np.float64,
        )
        eng_fold_mis.append(eng_mi_k)

    if not raw_fold_mis:
        # All folds degenerated. Fall back to plug-in MI on full frame so
        # caller still gets a usable ranking (documented behaviour).
        logger.warning("score_features_by_kfold_oof_mi: every fold collapsed to a " "single-class held-out set; falling back to plug-in MI.")
        raw_mi = _mi_classif_batch(raw_arr, y_arr, nbins=nbins)
        eng_mi = _mi_classif_batch(eng_arr, y_arr, nbins=nbins)
    else:
        # Average across folds, then clip the AGGREGATE at zero. Per-fold
        # clipping (see _mi_from_binned_xy clip_zero arg) would inflate
        # the OOF mean on noise columns above the plug-in baseline,
        # defeating the bias-regularisation contract.
        raw_mi = np.clip(np.mean(np.vstack(raw_fold_mis), axis=0), 0.0, None)
        eng_mi = np.clip(np.mean(np.vstack(eng_fold_mis), axis=0), 0.0, None)

    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))

    rows = []
    for j, eng_name in enumerate(eng_cols):
        baseline = float(raw_mi_map.get(src_map[eng_name], 0.0))
        emi = float(eng_mi[j])
        uplift = emi / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col": src_map[eng_name],
            "baseline_mi_oof": baseline,
            "engineered_mi_oof": emi,
            "uplift_oof": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift_oof", ascending=False).reset_index(drop=True)
    return df


def _quantile_bin_batched(mat: np.ndarray, nbins: int) -> np.ndarray:
    """Column-batched equivalent of ``_mi_greedy_cmi_fe._quantile_bin``: one ``np.quantile(mat, qs,
    axis=0)`` call over every ALL-FINITE column instead of ``p`` separate per-column ``np.quantile``
    calls (the O(n log n) partition-select cost ``_quantile_bin`` pays once per column).

    Bit-identical to looping ``_quantile_bin`` over columns (verified empirically over 500 randomized
    trials incl. skewed/tied/constant/NaN-bearing columns): same ``qs = linspace(0, 1, nbins + 1)``
    INCLUDING the 0th/100th percentile, same per-column ``np.unique`` dedup, same ``edges[1:-1]`` trim,
    same ``<=2``-edge degenerate handling, same ``side='right'`` searchsorted convention.

    NOT the same algorithm as :func:`_bin_with_train_edges_batched` (which computes ONLY the inner
    ``linspace(0, 1, nbins + 1)[1:-1]`` percentiles directly, never the 0th/100th) -- that convention
    silently diverges from ``_quantile_bin`` whenever a middle percentile collides with the true min/max
    (measured ~40% mismatch rate on tied/skewed synthetic columns), so it is deliberately NOT reused here.

    Columns containing any non-finite value fall back to the scalar ``_quantile_bin`` per column
    (mirrors that function's own finite-mask branch, not worth vectorizing for the rare partial-NaN case).
    """
    mat = np.ascontiguousarray(mat, dtype=np.float64)
    n_rows, p = mat.shape
    out = np.zeros((n_rows, p), dtype=np.int64)
    if n_rows == 0 or p == 0:
        return out
    finite_per_col = np.isfinite(mat).all(axis=0)
    dense_idx = np.where(finite_per_col)[0]
    partial_idx = np.where(~finite_per_col)[0]
    qs = np.linspace(0.0, 1.0, int(nbins) + 1)
    if dense_idx.size:
        dense_mat = mat[:, dense_idx] if dense_idx.size != p else mat
        edges_all = np.quantile(dense_mat, qs, axis=0)  # (nbins + 1, len(dense_idx))
        for k, j in enumerate(dense_idx):
            edges = np.unique(edges_all[:, k])
            if edges.size <= 2:
                if edges.size == 2:
                    out[:, j] = (dense_mat[:, k] >= edges[1]).astype(np.int64)
                continue
            out[:, j] = np.searchsorted(edges[1:-1], dense_mat[:, k], side="right").astype(np.int64)
    for j in partial_idx:
        out[:, j] = _quantile_bin(mat[:, j], nbins=nbins)
    return out


def _cmi_gate_scores(
    engineered_X: pd.DataFrame,
    y_int: np.ndarray,
    current_support: pd.DataFrame,
    *,
    nbins: int = 10,
) -> dict[str, float]:
    """``CMI(eng_col; y | support_joint)`` for every engineered column.

    Mirrors the Layer 60 marginal-MI / CMI computation path: bin every
    column with quantile cuts, build the support joint via dense
    renumbering, then call the shared Miller-Madow-corrected CMI helper.

    Notes
    -----
    * When ``current_support`` is empty, returns ``{}`` -- caller must
      skip Gate 3 in that case. We deliberately do NOT collapse to
      marginal MI here because the marginal MI is ALREADY the basis of
      Gate 1; calling it again would double-count.
    * When ``y_int`` has < 2 unique classes, CMI is ill-defined; we
      return zero for every engineered column (the gate will admit
      nothing, which is the safe default).
    """
    if current_support.empty or current_support.shape[1] == 0:
        return {}
    if np.unique(y_int).size < 2:
        return {c: 0.0 for c in engineered_X.columns}
    # Build the support joint key. Batched (2026-07-12) via _quantile_bin_batched -- one np.quantile call
    # per matrix instead of one per column; bit-identical to the scalar _quantile_bin loop (see that
    # function's docstring for the verified-equivalence note).
    sup_mat = current_support.to_numpy(dtype=np.float64)
    sup_bins_mat = _quantile_bin_batched(sup_mat, nbins)
    sup_bins = [np.ascontiguousarray(sup_bins_mat[:, j]) for j in range(sup_bins_mat.shape[1])]
    z_joint, _ = _renumber_joint(*sup_bins)
    _, y_bin = np.unique(y_int, return_inverse=True)
    y_bin = y_bin.astype(np.int64)
    eng_mat = engineered_X.to_numpy(dtype=np.float64)
    eng_bins_mat = _quantile_bin_batched(eng_mat, nbins)
    out: dict[str, float] = {}
    for k, c in enumerate(engineered_X.columns):
        out[c] = float(_cmi_from_binned(np.ascontiguousarray(eng_bins_mat[:, k]), y_bin, z_joint))
    return out


def hybrid_orth_mi_three_gate_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    current_support: Optional[pd.DataFrame] = None,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    cmi_min: float = 0.001,
    n_folds: int = 5,
    seed: int = 0,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three-gate variant of :func:`hybrid_orth_mi_fe` with OOF MI scoring.

    Gates:
      1. ``uplift_oof >= min_uplift`` (relative uplift on OOF MI).
      2. ``engineered_mi_oof >= max(legacy_floor, MAD_noise_floor)``
         (absolute floor on the OOF distribution; mirrors Layer 21).
      3. ``CMI(candidate; y | support) >= cmi_min`` (conditional gate;
         SKIPPED when ``current_support`` is None or empty -- in that
         case marginal MI from Gate 1 already covers the case).

    Parameters
    ----------
    current_support : DataFrame or None
        The features already in the selection support. When non-empty,
        Gate 3 evaluates CMI of each engineered column conditional on
        the JOINT of these columns; candidates whose CMI sits at or
        below ``cmi_min`` are excluded as duplicate-signal even if they
        passed Gates 1 and 2. Pass ``None`` to skip Gate 3 entirely.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : X with the three-gate top-K winners appended.
        scores : the full OOF ranking (winners + rejects) WITH a
            ``cmi_support`` column (NaN when support is empty),
            sorted by ``uplift_oof`` descending.
    """
    engineered = generate_univariate_basis_features(
        X, cols=cols, degrees=degrees, basis=basis,
    )
    empty_cols = [
        "engineered_col", "source_col",
        "baseline_mi_oof", "engineered_mi_oof", "uplift_oof",
        "cmi_support",
    ]
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=empty_cols)

    raw_X = X[[c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]]
    scores = score_features_by_kfold_oof_mi(
        raw_X, engineered, y,
        n_folds=n_folds, seed=seed, nbins=nbins,
    )
    if scores.empty:
        return X.copy(), pd.DataFrame(columns=empty_cols)

    # Gate 3: CMI conditional on current_support.
    y_int = _coerce_y_int64(y)
    use_cmi_gate = (
        current_support is not None and isinstance(current_support, pd.DataFrame) and current_support.shape[1] > 0 and len(current_support) == len(engineered)
    )
    if use_cmi_gate:
        cmi_map = _cmi_gate_scores(
            engineered, y_int, current_support, nbins=nbins,
        )
        scores["cmi_support"] = scores["engineered_col"].map(cmi_map).astype(np.float64)
    else:
        scores["cmi_support"] = np.nan

    # Gate 2: MAD floor on the OOF engineered MI distribution, plus the
    # legacy `frac * max(baseline_oof)` floor (Layer 21 construction
    # repointed at OOF columns). On all-noise frames the MAD floor sits
    # comfortably above the median noise band; on real-signal frames
    # legitimate signals are statistical outliers and pass through.
    max_baseline_oof = float(scores["baseline_mi_oof"].max())
    legacy_floor = float(min_abs_mi_frac) * max(0.0, max_baseline_oof)
    eng_oof = scores["engineered_mi_oof"].to_numpy()
    if eng_oof.size >= 4:
        med = float(np.median(eng_oof))
        mad = float(np.median(np.abs(eng_oof - med)))
        noise_floor = med + 3.5 * 1.4826 * mad
    else:
        noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor)

    mask = (scores["uplift_oof"] >= float(min_uplift)) & (scores["engineered_mi_oof"] >= abs_floor)
    if use_cmi_gate:
        mask = mask & (scores["cmi_support"] >= float(cmi_min))

    qualified = scores[mask]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_three_gate_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    current_support: Optional[pd.DataFrame] = None,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    cmi_min: float = 0.001,
    n_folds: int = 5,
    seed: int = 0,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_three_gate_fe` but additionally returns
    a list of ``orth_univariate`` recipes -- one per appended column -- so
    that ``MRMR.transform`` can recompute each engineered column on test
    data without re-running the OOF MI ranking or the CMI gate.

    Recipes are byte-identical to Layer 21 / Layer 62 (engineered VALUES
    are bit-equal; only the SELECTION rule differs).
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_three_gate_fe(
        X, y, current_support,
        cols=cols, degrees=degrees, basis=basis, top_k=top_k,
        min_uplift=min_uplift, min_abs_mi_frac=min_abs_mi_frac,
        cmi_min=cmi_min, n_folds=n_folds, seed=seed, nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}
    for name in appended:
        if "__" not in name:
            continue
        src, suffix = name.split("__", 1)
        chosen_basis = None
        chosen_degree = None
        for code in ("LL", "He", "T", "L"):
            if suffix.startswith(code):
                rest = suffix[len(code) :]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_three_gate_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
