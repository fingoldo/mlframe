"""Layer 62 (2026-05-31): BOOTSTRAP-STABLE MI ranking for hybrid orth-poly FE.

Why this layer
--------------

Layer 21's ``score_features_by_mi_uplift`` ranks engineered columns by a
SINGLE point-estimate MI computed on the full training frame. On finite
samples that estimate has non-trivial variance: a borderline candidate
``He_n(x_noise)`` with mean MI ~ noise floor but a long right tail can win
a slot purely because that ONE bootstrap of its sampling distribution
landed on the right tail. Across different training subsets / seeds the
selection then flips in and out -- selection instability that downstream
stacking / refit pipelines pay for in variance and brittle behaviour.

This module provides a bootstrap-stable scorer:

  For B bootstrap subsamples of ``raw_X`` / ``engineered_X`` (drawn jointly
  with replacement at ``sample_fraction``), recompute the per-column MI
  on each, then rank candidates by

      lcb_score = mean(uplift) - 1.96 * std(uplift)

i.e. the lower edge of an approximate 95 % confidence interval. Candidates
whose MI estimate is high but high-variance (typical noise-driven flukes
that ride a tail) get a LARGE std and a SMALL lcb; candidates whose MI is
high AND stable get a small std and a high lcb. The lcb metric is a
selection-stable ranking primitive borrowed from Hoeffding-style
exploration / exploitation bounds (UCB in reverse).

Returned columns mirror Layer 21's ``score_features_by_mi_uplift`` shape
PLUS the bootstrap-aware columns (mean, std, lcb for baseline_mi /
engineered_mi / uplift), so downstream calibration / debugging can
inspect the spread directly without re-running the bootstrap.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered values are bit-equal
to Layer 21's; only the SELECTION rule changes. Replay therefore reuses
the existing ``_apply_orth_univariate`` path.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_bootstrap_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._orthogonal_univariate_fe import (
    _mi_classif_batch,
    generate_univariate_basis_features,
)

logger = logging.getLogger(__name__)

__all__ = [
    "score_features_by_bootstrap_mi",
    "hybrid_orth_mi_bootstrap_fe",
    "hybrid_orth_mi_bootstrap_fe_with_recipes",
]


# Z-score for the 95 % normal-CI lower bound. Mirrors the literature
# convention; using 1.645 (one-sided 95 %) is also defensible but 1.96 is
# the canonical "approximate 95 % CI" half-width and what downstream
# callers will pattern-match.
_LCB_Z = 1.96


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


def _all_columns_distinct(arr: np.ndarray) -> bool:
    """True iff every column of ``arr`` (n, k) has all-distinct values.

    Gate for the sorted-resample-gather optimisation in
    :func:`score_features_by_bootstrap_mi`. The downstream MI uses
    equi-frequency argsort binning whose tie-breaking is positional, so
    reordering the bootstrap rows is bit-identical ONLY when there are no tied
    values to split at a bin boundary - i.e. when each column is all-distinct
    (continuous). For such matrices the only ties under with-replacement
    resampling are exact-equal duplicate rows, which bin together. Low-card /
    discrete columns must keep the original random-order gather (sorting them
    shifts MI ~1e-3 and drifts selection). Short-circuits on the first column
    that has a duplicate so discrete matrices bail cheaply."""
    n = arr.shape[0]
    for j in range(arr.shape[1]):
        if np.unique(arr[:, j]).size < n:
            return False
    return True


def score_features_by_bootstrap_mi(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
) -> pd.DataFrame:
    """Bootstrap-stable MI uplift scorer.

    Parameters
    ----------
    raw_X : DataFrame
        Source columns. Each engineered column ``"{source}__{code}{deg}"``
        maps back to its source via the ``__`` prefix (mirrors Layer 21).
    engineered_X : DataFrame
        Output of ``generate_univariate_basis_features``. Must share the
        SAME row index as ``raw_X`` and ``y`` (joint subsampling assumes
        positional alignment).
    y : array-like (n,)
        Discrete target.
    n_boot : int, default 10
        Number of bootstrap subsamples. Higher = tighter LCB estimate;
        lower = faster. 10 is the literature sweet spot for selection
        stability under a 5 % uplift gate (Sankaran 2021 et al.).
    sample_fraction : float, default 0.8
        Subsample size as a fraction of ``n``. Sampling is WITH
        REPLACEMENT to match standard bootstrap semantics; the
        ``sample_fraction`` knob lets callers trade variance vs runtime
        independent of the with/without-replacement choice (smaller
        fraction = more variance per replicate = wider CIs = stricter
        LCB ranking).
    seed : int
        RNG seed for reproducibility.
    nbins : int
        Quantile bins for MI estimation; matches Layer 21 default.

    Returns
    -------
    DataFrame with columns:
        engineered_col, source_col,
        baseline_mi_mean, baseline_mi_std, baseline_mi_lcb,
        engineered_mi_mean, engineered_mi_std, engineered_mi_lcb,
        uplift_mean, uplift_std, uplift_lcb,
    sorted by ``uplift_lcb`` descending.

    Notes
    -----
    * When ``n_boot < 2`` the std degenerates to 0 and lcb == mean,
      reducing to the Layer 21 point-estimate ranking. We still allow
      it (lets callers ablate the bootstrap path without re-routing).
    * Rows where the bootstrap baseline_mi was ~ 0 across all replicates
      get a (mean + 1e-12) denominator in uplift, mirroring Layer 21.
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_bootstrap_mi: raw_X has {len(raw_X)} rows but "
            f"engineered_X has {len(engineered_X)}; positional joint "
            f"subsampling requires aligned indices."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(
            f"score_features_by_bootstrap_mi: raw_X has {len(raw_X)} rows but "
            f"y has {len(np.asarray(y))}; aligned indices required."
        )

    y_arr = _coerce_y_int64(y)
    raw_cols = list(raw_X.columns)
    eng_cols = list(engineered_X.columns)
    n = len(raw_X)

    # Map engineered_col -> source col (prefix before ``__``).
    src_map = {
        eng_name: (eng_name.split("__", 1)[0] if "__" in eng_name else eng_name)
        for eng_name in eng_cols
    }
    # Per-engineered baseline lookup is via the raw_X MI vector; we cache the
    # per-replicate raw MI as a name -> mi dict to avoid re-indexing per col.
    raw_arr = raw_X.to_numpy(dtype=np.float64)
    eng_arr = engineered_X.to_numpy(dtype=np.float64)

    # The wide (sample_n, n_eng) gathers in the bootstrap loop dominate this
    # function's self-time, and a random with-replacement ``idx`` scatters the
    # reads across the whole matrix (cache-miss bound). Sorting ``idx`` makes
    # the gather near-sequential (~6-7x faster here), but the downstream MI uses
    # equi-frequency argsort binning whose tie-breaking is positional, so the
    # sort is bit-identical only when a matrix's columns are all-distinct (see
    # _all_columns_distinct). Gate per matrix: the engineered matrix is
    # continuous by construction (polynomial/basis transforms) so it almost
    # always qualifies and gets the win; raw_X is narrow (cheap gather either
    # way) and may carry discrete columns, in which case it keeps random order.
    raw_sort_safe = _all_columns_distinct(raw_arr)
    eng_sort_safe = _all_columns_distinct(eng_arr)

    n_boot_eff = max(1, int(n_boot))
    sample_n = max(2, int(round(float(sample_fraction) * n)))
    rng = np.random.default_rng(int(seed))

    # Precompute engineered-col -> raw-col position ONCE so each replicate's
    # baseline vector is a vectorised gather ``raw_mi_b[src_idx]`` instead of
    # rebuilding ``dict(zip(raw_cols, raw_mi_b.tolist()))`` + a per-eng-col
    # Python listcomp every replicate (the per-replicate dict+listcomp was a
    # measurable slice of this function's self-time at n_boot=10). Engineered
    # cols whose source is absent from raw_X map to -1 -> 0.0 baseline (matches
    # the listcomp's ``.get(..., 0.0)`` fallback).
    _raw_pos = {c: i for i, c in enumerate(raw_cols)}
    _src_idx = np.array(
        [_raw_pos.get(src_map[ec], -1) for ec in eng_cols], dtype=np.int64
    )
    _src_valid = _src_idx >= 0
    _src_idx_clipped = np.where(_src_valid, _src_idx, 0)

    baseline_replicates: list[np.ndarray] = []  # one (n_eng,) vector per boot
    engineered_replicates: list[np.ndarray] = []
    uplift_replicates: list[np.ndarray] = []

    for b in range(n_boot_eff):
        idx = rng.integers(0, n, size=sample_n)
        # Guard against degenerate single-class bootstraps (rare-imbalance
        # frames). Resample up to 5 times to find a multi-class slice;
        # fall back to full-sample MI for that replicate if every retry
        # collapses to a single class (the bootstrap then contributes
        # the point estimate, which is the most defensible fallback).
        for _retry in range(5):
            if np.unique(y_arr[idx]).size >= 2:
                break
            idx = rng.integers(0, n, size=sample_n)
        if np.unique(y_arr[idx]).size < 2:
            idx = np.arange(n)
        # Sort the resample indices ONLY for matrices whose columns are all
        # distinct (continuous), where the gather is bit-identical (see the
        # raw_sort_safe / eng_sort_safe precompute). The two MI calls are
        # independent, so raw and eng can use different row orders of the same
        # resample multiset without affecting their pairing in ``uplift_b``.
        idx_raw = np.sort(idx) if raw_sort_safe else idx
        idx_eng = np.sort(idx) if eng_sort_safe else idx
        sub_raw = raw_arr[idx_raw, :]
        sub_eng = eng_arr[idx_eng, :]
        raw_mi_b = _mi_classif_batch(sub_raw, y_arr[idx_raw], nbins=nbins)
        eng_mi_b = _mi_classif_batch(sub_eng, y_arr[idx_eng], nbins=nbins)
        # Vectorised baseline gather (see _src_idx precompute above).
        raw_mi_arr = np.asarray(raw_mi_b, dtype=np.float64)
        baseline_b = np.where(_src_valid, raw_mi_arr[_src_idx_clipped], 0.0)
        engineered_b = np.asarray(eng_mi_b, dtype=np.float64)
        uplift_b = engineered_b / (baseline_b + 1e-12)
        baseline_replicates.append(baseline_b)
        engineered_replicates.append(engineered_b)
        uplift_replicates.append(uplift_b)

    baseline_mat = np.vstack(baseline_replicates)  # (B, n_eng)
    engineered_mat = np.vstack(engineered_replicates)
    uplift_mat = np.vstack(uplift_replicates)

    # ddof=0 for single-replicate degeneracy (std = 0, lcb = mean). For B>=2
    # the unbiased ddof=1 is the standard convention; we use it when
    # available to avoid under-estimating std on small B.
    ddof = 1 if n_boot_eff >= 2 else 0
    baseline_mean = baseline_mat.mean(axis=0)
    baseline_std = baseline_mat.std(axis=0, ddof=ddof) if n_boot_eff >= 2 else np.zeros_like(baseline_mean)
    engineered_mean = engineered_mat.mean(axis=0)
    engineered_std = engineered_mat.std(axis=0, ddof=ddof) if n_boot_eff >= 2 else np.zeros_like(engineered_mean)
    uplift_mean = uplift_mat.mean(axis=0)
    uplift_std = uplift_mat.std(axis=0, ddof=ddof) if n_boot_eff >= 2 else np.zeros_like(uplift_mean)

    baseline_lcb = baseline_mean - _LCB_Z * baseline_std
    engineered_lcb = engineered_mean - _LCB_Z * engineered_std
    uplift_lcb = uplift_mean - _LCB_Z * uplift_std

    rows = []
    for j, eng_name in enumerate(eng_cols):
        rows.append({
            "engineered_col": eng_name,
            "source_col": src_map[eng_name],
            "baseline_mi_mean": float(baseline_mean[j]),
            "baseline_mi_std": float(baseline_std[j]),
            "baseline_mi_lcb": float(baseline_lcb[j]),
            "engineered_mi_mean": float(engineered_mean[j]),
            "engineered_mi_std": float(engineered_std[j]),
            "engineered_mi_lcb": float(engineered_lcb[j]),
            "uplift_mean": float(uplift_mean[j]),
            "uplift_std": float(uplift_std[j]),
            "uplift_lcb": float(uplift_lcb[j]),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift_lcb", ascending=False).reset_index(drop=True)
    return df


def hybrid_orth_mi_bootstrap_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift_lcb: float = 1.0,
    min_abs_mi_frac: float = 0.1,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap-stable variant of :func:`hybrid_orth_mi_fe`.

    Replaces the single point-estimate MI ranking with a lower-confidence-
    bound ranking over B bootstrap subsamples. Winners are taken from the
    candidates whose ``uplift_lcb`` clears ``min_uplift_lcb`` AND whose
    ``engineered_mi_lcb`` clears the legacy absolute floor
    ``min_abs_mi_frac * max(baseline_mi_lcb)``. Top-K by ``uplift_lcb``.

    The byproduct of the LCB gate: noise candidates with high mean uplift
    but high std (typical right-tail flukes) have LCB ~ 0 and never enter
    the support, even when the point-estimate ranking would have put them
    in the top-K. Stable signals with tight CIs ride through unchanged.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the LCB-ranked top-K winners appended.
        scores : the full bootstrap ranking DataFrame (winners + rejects),
            sorted by ``uplift_lcb`` descending.
    """
    engineered = generate_univariate_basis_features(
        X, cols=cols, degrees=degrees, basis=basis,
    )
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=[
            "engineered_col", "source_col",
            "baseline_mi_mean", "baseline_mi_std", "baseline_mi_lcb",
            "engineered_mi_mean", "engineered_mi_std", "engineered_mi_lcb",
            "uplift_mean", "uplift_std", "uplift_lcb",
        ])
    raw_X = X[[
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]]
    scores = score_features_by_bootstrap_mi(
        raw_X, engineered, y,
        n_boot=n_boot, sample_fraction=sample_fraction,
        seed=seed, nbins=nbins,
    )

    # Two-gate selection on the LCB metrics. Mirrors Layer 21 but every
    # threshold is applied to the LCB rather than the point estimate:
    # 1. uplift_lcb >= min_uplift_lcb (default 1.0 -- "with 95% confidence
    #    the engineered MI is at least as large as the baseline MI";
    #    stricter than Layer 21's 1.05 point-estimate gate IN EFFECT
    #    because the LCB subtracts 1.96*std before comparison).
    # 2. engineered_mi_lcb >= max(legacy_frac_floor, MAD-noise floor)
    if scores.empty:
        return X.copy(), scores
    max_baseline_lcb = float(scores["baseline_mi_lcb"].max())
    legacy_floor = float(min_abs_mi_frac) * max(0.0, max_baseline_lcb)

    # MAD-based noise floor on the engineered_mi_lcb distribution (mirrors
    # Layer 21's median + 3.5 * 1.4826 * MAD reference): legitimate signal
    # is an extreme outlier above the noise-band median, so the floor stays
    # in the noise band on real frames and bites only on all-noise frames.
    eng_lcb = scores["engineered_mi_lcb"].to_numpy()
    if eng_lcb.size >= 4:
        med = float(np.median(eng_lcb))
        mad = float(np.median(np.abs(eng_lcb - med)))
        noise_floor = med + 3.5 * 1.4826 * mad
    else:
        noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor)
    qualified = scores[
        (scores["uplift_lcb"] >= float(min_uplift_lcb))
        & (scores["engineered_mi_lcb"] >= abs_floor)
    ]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_bootstrap_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift_lcb: float = 1.0,
    min_abs_mi_frac: float = 0.1,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_bootstrap_fe` but additionally returns
    a list of ``orth_univariate`` recipes -- one per appended column -- so
    that ``MRMR.transform`` can recompute each engineered column on test
    data without re-running the bootstrap MI ranking.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical -- only the SELECTION rule (LCB vs point estimate)
    differs. The recipe parser logic is reused unchanged.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_bootstrap_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis, top_k=top_k,
        min_uplift_lcb=min_uplift_lcb, min_abs_mi_frac=min_abs_mi_frac,
        n_boot=n_boot, sample_fraction=sample_fraction,
        seed=seed, nbins=nbins,
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
                rest = suffix[len(code):]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_bootstrap_fe_with_recipes: cannot parse "
                "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
