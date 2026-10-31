"""Layer 62: BOOTSTRAP-STABLE MI ranking for hybrid orth-poly FE.

Why this layer
--------------

Layer 21's ``score_features_by_mi_uplift`` ranks engineered columns by a
SINGLE point-estimate MI computed on the full training frame. On finite
samples that estimate has non-trivial variance: a borderline candidate
``He_n(x_noise)`` with mean MI ~ noise floor but a long right tail can win
a slot purely because that ONE bootstrap of its sampling distribution
landed on the right tail. Across different training subsets / seeds the
selection then flips in and out - selection instability that downstream
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

Each emitted column is backed by an ``orth_univariate`` recipe - the
SAME kind Layer 21 uses - because the engineered values are bit-equal
to Layer 21's; only the SELECTION rule changes. Replay therefore reuses
the existing ``_apply_orth_univariate`` path.

NOT wired into ``MRMR.fit`` by default - opt-in via
``fe_hybrid_orth_bootstrap_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

from mlframe.feature_selection.filters._relative_uplift import relative_uplift_array
import pandas as pd

from ._orthogonal_shared import coerce_y_classif
from ._orthogonal_univariate_fe import (
    _mi_classif_batch,
    generate_univariate_basis_features,
)
from mlframe.utils.log_throttle import log_throttle
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)

from ._cache_release import register_cache_releaser

__all__ = [
    "baseline_lcb_by_bootstrap_mi",
    "score_features_by_bootstrap_mi",
    "hybrid_orth_mi_bootstrap_fe",
    "hybrid_orth_mi_bootstrap_fe_with_recipes",
]

# Z-score for the 95 % normal-CI lower bound. Mirrors the literature
# convention; using 1.645 (one-sided 95 %) is also defensible but 1.96 is
# the canonical "approximate 95 % CI" half-width and what downstream
# callers will pattern-match.
_LCB_Z = 1.96

def _bootstrap_rows(rng: np.random.Generator, y_arr: np.ndarray, n: int, sample_n: int) -> np.ndarray:
    """Row indices of one bootstrap replicate: ``sample_n`` rows drawn with replacement.

    Guard against degenerate single-class bootstraps (rare-imbalance frames): resample up to 5 times to find a multi-class slice; fall back to the full sample for that
    replicate if every retry collapses to a single class (the bootstrap then contributes the point estimate, which is the most defensible fallback).
    """
    idx = rng.integers(0, n, size=sample_n)
    for _retry in range(5):
        if np.unique(y_arr[idx]).size >= 2:
            break
        idx = rng.integers(0, n, size=sample_n)
    if np.unique(y_arr[idx]).size < 2:
        idx = np.arange(n)
    return idx


_REPS_CACHE: dict = {}  # the latest resample set: (seed, rows, sample rows, resamples, y fingerprint) -> the list of row-index vectors


def _release_reps_cache() -> int:
    """Drop the cached resample list (registered for release under memory pressure); returns the bytes freed."""
    hit = _REPS_CACHE.get("latest")
    freed = sum(int(r.nbytes) for r in hit[1]) if hit else 0
    _REPS_CACHE.clear()
    return freed


register_cache_releaser(_release_reps_cache)


def seeded_bootstrap_rows(rng_seed: int, y_arr: np.ndarray, n: int, sample_n: int, n_boot: int) -> "list[np.ndarray]":
    """The ``n_boot`` resamples of one seeded stream (:func:`_bootstrap_rows` after :func:`_bootstrap_rows`, from ``default_rng(rng_seed)``), kept for the next call with the same seed, sizes and labels.

    The scorers of one fit ask for the same resamples again and again (one call per chunk of source columns, the seed and the labels never change); drawing 10 resamples of 800000 rows each time
    (and checking each for a single class) cost 54 s of a 1M-row modular fit. The same list object comes back, which lets the rank-MI preparation recognise it."""
    from mlframe.utils.disk_cache import hash_array_content

    key = (int(rng_seed), int(n), int(sample_n), int(n_boot), hash_array_content(np.ascontiguousarray(y_arr)))
    hit = _REPS_CACHE.get("latest")
    if hit is not None and hit[0] == key:
        reps: "list[np.ndarray]" = hit[1]
        return reps
    rng = np.random.default_rng(int(rng_seed))
    reps = [_bootstrap_rows(rng, y_arr, n, sample_n) for _ in range(n_boot)]
    _REPS_CACHE.clear()
    _REPS_CACHE["latest"] = (key, reps)
    return reps


def _replicate_mi(arr: np.ndarray, y_arr: np.ndarray, rep_rows: "list[np.ndarray]", nbins: int) -> np.ndarray:
    """``(B, k)`` rank-binned MI of every column of ``arr`` on every resample in ``rep_rows``.

    Columns that are finite and (almost) all distinct are scored by :func:`bootstrap_rank_mi` (sort once, walk the order with the resample's multiplicities; bit-identical to the sorted-gather
    path for a fully distinct column); the other columns are gathered per resample in random order and scored by the batch MI, as before."""
    from ._bootstrap_rank_mi_njit import bootstrap_rank_mi

    k = arr.shape[1]
    out = np.zeros((len(rep_rows), k), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs)
    mi, fast = bootstrap_rank_mi(arr, y_arr, rep_rows, nbins)
    out[:, fast] = mi[:, fast]
    slow_idx = np.flatnonzero(~fast)
    if slow_idx.size:
        sub = arr if slow_idx.size == k else arr[:, slow_idx]
        for b, idx in enumerate(rep_rows):
            out[b, slow_idx] = np.asarray(_mi_classif_batch(sub[idx, :], y_arr[idx], nbins=nbins), dtype=np.float64)  # Intentional float64: statistic vector
    return out


def baseline_lcb_by_bootstrap_mi(raw_X: pd.DataFrame, y: np.ndarray, *, n_boot: int = 10, sample_fraction: float = 0.8, seed: int = 0, nbins: int = 10) -> np.ndarray:
    """``baseline_mi_lcb`` of every column of ``raw_X``, exactly as :func:`score_features_by_bootstrap_mi` reports it for the engineered columns built from that column.

    The baseline of a source column depends only on the column, the seeded resamples and ``y``, not on any engineered column, so a caller that needs only the noise floor of the raw
    baseline distribution (the modular and decomposition gates) does not have to score a candidate per source to get it.
    """
    y_arr = coerce_y_classif(y)
    n = len(raw_X)
    from ._fe_usability_signal import _crit_np_dtype

    raw_arr = raw_X.to_numpy(dtype=_crit_np_dtype())
    n_boot_eff = max(1, int(n_boot))
    sample_n = max(2, round(float(sample_fraction) * n))
    reps = seeded_bootstrap_rows(int(seed), y_arr, n, sample_n, n_boot_eff)
    mat = _replicate_mi(raw_arr, y_arr, reps, nbins)
    ddof = 1 if n_boot_eff >= 2 else 0
    std = mat.std(axis=0, ddof=ddof) if n_boot_eff >= 2 else np.zeros(mat.shape[1])
    return np.asarray(mat.mean(axis=0) - _LCB_Z * std)


def _bootstrap_summary_table(eng_cols, src_map, baseline_mean, baseline_std, baseline_lcb,
                             engineered_mean, engineered_std, engineered_lcb, uplift_mean, uplift_std, uplift_lcb) -> pd.DataFrame:
    """One row per engineered column (bootstrap means, stds and lower bounds), sorted by the uplift lower bound."""
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
        raise ValueError(f"score_features_by_bootstrap_mi: raw_X has {len(raw_X)} rows but " f"y has {len(np.asarray(y))}; aligned indices required.")

    y_arr = coerce_y_classif(y)
    raw_cols = list(raw_X.columns)
    eng_cols = list(engineered_X.columns)
    n = len(raw_X)

    # Map engineered_col -> source col (prefix before ``__``).
    # The source is the longest raw column the name starts with ("<raw>__<suffix>"): a raw column that itself contains "__" (an engineered column from an earlier stage, x1*x2*x3__T1_T1_T1)
    # is not cut at its own first "__".
    _raw_by_len = sorted((str(c) for c in raw_cols), key=len, reverse=True)

    def _source_of(eng_name: str) -> str:
        """Raw column an engineered name was built from."""
        for c in _raw_by_len:
            if eng_name.startswith(c + "__"):
                return c
        return eng_name.split("__", 1)[0] if "__" in eng_name else eng_name

    src_map = {eng_name: _source_of(eng_name) for eng_name in eng_cols}
    # Per-engineered baseline lookup is via the raw_X MI vector; we cache the
    # per-replicate raw MI as a name -> mi dict to avoid re-indexing per col.
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    raw_arr = raw_X.to_numpy(dtype=_dt)
    eng_arr = engineered_X.to_numpy(dtype=_dt)

    n_boot_eff = max(1, int(n_boot))
    sample_n = max(2, round(float(sample_fraction) * n))

    # Precompute engineered-col -> raw-col position ONCE so the baseline matrix is a vectorised column gather ``raw_mi[:, src_idx]``. Engineered cols whose source is absent from raw_X
    # map to -1 -> 0.0 baseline.
    _raw_pos = {c: i for i, c in enumerate(raw_cols)}
    _src_idx = np.array([_raw_pos.get(src_map[ec], -1) for ec in eng_cols], dtype=np.int64)
    _src_valid = _src_idx >= 0
    _src_idx_clipped = np.where(_src_valid, _src_idx, 0)

    reps = seeded_bootstrap_rows(int(seed), y_arr, n, sample_n, n_boot_eff)
    raw_mi = _replicate_mi(raw_arr, y_arr, reps, nbins)
    eng_mi = _replicate_mi(eng_arr, y_arr, reps, nbins)
    baseline_replicates: list[np.ndarray] = []  # one (n_eng,) vector per boot
    engineered_replicates: list[np.ndarray] = []
    uplift_replicates: list[np.ndarray] = []
    for b in range(n_boot_eff):
        baseline_b = np.where(_src_valid, raw_mi[b][_src_idx_clipped], 0.0)
        engineered_b = eng_mi[b]
        uplift_b = relative_uplift_array(engineered_b, baseline_b)
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

    return _bootstrap_summary_table(eng_cols, src_map, baseline_mean, baseline_std, baseline_lcb,
                                    engineered_mean, engineered_std, engineered_lcb, uplift_mean, uplift_std, uplift_lcb)

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
        return X, pd.DataFrame(columns=[
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
    # 1. uplift_lcb >= min_uplift_lcb (default 1.0 - "with 95% confidence
    #    the engineered MI is at least as large as the baseline MI";
    #    stricter than Layer 21's 1.05 point-estimate gate IN EFFECT
    #    because the LCB subtracts 1.96*std before comparison).
    # 2. engineered_mi_lcb >= max(legacy_frac_floor, MAD-noise floor)
    if scores.empty:
        return X, scores
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
    qualified = scores[(scores["uplift_lcb"] >= float(min_uplift_lcb)) & (scores["engineered_mi_lcb"] >= abs_floor)]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy(deep=False)
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
) -> tuple[pd.DataFrame, pd.DataFrame, list[EngineeredRecipe]]:
    """Same as :func:`hybrid_orth_mi_bootstrap_fe` but additionally returns
    a list of ``orth_univariate`` recipes - one per appended column - so
    that ``MRMR.transform`` can recompute each engineered column on test
    data without re-running the bootstrap MI ranking.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical - only the SELECTION rule (LCB vs point estimate)
    differs. The recipe parser logic is reused unchanged.
    """
    from .engineered_recipes import build_orth_univariate_recipe
    from ._orthogonal_univariate_fe import _evaluate_basis_column

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
                rest = suffix[len(code) :]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            log_throttle(
                logger,
                "orth_bootstrap_mi_fe_cannot_parse_basis_degree",
                logging.WARNING,
                "hybrid_orth_mi_bootstrap_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        # freeze the fit-time basis-preprocess params (mirrors the
        # canonical Layer-21 hybrid_orth_mi_fe_with_recipes fix); recomputing on the FULL fit-time
        # source column is safe/exact - it reproduces, not refits, the fit-time params.
        _pp = None
        try:
            _col_full = np.asarray(X[src].to_numpy(), dtype=_crit_np_dtype())
            _, _pp = _evaluate_basis_column(_col_full, chosen_basis, int(chosen_degree), return_params=True)
        except Exception as exc:
            # Was a bare except with zero logging,
            # silently reverting this column to the pre-B-17 refit-at-replay behaviour on any
            # exception (including a genuine programming bug), with no diagnostic trace.
            log_throttle(logger, "orthogonal_bootstrap_mi_fe.hot_loop_warning", logging.WARNING, "failed to freeze fit-time basis preprocess_params (falling back to refit-at-replay): %r", exc)
            _pp = None
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
            preprocess_params=_pp,
        ))
    return X_aug, scores, recipes
