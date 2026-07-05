"""Layer 68 (2026-06-01): per-column scorer AUTO-SELECTION for hybrid orth-poly FE.

Why this layer
--------------

Layers 21 / 65 / 66 / 67 each ship a different dependence scorer for ranking
orth-poly engineered columns:

* Layer 21 -- plug-in quantile-binned MI (fast; weak on smooth signal
  below bin resolution, weak on heavy tails);
* Layer 65 -- Kraskov-Stoegbauer-Grassberger k-NN MI (binning-free; wins
  on smooth continuous signal);
* Layer 66 -- copula MI on rank-uniformised pairs (marginal-invariant;
  wins on heavy-tailed / skewed marginals);
* Layer 67 -- Szekely-Rizzo distance correlation (NON-MI; wins on
  non-monotone / non-functional / oscillatory dependence).

Each is a separate opt-in flag, and the user has to KNOW which scorer is
right for which column. In production with mixed-character columns
(some smooth, some heavy-tailed, some non-monotone, some discrete-binned)
the single-scorer choice is wrong for SOME columns no matter which one
the user picks.

This module runs ALL FOUR scorers on every engineered column under a
small bootstrap budget, picks the per-column scorer with the highest
LOWER CONFIDENCE BOUND (``mean - 1.96 * std`` across ``n_boot`` resamples),
and uses ITS score for the cross-column ranking + selection. The
per-column dispatch matches each engineered column to its best estimator
WITHOUT requiring the user to specify which family of signal they expect.

Lower confidence bound (LCB) over a small bootstrap is the standard
robust-selection criterion when scorers have heterogeneous variance: a
high-mean / high-variance scorer can be dominated by a slightly lower-mean
/ low-variance one because the LCB tilts toward the steadier estimate.
This is the same principle Layer 62's bootstrap LCB applies to a SINGLE
scorer; Layer 68 applies it ACROSS scorers per column.

Layer 68 vs Layers 65 / 66 / 67
-------------------------------

* Layers 65 / 66 / 67: single-scorer opt-in, user must pick.
* Layer 68 (this): per-column auto-select from {plug-in, KSG, copula,
  dCor} via bootstrap LCB. Right scorer on each column without prior
  knowledge of the signal family.

The four are complementary -- this layer materialises the COMPLEMENTARITY.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered VALUES are bit-equal to
Layer 21; only the SCORING (and therefore the selection) changes.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_auto_scorer_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._orthogonal_univariate_fe import generate_univariate_basis_features
from ._orth_auto_scorer_fe import (
    SCORER_NAMES,
    _score_copula,
    _score_dcor,
    _score_hsic,
    _score_ksg,
    _score_plug_in,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SCORER_NAMES",
    "ENSEMBLE_AGGREGATORS",
    "MUTUAL_RANK_AGGREGATORS",
    "select_best_scorer_per_column",
    "score_features_by_auto_scorer_uplift",
    "score_features_by_ensemble_uplift",
    "hybrid_orth_mi_auto_scorer_fe",
    "hybrid_orth_mi_auto_scorer_fe_with_recipes",
    "hybrid_orth_mi_ensemble_fe",
    "hybrid_orth_mi_ensemble_fe_with_recipes",
]


# Canonical aggregator identifiers for ensemble-of-scorers rank fusion.
# - mean_rank: average of per-scorer ranks (1 = best); the standard rank-
#   fusion baseline, robust to monotone score transforms because it only
#   uses ordinal information from each scorer.
# - borda_count: each scorer contributes (N - rank) "points" per column;
#   equivalent to mean_rank up to an affine transform on a fixed-N pool,
#   but kept as an explicit aggregator so downstream replay can pin it.
# - reciprocal_rank: sum of 1 / (k + rank) across scorers (k = 60 per
#   Cormack/Clarke/Buettcher 2009); strongly downweights agreement on
#   already-low-ranked columns so a single scorer with a strong top-1
#   does not get drowned by three scorers agreeing on a tail column.
ENSEMBLE_AGGREGATORS = (
    "mean_rank", "borda_count", "reciprocal_rank", "mutual_top_k",
)

# 2026-06-01 Layer 82: MUTUAL-RANK fusion aggregators. The "mutual_top_k"
# strategy keeps a candidate ONLY if it lies in the top-K of EVERY
# participating scorer (strict conjunction). Complements the existing
# mean/Borda/RR aggregators which are all UNION-flavoured (one strong scorer
# can carry a candidate). The strict-conjunction rule trades recall for
# precision -- useful when the cost of admitting a noise column is higher
# than the cost of missing a borderline signal.
MUTUAL_RANK_AGGREGATORS = ("mutual_top_k",)


# ---------------------------------------------------------------------------
# Layer 69: ENSEMBLE-OF-SCORERS rank fusion
# ---------------------------------------------------------------------------
#
# Layer 68 picks ONE scorer per column via bootstrap LCB. Layer 69 is the
# complementary path: instead of selecting a single winner, AGGREGATE the
# per-scorer rankings into a consensus ranking. Rank fusion smooths over
# bootstrap-LCB noise that can flip the per-column winner on borderline
# data, and surfaces columns that are uniformly ranked high by MULTIPLE
# scorers (a robustness signal that no individual scorer's LCB exposes).
#
# When to use Layer 68 vs Layer 69
# --------------------------------
# * Layer 68 wins on heterogeneous frames with CLEARLY-SEPARATED signal
#   families (some smooth columns, some heavy-tailed, some non-monotone).
#   The per-column auto-select matches each column to its native scorer.
# * Layer 69 wins on AMBIGUOUS frames where no single scorer is best on
#   any single column AND the bootstrap-LCB-pick is unstable across
#   seeds. Rank fusion averages the noise out and the consensus rank
#   stabilises faster than the per-column-winner identity.


def _compute_per_scorer_rank_table(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    scorers: Sequence[str],
    random_state: int,
    nbins: int,
    n_neighbors: int,
    copula_n_bins: int,
    dcor_n_sample: int,
) -> tuple[pd.DataFrame, dict, dict]:
    """Score every engineered column with every requested scorer ONCE.

    Returns
    -------
    score_table : DataFrame
        Rows = engineered columns; columns = ``score_<scorer>`` (raw
        dependence value, higher = better) plus ``source_col``.
    baseline_per_source : dict
        ``{source_col: {scorer: raw_baseline_score}}``; the per-source
        baseline used to compute the ensemble uplift below.
    rank_per_scorer : dict
        ``{scorer: dict_engineered_col_to_int_rank}`` with rank 1 = best
        WITHIN the engineered pool (NOT mixed with raw_X baselines).
        Ties broken by stable column order so the rank table is
        deterministic for replay.
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"_compute_per_scorer_rank_table: raw_X has {len(raw_X)} rows " f"but engineered_X has {len(engineered_X)}; positional row " f"alignment required."
        )
    y_arr = np.asarray(y).ravel()
    raw_cols = list(raw_X.columns)
    eng_cols = list(engineered_X.columns)

    def _call(name, x_vec, y_vec):
        if name == "plug_in":
            return _score_plug_in(x_vec, y_vec, nbins=int(nbins))
        if name == "ksg":
            return _score_ksg(
                x_vec, y_vec, n_neighbors=int(n_neighbors),
                random_state=int(random_state),
            )
        if name == "copula":
            return _score_copula(x_vec, y_vec, n_bins=int(copula_n_bins))
        if name == "dcor":
            return _score_dcor(
                x_vec, y_vec, n_sample=int(dcor_n_sample),
                random_state=int(random_state),
            )
        if name == "hsic":
            return _score_hsic(
                x_vec, y_vec, n_sample=int(dcor_n_sample),
                random_state=int(random_state),
            )
        raise ValueError(f"unknown scorer name: {name!r}")

    # Batch the column-separable scorers (plug-in MI, copula MI) across all
    # columns in one kernel call instead of one _mi_classif_batch / copula call
    # per column (each rebuilds the same dispatch + per-column quantile). The
    # per-column reshape(-1,1)/out[0] path the siblings used is bit-identical to
    # the full-frame batch (per-column quantile is independent; verified 0.0
    # diff). Non-separable scorers (ksg/dcor/hsic carry a per-call random_state
    # subsample) stay per-column via _call.
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    _BATCHABLE = {"plug_in", "copula"}
    _batch_scorers = [s for s in scorers if s in _BATCHABLE]

    def _batch_scores(frame: pd.DataFrame, cols: list) -> dict:
        """{scorer: np.ndarray aligned to ``cols``} for the batchable scorers."""
        from ._fe_usability_signal import _crit_np_dtype
        _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
        if not _batch_scorers or not cols:
            return {}
        Xmat = frame[cols].to_numpy(dtype=_dt)
        res: dict = {}
        for s in _batch_scorers:
            if s == "plug_in":
                from ._orthogonal_univariate_fe import _mi_classif_batch
                y_mi = y_arr
                if not np.issubdtype(y_mi.dtype, np.integer):
                    uniq = np.unique(y_mi[np.isfinite(y_mi)] if y_mi.dtype.kind in "fc" else y_mi)
                    if uniq.size <= 32:
                        y_mi = y_mi.astype(np.int64)
                res[s] = np.asarray(_mi_classif_batch(Xmat, y_mi, nbins=int(nbins)), dtype=np.float64)
            elif s == "copula":
                from ._orthogonal_copula_mi_fe import _copula_mi_batch

                res[s] = np.asarray(_copula_mi_batch(Xmat, y_arr, n_bins=int(copula_n_bins)), dtype=np.float64)
        return res

    raw_batched = _batch_scores(raw_X, raw_cols)
    baseline_per_source: dict = {}
    for ci, src in enumerate(raw_cols):
        x_full = raw_X[src].to_numpy(dtype=_dt)
        baseline_per_source[src] = {s: (float(raw_batched[s][ci]) if s in _batch_scorers else float(_call(s, x_full, y_arr))) for s in scorers}

    eng_batched = _batch_scores(engineered_X, eng_cols)
    rows = []
    for ci, eng_name in enumerate(eng_cols):
        src = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        x_full = engineered_X[eng_name].to_numpy(dtype=_dt)
        row = {"engineered_col": eng_name, "source_col": src}
        for s in scorers:
            if s in _batch_scorers:
                row[f"score_{s}"] = float(eng_batched[s][ci])
            else:
                row[f"score_{s}"] = float(_call(s, x_full, y_arr))
        rows.append(row)
    score_table = pd.DataFrame(rows)

    rank_per_scorer: dict = {}
    for s in scorers:
        col = f"score_{s}"
        if score_table.empty:
            rank_per_scorer[s] = {}
            continue
        # rank descending (higher score = better = rank 1); 'first' tie
        # breaking keeps the rank table deterministic across replays.
        ranks = (
            score_table[col]
            .rank(
                ascending=False,
                method="first",
            )
            .astype(int)
        )
        rank_per_scorer[s] = dict(zip(score_table["engineered_col"], ranks))
    return score_table, baseline_per_source, rank_per_scorer


def _aggregate_ranks(
    rank_per_scorer: dict,
    eng_cols: Sequence[str],
    aggregator: str,
    *,
    mutual_top_k: int = 5,
) -> dict:
    """Fuse per-scorer ranks into a single aggregate score (lower = better).

    * ``mean_rank``: arithmetic mean of integer ranks across scorers.
    * ``borda_count``: ``sum(N + 1 - rank)`` across scorers (higher =
      better). Returned as ``-points`` so the lower-is-better contract
      downstream matches the other aggregators.
    * ``reciprocal_rank``: ``sum(1 / (k + rank))`` (k = 60, the
      Cormack/Clarke/Buettcher 2009 default). Returned negated so lower
      = better.
    * ``mutual_top_k`` (Layer 82): strict conjunction -- a candidate is
      tagged "qualified" (aggregate score = max-of-ranks, lower = better)
      only if it sits in the top-K of EVERY participating scorer. Columns
      that miss the top-K of ANY scorer get a huge sentinel score so the
      downstream selector drops them. Trades recall for precision.

    Returns
    -------
    dict mapping ``engineered_col -> aggregate_score`` (lower = better).
    """
    scorers = list(rank_per_scorer.keys())
    if not scorers or not eng_cols:
        return {}
    n_cols = len(eng_cols)
    out: dict = {}
    if aggregator == "mean_rank":
        for col in eng_cols:
            ranks = [rank_per_scorer[s].get(col, n_cols + 1) for s in scorers]
            out[col] = float(np.mean(ranks))
        return out
    if aggregator == "borda_count":
        for col in eng_cols:
            pts = 0.0
            for s in scorers:
                r = rank_per_scorer[s].get(col, n_cols + 1)
                pts += n_cols + 1 - r
            # Borda points are higher-is-better; negate so the downstream
            # "lower = better" contract is uniform across all aggregators.
            out[col] = -float(pts)
        return out
    if aggregator == "reciprocal_rank":
        k = 60.0  # Cormack/Clarke/Buettcher 2009 constant.
        for col in eng_cols:
            rr = 0.0
            for s in scorers:
                r = rank_per_scorer[s].get(col, n_cols + 1)
                rr += 1.0 / (k + r)
            out[col] = -float(rr)
        return out
    if aggregator == "mutual_top_k":
        # Strict conjunction: a candidate must be top-K under EVERY scorer.
        # Columns that miss the top-K of any scorer get sentinel = n_cols
        # * 1000 so they sink to the bottom of the ranking. Among the
        # qualified set, smaller worst-case rank wins (the "weakest-link"
        # rank across scorers).
        k_thr = max(1, int(mutual_top_k))
        sentinel = float(n_cols * 1000 + 1)
        for col in eng_cols:
            ranks = [rank_per_scorer[s].get(col, n_cols + 1) for s in scorers]
            worst = max(ranks) if ranks else (n_cols + 1)
            if worst <= k_thr:
                # Qualified: rank by worst-case (weakest-link) rank.
                out[col] = float(worst)
            else:
                out[col] = sentinel
        return out
    raise ValueError(f"unknown aggregator {aggregator!r}; expected one of " f"{ENSEMBLE_AGGREGATORS}")


def score_features_by_ensemble_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    scorers: Sequence[str] = SCORER_NAMES,
    aggregator: str = "mean_rank",
    mutual_top_k: int = 5,
    random_state: int = 0,
    nbins: int = 10,
    n_neighbors: int = 3,
    copula_n_bins: int = 20,
    dcor_n_sample: int = 500,
) -> pd.DataFrame:
    """Score engineered columns by ENSEMBLE-of-scorers rank fusion.

    Each scorer in ``scorers`` ranks every engineered column independently.
    Per-column ranks are then aggregated via ``aggregator`` into a single
    consensus score; the cross-column ranking + selection uses this
    consensus instead of any individual scorer's LCB.

    Parameters
    ----------
    raw_X : DataFrame
        Source columns aligned positionally with ``y``. Used to compute
        the per-source / per-scorer raw baseline that anchors the uplift.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`.
    y : array-like (n,)
        Target.
    scorers : sequence of {"plug_in", "ksg", "copula", "dcor"}
        Subset of scorers participating in the ensemble. Default = all
        four. Empty sequence is rejected; passing a single scorer falls
        through to that scorer's raw ranking (a degenerate but valid
        case useful for ablation).
    aggregator : str
        One of ``ENSEMBLE_AGGREGATORS``. See module docstring.
    random_state, nbins, n_neighbors, copula_n_bins, dcor_n_sample
        Passed to the underlying scorers verbatim. No bootstrap here --
        the ensemble's robustness comes from cross-scorer consensus, not
        cross-resample averaging. Callers wanting both can stack this
        function on top of bootstrap-averaged scorers; the current API
        keeps the two robustness levers orthogonal.

    Returns
    -------
    DataFrame with columns
    ``[engineered_col, source_col, baseline_mi, engineered_mi, uplift,
    aggregate_rank, per_scorer_rank, per_scorer_score]``
    sorted by ``uplift`` descending.

    Notes
    -----
    * ``engineered_mi`` is the MEAN score across participating scorers
      after per-scorer normalisation by that scorer's max raw-source
      baseline -- the same headroom rescaling Layer 68 uses cross-scorer.
      This keeps the metric dimensionless so MRMR's two-gate selection
      (``uplift >= min_uplift`` + abs MI floor) is comparable to Layers
      65 / 66 / 67 / 68.
    * ``baseline_mi`` is the same MEAN-normalised score but for the
      engineered column's SOURCE row -- the natural uplift denominator.
    * ``aggregate_rank`` is the per-aggregator fused rank (lower =
      better; ``mean_rank`` is the literal mean; ``borda_count`` and
      ``reciprocal_rank`` are stored NEGATED so a uniform "lower = better"
      contract holds across aggregators).
    """
    if aggregator not in ENSEMBLE_AGGREGATORS:
        raise ValueError(f"score_features_by_ensemble_uplift: aggregator {aggregator!r} " f"not in {ENSEMBLE_AGGREGATORS}")
    scorers_tuple = tuple(scorers)
    if not scorers_tuple:
        raise ValueError("score_features_by_ensemble_uplift: scorers must be non-empty")
    for s in scorers_tuple:
        if s not in SCORER_NAMES:
            raise ValueError(f"score_features_by_ensemble_uplift: unknown scorer " f"{s!r}; expected one of {SCORER_NAMES}")
    if engineered_X.empty:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift", "aggregate_rank",
            "per_scorer_rank", "per_scorer_score",
        ])

    score_table, baseline_per_source, rank_per_scorer = (
        _compute_per_scorer_rank_table(
            raw_X, engineered_X, y,
            scorers=scorers_tuple, random_state=int(random_state),
            nbins=int(nbins), n_neighbors=int(n_neighbors),
            copula_n_bins=int(copula_n_bins),
            dcor_n_sample=int(dcor_n_sample),
        )
    )

    raw_cols = list(raw_X.columns)
    per_scorer_max = {}
    for s in scorers_tuple:
        max_s = max(
            (baseline_per_source[src].get(s, 0.0) for src in raw_cols),
            default=0.0,
        )
        per_scorer_max[s] = float(max(max_s, 1e-12))

    eng_cols = list(score_table["engineered_col"])
    agg_scores = _aggregate_ranks(
        rank_per_scorer, eng_cols, aggregator=aggregator,
        mutual_top_k=int(mutual_top_k),
    )

    rows = []
    for _, srow in score_table.iterrows():
        eng_name = srow["engineered_col"]
        src = srow["source_col"]
        norm_eng = []
        norm_base = []
        per_score_dict = {}
        per_rank_dict = {}
        for s in scorers_tuple:
            raw_eng = float(srow[f"score_{s}"])
            raw_base = float(baseline_per_source.get(src, {}).get(s, 0.0))
            norm_eng.append(raw_eng / per_scorer_max[s])
            norm_base.append(raw_base / per_scorer_max[s])
            per_score_dict[s] = raw_eng
            per_rank_dict[s] = int(rank_per_scorer[s].get(eng_name, -1))
        eng_mi = float(np.mean(norm_eng)) if norm_eng else 0.0
        base_mi = float(np.mean(norm_base)) if norm_base else 0.0
        uplift = eng_mi / (base_mi + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col": src,
            "baseline_mi": base_mi,
            "engineered_mi": eng_mi,
            "uplift": uplift,
            "aggregate_rank": float(agg_scores.get(eng_name, float("inf"))),
            "per_scorer_rank": dict(per_rank_dict),
            "per_scorer_score": dict(per_score_dict),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort by aggregate_rank ascending (lower = better) so the
        # consensus rank drives selection; uplift is the cross-layer-
        # comparable metric but the ensemble's distinguishing signal is
        # the fused rank, which is what downstream selection should use.
        df = df.sort_values(
            ["aggregate_rank", "uplift"], ascending=[True, False],
        ).reset_index(drop=True)
    return df


def hybrid_orth_mi_ensemble_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    scorers: Sequence[str] = SCORER_NAMES,
    aggregator: str = "mean_rank",
    mutual_top_k: int = 5,
    random_state: int = 0,
    nbins: int = 10,
    n_neighbors: int = 3,
    copula_n_bins: int = 20,
    dcor_n_sample: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensemble-of-scorers variant of :func:`hybrid_orth_mi_auto_scorer_fe`.

    Generates the orth-poly basis expansion, runs the rank-fusion
    ENSEMBLE scoring across ``scorers`` (default: all four), then keeps
    the top-K columns by fused rank subject to the same two-gate
    selection as Layers 65 / 66 / 67 / 68 (uplift >= ``min_uplift``,
    engineered_mi >= ``min_abs_mi_frac * max_raw_baseline`` + MAD floor).

    The ensemble's win over Layer 68's per-column auto-select: when
    bootstrap-LCB noise makes the per-column winner unstable across
    seeds, rank fusion smooths over the instability because a column
    that is consistently top-ranked by ANY of the participating scorers
    keeps its high consensus rank even if no single scorer "wins" the
    LCB tournament on every seed.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the ensemble-ranked top-K winners.
        scores : the full ranking DataFrame.
    """
    engineered = generate_univariate_basis_features(
        X, cols=cols, degrees=degrees, basis=basis,
    )
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift", "aggregate_rank",
            "per_scorer_rank", "per_scorer_score",
        ])
    raw_X = X[[
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]]
    scores = score_features_by_ensemble_uplift(
        raw_X, engineered, y,
        scorers=scorers, aggregator=aggregator,
        mutual_top_k=int(mutual_top_k),
        random_state=int(random_state), nbins=int(nbins),
        n_neighbors=int(n_neighbors),
        copula_n_bins=int(copula_n_bins),
        dcor_n_sample=int(dcor_n_sample),
    )
    if scores.empty:
        return X.copy(), scores
    # Same two-gate calibration as the auto-scorer path for cross-layer
    # parity (uplift + MAD floor on baseline_mi and engineered_mi).
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max(0.0, max_raw_baseline)
    if raw_baselines.size >= 4:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        noise_floor = med + 3.5 * 1.4826 * mad
    else:
        noise_floor = 0.0
    eng_mis = scores["engineered_mi"].to_numpy()
    if eng_mis.size >= 4:
        med_e = float(np.median(eng_mis))
        mad_e = float(np.median(np.abs(eng_mis - med_e)))
        eng_noise_floor = med_e + 3.5 * 1.4826 * mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = scores[(scores["uplift"] >= float(min_uplift)) & (scores["engineered_mi"] >= abs_floor)]
    # Layer 82 mutual_top_k: drop rows the strict-conjunction aggregator
    # disqualified (they carry a sentinel aggregate_rank far above n_cols).
    # The uplift/abs gates can still admit them on borderline data, so the
    # explicit sentinel cut is required to honour the strict-conjunction
    # contract.
    if aggregator == "mutual_top_k" and "aggregate_rank" in qualified.columns:
        n_total = len(scores)
        mutual_cut = float(n_total * 1000)
        qualified = qualified[qualified["aggregate_rank"] <= mutual_cut]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_ensemble_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    scorers: Sequence[str] = SCORER_NAMES,
    aggregator: str = "mean_rank",
    mutual_top_k: int = 5,
    random_state: int = 0,
    nbins: int = 10,
    n_neighbors: int = 3,
    copula_n_bins: int = 20,
    dcor_n_sample: int = 500,
):
    """:func:`hybrid_orth_mi_ensemble_fe` + ``orth_univariate`` recipes.

    Recipes are byte-identical to Layer 21 because the engineered
    VALUES are identical -- only the SCORING (and therefore the
    selection) differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_ensemble_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        scorers=scorers, aggregator=aggregator,
        mutual_top_k=int(mutual_top_k),
        random_state=int(random_state),
        nbins=int(nbins), n_neighbors=int(n_neighbors),
        copula_n_bins=int(copula_n_bins),
        dcor_n_sample=int(dcor_n_sample),
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    code_to_basis = {
        "He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre",
    }
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
                "hybrid_orth_mi_ensemble_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes


from ._orth_auto_scorer_fe import (  # noqa: E402,F401
    hybrid_orth_mi_auto_scorer_fe,
    hybrid_orth_mi_auto_scorer_fe_with_recipes,
    score_features_by_auto_scorer_uplift,
    select_best_scorer_per_column,
)
