"""Per-column scorer AUTO-SELECTION for hybrid orth-poly FE.

Runs all dependence scorers (plug-in / KSG / copula / dCor / HSIC) on every
engineered column under a small bootstrap budget and picks the per-column
scorer with the highest lower confidence bound (``mean - 1.96 * std``), using
its score for cross-column ranking + selection. The per-column dispatch
matches each engineered column to its best estimator without requiring the
user to specify which family of signal they expect.

The ensemble-of-scorers rank-fusion path lives in the parent module
``_orthogonal_scorer_auto_fe`` and reuses the ``_score_*`` scorers here.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._orthogonal_univariate_fe import generate_univariate_basis_features

logger = logging.getLogger(__name__)

__all__ = [
    "SCORER_NAMES",
    "select_best_scorer_per_column",
    "score_features_by_auto_scorer_uplift",
    "hybrid_orth_mi_auto_scorer_fe",
    "hybrid_orth_mi_auto_scorer_fe_with_recipes",
]


# Canonical scorer identifiers used in the auto-selection table. HSIC joined
# the pool as kernel-based dependence with the same iff-independence guarantee
# dCor offers but at a bandwidth-tuned length scale -- complementary to dCor on
# sharp local non-linearities and high-frequency oscillation. Adding HSIC is
# back-compat: callers that pinned the legacy 4-tuple via
# ``fe_hybrid_orth_ensemble_scorers`` keep the old behaviour; the auto pool
# picks HSIC when its bootstrap LCB dominates the other four.
SCORER_NAMES = ("plug_in", "ksg", "copula", "dcor", "hsic", "xi", "tail_dep")


def _score_plug_in(x: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> float:
    """Plug-in quantile-binned MI(x; y) -- the Layer 21 scorer.

    Routes through ``_mi_classif_batch`` on a single column. Returns 0.0
    on degenerate inputs (constant column / empty y); this is the same
    convention the Layer 21 batch path uses.
    """
    from ._orthogonal_univariate_fe import _mi_classif_batch

    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0
    if not np.isfinite(arr).any():
        return 0.0
    y_arr = np.asarray(y).ravel()
    if not np.issubdtype(y_arr.dtype, np.integer):
        # ORTH_SCORING_B-1 fix: was a bare `y_arr.astype(np.int64)` -- TRUNCATES,
        # does not densify -- the exact B-18 bug class, reintroduced because this function was carved into
        # this sibling file (2026-06-06) BEFORE the 2026-07-20 B-18 fix pass, which patched the parent
        # module but never followed the split here. A fractional low-cardinality y (e.g. [0.1, 0.2, ...])
        # perfectly separated by x truncates to all-0.0 -> MI=0.0 instead of the correct densified MI.
        # _mi_classif_batch's sklearn fallback handles float y via class binning; the numba dispatch needs
        # int64 y. Densify via np.unique(return_inverse=True) if all unique values fit (small-cardinality
        # classif target), matching every sibling scorer file's _coerce_y_int64 convention.
        uniq = np.unique(y_arr[np.isfinite(y_arr)] if y_arr.dtype.kind in "fc" else y_arr)
        if uniq.size <= 32:
            _, y_arr = np.unique(y_arr, return_inverse=True)
            y_arr = y_arr.astype(np.int64)
    out = _mi_classif_batch(
        arr.reshape(-1, 1), y_arr, nbins=int(nbins),
    )
    return float(out[0])


def _score_ksg(x: np.ndarray, y: np.ndarray, *, n_neighbors: int = 3, random_state: int = 0) -> float:
    """KSG / k-NN MI(x; y) -- Layer 65's Kraskov estimator.

    Delegates to the ``_ksg_mi_batch`` helper in the Layer 65 module so
    the discrete-vs-continuous y routing stays consistent.
    """
    from ._orthogonal_ksg_mi_fe import _ksg_mi_batch

    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0
    out = _ksg_mi_batch(
        arr.reshape(-1, 1), np.asarray(y).ravel(),
        n_neighbors=int(n_neighbors), random_state=int(random_state),
    )
    return float(out[0])


def _score_copula(x: np.ndarray, y: np.ndarray, *, n_bins: int = 20) -> float:
    """Copula MI(rank(x); rank(y)) -- Layer 66 rank-uniform MI."""
    from ._orthogonal_copula_mi_fe import copula_mi

    return float(copula_mi(
        np.asarray(x).ravel(), np.asarray(y).ravel(), n_bins=int(n_bins),
    ))


def _score_dcor(x: np.ndarray, y: np.ndarray, *, n_sample: int = 500, random_state: int = 0) -> float:
    """Szekely-Rizzo distance correlation -- Layer 67's non-MI dependence."""
    from ._orthogonal_dcor_fe import distance_correlation

    return float(distance_correlation(
        np.asarray(x).ravel(), np.asarray(y).ravel(),
        n_sample=int(n_sample), random_state=int(random_state),
    ))


def _score_hsic(x: np.ndarray, y: np.ndarray, *, n_sample: int = 500, random_state: int = 0) -> float:
    """HSIC -- Layer 71's kernel-based non-MI dependence (RBF, median heuristic)."""
    from ._orthogonal_hsic_fe import hsic

    return float(hsic(
        np.asarray(x).ravel(), np.asarray(y).ravel(),
        kernel="rbf", n_sample=int(n_sample),
        random_state=int(random_state),
    ))


def _score_xi(x: np.ndarray, y: np.ndarray, *, random_state: int = 0) -> float:
    """Chatterjee's Xi rank correlation -- Layer 72's sort-then-walk dependence measure, distinct
    from every distance/kernel/binning scorer above (catches high-frequency oscillatory signal a
    fixed-scale scorer averages away)."""
    from ._orthogonal_xi_fe import xi_correlation

    return float(xi_correlation(
        np.asarray(x).ravel(), np.asarray(y).ravel(), random_state=int(random_state),
    ))


def _score_tail_dep(x: np.ndarray, y: np.ndarray, *, q: float = 0.95, n_perm: int = 50, random_state: int = 0) -> float:
    """Tail-dependence coefficient -- Layer 73's co-exceedance-rate scorer, distinct from every
    scorer above (catches a column co-dependent with y ONLY in their joint extreme tail, diluted
    to a middling value by the full-distribution copula-MI average). Takes the max of the upper
    and lower tail-dependence coefficients since the extreme co-movement's sign is not known a
    priori."""
    from ._orthogonal_tail_dependence_fe import tail_dependence_score

    x_arr = np.asarray(x).ravel()
    y_arr = np.asarray(y).ravel()
    upper = tail_dependence_score(x_arr, y_arr, q=q, tail="upper", n_perm=n_perm, random_state=random_state)
    lower = tail_dependence_score(x_arr, y_arr, q=q, tail="lower", n_perm=n_perm, random_state=random_state)
    return float(max(upper, lower))


def _compute_lcb(values: np.ndarray) -> float:
    """LCB = mean - 1.96 * std (95 % lower confidence bound).

    Returns the mean alone when only one sample is available (std is
    undefined). Floored at 0.0 because every backing scorer is a
    non-negative dependence measure -- a negative LCB only means the
    bootstrap variance dominates the mean, in which case 0 is the
    right "no detectable signal" reading.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return max(0.0, float(arr[0]))
    mean = float(np.mean(arr))
    # Use ddof=1 (sample std) so the LCB is unbiased for the bootstrap
    # variance estimate, matching Layer 62's bootstrap-LCB convention.
    std = float(np.std(arr, ddof=1))
    return max(0.0, mean - 1.96 * std)


def _bootstrap_subsample_indices(
    n: int, n_boot: int, random_state: int,
) -> list:
    """Build ``n_boot`` deterministic row-index sets for the bootstrap.

    Each set is a uniform random subsample WITHOUT replacement of size
    ``max(50, n // 2)`` -- subsampling-without-replacement is the
    "m-out-of-n bootstrap" (Politis & Romano 1994) and is the standard
    choice for variance estimation when the underlying scorer is itself
    biased on small samples (KSG / dCor both are).

    Returns a list of length ``n_boot`` of 1-D ``np.ndarray[int64]`` row
    indices; deterministic given ``random_state`` so the LCB table
    replays bit-equal.
    """
    rng = np.random.default_rng(int(random_state))
    sub_n = max(50, n // 2)
    if sub_n >= n:
        # Tiny-fixture fallback: the bootstrap collapses to ``n_boot``
        # identical full-population draws -- LCB equals the mean, which
        # is still a valid (if conservative) ordering criterion.
        return [np.arange(n, dtype=np.int64) for _ in range(int(n_boot))]
    out = []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, size=sub_n, replace=False)
        idx.sort()
        out.append(idx.astype(np.int64))
    return out


def select_best_scorer_per_column(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_boot: int = 5,
    random_state: int = 0,
    nbins: int = 10,
    n_neighbors: int = 3,
    copula_n_bins: int = 20,
    dcor_n_sample: int = 500,
) -> pd.DataFrame:
    """Per-column AUTO-SELECTION of the best dependence scorer via bootstrap LCB.

    For each engineered column ``E``:

    * For each scorer ``s`` in ``{plug_in, ksg, copula, dcor}``:
        * Compute ``s(E_sub; y_sub)`` on each of ``n_boot`` subsamples;
        * LCB(s, E) = ``mean - 1.96 * std`` over the bootstrap.
    * Best scorer for ``E`` = ``argmax_s LCB(s, E)``.
    * Best LCB for ``E`` = the chosen scorer's LCB.
    * Uplift for ``E`` = ``LCB(E) / LCB(source(E))`` under the SAME
      best-scorer choice (so the uplift is comparable across scorers --
      see Layer 62 for the bootstrap-LCB justification).

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns (used to compute the per-source baseline
        under each scorer). Positionally aligned with ``y``.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`. Column
        names must carry the ``"{source}__{basis_code}{degree}"`` suffix
        so the baseline can be looked up by source.
    y : array-like (n,)
        Target. Discrete or continuous -- each scorer handles its own
        routing.
    n_boot : int
        Number of bootstrap (m-out-of-n) subsamples per scorer per
        column. Five is the recommended floor: smaller and the variance
        estimate is too noisy for the LCB to discriminate scorers;
        larger and the cost climbs ~linearly without proportionate
        accuracy gain at typical SNR.
    random_state : int
        Deterministic bootstrap seed.

    Returns
    -------
    DataFrame with one row per engineered column and columns
    ``[engineered_col, source_col, best_scorer, lcb, baseline_lcb,
    uplift, lcb_per_scorer]``. ``lcb_per_scorer`` is a dict mapping each
    scorer name to its LCB on the engineered column (useful for diagnostics
    + downstream cross-scorer audit).
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"select_best_scorer_per_column: raw_X has {len(raw_X)} rows " f"but engineered_X has {len(engineered_X)}; positional row " f"alignment required."
        )
    y_arr = np.asarray(y).ravel()
    if len(raw_X) != len(y_arr):
        raise ValueError(f"select_best_scorer_per_column: raw_X has {len(raw_X)} rows " f"but y has {len(y_arr)}; positional row alignment required.")
    if engineered_X.empty:
        return pd.DataFrame(
            columns=[
                "engineered_col",
                "source_col",
                "best_scorer",
                "lcb",
                "baseline_lcb",
                "uplift",
                "lcb_per_scorer",
                "lcb_norm_per_scorer",
            ]
        )
    n = len(raw_X)
    boot_idx = _bootstrap_subsample_indices(
        n, int(n_boot), int(random_state),
    )

    raw_cols = list(raw_X.columns)
    eng_cols = list(engineered_X.columns)

    def _scorer_call(name, x_vec, y_vec, seed):
        """Dispatch to the named dependence scorer (plug_in/ksg/copula/dcor/hsic) with this function's shared hyperparameters, so the bootstrap/LCB loop can call scorers generically by name."""
        if name == "plug_in":
            return _score_plug_in(x_vec, y_vec, nbins=int(nbins))
        if name == "ksg":
            return _score_ksg(
                x_vec, y_vec, n_neighbors=int(n_neighbors),
                random_state=int(seed),
            )
        if name == "copula":
            return _score_copula(x_vec, y_vec, n_bins=int(copula_n_bins))
        if name == "dcor":
            return _score_dcor(
                x_vec, y_vec, n_sample=int(dcor_n_sample),
                random_state=int(seed),
            )
        if name == "hsic":
            return _score_hsic(
                x_vec, y_vec, n_sample=int(dcor_n_sample),
                random_state=int(seed),
            )
        if name == "xi":
            return _score_xi(x_vec, y_vec, random_state=int(seed))
        if name == "tail_dep":
            return _score_tail_dep(x_vec, y_vec, random_state=int(seed))
        raise ValueError(f"unknown scorer name: {name!r}")

    # Per-source baseline: { source_col: { scorer: lcb } }. Computed
    # ONCE per source -- many engineered columns share the same source.
    source_lcb = {}
    for src in raw_cols:
        x_full = raw_X[src].to_numpy(dtype=np.float64)
        per_scorer = {}
        for s in SCORER_NAMES:
            vals = np.empty(len(boot_idx), dtype=np.float64)
            for b, idx in enumerate(boot_idx):
                vals[b] = _scorer_call(
                    s, x_full[idx], y_arr[idx],
                    int(random_state) + b,
                )
            per_scorer[s] = _compute_lcb(vals)
        source_lcb[src] = per_scorer

    # Per-scorer normalisation constant: each scorer is on a different
    # natural scale (plug-in MI in nats, KSG MI in nats but with k-NN
    # bias correction, copula MI in nats on rank-uniform pairs, dCor on
    # [0, 1]). Comparing raw LCBs across scorers systematically favours
    # whichever scorer has the larger natural support (dCor's [0, 1]
    # dominates plug-in MI capped near H(Y)). We normalise each scorer's
    # LCB by the MAX LCB it achieves on the raw_X source columns under
    # the same bootstrap -- this is the per-scorer "headroom" against
    # the strongest raw signal available, which is the right ratio to
    # compare cross-scorer. Equivalent to asking "how close did this
    # scorer get to the strongest raw-signal LCB it could possibly hit"
    # rather than "which scorer's natural support is largest".
    per_scorer_max = {}
    for s in SCORER_NAMES:
        max_s = max(
            (source_lcb[src].get(s, 0.0) for src in raw_cols),
            default=0.0,
        )
        per_scorer_max[s] = float(max(max_s, 1e-12))
    # Per-scorer self-referential scale for the headroom denominator below. Each scorer's natural magnitude differs by orders (HSIC's RKHS
    # statistic vs nat-scale MI), so the floor on the denominator MUST live on the scorer's OWN scale, never a global cross-scorer ceiling.
    # We take the larger of the scorer's raw-pool max and the median of its raw-pool LCBs; a scorer whose entire raw pool is its noise floor
    # then has a denominator that tracks that noise floor rather than a near-zero value that would inflate any engineered LCB into a win.
    per_scorer_scale = {}
    for s in SCORER_NAMES:
        raw_vals = [float(source_lcb[src].get(s, 0.0)) for src in raw_cols]
        med_s = float(np.median(raw_vals)) if raw_vals else 0.0
        per_scorer_scale[s] = float(max(per_scorer_max[s], med_s, 1e-12))

    rows = []
    for eng_name in eng_cols:
        src = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        x_full = engineered_X[eng_name].to_numpy(dtype=np.float64)
        per_scorer = {}
        for s in SCORER_NAMES:
            vals = np.empty(len(boot_idx), dtype=np.float64)
            for b, idx in enumerate(boot_idx):
                vals[b] = _scorer_call(
                    s, x_full[idx], y_arr[idx],
                    int(random_state) + b,
                )
            per_scorer[s] = _compute_lcb(vals)
        # Normalised score = per-scorer HEADROOM over its own raw baseline, on its own scale: ``(engineered_lcb - raw_max) / scale``. This is
        # dimensionless and self-calibrating, so the winner is the scorer that demonstrates the most REAL additional signal beyond the strongest
        # raw signal IT could detect -- not the scorer with the largest natural scale, nor a scorer whose tiny raw ceiling would let any engineered
        # value inflate into a degenerate ratio. A scorer whose engineered LCB merely matches its own noise floor scores ~0 and cannot win a column;
        # a scorer that genuinely lifts above its own raw baseline (HSIC on a non-monotone signal) scores high and can win where it is strongest.
        per_scorer_norm = {s: (per_scorer[s] - per_scorer_max[s]) / per_scorer_scale[s] for s in SCORER_NAMES}
        best_scorer = max(
            SCORER_NAMES, key=lambda s: per_scorer_norm[s],
        )
        best_lcb = float(per_scorer[best_scorer])
        baseline_lcb = float(source_lcb.get(src, {}).get(best_scorer, 0.0))
        uplift = best_lcb / (baseline_lcb + 1e-12)
        rows.append(
            {
                "engineered_col": eng_name,
                "source_col": src,
                "best_scorer": best_scorer,
                "lcb": best_lcb,
                "baseline_lcb": baseline_lcb,
                "uplift": uplift,
                "lcb_per_scorer": dict(per_scorer),
                "lcb_norm_per_scorer": dict(per_scorer_norm),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def score_features_by_auto_scorer_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_boot: int = 5,
    random_state: int = 0,
    nbins: int = 10,
    n_neighbors: int = 3,
    copula_n_bins: int = 20,
    dcor_n_sample: int = 500,
) -> pd.DataFrame:
    """Score engineered columns under the per-column best-scorer choice.

    Thin alias of :func:`select_best_scorer_per_column` that re-emits the
    table under the column names downstream gates expect: the LCB +
    uplift columns are renamed to ``engineered_mi`` / ``baseline_mi``
    so Layer 21 / 65 / 66 / 67 reuse the same selection helpers.
    The values are dependence-measure LCBs (not MI in nats), but the
    ranking semantics are identical (higher = better).

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift, best_scorer, lcb_per_scorer]`` sorted by
    ``uplift`` descending.
    """
    raw_table = select_best_scorer_per_column(
        raw_X, engineered_X, y,
        n_boot=int(n_boot), random_state=int(random_state),
        nbins=int(nbins), n_neighbors=int(n_neighbors),
        copula_n_bins=int(copula_n_bins), dcor_n_sample=int(dcor_n_sample),
    )
    if raw_table.empty:
        return pd.DataFrame(
            columns=[
                "engineered_col",
                "source_col",
                "baseline_mi",
                "engineered_mi",
                "uplift",
                "best_scorer",
                "lcb_per_scorer",
                "lcb_norm_per_scorer",
            ]
        )
    out = raw_table.rename(
        columns={
            "lcb": "engineered_mi",
            "baseline_lcb": "baseline_mi",
        }
    )
    return out[
        [
            "engineered_col",
            "source_col",
            "baseline_mi",
            "engineered_mi",
            "uplift",
            "best_scorer",
            "lcb_per_scorer",
            "lcb_norm_per_scorer",
        ]
    ]


def hybrid_orth_mi_auto_scorer_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    n_boot: int = 5,
    random_state: int = 0,
    nbins: int = 10,
    n_neighbors: int = 3,
    copula_n_bins: int = 20,
    dcor_n_sample: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Auto-scorer variant of :func:`hybrid_orth_mi_fe`.

    Generates the orth-poly basis expansion, runs the per-column AUTO-
    scorer selection (bootstrap LCB across {plug-in, KSG, copula, dCor}),
    then applies the same two-gate selection as Layer 21 / 65 / 66 / 67:
    (1) uplift >= ``min_uplift``, (2) engineered_mi >=
    ``min_abs_mi_frac * max(raw_baseline_mi)`` plus the MAD-noise floor,
    then top-K by uplift.

    The auto-scorer win: heterogeneous frames (some smooth columns, some
    heavy-tailed, some non-monotone) get the RIGHT scorer per column
    without the user having to know which signal family lives where.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the auto-scorer-ranked top-K winners.
        scores : the full ranking DataFrame (winners + rejects) carrying
                 the ``best_scorer`` column for downstream audit.
    """
    engineered = generate_univariate_basis_features(
        X, cols=cols, degrees=degrees, basis=basis,
    )
    if engineered.empty:
        return X, pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi", "engineered_mi",
            "uplift", "best_scorer", "lcb_per_scorer",
            "lcb_norm_per_scorer",
        ])
    raw_X = X[[
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]]
    scores = score_features_by_auto_scorer_uplift(
        raw_X, engineered, y,
        n_boot=int(n_boot), random_state=int(random_state),
        nbins=int(nbins), n_neighbors=int(n_neighbors),
        copula_n_bins=int(copula_n_bins), dcor_n_sample=int(dcor_n_sample),
    )
    if scores.empty:
        return X, scores
    # Two-gate selection identical to Layers 65 / 66 / 67 for cross-layer
    # parity. Baseline LCB scales DIFFER across scorers (KSG MI is in
    # nats, dCor in [0, 1]) but every row in ``scores`` uses ITS column's
    # chosen scorer for BOTH the engineered and baseline LCB -- the
    # ratio is dimensionless and the MAD floor is computed on the
    # heterogeneous baseline column with no additional normalisation.
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
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X
    return X_aug, scores


def hybrid_orth_mi_auto_scorer_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    n_boot: int = 5,
    random_state: int = 0,
    nbins: int = 10,
    n_neighbors: int = 3,
    copula_n_bins: int = 20,
    dcor_n_sample: int = 500,
):
    """Same as :func:`hybrid_orth_mi_auto_scorer_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so
    ``MRMR.transform`` can recompute each engineered column on test data
    without re-running the auto-scorer selection.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical -- only the SCORING (and therefore the selection)
    differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe
    from ._orthogonal_univariate_fe import _evaluate_basis_column

    X_aug, scores = hybrid_orth_mi_auto_scorer_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        n_boot=int(n_boot), random_state=int(random_state),
        nbins=int(nbins), n_neighbors=int(n_neighbors),
        copula_n_bins=int(copula_n_bins), dcor_n_sample=int(dcor_n_sample),
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
                "hybrid_orth_mi_auto_scorer_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        # ORTH_SCORING_B-2 fix: freeze the fit-time basis-preprocess params
        # (the B-17 fix, applied to the parent module's ensemble builder but never followed into this
        # carved-out sibling) so MRMR.transform() on a row-sliced/distribution-shifted test frame replays
        # the fit-time z-score/min-max axis instead of silently refitting it.
        _pp = None
        try:
            _col_full = np.asarray(X[src].to_numpy(), dtype=np.float64)
            _, _pp = _evaluate_basis_column(_col_full, chosen_basis, int(chosen_degree), return_params=True)
        except Exception as exc:
            logger.debug("failed to freeze fit-time basis preprocess_params (falling back to refit-at-replay): %r", exc)
            _pp = None
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
            preprocess_params=_pp,
        ))
    return X_aug, scores, recipes
