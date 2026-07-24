"""Layer 76 (2026-06-01): META-SCORER auto-selection that LEARNS from cheap
signal characteristics ("data fingerprints") and dispatches to the
predicted-best scorer of the Layer 21 / 65 / 66 / 67 / 71 / 72 / 73 / 74
family.

Why this layer
--------------

Layer 75 pinned the 5-fixture x 8-scorer empirical AUC matrix and showed
that different scorers win on different signal characters:

* plug-in (Layer 21) wins on smooth monotone with adequate sample;
* HSIC (Layer 71) wins on Pearson-blind quadratic / non-monotone;
* CMIM (Layer 74) wins on heavily-duplicating candidate pools (XOR-style
  multi-redundant sources);
* JMIM (Layer 72) is the runner-up redundancy filter on the same pool;
* dCor (Layer 67) edges plug-in on non-monotone cubic;
* copula (Layer 66) is the rank-invariant champion on heavy-tailed
  marginals when the signal is non-monotone in raw space.

The L75 matrix is hard-coded knowledge -- useful as documentation but a
USER calling MRMR can't reach into it and pick the right scorer for a
NEW dataset they haven't fingerprinted. Layer 68 (per-column bootstrap
LCB) and Layer 69 (rank fusion) are the brute-force ways to handle this
-- run ALL scorers and let a meta-criterion choose. Both are expensive:
Layer 68 runs every scorer ``n_boot=5`` times on every column, Layer 69
runs every scorer ONCE on every column. Either way the cost is O(n_scorers).

Layer 76 takes the OPPOSITE approach: spend a small fixed budget on
CHEAP data fingerprints (skew, kurtosis, n_unique, mean abs Pearson,
mean abs dCor on a subsample), use simple deterministic rules
(distilled from the L75 matrix) to PREDICT which scorer will win, and
run ONLY that scorer. The wall-clock saving versus Layer 68 / 69 is
roughly n_scorers - 1 (one scorer dispatched out of eight), at the
cost of a meta-prediction that can be wrong on edge cases. Layer 76
ships an explicit ``force_scorer`` override for the cases where the
user has more knowledge than the rules.

Fingerprint -> rule -> scorer
-----------------------------

``fingerprint_signal(X, y)`` returns the following statistics (all cheap;
sub-second on typical mlframe pipelines):

* ``n``  -- row count;
* ``unique_y_count``  -- ``nunique(y)`` -- classification vs regression;
* ``x_unique_avg``  -- average ``nunique`` across numeric columns;
* ``mean_abs_pearson``  -- mean absolute Pearson correlation
  ``|corr(X[c], y)|`` across numeric columns (when ``y`` is numeric or
  encodable); cheap O(n * d) signal-monotonicity proxy;
* ``mean_abs_skew``  -- mean absolute skewness across numeric columns
  (heavy-tail / asymmetric marginal proxy);
* ``mean_kurtosis``  -- mean ``Fisher`` kurtosis (heavy-tail proxy);
* ``inter_x_max_corr``  -- max absolute Pearson correlation between
  numeric column pairs (redundancy proxy);
* ``dcor_proxy``  -- mean absolute ``rho_spearman`` on a subsample
  of up to 500 rows; cheap non-monotone-dependence proxy that DOES NOT
  require the full O(n^2) dCor compute. Equal to mean_abs_pearson on
  monotone signal; LARGER than mean_abs_pearson when the signal is rank-
  monotone but not Pearson-monotone.

``predict_best_scorer(fp_dict)`` is a deterministic rule cascade -- each
rule is documented in-line with the L75 empirical justification:

1. Redundancy first: ``inter_x_max_corr >= 0.85`` -> ``cmim`` (L75
   xor_redundant winner with the largest margin);
2. Non-monotone signal: ``dcor_proxy - mean_abs_pearson >= 0.05`` AND
   ``mean_abs_pearson < 0.20`` -> ``hsic`` (L75 quadratic winner; HSIC
   beats dCor and plug-in on Pearson-blind cases);
3. Heavy-tail marginals: ``mean_abs_skew >= 1.5 OR mean_kurtosis >= 3.0``
   AND continuous y -> ``copula`` (L66 design claim; L75 heavy-tail
   confirms copula matches plug-in on rank-monotone heavy-tail);
4. Continuous y: ``unique_y_count >= 20`` -> ``ksg`` (L65 design claim;
   binning-free MI for continuous-y regression);
5. Default: ``plug_in`` (L21 cheap baseline; L75 linear_monotone winner).

The rule cascade is INTENTIONALLY simple: a five-rule decision tree is
auditable, debuggable, and -- the most important property -- predictable
across pickle / clone / replay. A learned meta-classifier (e.g.
gradient-boost over fingerprints) was considered and REJECTED: meta-
classifier weights would need to be retrained on every L75-style
benchmark refresh, and a misclassified column at fit time would break
``transform`` replay because the meta-prediction would be inputs-dependent
(fit-time X.shape vs transform-time X.shape).

Layer 76 vs Layers 68 / 69
--------------------------

* Layer 68 (per-column bootstrap LCB): runs 4-5 scorers x ``n_boot=5``
  bootstraps per column; ~O(20 * n_cols) MI compute. Right when the
  scorer winner is column-specific AND bootstrap noise is small.
* Layer 69 (rank-fusion ensemble): runs each scorer ONCE per column,
  fuses ranks via mean_rank / borda / reciprocal_rank; ~O(5 * n_cols).
  Right when no single scorer dominates but the consensus rank is
  reliable.
* Layer 76 (THIS): runs FINGERPRINT once (O(n * d) for Pearson; O(min(n,
  500)^2) for dCor proxy -- both fast), then ONE scorer end-to-end;
  ~O(1 * n_cols + fingerprint_cost). Right when the dataset has a
  clear signal character that the rule cascade can recognise.

Not wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_meta_enable=True``. ``force_scorer`` override available
via ``fe_hybrid_orth_meta_force_scorer=<name>`` for users who want to
pin a specific scorer regardless of the fingerprint.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Only numeric / data-shape failures are an acceptable "no signal -> 0.0"
# outcome for the meta-feature probes. A genuine programming error
# (AttributeError, KeyError, NameError, ...) must propagate, not be silently
# coerced to 0.0 -- that would misroute the scorer by faking absence of signal.
_NUMERIC_ERRORS = (ValueError, TypeError, ZeroDivisionError, FloatingPointError, ArithmeticError)

__all__ = [
    "META_SCORER_NAMES",
    "fingerprint_signal",
    "predict_best_scorer",
    "hybrid_orth_mi_meta_fe",
    "hybrid_orth_mi_meta_fe_with_recipes",
]


# Canonical scorer identifiers the meta-predictor can dispatch to.
# Mirrors the Layer 21 / 65 / 66 / 67 / 71 / 72 / 74 set (no TC -- TC
# is dominated by JMIM / CMIM on every L75 fixture so the meta-cascade
# routes redundancy cases to CMIM, never TC; user can still force_scorer
# to "tc" if desired but the meta-cascade never picks it).
META_SCORER_NAMES = (
    "plug_in", "ksg", "copula", "dcor", "hsic", "jmim", "cmim",
)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _numeric_cols(X: pd.DataFrame, cols: Optional[Sequence[str]] = None):
    """Return the numeric columns of ``X`` (optionally filtered by ``cols``)."""
    if cols is None:
        cols = list(X.columns)
    return [c for c in cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]


def fingerprint_signal(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    dcor_proxy_sample: int = 500,
    random_state: int = 0,
) -> dict:
    """Compute a cheap fingerprint of the (X, y) signal character.

    Returns
    -------
    dict
        Keys: ``n``, ``unique_y_count``, ``x_unique_avg``,
        ``mean_abs_pearson``, ``mean_abs_skew``, ``mean_kurtosis``,
        ``inter_x_max_corr``, ``dcor_proxy``. All floats / ints; NaN
        for empty-input edge cases.

    The compute is intentionally sub-second on typical mlframe frames:
    Pearson and Spearman are O(n * d); skew / kurtosis are O(n * d)
    per column; the dCor proxy uses Spearman rank correlation on a
    subsample of up to ``dcor_proxy_sample`` rows so the cost is
    capped regardless of frame size.
    """
    num_cols = _numeric_cols(X, cols)
    y_arr = np.asarray(y).ravel()
    n = len(y_arr)
    unique_y_count = int(np.unique(y_arr[~pd.isna(y_arr)]).size)

    if not num_cols:
        return {
            "n": n,
            "unique_y_count": unique_y_count,
            "n_source_cols": 0,
            "x_unique_avg": 0.0,
            "mean_abs_pearson": 0.0,
            "mean_abs_skew": 0.0,
            "mean_kurtosis": 0.0,
            "inter_x_max_corr": 0.0,
            "dcor_proxy": 0.0,
        }

    X_num = X[num_cols].copy()
    # x_unique_avg: mean column-cardinality (capped at n to avoid weird
    # values on object cols that snuck through dtype check).
    x_unique_avg = float(np.mean([min(int(X_num[c].nunique(dropna=True)), n) for c in num_cols]))

    # mean_abs_pearson: |corr(X[c], y)| averaged across numeric columns.
    # When y is constant or all-NaN we fall back to 0.0.
    try:
        y_series = pd.Series(y_arr, index=X_num.index)
        pears = []
        if y_series.nunique(dropna=True) >= 2:
            for c in num_cols:
                try:
                    r = float(X_num[c].corr(y_series, method="pearson"))
                except _NUMERIC_ERRORS as exc:
                    # ORTH_SCORING_B-9 fix: was a bare `except Exception`, broader
                    # than this module's own declared _NUMERIC_ERRORS convention (and the module docstring's
                    # explicit invariant that a genuine programming error must propagate); unlogged.
                    logger.debug("meta_scorer pearson corr failed for column %r: %r", c, exc)
                    r = float("nan")
                if np.isfinite(r):
                    pears.append(abs(r))
        mean_abs_pearson = float(np.mean(pears)) if pears else 0.0
    except _NUMERIC_ERRORS as exc:
        logger.warning("meta_scorer mean_abs_pearson failed numerically: %r", exc)
        mean_abs_pearson = 0.0

    # mean_abs_skew and mean_kurtosis on numeric columns. skew() returns
    # NaN on constant cols; we drop those.
    try:
        skews = X_num.skew(axis=0, numeric_only=True)
        mean_abs_skew = float(skews.abs().replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(skews) else 0.0
    except _NUMERIC_ERRORS as exc:
        logger.warning("meta_scorer mean_abs_skew failed numerically: %r", exc)
        mean_abs_skew = 0.0
    try:
        kurts = X_num.kurt(axis=0, numeric_only=True)
        mean_kurtosis = float(kurts.replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(kurts) else 0.0
    except _NUMERIC_ERRORS as exc:
        logger.warning("meta_scorer mean_kurtosis failed numerically: %r", exc)
        mean_kurtosis = 0.0

    # inter_x_max_corr: max |Pearson| among off-diagonal pairs. Redundancy
    # proxy that powers the L75 xor_redundant -> CMIM routing.
    try:
        if X_num.shape[1] >= 2:
            corr_mat = X_num.corr(method="pearson").to_numpy()
            d = corr_mat.shape[0]
            mask = ~np.eye(d, dtype=bool)
            finite_off = corr_mat[mask]
            finite_off = finite_off[np.isfinite(finite_off)]
            inter_x_max_corr = float(np.abs(finite_off).max()) if finite_off.size else 0.0
        else:
            inter_x_max_corr = 0.0
    except _NUMERIC_ERRORS as exc:
        logger.warning("meta_scorer inter_x_max_corr failed numerically: %r", exc)
        inter_x_max_corr = 0.0

    # dcor_proxy: mean MAX(|Spearman|, |Pearson(|x - mean|, y)|) of each
    # column with y, on a subsample of up to ``dcor_proxy_sample`` rows.
    # We combine TWO cheap non-Pearson dependence proxies:
    #   * Spearman rank correlation -- catches rank-monotone non-linear
    #     dependence (e.g. y = sigmoid(x), heavy-tailed monotone signal);
    #   * Pearson(|x - mean(x)|, y) -- catches SYMMETRIC non-monotone
    #     dependence (e.g. y = x^2 -- |x| is rank-correlated with y even
    #     though x itself is not). This is the cheap non-monotone proxy
    #     that the L75 quadratic-fixture rule needs because pure Pearson
    #     and pure Spearman both drop to 0 on the symmetric quadratic
    #     fixture.
    # The COMBINED proxy stays high on EITHER rank-monotone or symmetric
    # non-monotone dependence, which is the union the L76 rule cascade
    # uses to detect Pearson-blind signals.
    try:
        if n > 1 and pd.Series(y_arr).nunique(dropna=True) >= 2:
            sample_n = min(int(dcor_proxy_sample), n)
            if sample_n < n:
                rng = np.random.default_rng(int(random_state))
                idx = rng.choice(n, size=sample_n, replace=False)
            else:
                idx = np.arange(n)
            X_sub = X_num.iloc[idx]
            y_sub = pd.Series(y_arr[idx], index=X_sub.index)
            sps = []
            for c in num_cols:
                col_vals = X_sub[c]
                try:
                    r_sp = float(col_vals.corr(y_sub, method="spearman"))
                except _NUMERIC_ERRORS as exc:
                    # ORTH_SCORING_B-9 fix: see the mean_abs_pearson site's matching
                    # fix above for the full rationale.
                    logger.debug("meta_scorer spearman corr failed for column %r: %r", c, exc)
                    r_sp = float("nan")
                try:
                    centered = (col_vals - float(col_vals.mean())).abs()
                    r_sym = float(centered.corr(y_sub, method="pearson"))
                except _NUMERIC_ERRORS as exc:
                    logger.debug("meta_scorer symmetric-pearson corr failed for column %r: %r", c, exc)
                    r_sym = float("nan")
                candidates = [abs(r) for r in (r_sp, r_sym) if np.isfinite(r)]
                if candidates:
                    sps.append(max(candidates))
            dcor_proxy = float(np.mean(sps)) if sps else 0.0
        else:
            dcor_proxy = 0.0
    except _NUMERIC_ERRORS as exc:
        logger.warning("meta_scorer dcor_proxy failed numerically: %r", exc)
        dcor_proxy = 0.0

    return {
        "n": n,
        "unique_y_count": unique_y_count,
        "n_source_cols": len(num_cols),
        "x_unique_avg": x_unique_avg,
        "mean_abs_pearson": mean_abs_pearson,
        "mean_abs_skew": mean_abs_skew,
        "mean_kurtosis": mean_kurtosis,
        "inter_x_max_corr": inter_x_max_corr,
        "dcor_proxy": dcor_proxy,
    }


# ---------------------------------------------------------------------------
# Predict best scorer
# ---------------------------------------------------------------------------


def predict_best_scorer(fp: dict) -> str:
    """Return the predicted-best scorer name for a fingerprint dict.

    Rule cascade (first match wins). Each rule is justified by the L75
    empirical AUC matrix; numeric thresholds are calibrated to the
    fixture span -- xor_redundant has inter_x_max_corr >= 0.95 so the
    0.85 floor admits real-world borderline cases without false-firing
    on independent-source frames. ``mean_abs_pearson < 0.20`` for
    quadratic / non-monotone is the L75 quadratic fingerprint signature.

    1. ``inter_x_max_corr >= 0.85`` -> ``cmim``
       (heavily-duplicating candidate pool; L75 xor_redundant winner).
    2. ``mean_abs_skew >= 5.0 OR mean_kurtosis >= 20.0`` -> ``copula``
       (heavy-tailed marginals; L66 design claim). Placed BEFORE the
       HSIC rule because heavy-tail fixtures trip the dcor_proxy gap
       too (Spearman is rank-invariant on Pareto), but rank-invariant
       MI is the correct dispatch, not kernel HSIC.
    3. ``dcor_proxy - mean_abs_pearson >= 0.05`` AND
       ``mean_abs_pearson < 0.20`` -> ``hsic``
       (Pearson-blind non-monotone; L75 quadratic / cubic).
    4. ``unique_y_count < 15`` AND ``inter_x_max_corr < 0.6`` AND
       ``n_source_cols >= 5`` AND ``mean_abs_pearson < 0.20`` ->
       ``cmim`` (Layer 84 addition; L83 7-dataset benchmark showed CMIM
       as real-data winner on tabular sklearn-style classification
       frames where no single column has a strong linear handle on y;
       placed AFTER heavy-tail / Pearson-blind rules because those have
       a more specific dispatch).
    5. ``unique_y_count >= 20`` -> ``ksg``
       (continuous y, no heavy tail; L65 design claim).
    6. Default -> ``plug_in`` (L21 cheap baseline; linear_monotone winner).

    Parameters
    ----------
    fp : dict
        Output of :func:`fingerprint_signal`.

    Returns
    -------
    str
        One of ``META_SCORER_NAMES``.
    """
    inter_x = float(fp.get("inter_x_max_corr", 0.0))
    dcor_p = float(fp.get("dcor_proxy", 0.0))
    pears = float(fp.get("mean_abs_pearson", 0.0))
    skew = float(fp.get("mean_abs_skew", 0.0))
    kurt = float(fp.get("mean_kurtosis", 0.0))
    uy = int(fp.get("unique_y_count", 0))
    n_src = int(fp.get("n_source_cols", 0))

    # Rule 1: redundancy among candidates -> CMIM.
    if inter_x >= 0.85:
        return "cmim"
    # Rule 2: heavy-tail marginals -> copula.
    # L66 design claim is about heavy-tailed MARGINALS, regardless of y
    # type -- copula's rank-uniformisation strips out the tail before
    # the MI compute. Placed BEFORE the HSIC rule because heavy-tail
    # fixtures also trip the dcor_proxy > pearson gap (Spearman is
    # rank-invariant so dwarfs Pearson on Pareto-distributed columns)
    # but rank-invariant MI is the correct dispatch, not HSIC.
    if skew >= 5.0 or kurt >= 20.0:
        return "copula"
    # Rule 3: Pearson-blind non-monotone -> HSIC.
    if (dcor_p - pears) >= 0.05 and pears < 0.20:
        return "hsic"
    # Rule 4 (Layer 84): tabular sklearn-like classification with mild
    # redundancy -> CMIM. L83's 7-dataset x 10-mechanism benchmark
    # established CMIM as the real-data winner on tabular sklearn-style
    # frames (5/7 wins or ties). The signature is:
    #   * ``unique_y_count < 15`` (classification with a small label
    #     alphabet -- excludes regression which falls through to KSG);
    #   * ``inter_x_max_corr < 0.6`` (mild redundancy, not the extreme
    #     near-copy regime Rule 1 already routes; CMIM still earns its
    #     keep on tabular frames at lower correlations because the
    #     conditional-MI redundancy filter discounts engineered columns
    #     whose information about y is already covered by some raw
    #     source);
    #   * ``n_source_cols >= 5`` (enough source columns for the
    #     redundancy filter to have non-trivial conditioning sets;
    #     below 5 the MI estimator noise dominates the per-pair CMI);
    #   * ``mean_abs_pearson < 0.20`` (no easy linear signal -- a clean
    #     linear-monotone signal where Pearson catches the dependence is
    #     better routed to plug_in, which is the L75 linear_monotone
    #     winner. CMIM's redundancy filter pays off when no single column
    #     has a clear linear handle on y, so the engineered candidates
    #     have to be ranked on conditional-MI redundancy. This bound also
    #     keeps the L76 linear_monotone fingerprint at Pearson ~ 0.26
    #     routed to plug_in rather than CMIM.).
    # Heavy-tail (Rule 2) and Pearson-blind (Rule 3) take precedence
    # because those fixtures have a more specific scorer dispatch.
    if uy < 15 and inter_x < 0.6 and n_src >= 5 and pears < 0.20:
        return "cmim"
    # Rule 5: continuous y, no heavy tail -> KSG.
    if uy >= 20:
        return "ksg"
    # Rule 6: default -> plug-in.
    return "plug_in"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _dispatch_scorer(
    scorer: str,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]],
    degrees: Sequence[int],
    basis: str,
    top_k: int,
    min_uplift: float,
    min_abs_mi_frac: float,
    random_state: int,
    with_recipes: bool,
):
    """Run the named scorer with consistent kwargs. Returns either
    ``(X_aug, scores)`` or ``(X_aug, scores, recipes)`` depending on
    ``with_recipes``.
    """
    scorer = scorer.lower()
    if scorer == "plug_in":
        from ._orthogonal_univariate_fe import (
            hybrid_orth_mi_fe,
            hybrid_orth_mi_fe_with_recipes,
        )
        if with_recipes:
            return hybrid_orth_mi_fe_with_recipes(
                X, y, cols=cols, degrees=degrees, basis=basis,
                top_k=top_k, min_uplift=min_uplift,
                min_abs_mi_frac=min_abs_mi_frac, nbins=10,
            )
        return hybrid_orth_mi_fe(
            X, y, cols=cols, degrees=degrees, basis=basis,
            top_k=top_k, min_uplift=min_uplift,
            min_abs_mi_frac=min_abs_mi_frac, nbins=10,
        )
    if scorer == "ksg":
        from ._orthogonal_ksg_mi_fe import (
            hybrid_orth_mi_ksg_fe,
            hybrid_orth_mi_ksg_fe_with_recipes,
        )
        if with_recipes:
            return hybrid_orth_mi_ksg_fe_with_recipes(
                X, y, cols=cols, degrees=degrees, basis=basis,
                top_k=top_k, min_uplift=min_uplift,
                min_abs_mi_frac=min_abs_mi_frac,
                n_neighbors=3, random_state=random_state,
            )
        return hybrid_orth_mi_ksg_fe(
            X, y, cols=cols, degrees=degrees, basis=basis,
            top_k=top_k, min_uplift=min_uplift,
            min_abs_mi_frac=min_abs_mi_frac,
            n_neighbors=3, random_state=random_state,
        )
    if scorer == "copula":
        from ._orthogonal_copula_mi_fe import (
            hybrid_orth_mi_copula_fe,
            hybrid_orth_mi_copula_fe_with_recipes,
        )
        if with_recipes:
            return hybrid_orth_mi_copula_fe_with_recipes(
                X, y, cols=cols, degrees=degrees, basis=basis,
                top_k=top_k, min_uplift=min_uplift,
                min_abs_mi_frac=min_abs_mi_frac, n_bins=20,
            )
        return hybrid_orth_mi_copula_fe(
            X, y, cols=cols, degrees=degrees, basis=basis,
            top_k=top_k, min_uplift=min_uplift,
            min_abs_mi_frac=min_abs_mi_frac, n_bins=20,
        )
    if scorer == "dcor":
        from ._orthogonal_dcor_fe import (
            hybrid_orth_mi_dcor_fe,
            hybrid_orth_mi_dcor_fe_with_recipes,
        )
        if with_recipes:
            return hybrid_orth_mi_dcor_fe_with_recipes(
                X, y, cols=cols, degrees=degrees, basis=basis,
                top_k=top_k, min_uplift=min_uplift,
                min_abs_mi_frac=min_abs_mi_frac,
                n_sample=500, random_state=random_state,
            )
        return hybrid_orth_mi_dcor_fe(
            X, y, cols=cols, degrees=degrees, basis=basis,
            top_k=top_k, min_uplift=min_uplift,
            min_abs_mi_frac=min_abs_mi_frac,
            n_sample=500, random_state=random_state,
        )
    if scorer == "hsic":
        from ._orthogonal_hsic_fe import (
            hybrid_orth_mi_hsic_fe,
            hybrid_orth_mi_hsic_fe_with_recipes,
        )
        if with_recipes:
            return hybrid_orth_mi_hsic_fe_with_recipes(
                X, y, cols=cols, degrees=degrees, basis=basis,
                top_k=top_k, min_uplift=min_uplift,
                min_abs_mi_frac=min_abs_mi_frac,
                n_sample=500, random_state=random_state,
            )
        return hybrid_orth_mi_hsic_fe(
            X, y, cols=cols, degrees=degrees, basis=basis,
            top_k=top_k, min_uplift=min_uplift,
            min_abs_mi_frac=min_abs_mi_frac,
            n_sample=500, random_state=random_state,
        )
    if scorer == "jmim":
        from ._orthogonal_jmim_fe import (
            hybrid_orth_mi_jmim_fe,
            hybrid_orth_mi_jmim_fe_with_recipes,
        )
        if with_recipes:
            return hybrid_orth_mi_jmim_fe_with_recipes(
                X, y, cols=cols, degrees=degrees, basis=basis,
                top_k=top_k, min_uplift=min_uplift,
                min_abs_mi_frac=min_abs_mi_frac, n_bins=10,
            )
        return hybrid_orth_mi_jmim_fe(
            X, y, cols=cols, degrees=degrees, basis=basis,
            top_k=top_k, min_uplift=min_uplift,
            min_abs_mi_frac=min_abs_mi_frac, n_bins=10,
        )
    if scorer == "cmim":
        from ._orthogonal_cmim_fe import (
            hybrid_orth_mi_cmim_fe,
            hybrid_orth_mi_cmim_fe_with_recipes,
        )
        if with_recipes:
            return hybrid_orth_mi_cmim_fe_with_recipes(
                X, y, cols=cols, degrees=degrees, basis=basis,
                top_k=top_k, min_uplift=min_uplift,
                min_abs_mi_frac=min_abs_mi_frac, n_bins=10,
            )
        return hybrid_orth_mi_cmim_fe(
            X, y, cols=cols, degrees=degrees, basis=basis,
            top_k=top_k, min_uplift=min_uplift,
            min_abs_mi_frac=min_abs_mi_frac, n_bins=10,
        )
    if scorer == "tc":
        from ._orthogonal_total_correlation_fe import (
            hybrid_orth_mi_tc_fe,
            hybrid_orth_mi_tc_fe_with_recipes,
        )
        if with_recipes:
            return hybrid_orth_mi_tc_fe_with_recipes(
                X, y, cols=cols, degrees=degrees, basis=basis,
                top_k=top_k, min_uplift=min_uplift,
                min_abs_mi_frac=min_abs_mi_frac, n_bins=10,
            )
        return hybrid_orth_mi_tc_fe(
            X, y, cols=cols, degrees=degrees, basis=basis,
            top_k=top_k, min_uplift=min_uplift,
            min_abs_mi_frac=min_abs_mi_frac, n_bins=10,
        )
    raise ValueError(f"Unknown scorer {scorer!r}; expected one of " f"{(*META_SCORER_NAMES, 'tc')} (tc only via force_scorer).")


def hybrid_orth_mi_meta_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 0.95,
    min_abs_mi_frac: float = 0.05,
    force_scorer: Optional[str] = None,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict]:
    """Meta-scorer hybrid orth-poly FE.

    Fingerprint -> predict_best_scorer -> dispatch. ``force_scorer``
    bypasses the rule cascade and pins a specific scorer.

    Returns
    -------
    (X_augmented, scores, chosen_scorer, fingerprint)
        X_augmented : ``X`` with the chosen scorer's top-K winners appended.
        scores : the chosen scorer's full ranking DataFrame.
        chosen_scorer : the scorer name that was dispatched.
        fingerprint : the fingerprint dict (always returned, even when
            ``force_scorer`` is set, so callers can audit the rule
            cascade's prediction vs the override).
    """
    fp = fingerprint_signal(X, y, cols=cols, random_state=int(random_state))
    if force_scorer is not None:
        chosen = str(force_scorer).lower()
    else:
        chosen = predict_best_scorer(fp)
    result = _dispatch_scorer(
        chosen, X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        random_state=int(random_state),
        with_recipes=False,
    )
    X_aug, scores = result
    return X_aug, scores, chosen, fp


def hybrid_orth_mi_meta_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 0.95,
    min_abs_mi_frac: float = 0.05,
    force_scorer: Optional[str] = None,
    random_state: int = 0,
):
    """Same as :func:`hybrid_orth_mi_meta_fe` plus a list of
    ``orth_univariate`` recipes from the dispatched scorer -- one per
    appended column -- so that ``MRMR.transform`` can recompute each
    engineered column on test data without re-running the meta-cascade.
    """
    fp = fingerprint_signal(X, y, cols=cols, random_state=int(random_state))
    if force_scorer is not None:
        chosen = str(force_scorer).lower()
    else:
        chosen = predict_best_scorer(fp)
    result = _dispatch_scorer(
        chosen, X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        random_state=int(random_state),
        with_recipes=True,
    )
    X_aug, scores, recipes = result
    return X_aug, scores, recipes, chosen, fp
