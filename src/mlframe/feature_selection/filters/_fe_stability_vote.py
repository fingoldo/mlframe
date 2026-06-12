"""Cross-fold recipe stability voting (backlog #15, 2026-06-10).

A near-free CONSENSUS layer over the existing FE gates. The expensive MRMR FE
search runs ONCE on the full data (unchanged); this module adds a cheap K-fold
*confirmation*: every surviving ``unary_binary`` recipe is REPLAYED (leak-safe --
the recipe is fixed, only the rows change) on K held-out folds, its uplift gate
statistic recomputed on each fold, and the recipe admitted into the support only
if it clears the gate in ``>= ceil(quorum * K)`` folds.

Why this is orthogonal to what already ships
--------------------------------------------
* ``_stability_fe.py`` (Layer 36) does bootstrap voting but as a separate opt-in
  estimator that REFITS MRMR N times -- expensive, not default-wired. THIS reuses
  the single full-fit's recipes and only REPLAYS them on folds -> no refit, so the
  added cost is K cheap quantile-bin + plug-in-MI passes per recipe.
* The order-2 / order-3 maxT permutation floors kill the chance-MAX candidate
  WITHIN a fold (best-of-pool selection bias). This kills a different failure: a
  recipe that won only on a fold-specific QUIRK of the full-data split -- its
  uplift is carried by a handful of rows that happen to sit in the training
  split, and collapses on the held-out folds. maxT cannot see that; cross-fold
  voting can.

Leak-safety
-----------
``apply_recipe`` replays a recipe from its frozen ``extra`` (quantile edges,
prewarp coeffs, gate medians -- all pinned at full-fit time) WITHOUT any y
reference, so replaying on a held-out fold is leakage-free by construction. The
per-fold gate statistic compares ``MI(engineered_fold; y_fold)`` against the
fold's source-operand marginal MIs -- a held-out re-evaluation of the same
uplift the in-fit gate used.

Scope
-----
Only ``unary_binary`` recipes (the numeric pair-FE survivors built in
``_mrmr_fe_step``) are voted on -- they are the fold-specific-quirk-prone family
the backlog targets. Other recipe kinds (cat-FE encoders, orth-basis, cluster
aggregates, hinge / wavelet) are passed through untouched: they either carry no
per-fold gate analogue here or are produced by separate stages with their own
held-out validation. The voter is a no-op when fewer than 2 such recipes exist.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _fold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """K disjoint held-out fold index arrays (shuffled, near-equal sizes).

    A shuffled KFold partition: every row is the held-out test row of exactly
    one fold. ``rng`` makes the partition reproducible from the MRMR seed.
    """
    perm = rng.permutation(n)
    return [np.sort(part) for part in np.array_split(perm, k)]


def _marginal_mi(x_codes: np.ndarray, y_codes: np.ndarray) -> float:
    """Miller-Madow-debiased plug-in ``MI(X; Y)`` from integer bin codes.

    Reuses the exact CMI primitive the in-fit S5 engineered-redundancy gate
    scores with (``z_joint=None`` reduces it to marginal MI), so the held-out
    statistic is on the SAME debiased scale as the production estimator -- no
    second MI kernel, no scale skew.
    """
    from ._mi_greedy_cmi_fe import _cmi_from_binned

    return float(_cmi_from_binned(x_codes, y_codes, None))


def _recipe_clears_fold(
    *,
    eng_codes: np.ndarray,
    src_a_codes: Optional[np.ndarray],
    src_b_codes: Optional[np.ndarray],
    y_codes: np.ndarray,
    prevalence: float,
    alt_acceptance: bool = False,
) -> bool:
    """Whether ``eng_codes`` clears the held-out uplift gate on one fold.

    Mirrors the in-fit gate (``_mrmr_fe_step``): the engineered column's joint
    information with y must beat the sum of its source operands' marginal MIs by
    the prevalence ratio. When both operands' marginal MIs are ~0 (the canonical
    zero-marginal / XOR pair), any positive engineered MI clears -- exactly the
    in-fit ``ind_elems_mi_sum <= 0`` branch. Source codes may be ``None`` (the
    operand was an engineered parent whose raw column is not directly available);
    a missing operand contributes 0 to the marginal sum, matching the in-fit
    treatment of a zero-MI operand.

    ``alt_acceptance`` (2026-06-11): the recipe was admitted in-fit via the
    ALTERNATIVE acceptance path -- the per-operand learned PRE-WARP uplift gate or
    the gate_med path -- NOT the elementary ``eng_mi > sum_marg`` joint-prevalence
    gate. A prewarp recipe is a 1-D summary of a 2-D NON-monotone product whose
    whole reason for its dedicated acceptance path is that it CANNOT beat the raw
    operand marginal sum (the elementary library is representationally blind to the
    non-monotone inner); applying the ``eng_mi > sum_marg`` bar here would
    STRUCTURALLY drop every genuine prewarp recovery. For these recipes the
    held-out confirmation is the SAME signal test the prewarp gate uses -- the
    engineered column must carry genuine information about y -- so any positive
    held-out MI clears (the marginal-sum comparison does not apply). Empirically
    this only becomes load-bearing once a SECOND unary_binary recipe exists (the
    voter is a <2-recipe no-op otherwise); the auto-escalation residual complement
    introduced that second recipe and wrongly dropped the genuine prewarp capture.
    """
    eng_mi = _marginal_mi(eng_codes, y_codes)
    if eng_mi <= 0.0:
        return False
    if alt_acceptance:
        # Prewarp / gate_med alternative-acceptance recipe: genuine positive
        # held-out MI is the confirmation (the elementary marginal-sum bar does
        # not apply -- see docstring).
        return True
    sum_marg = 0.0
    if src_a_codes is not None:
        sum_marg += _marginal_mi(src_a_codes, y_codes)
    if src_b_codes is not None:
        sum_marg += _marginal_mi(src_b_codes, y_codes)
    if sum_marg <= 0.0:
        # Zero-marginal / hidden-pair (XOR): any positive engineered MI is signal.
        return True
    return eng_mi > sum_marg * prevalence


def confirm_recipes_cross_fold(
    *,
    recipes: dict,
    X: Any,
    y_codes: np.ndarray,
    feature_names_in: list,
    nbins: int,
    k: int = 5,
    quorum: float = 0.6,
    rng: Optional[np.random.Generator] = None,
    verbose: int = 0,
    # REJECTION-LEDGER out-param (additive, 2026-06-11): when a dict is passed, the voter
    # records ``{eng_name -> {"passes", "evaluated", "need_eff", "src_names"}}`` for every
    # FAILED recipe so the caller can append a per-gate rejection record with the real margin
    # (passes vs the quorum bar) WITHOUT recomputing the vote. Pure-record; never changes which
    # recipes fail. ``None`` (default) = legacy behaviour, no diagnostics captured.
    diagnostics_out: Optional[dict] = None,
) -> set:
    """Vote each ``unary_binary`` recipe across K held-out folds; return the
    set of engineered names that FAILED the quorum (caller drops them).

    Parameters
    ----------
    recipes
        ``{engineered_name -> EngineeredRecipe}`` (the live ``engineered_recipes``
        dict built during the FE step). Only ``unary_binary`` recipes are voted.
    X
        The (possibly augmented) fit-time frame. Must still carry every RAW
        ``feature_names_in`` column -- ``apply_recipe`` extracts source columns by
        name. A DataFrame (pandas / polars) or ndarray (column-name lookup then
        falls back through ``_extract_column``).
    y_codes
        Discretised target codes (``classes_y`` -- the SAME codes the MI sweep
        scored against), length == n_rows.
    feature_names_in
        Raw input feature names; a recipe whose ``src_names`` are all raw is
        directly fold-replayable. Nested-engineered operands replay recursively
        via ``apply_recipe`` and are scored with a ``None`` source-MI leg.
    nbins
        Quantile-bin count for re-binning the held-out source operands + the
        engineered column (matches ``quantization_nbins``).
    k
        Number of folds (>= 2). Below 2 the voter is a structural no-op.
    quorum
        Fraction of folds a recipe must clear. The pass bar is
        ``ceil(quorum * k)`` folds. ``quorum <= 0`` admits everything (no-op);
        values are clamped to (0, 1].
    rng
        Reproducible fold partition; defaults to a fresh default_rng.

    Returns
    -------
    Set of engineered names that did NOT clear the quorum (to be dropped). Empty
    when voting is a no-op (k < 2, quorum <= 0, fewer than 2 unary_binary
    recipes, or n too small to fold).
    """
    from .engineered_recipes import apply_recipe

    failed: set = set()
    if not recipes or k is None or int(k) < 2:
        return failed
    q = float(quorum)
    if q <= 0.0:
        return failed
    if q > 1.0:
        q = 1.0
    k = int(k)

    # Only the numeric pair-FE family is fold-prone enough to vote (see module docstring).
    voted = {nm: r for nm, r in recipes.items() if getattr(r, "kind", None) == "unary_binary"}
    if len(voted) < 2:
        return failed

    n = int(getattr(X, "shape", (0,))[0]) if hasattr(X, "shape") else 0
    y_codes = np.asarray(y_codes).ravel()
    if n <= 0 or y_codes.shape[0] != n:
        return failed
    # Need at least 2 rows per fold for a meaningful MI estimate.
    if n < 2 * k:
        return failed

    from ._mi_greedy_cmi_fe import _quantile_bin

    # TARGET ANCHOR RE-BINNING (IRON RULE / drop_redundant_raw_operands precedent,
    # _mrmr_fit_impl.py:6603). The ``classes_y`` codes the FE sweep scored against
    # are the SCREENING-level target binning -- on a skewed regression target that
    # is value-edge, HIGH-cardinality and HEAVILY imbalanced (e.g. 26 unequal
    # levels for ``y=a**2/b + log(c)*sin(d)``). Scoring the held-out uplift gate
    # with that anchor while the engineered column + source operands are re-binned
    # equi-frequency to ``nbins`` mismatches the two MI scales: the high-cardinality
    # target inflates the finite-sample joint-MI bias ASYMMETRICALLY (more for the
    # 2-D engineered leg than the 1-D source legs), which collapsed the uplift ratio
    # and DROPPED the genuine div(sqr(a),abs(b)) / mul(log(c),sin(d)) recipes (passes
    # 1/5). Re-bin the target equi-frequency to ``nbins`` so y, the engineered column
    # and the source operands share ONE balanced binning -> the ratio is faithful and
    # the genuine recipes clear the quorum. Monotone re-binning of integer codes
    # preserves the target's information ordering.
    y_codes = _quantile_bin(np.asarray(y_codes, dtype=np.float64), nbins=nbins)

    if rng is None:
        rng = np.random.default_rng()
    folds = _fold_indices(n, k, rng)
    need = int(math.ceil(q * k))

    raw_set = set(feature_names_in or [])

    # Helper: row-slice X for one fold, preserving the frame type apply_recipe expects.
    _is_pandas = hasattr(X, "iloc")
    _is_polars = hasattr(X, "schema") and not _is_pandas

    def _slice(idx: np.ndarray):
        if _is_pandas:
            return X.iloc[idx]
        if _is_polars:
            return X[idx]
        return X[idx]

    # Pre-extract per-source RAW column codes per fold is wasteful; instead bin
    # the fold's source operand on demand. Cache the full-frame raw column float
    # values once per source name so each fold only re-bins its slice.
    from .engineered_recipes._recipe_extract import _extract_column

    _raw_col_cache: dict = {}

    def _raw_codes(name: str, idx: np.ndarray) -> Optional[np.ndarray]:
        if name not in raw_set:
            return None  # engineered parent: scored with a None leg
        vals = _raw_col_cache.get(name)
        if vals is None:
            try:
                vals = np.asarray(_extract_column(X, name), dtype=np.float64).ravel()
            except Exception:
                return None
            _raw_col_cache[name] = vals
        if vals.shape[0] != n:
            return None
        return _quantile_bin(vals[idx], nbins=nbins)

    for eng_name, recipe in voted.items():
        src = tuple(getattr(recipe, "src_names", ()) or ())
        src_a = src[0] if len(src) >= 1 else None
        src_b = src[1] if len(src) >= 2 else None
        # ALTERNATIVE-ACCEPTANCE recipe detection (2026-06-11): a recipe whose
        # operand used the learned ``prewarp`` (or ``gate_med``) pseudo-unary was
        # admitted in-fit via the prewarp-uplift / gate alternative path, NOT the
        # elementary ``eng_mi > sum_marg`` gate -- so it must not be voted against
        # that bar (see ``_recipe_clears_fold``). The fit-time spec is persisted
        # flat in ``recipe.extra`` as ``prewarp_<side>_coef`` / ``gate_med_<side>_median``.
        _rx = getattr(recipe, "extra", None) or {}
        _alt_acceptance = any(
            _k in _rx for _k in (
                "prewarp_a_coef", "prewarp_b_coef",
                "gate_med_a_median", "gate_med_b_median",
            )
        )
        passes = 0
        evaluated = 0
        for idx in folds:
            X_fold = _slice(idx)
            try:
                eng_fold = np.asarray(apply_recipe(recipe, X_fold)).ravel()
            except Exception as exc:
                # A recipe that cannot replay on a fold is not penalised here --
                # transform()-time replay safety is a separate concern. Skip the
                # fold; if every fold fails to replay the recipe is left admitted.
                if verbose:
                    logger.info(
                        "Stability vote: recipe '%s' failed to replay on a fold (%s: %s); skipping fold.",
                        eng_name, type(exc).__name__, exc,
                    )
                continue
            if eng_fold.shape[0] != idx.shape[0]:
                continue
            eng_codes = _quantile_bin(np.asarray(eng_fold, dtype=np.float64), nbins=nbins)
            a_codes = _raw_codes(src_a, idx) if src_a is not None else None
            b_codes = _raw_codes(src_b, idx) if src_b is not None else None
            y_fold = y_codes[idx]
            evaluated += 1
            if _recipe_clears_fold(
                eng_codes=eng_codes,
                src_a_codes=a_codes,
                src_b_codes=b_codes,
                y_codes=y_fold,
                prevalence=1.0,
                alt_acceptance=_alt_acceptance,
            ):
                passes += 1
        # No fold could be evaluated (replay failed everywhere) -> leave admitted.
        if evaluated == 0:
            continue
        # Scale the quorum to the number of folds actually evaluated so a recipe
        # that could not replay on a fold is not unfairly penalised.
        need_eff = need if evaluated == k else int(math.ceil(q * evaluated))
        if passes < need_eff:
            failed.add(eng_name)
            if diagnostics_out is not None:
                diagnostics_out[eng_name] = {
                    "passes": int(passes),
                    "evaluated": int(evaluated),
                    "need_eff": int(need_eff),
                    "src_names": src,
                }
            if verbose:
                logger.info(
                    "Stability vote: DROPPED engineered '%s' -- cleared the held-out uplift "
                    "gate in only %d/%d folds (quorum %d).",
                    eng_name, passes, evaluated, need_eff,
                )

    if failed and verbose:
        logger.info(
            "Cross-fold stability vote: dropped %d/%d unary_binary recipe(s) as fold-specific "
            "(K=%d, quorum=%.2f).",
            len(failed), len(voted), k, q,
        )
    return failed
