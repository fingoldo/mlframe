"""Engineered-candidate helpers for single-predictor confirmation.

Directed-FE tie-break + per-signal prefer-engineered substitution + the RC2
conditioning-joint undersampling diagnostic. These read only ``ScreenContext``
fields and static name sets; they hold no ``ScreenContext``-entangled scoring
state, so they live apart from the score/confirm bodies in ``_confirm_predictor``.
"""

from __future__ import annotations

import re

import numpy as np

from .info_theory import merge_vars


def _conditioning_rows_per_cell(ctx, X: tuple) -> float:
    """Rows-per-nonempty-cell of the conditioning joint ``(X u selected_vars)``.

    RC2 sample-size diagnostic for the Fleuret conditional-MI permutation test.
    The conditional test estimates ``I(X; Y | Z)`` over the joint ``(X, Y, Z)``
    histogram; when that joint is severely undersampled (few rows per occupied
    cell) the plug-in conditional-MI estimate is dominated by finite-sample bias
    and the SHUFFLED-y null conditional MI is ~= the REAL conditional MI, so the
    permutation gate over-rejects every genuine feature after the first
    (proven on sklearn diabetes, n=442, s5 10-bin -> ~0.4 rows/cell).

    We measure the occupied-cell count of the FULL ``(X, Y, selected_vars)``
    joint - the exact support over which the conditional-MI permutation test
    operates (``I(X; Y | Z)`` plug-in needs the H(X,Y,Z) histogram). Counting
    only ``(X, Z)`` is too coarse: on diabetes after MDLP binning the
    ``(bmi, s5)`` joint occupies ~25 cells (~17.7 rows/cell, looks healthy) but
    adding the 10-bin target Y fragments it to ~1000 cells (~0.4 rows/cell) -
    the regime where the shuffled-y null CMI matches the real CMI. ``merge_vars``
    returns ``current_nclasses`` = number of NON-empty bins after pruning, which
    is exactly the occupied-cell count. Returns ``+inf`` when the joint is
    empty/degenerate so the strict conditional path is kept.
    """
    factors_data = ctx.factors_data
    n_samples = len(factors_data)
    if n_samples == 0:
        return float("inf")
    n_cols = factors_data.shape[1] if factors_data.ndim == 2 else 0
    cond_indices: list = []
    cond_indices.extend(int(c) for c in X)
    # Include the target Y: the conditional-MI estimator histograms (X, Y, Z).
    # In the standard MRMR setup ``targets_data is factors_data`` so y indexes
    # into ``factors_data``; guard the (rare) separate-target-array case by
    # skipping out-of-range y indices (the (X, Z) cell count is then a lower
    # bound on the support, still a valid undersampling signal).
    for yi in ctx.y:
        if 0 <= int(yi) < n_cols:
            cond_indices.append(int(yi))
    for z in ctx.selected_vars:
        if hasattr(z, "__len__"):
            cond_indices.extend(int(c) for c in z)
        else:
            cond_indices.append(int(z))
    try:
        _, _, n_nonempty_cells = merge_vars(
            factors_data=factors_data,
            vars_indices=cond_indices,
            var_is_nominal=None,
            factors_nbins=ctx.factors_nbins,
            dtype=np.int32,
        )
    except Exception:
        return float("inf")
    if n_nonempty_cells <= 0:
        return float("inf")
    return n_samples / float(n_nonempty_cells)


def _candidate_is_engineered(X, factors_names, raw_feature_names) -> bool:
    """True iff candidate ``X`` (single index or k-way tuple of cols-indices) is
    engineered, i.e. at least one of its component columns is NOT a raw input.

    When ``raw_feature_names`` (a set of the original pre-FE column names) is
    provided, a component is engineered iff its ``factors_names`` entry is not in
    that set -- the authoritative test. When it is ``None`` (direct callers that
    don't thread the raw-name set), fall back to the syntactic convention that
    engineered names contain ``(`` (functional forms like ``add(x0,x2)``,
    ``div(sqr(a),b)``) or ``__`` (basis transforms like ``x1__He2``); raw column
    names never contain those tokens in any mlframe FE producer.
    """
    for sub in X:
        try:
            name = factors_names[sub]
        except (IndexError, TypeError, KeyError):
            continue
        if raw_feature_names is not None:
            if name not in raw_feature_names:
                return True
        else:
            if ("(" in name) or ("__" in name):
                return True
    return False


def _prefer_engineered_order(order, expected_gains, ctx) -> np.ndarray:
    """Reorder the descending-gain ``order`` so that, among the leading run of
    candidates whose gain is within ``ctx.prefer_engineered_rel_eps`` (relative)
    of the top gain, engineered candidates precede raw ones.

    Gated and minimal: only the candidates tied (within rel-eps) with the current
    front-runner are eligible to move, and the relative order INSIDE the
    engineered group and inside the raw group is preserved (stable), so a clear
    winner -- one with no engineered peer within eps -- is never displaced. The
    promotion is deterministic and independent of the scoring backend because it
    reads only the (backend-identical) ``expected_gains`` array and the static
    raw-name set. Returns the (possibly) reordered index array.
    """
    rel_eps = float(getattr(ctx, "prefer_engineered_rel_eps", 0.0) or 0.0)
    if rel_eps <= 0.0 or len(order) < 2:
        return np.asarray(order)

    gains = np.asarray(expected_gains, dtype=np.float64)
    top_idx = order[0]
    top_gain = gains[top_idx]
    # Only meaningful when the leader carries a positive gain; near-ties at or
    # below zero are noise and must keep the legacy (index) ordering.
    if not (top_gain > 0.0):
        return np.asarray(order)

    factors_names = ctx.factors_names
    raw_names = getattr(ctx, "raw_feature_names", None)
    candidates = ctx.candidates
    tol = rel_eps * abs(top_gain)
    # An already-engineered leader stays first: it lands at the head of the
    # stable ``engineered_in_band`` group below, so the order is unchanged.

    leading, engineered_in_band, rest = [], [], []
    seen_band = True
    for _pos, cand_idx in enumerate(order):
        if seen_band and (top_gain - gains[cand_idx]) <= tol:
            # Within the relative-eps band of the leader.
            if _candidate_is_engineered(candidates[cand_idx], factors_names, raw_names):
                engineered_in_band.append(cand_idx)
            else:
                leading.append(cand_idx)
        else:
            seen_band = False
            rest.append(cand_idx)

    if not engineered_in_band:
        return np.asarray(order)  # no engineered peer in the band -> clear winner, untouched

    # Engineered band-members first (stable), then the raw band-members (stable),
    # then the untouched tail.
    return np.asarray(engineered_in_band + leading + rest, dtype=order.dtype)


# Split an engineered name into identifier-ish sub-tokens. ``__`` is a word char
# so a basis transform ``b__He2`` survives as a single token (its raw parent is
# the part before the FIRST ``__``); functional / cross forms ``div(sqr(a),b)``,
# ``c1*c2`` split on the operator/paren/comma punctuation into their operands.
_PARENT_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")


def _extract_single_raw_parent(cand, factors_names, raw_names):
    """Return the raw parent NAME iff candidate ``cand`` (an iterable of column
    indices) is ENGINEERED and is a function of EXACTLY ONE raw input column;
    otherwise ``None``.

    A component column counts as raw when its ``factors_names`` entry is in
    ``raw_names``. An engineered component contributes the raw column(s) it is
    built from: the operands of a functional/cross form (``div(sqr(a),b)`` ->
    {a, b}; ``c1*c2`` -> {c1, c2}) and, for a basis transform ``b__He2``, the
    prefix before the first ``__`` (``b``). Token equality against ``raw_names``
    is EXACT (``x1`` never matches inside ``x10``); the ``__``-prefix strip only
    fires when the whole token is not itself a raw name, so a raw column that
    legitimately contains ``__`` is preserved. Returns the parent only when the
    candidate references a single distinct raw column -- the unambiguous
    "transform of one parent" case the substitution is allowed to act on.
    """
    parents = set()
    is_engineered = False
    for sub in cand:
        try:
            name = factors_names[sub]
        except (IndexError, TypeError, KeyError):
            return None
        if name in raw_names:
            parents.add(name)
            continue
        is_engineered = True
        for tok in _PARENT_TOKEN_SPLIT.split(name):
            if not tok:
                continue
            if tok in raw_names:
                parents.add(tok)
            elif "__" in tok:
                base = tok.split("__", 1)[0]
                if base in raw_names:
                    parents.add(base)
        if len(parents) > 1:
            return None  # multi-parent -> not a single-parent transform
    if not is_engineered or len(parents) != 1:
        return None
    return next(iter(parents))


def _confirmable_engineered_child(ctx, X, winner_idx, winner_gain, expected_gains):
    """When a RAW single-feature candidate ``X`` just won confirmation, find an
    engineered single-parent transform of ``X`` that is within the
    prefer-engineered relative band of the raw winner's gain, and return it as a
    drop-in substitute. Returns ``(child_idx, child_cand, child_gain)`` or
    ``None``.

    Why a substitution and not a reorder: by the data-processing inequality an
    engineered transform ``E = f(P)`` can NEVER carry more mutual information
    about ``y`` than its raw parent ``P`` (``I(E; y) <= I(P; y)`` for any
    binning), so an MI/CMIM criterion ALWAYS scores the raw parent at or above
    its transform and the parent wins every near-tie. Yet ``E`` is strictly
    richer for a shallow downstream model (a linear model can exploit ``b**2-1``
    but not raw ``b`` for a quadratic signal) -- the whole point of the
    engineered feature. ``_prefer_engineered_order`` already encodes this
    preference, but only within the GLOBAL leader band; a secondary-signal pair
    (``b`` vs ``b__He2`` while some unrelated feature leads) never enters that
    band. This applies the SAME preference per-signal at the decision point:
    when the raw parent is the confirmed winner and its transform sits within
    ``prefer_engineered_rel_eps`` below it (the only side DPI allows) and the
    transform independently confirms, prefer the transform.

    Scope is narrow and raw-safe: only fires for a single-feature raw winner
    with a configured ``prefer_engineered_rel_eps``; the transform must reference
    exactly this one raw parent, be unselected/unfailed, and clear the band
    floor. The transform's effective gain is recovered from ``partial_gains``
    when CMIM pruning zeroed its ``expected_gains`` slot. No-op (returns ``None``)
    when prefer-engineered is disabled, no raw-name set is threaded, or no such
    transform exists -- legacy bit-stable.
    """
    raw_names = getattr(ctx, "raw_feature_names", None)
    rel_eps = float(getattr(ctx, "prefer_engineered_rel_eps", 0.0) or 0.0)
    if not raw_names or rel_eps <= 0.0 or len(X) != 1 or not (winner_gain > 0.0):
        return None
    factors_names = ctx.factors_names
    try:
        parent_name = factors_names[X[0]]
    except (IndexError, TypeError, KeyError):
        return None
    if parent_name not in raw_names:
        return None  # the winner is itself engineered -> nothing to substitute

    candidates = ctx.candidates
    partial_gains = getattr(ctx, "partial_gains", None) or {}
    added = ctx.added_candidates or set()
    failed = ctx.failed_candidates or set()
    selected = ctx.selected_vars or []
    band_floor = winner_gain * (1.0 - rel_eps)

    best_child = None  # (gain, idx, cand)
    for idx, cand in enumerate(candidates):
        if idx == winner_idx or idx in added or idx in failed:
            continue
        if any(sub in selected for sub in cand):
            continue
        if _extract_single_raw_parent(cand, factors_names, raw_names) != parent_name:
            continue
        g = float(expected_gains[idx]) if idx < len(expected_gains) else 0.0
        pg = partial_gains.get(idx)
        if pg is not None:
            try:
                g = max(g, float(pg[0]))
            except (TypeError, IndexError, ValueError):
                pass
        if g < band_floor or g >= 1e29:
            continue
        if best_child is None or g > best_child[0]:
            best_child = (g, idx, cand)

    if best_child is None:
        return None
    return best_child[1], best_child[2], best_child[0]
