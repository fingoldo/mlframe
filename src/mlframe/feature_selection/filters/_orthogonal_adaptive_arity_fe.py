"""Layer 78 (2026-06-01): ADAPTIVE-ARITY cross-basis FE.

Layers 22 (pair) / 56 (triplet) / 77 (quadruplet) each fix the cross-basis
arity at construction time. The user has to know in advance whether the
signal is 2-way, 3-way, or 4-way to pick the right module. For real-world
data the arity of the dominant interaction is unknown -- some signals are
genuinely bilinear, others 3-way XOR, others 4-way.

Layer 78 (adaptive arity) tries arity 2..max_arity for each seed tuple
and emits ONLY the winning arity per source set. Concretely, for the
source columns (x1, x2, x3, x4) we evaluate

    arity-2 cells: He_a(x1)*He_b(x2), He_a(x1)*He_b(x3), ...   (all C(4,2) pairs)
    arity-3 cells: He_a(x1)*He_b(x2)*He_c(x3), ...             (all C(4,3) triplets)
    arity-4 cell : He_a(x1)*He_b(x2)*He_c(x3)*He_d(x4)         (the C(4,4) quadruplet)

For each tuple at arity k we keep it ONLY if its MI strictly beats the
MI of every (k-1)-subset of the same tuple ("the lower-arity prefix
already explains it" check). The emit list collapses to exactly the
arity at which the new product carries information the smaller products
do not.

Why this matters
----------------

* Caller does not have to pick arity by hand; the right arity surfaces
  per source set.
* For a 2-way XOR target ``y = sign(x1*x2)`` the adaptive path emits
  the He_1(x1)*He_1(x2) pair and prunes the (x1,x2,x3)/(x1,x2,x3,x4)
  cells whose MI does NOT beat the pair's MI -- the pair already
  carries the signal.
* For a 3-way XOR target ``y = sign(x1*x2*x3)`` no pair beats the
  triplet (each pair has zero marginal MI). The triplet is emitted
  and the quadruplet x1*x2*x3*x4 is pruned because its MI does not
  exceed the (x1,x2,x3) triplet's MI -- the 4th leg is noise.
* For a 4-way XOR target ``y = sign(x1*x2*x3*x4)`` no triplet beats
  the quadruplet (each triplet has zero marginal MI). The quadruplet
  is the winner.

Cost guard
----------

Arity enumeration is O(sum_{k=2..max_arity} C(seed_k, k) * deg^k).
With defaults max_arity=3, seed_k=4, max_degree=1 we evaluate
C(4,2)+C(4,3)=6+4=10 candidates -- bounded.

Recipe parity
-------------

* arity-2 winners reuse the Layer 22 ``orth_pair_cross`` recipe.
* arity-3 winners reuse the Layer 56 ``orth_triplet_cross`` recipe.
* arity-4 winners reuse the Layer 77 ``orth_quadruplet_cross`` recipe.

No new recipe kind needed: the per-arity sibling modules already pin the
replay path.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .hermite_fe import basis_route_by_moments, _POLY_BASES
from ._orthogonal_univariate_fe import (
    _evaluate_basis_column,
    _mi_classif_batch, mi_classif_batch_chunked,
    _BASIS_CODE,
    hybrid_orth_mi_fe,
)

logger = logging.getLogger(__name__)

__all__ = [
    "generate_adaptive_arity_cross_basis",
    "score_adaptive_arity_cross_basis",
    "hybrid_orth_mi_adaptive_arity_fe",
    "hybrid_orth_mi_adaptive_arity_fe_with_recipes",
]


_ADAPTIVE_SCORE_EMPTY_COLS = [
    "engineered_col",
    "source_cols",
    "arity",
    "baseline_mi",          # max MI over lower-arity prefixes for THIS tuple
    "engineered_mi",
    "uplift",
]


def _adaptive_eng_col_name(
    cols: Sequence[str], basis: str, degrees: Sequence[int],
) -> str:
    """Stable name shared with the per-arity modules:
    ``"{c1}*{c2}*...__He{d1}_He{d2}_..."``.

    Matches the Layer 22 / 56 / 77 naming convention so the existing
    parsers (``score_*_cross_basis_by_mi_uplift``) and recipe builders
    can be reused without changes.
    """
    code = _BASIS_CODE.get(basis, basis)
    star = "*".join(cols)
    deg = "_".join(f"{code}{d}" for d in degrees)
    return f"{star}__{deg}"


def _coerce_y_classif(y) -> np.ndarray:
    """Match _mi_classif_batch contract: dense int64 labels.

    Non-integer y is densified via ``np.unique(return_inverse=...)`` rather than
    truncated with ``.astype(int64)`` -- plain truncation merges distinct labels
    and destroys continuous-y signal (everything in [0, 1) collapses to 0)."""
    y_arr = np.asarray(y).ravel()
    if np.issubdtype(y_arr.dtype, np.integer):
        return y_arr.astype(np.int64, copy=False)
    _, inv = np.unique(y_arr, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def generate_adaptive_arity_cross_basis(
    X: pd.DataFrame,
    y,
    source_cols: Sequence[str],
    *,
    max_arity: int = 3,
    max_degree: int = 1,
    basis: str = "hermite",
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate arity-2..max_arity cross-basis products over
    ``source_cols`` and keep ONLY the tuples whose MI strictly beats the
    MI of every lower-arity (k-1)-subset of the same tuple.

    For each candidate tuple at arity k:

    1. Evaluate every ``(deg_1, ..., deg_k)`` in ``[1..max_degree]^k``
       cell as ``prod_i basis(x_i)^{deg_i}``.
    2. Take the MAX MI across all degree combinations for the tuple
       (the tuple "wins" via its best degree assignment).
    3. Compare that to the MAX MI of every (k-1)-subset of the tuple
       evaluated the same way at arity k-1. If the arity-k MI strictly
       exceeds the max over (k-1)-subsets, KEEP the arity-k best cell;
       otherwise PRUNE the arity-k tuple.

    Parameters
    ----------
    X : DataFrame
        Source frame (numeric columns).
    y : array-like
        Classification target. Coerced to int64 for the MI batch path.
    source_cols : sequence of str
        Pool of source columns to enumerate cross-products over.
    max_arity : int
        Maximum arity. Default 3 keeps C(seed_k,2)+C(seed_k,3) bounded.
        max_arity in {2, 3, 4} is supported.
    max_degree : int
        Max per-leg degree. Default 1 -- He_1^k IS the dominant
        multiplicative k-way signal for every XOR-family target.
    basis : {'hermite', 'legendre', 'chebyshev', 'laguerre', 'auto'}
        Polynomial basis. 'auto' routes per-column via
        ``basis_route_by_moments``.
    nbins : int
        Quantisation bins for ``_mi_classif_batch``.

    Returns
    -------
    eng_X : DataFrame of the kept engineered columns (one per winning
        tuple at its winning arity), index aligned to ``X``.
    score_df : DataFrame with columns
        ``[engineered_col, source_cols, arity, baseline_mi,
        engineered_mi, uplift]`` sorted by uplift descending. Includes
        ONLY the winning tuples (lower-arity prefixes that won are
        listed; tuples whose MI did not beat their prefix are pruned).
    """
    if max_arity < 2:
        raise ValueError(f"max_arity must be >= 2, got {max_arity}")
    if max_arity > 4:
        raise ValueError(
            f"max_arity must be <= 4 (no per-arity module beyond Layer 77); "
            f"got {max_arity}"
        )
    src = [c for c in source_cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if len(src) < 2:
        return pd.DataFrame(index=X.index), pd.DataFrame(columns=_ADAPTIVE_SCORE_EMPTY_COLS)

    y_arr = _coerce_y_classif(y)
    max_d = max(1, int(max_degree))

    # Precompute per-column basis routing + the deg-by-deg basis values.
    basis_per_col: dict[str, str] = {}
    leg_cache: dict[tuple[str, int], np.ndarray] = {}
    for col in src:
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not finite_mask.all():
            fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
            x = np.where(finite_mask, x, fill)
        chosen = basis_route_by_moments(x) if basis == "auto" else basis
        if chosen not in _POLY_BASES:
            logger.warning(
                "generate_adaptive_arity_cross_basis: unknown basis %r for "
                "column %r; skipping", chosen, col,
            )
            continue
        basis_per_col[col] = chosen
        for d in range(1, max_d + 1):
            leg_cache[(col, d)] = _evaluate_basis_column(x, chosen, d)

    valid_src = [c for c in src if c in basis_per_col]
    if len(valid_src) < 2:
        return pd.DataFrame(index=X.index), pd.DataFrame(columns=_ADAPTIVE_SCORE_EMPTY_COLS)

    # tuple_best[arity][tuple-key] = (best_mi, best_deg_combo, best_col_name, product_array)
    # tuple-key is the frozenset of column names (commutative product).
    tuple_best: dict[int, dict[frozenset, tuple[float, tuple[int, ...], str, np.ndarray]]] = {
        k: {} for k in range(2, max_arity + 1)
    }

    def _all_deg_combos(k: int) -> list[tuple[int, ...]]:
        combos = [()]
        for _ in range(k):
            combos = [c + (d,) for c in combos for d in range(1, max_d + 1)]
        return combos

    # Pass 1: enumerate every (tuple, degree-combo) cell at every arity,
    # evaluate the product, compute MI in batch, find the best degree
    # combo per tuple.
    eval_results: dict[int, list[tuple[tuple[str, ...], tuple[int, ...], str, np.ndarray]]] = {
        k: [] for k in range(2, max_arity + 1)
    }
    for k in range(2, max_arity + 1):
        if len(valid_src) < k:
            continue
        deg_combos = _all_deg_combos(k)
        for tup in combinations(valid_src, k):
            for degs in deg_combos:
                prod = leg_cache[(tup[0], degs[0])].copy()
                for leg_idx in range(1, k):
                    prod = prod * leg_cache[(tup[leg_idx], degs[leg_idx])]
                # Per-leg basis for the name; arity-uniform basis code in the
                # name (matches Layer 22 / 56 / 77 stable naming convention).
                name_basis = basis_per_col[tup[0]]
                name = _adaptive_eng_col_name(tup, name_basis, degs)
                eval_results[k].append((tup, degs, name, prod))

    # Batch MI per arity to amortise the histogram + plug-in cost.
    for k in range(2, max_arity + 1):
        if not eval_results[k]:
            continue
        cols_block = np.column_stack([r[3] for r in eval_results[k]]).astype(np.float64, copy=False)
        mi_block = _mi_classif_batch(cols_block, y_arr, nbins=nbins)
        for (tup, degs, name, prod), mi_val in zip(eval_results[k], mi_block.tolist()):
            key = frozenset(tup)
            prev = tuple_best[k].get(key)
            if prev is None or float(mi_val) > prev[0]:
                tuple_best[k][key] = (float(mi_val), degs, name, prod)

    # Pass 2: for each arity k >= 3, prune tuples whose best MI does not
    # strictly exceed the max best MI over its (k-1)-subsets.
    out_cols: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    # Track which tuples win at SOME arity so the lower-arity prefix that
    # is dominated by a higher-arity winner can still be reported.
    # Adaptive contract: emit the arity at which MI is best AND lower
    # arity does not already match. Concretely we emit a tuple at arity k
    # iff (a) MI(k) > max MI(k-1 subsets), AND (b) the tuple is not
    # itself a strict subset of a higher-arity winner that ALSO beats it.
    # (b) is automatic when (a) holds at the higher arity.
    winners: dict[frozenset, tuple[int, float, tuple[int, ...], str, np.ndarray]] = {}
    # arity 2: always candidates (no lower arity to compare against). We
    # still gate on absolute MI > 0 to skip degenerate / constant cells.
    for key, (mi_val, degs, name, prod) in tuple_best[2].items():
        if mi_val > 0.0:
            winners[key] = (2, mi_val, degs, name, prod)
    # arity k >= 3: keep ONLY if MI strictly beats every (k-1)-subset MI.
    for k in range(3, max_arity + 1):
        for key, (mi_val, degs, name, prod) in tuple_best[k].items():
            # Max MI over (k-1)-subsets that themselves were evaluated.
            sub_max = 0.0
            for sub in combinations(tuple(key), k - 1):
                sub_key = frozenset(sub)
                sub_best = tuple_best[k - 1].get(sub_key)
                if sub_best is not None and sub_best[0] > sub_max:
                    sub_max = sub_best[0]
            if mi_val > sub_max:
                winners[key] = (k, mi_val, degs, name, prod)

    # Now collapse to ONE winner per maximal signal set. Two-way pruning:
    # (a) a HIGHER-arity winner with strictly larger MI eclipses every
    #     proper SUBSET winner -- "the triplet already explains the pair";
    # (b) a LOWER-arity winner with strictly larger MI eclipses every
    #     proper SUPERSET winner -- "the pair already explains the
    #     quadruplet, the extra legs are noise".
    # We process winners by MI descending so each high-MI cell eclipses
    # both directions before we evaluate lower-MI candidates.
    eclipsed: set[frozenset] = set()
    for key, (k, mi_val, degs, name, prod) in sorted(
        winners.items(), key=lambda kv: -kv[1][0],
    ):
        if key in eclipsed:
            continue
        # Eclipse proper SUBSETS (sub_k < k) with strictly smaller MI.
        for sub_k in range(2, k):
            for sub in combinations(tuple(key), sub_k):
                sub_key = frozenset(sub)
                if sub_key in winners and sub_key not in eclipsed:
                    if winners[sub_key][1] < mi_val:
                        eclipsed.add(sub_key)
        # Eclipse proper SUPERSETS (sup_k > k) with strictly smaller MI.
        # Enumerate stored winners at higher arities and test set
        # containment; cheaper than enumerating C(N, sup_k) supersets.
        for sup_key, (sup_k, sup_mi, _sd, _sn, _sp) in winners.items():
            if sup_key is key or sup_key in eclipsed:
                continue
            if sup_k > k and key.issubset(sup_key) and sup_mi < mi_val:
                eclipsed.add(sup_key)

    # Emit kept winners.
    for key, (k, mi_val, degs, name, prod) in winners.items():
        if key in eclipsed:
            continue
        # Recover the original tuple ORDER (sorted by valid_src order so
        # the name matches across calls deterministically).
        tup = tuple(c for c in valid_src if c in key)
        baseline_mi = 0.0
        for sub_k in range(2, k):
            for sub in combinations(tup, sub_k):
                sb = tuple_best[sub_k].get(frozenset(sub))
                if sb is not None and sb[0] > baseline_mi:
                    baseline_mi = sb[0]
        out_cols[name] = prod
        rows.append({
            "engineered_col": name,
            "source_cols": tup,
            "arity": k,
            "baseline_mi": baseline_mi,
            "engineered_mi": mi_val,
            "uplift": mi_val / (baseline_mi + 1e-12),
        })

    eng_X = pd.DataFrame(out_cols, index=X.index)
    score_df = pd.DataFrame(rows)
    if not score_df.empty:
        score_df = score_df.sort_values("uplift", ascending=False).reset_index(drop=True)
    else:
        score_df = pd.DataFrame(columns=_ADAPTIVE_SCORE_EMPTY_COLS)
    return eng_X, score_df


def score_adaptive_arity_cross_basis(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y,
    *,
    nbins: int = 10,
) -> pd.DataFrame:
    """Score every engineered adaptive-arity column against y. Mirrors
    the per-arity ``score_*_cross_basis_by_mi_uplift`` helpers but the
    baseline is the BEST individual raw leg MI, matching Layer 56/77.
    """
    y_arr = _coerce_y_classif(y)
    if engineered_X.empty:
        return pd.DataFrame(columns=_ADAPTIVE_SCORE_EMPTY_COLS)
    raw_cols = list(raw_X.columns)
    raw_mi = _mi_classif_batch(raw_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    eng_mi = mi_classif_batch_chunked(engineered_X, y_arr, nbins=nbins)
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        head = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        legs = head.split("*")
        arity = len(legs)
        baseline = max((float(raw_mi_map.get(leg, 0.0)) for leg in legs), default=0.0)
        emi = float(eng_mi[j])
        rows.append({
            "engineered_col": eng_name,
            "source_cols": tuple(legs),
            "arity": arity,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": emi / (baseline + 1e-12),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def hybrid_orth_mi_adaptive_arity_fe(
    X: pd.DataFrame,
    y,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    seed_k: int = 4,
    max_arity: int = 3,
    max_degree: int = 1,
    top_count: int = 3,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    adaptive_min_uplift: float = 1.05,
    adaptive_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage hybrid: (1) univariate Layer 21 ``hybrid_orth_mi_fe``,
    then (2) adaptive-arity cross-basis on the top-N raw source columns.

    Stage 2 evaluates arity 2..max_arity per seed tuple and keeps the
    winning arity per maximal signal set (see
    ``generate_adaptive_arity_cross_basis``).

    Parameters
    ----------
    X, y, cols, degrees, basis, top_k, min_uplift, min_abs_mi_frac, nbins
        Forwarded to the univariate ``hybrid_orth_mi_fe`` stage.
    seed_k : int
        Source pool cap for the adaptive stage. Default 4.
    max_arity : int
        Maximum arity in the adaptive stage. Default 3.
    max_degree : int
        Max per-leg degree. Default 1.
    top_count : int
        How many adaptive winners to append after the univariate winners.
    adaptive_min_uplift, adaptive_min_abs_mi_frac : float
        Two-gate thresholds for the adaptive stage. Compared against the
        BEST individual leg MI as the baseline.

    Returns
    -------
    (X_augmented, univariate_scores, adaptive_scores)
    """
    # Stage 1: univariate hybrid (Layer 21).
    X_aug_uni, uni_scores = hybrid_orth_mi_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
    )

    # Build the seed pool. Same caveat as Layer 56/77: on pure k-way XOR
    # every leg has zero univariate MI, so a raw-MI top-k cut would drop
    # signal legs. When the caller passes ``cols``, respect that order;
    # only fall back to raw-MI ranking when ``cols=None`` AND input
    # width > seed_k.
    raw_cols_all = [
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    seed_sources: list[str] = []
    if len(raw_cols_all) >= 2:
        if cols is not None:
            seed_sources = list(raw_cols_all[: int(seed_k)])
        else:
            y_arr = _coerce_y_classif(y)
            raw_X_all = X[raw_cols_all]
            raw_mi_arr = _mi_classif_batch(
                raw_X_all.to_numpy(dtype=np.float64), y_arr, nbins=nbins,
            )
            order = np.argsort(-raw_mi_arr)
            seed_sources = [raw_cols_all[i] for i in order[: int(seed_k)]]

    if len(seed_sources) < 2 or int(top_count) <= 0:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=_ADAPTIVE_SCORE_EMPTY_COLS)

    eng_X, scores = generate_adaptive_arity_cross_basis(
        X, y, seed_sources,
        max_arity=int(max_arity),
        max_degree=int(max_degree),
        basis=basis,
        nbins=nbins,
    )
    if eng_X.empty or scores.empty:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=_ADAPTIVE_SCORE_EMPTY_COLS)

    # Two-gate selection mirrors Layer 22 / 56 / 77. Re-score against the
    # RAW seed pool so the baseline reflects single-leg MI rather than
    # lower-arity-product MI.
    raw_X_seed = X[seed_sources]
    reranked = score_adaptive_arity_cross_basis(
        raw_X_seed, eng_X, y, nbins=nbins,
    )
    # Floor anchored on the largest raw + engineered MI signal.
    max_raw_baseline = float(reranked["baseline_mi"].max()) if not reranked.empty else 0.0
    if not uni_scores.empty:
        max_raw_baseline = max(max_raw_baseline, float(uni_scores["baseline_mi"].max()))
    max_eng = float(reranked["engineered_mi"].max()) if not reranked.empty else 0.0
    abs_floor = float(adaptive_min_abs_mi_frac) * max(max_raw_baseline, max_eng)
    qualified = reranked[
        (reranked["uplift"] >= float(adaptive_min_uplift))
        & (reranked["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_count))
    keep = list(winners["engineered_col"])
    if keep:
        X_aug = pd.concat([X_aug_uni, eng_X[keep]], axis=1)
    else:
        X_aug = X_aug_uni
    return X_aug, uni_scores, reranked


def hybrid_orth_mi_adaptive_arity_fe_with_recipes(
    X: pd.DataFrame,
    y,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    seed_k: int = 4,
    max_arity: int = 3,
    max_degree: int = 1,
    top_count: int = 3,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    adaptive_min_uplift: float = 1.05,
    adaptive_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_adaptive_arity_fe` plus a flat list of
    recipes so ``MRMR.transform`` can replay each engineered column.

    Per-arity recipes are routed to the existing Layer 22 / 56 / 77
    builders. No new recipe kind is introduced.
    """
    from .engineered_recipes import build_orth_univariate_recipe
    from ._orthogonal_quadruplet_fe_recipes import build_orth_quadruplet_cross_recipe

    X_aug, uni_scores, adaptive_scores = hybrid_orth_mi_adaptive_arity_fe(
        X, y, cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, seed_k=seed_k,
        max_arity=max_arity, max_degree=max_degree, top_count=top_count,
        min_uplift=min_uplift, min_abs_mi_frac=min_abs_mi_frac,
        adaptive_min_uplift=adaptive_min_uplift,
        adaptive_min_abs_mi_frac=adaptive_min_abs_mi_frac,
        nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}

    def _parse_code_deg(s: str):
        for code in ("LL", "He", "T", "L"):
            if s.startswith(code):
                rest = s[len(code):]
                if rest.isdigit():
                    return code_to_basis[code], int(rest)
        return None, None

    def _route_basis(col: str) -> str:
        if basis != "auto":
            return basis
        try:
            x = X[col].to_numpy(dtype=np.float64)
            return basis_route_by_moments(x)
        except Exception:
            return "hermite"

    recipes = []
    for name in appended:
        head = name.split("__", 1)[0]
        legs = head.split("*")
        arity = len(legs)
        suffix = name.split("__", 1)[1] if "__" in name else ""
        parts = suffix.split("_")
        if arity == 1:
            chosen_basis, chosen_degree = _parse_code_deg(parts[0]) if parts else (None, None)
            if chosen_basis is None or chosen_degree is None:
                logger.warning(
                    "hybrid_orth_mi_adaptive_arity_fe_with_recipes: cannot parse "
                    "univariate suffix in %r; skipping recipe.", name,
                )
                continue
            recipes.append(build_orth_univariate_recipe(
                name=name, src_name=legs[0],
                basis=chosen_basis, degree=chosen_degree,
            ))
        elif arity == 2:
            if len(parts) != 2:
                logger.warning(
                    "hybrid_orth_mi_adaptive_arity_fe_with_recipes: expected 2 "
                    "deg parts in %r; skipping recipe.", name,
                )
                continue
            basis_a, deg_a = _parse_code_deg(parts[0])
            basis_b, deg_b = _parse_code_deg(parts[1])
            if basis_a is None or basis_b is None:
                continue
            basis_a = _route_basis(legs[0])
            basis_b = _route_basis(legs[1])
            from .engineered_recipes import EngineeredRecipe
            recipes.append(EngineeredRecipe(
                name=name,
                kind="orth_pair_cross",
                src_names=(legs[0], legs[1]),
                extra={
                    "basis_i": basis_a, "basis_j": basis_b,
                    "deg_a": int(deg_a), "deg_b": int(deg_b),
                },
            ))
        elif arity == 3:
            if len(parts) != 3:
                continue
            basis_a, deg_a = _parse_code_deg(parts[0])
            basis_b, deg_b = _parse_code_deg(parts[1])
            basis_c, deg_c = _parse_code_deg(parts[2])
            if basis_a is None or basis_b is None or basis_c is None:
                continue
            basis_a = _route_basis(legs[0])
            basis_b = _route_basis(legs[1])
            basis_c = _route_basis(legs[2])
            from .engineered_recipes import EngineeredRecipe
            recipes.append(EngineeredRecipe(
                name=name,
                kind="orth_triplet_cross",
                src_names=(legs[0], legs[1], legs[2]),
                extra={
                    "basis_i": basis_a, "basis_j": basis_b, "basis_k": basis_c,
                    "deg_a": int(deg_a), "deg_b": int(deg_b), "deg_c": int(deg_c),
                },
            ))
        elif arity == 4:
            if len(parts) != 4:
                continue
            basis_a, deg_a = _parse_code_deg(parts[0])
            basis_b, deg_b = _parse_code_deg(parts[1])
            basis_c, deg_c = _parse_code_deg(parts[2])
            basis_d, deg_d = _parse_code_deg(parts[3])
            if (basis_a is None or basis_b is None
                or basis_c is None or basis_d is None):
                continue
            basis_a = _route_basis(legs[0])
            basis_b = _route_basis(legs[1])
            basis_c = _route_basis(legs[2])
            basis_d = _route_basis(legs[3])
            recipes.append(build_orth_quadruplet_cross_recipe(
                name=name,
                src_a_name=legs[0], src_b_name=legs[1],
                src_c_name=legs[2], src_d_name=legs[3],
                basis_i=basis_a, basis_j=basis_b,
                basis_k=basis_c, basis_l=basis_d,
                deg_a=deg_a, deg_b=deg_b, deg_c=deg_c, deg_d=deg_d,
            ))
        else:
            logger.warning(
                "hybrid_orth_mi_adaptive_arity_fe_with_recipes: unexpected arity "
                "%d in %r; skipping recipe.", arity, name,
            )
            continue
    return X_aug, uni_scores, adaptive_scores, recipes
