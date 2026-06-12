"""Generic MI-greedy feature constructor (Layer 26, 2026-05-31).

Sibling to ``_orthogonal_univariate_fe.py``. Where the hybrid orthogonal-
polynomial constructor enumerates basis_n(z) terms over a small library of
classical polynomials, THIS module enumerates a broad library of generic
unary / binary transforms (``log_abs(x)``, ``sqrt_abs(x)``, ``x_i / x_j``,
``|x_i - x_j|``, etc.) and ranks the candidates by MI uplift over their
source baseline -- the same two-gate selection the orthogonal pipeline uses.

The two constructors complement each other:

* orthogonal-poly path captures *smooth* low-degree non-linearities driven
  by the basis structure (``He_2``, ``T_3``, etc.). Excellent for
  ``y = sign(x^2 - c)`` / saddle / circle families.
* MI-greedy generic path captures *scale-changing* transforms (``log``,
  ``sqrt``, ``tanh``, ``expm1``) and *ratio / difference / max / min* pair
  interactions that polynomial bases miss. Excellent for
  ``y = sign(log|x|>c)``, ``y = sign(x_revenue / x_cost > 1)``,
  ``y = sign(max(x_i, x_j) > c)``.

Both share the same MI scorer (``score_features_by_mi_uplift`` is reused
from ``_orthogonal_univariate_fe``) and the same two-gate selection (relative
uplift + absolute MI floor) so the screening floors stay calibrated.

Recipe-based replay: every appended column carries an ``EngineeredRecipe``
of kind ``"mi_greedy_transform"`` so ``MRMR.transform`` can recompute the
column on test data deterministically (X-only, never references y).
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "UNARY_TRANSFORMS",
    "BINARY_TRANSFORMS",
    "TRIG_BOUNDED_TRANSFORMS",
    "iter_candidates",
    "generate_mi_greedy_features",
    "greedy_mi_fe_construct",
    "greedy_mi_fe_construct_with_recipes",
    "apply_mi_greedy_transform",
    "engineered_name_unary",
    "engineered_name_binary",
]


# ---------------------------------------------------------------------------
# Transform registry. Each callable takes ndarray input(s) and returns one
# ndarray of the same length. Outputs are scrubbed of NaN/Inf downstream.
# ---------------------------------------------------------------------------


def _log_abs(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(np.asarray(x, dtype=np.float64)))


def _sqrt_abs(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(np.asarray(x, dtype=np.float64)))


def _square(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a * a


def _cube(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a * a * a


def _reciprocal_safe(x: np.ndarray) -> np.ndarray:
    """``1 / (x + eps_with_sign)``. ``eps`` matches the sign of x so the
    sign of the output mirrors the sign of x for tiny / zero entries
    (cleanest for downstream MI binning)."""
    a = np.asarray(x, dtype=np.float64)
    # 1e-12 floor with sign tracking. ``np.sign(0) = 0`` so coerce to +1
    # so the denom is non-zero.
    sgn = np.sign(a)
    sgn = np.where(sgn == 0.0, 1.0, sgn)
    return 1.0 / (a + 1e-12 * sgn)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(x, dtype=np.float64))


def _expm1_clip(x: np.ndarray) -> np.ndarray:
    """``expm1`` with input clipped to [-20, 20] so ``expm1`` doesn't
    overflow on extreme inputs. The clip is wide enough to not lose any
    practical signal (expm1(20) ~= 4.85e8)."""
    a = np.clip(np.asarray(x, dtype=np.float64), -20.0, 20.0)
    return np.expm1(a)


def _abs(x: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(x, dtype=np.float64))


def _sin(x: np.ndarray) -> np.ndarray:
    return np.sin(np.asarray(x, dtype=np.float64))


def _cos(x: np.ndarray) -> np.ndarray:
    return np.cos(np.asarray(x, dtype=np.float64))


UNARY_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "log_abs": _log_abs,
    "sqrt_abs": _sqrt_abs,
    "square": _square,
    "cube": _cube,
    "reciprocal_safe": _reciprocal_safe,
    "tanh": _tanh,
    "expm1_clip": _expm1_clip,
    "abs": _abs,
}


# Trigonometric: only meaningful on BOUNDED columns. Auto-iter applies these
# only when the column range / std fingerprint says "bounded".
TRIG_BOUNDED_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": _sin,
    "cos": _cos,
}


def _bin_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)


def _bin_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)


def _bin_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) * np.asarray(b, dtype=np.float64)


def _bin_div_safe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a / (b + eps_with_sign)`` -- mirrors ``_reciprocal_safe`` to keep
    sign monotonicity on small denominators."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    sgn = np.sign(bv)
    sgn = np.where(sgn == 0.0, 1.0, sgn)
    return av / (bv + 1e-12 * sgn)


def _bin_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _bin_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _bin_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def _bin_ratio_log(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``log1p(|a|) - log1p(|b|)`` -- scale-stable ratio surrogate that
    never overflows and gives the same sign as ``log(|a/b|)`` for finite
    inputs. Often outperforms raw ``a / b`` on heavy-tailed positive cols
    because MI binning of the bare ratio is dominated by the tail."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    return np.log1p(np.abs(av)) - np.log1p(np.abs(bv))


BINARY_TRANSFORMS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "add": _bin_add,
    "sub": _bin_sub,
    "mul": _bin_mul,
    "div": _bin_div_safe,
    "max": _bin_max,
    "min": _bin_min,
    "abs_diff": _bin_abs_diff,
    "ratio_log": _bin_ratio_log,
}


# ---------------------------------------------------------------------------
# Naming helpers. The names below are PARSED at recipe-build time to recover
# (source_cols_tuple, transform_name); keep them stable.
# ---------------------------------------------------------------------------


def engineered_name_unary(col: str, transform: str) -> str:
    return f"{transform}({col})"


def engineered_name_binary(col_i: str, col_j: str, transform: str) -> str:
    return f"({col_i}__{transform}__{col_j})"


def _parse_unary_name(name: str) -> Optional[tuple[str, str]]:
    """``"transform(col)"`` -> ``(transform, col)`` or None on miss."""
    if not (name.endswith(")") and "(" in name):
        return None
    head, inner = name.split("(", 1)
    inner = inner[:-1]  # strip trailing ')'
    if head not in UNARY_TRANSFORMS and head not in TRIG_BOUNDED_TRANSFORMS:
        return None
    return head, inner


def _parse_binary_name(name: str) -> Optional[tuple[str, str, str]]:
    """``"(col_i__transform__col_j)"`` -> ``(transform, col_i, col_j)``
    or None on miss. The OUTER parens disambiguate from unary
    ``transform(col)`` so a column literally named ``"a__b"`` doesn't
    fool the parser.
    """
    if not (name.startswith("(") and name.endswith(")")):
        return None
    inner = name[1:-1]
    # Split on the FIRST ``__`` of every candidate transform key. Because the
    # transforms include ``add``, ``sub`` etc., we look for ``__<transform>__``
    # anchored at known transform names.
    for tname in BINARY_TRANSFORMS:
        token = f"__{tname}__"
        idx = inner.find(token)
        if idx >= 0:
            col_i = inner[:idx]
            col_j = inner[idx + len(token):]
            return tname, col_i, col_j
    return None


# ---------------------------------------------------------------------------
# Bounded-column fingerprint for trig auto-application.
# ---------------------------------------------------------------------------


def _is_bounded_column(x: np.ndarray) -> bool:
    """Heuristic ``bounded?`` check used to decide whether to enumerate
    ``sin`` / ``cos`` on a column.

    ``True`` when:
      * finite range / std <= 5.0 (a uniform U[a,b] hits ~3.46; bimodal
        mixture ~4; standard normal ~6 with one outlier; lognormal ~10+)
      * AND values stay within a closed bounded box: ``max - min <= 100``
        (catches "engineered angle in radians but values in degrees"
        cases where sin(degrees) is meaningless)

    Conservative on purpose: if uncertain, skip trig (extra deps are not
    free in the MI ranking; emitting cos(noise) only adds noise to the
    candidate pool).
    """
    a = np.asarray(x, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size < 8:
        return False
    rng = float(finite.max() - finite.min())
    sd = float(finite.std())
    if sd <= 0.0 or not np.isfinite(rng):
        return False
    if rng > 100.0:
        return False
    return (rng / sd) <= 5.0


# ---------------------------------------------------------------------------
# Candidate iterator.
# ---------------------------------------------------------------------------


def iter_candidates(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    include_unary: bool = True,
    include_binary: bool = True,
    include_trig_on_bounded: bool = True,
) -> Iterator[tuple[tuple[str, ...], str]]:
    """Yield ``(source_cols_tuple, transform_name)`` candidates.

    * Unary: ``((col,), transform_name)`` for every transform in
      ``UNARY_TRANSFORMS`` (and trig transforms when the column is
      ``_is_bounded_column``).
    * Binary: ``((col_i, col_j), transform_name)`` for every unordered
      pair of distinct numeric columns and every transform in
      ``BINARY_TRANSFORMS``. Pair ordering is canonicalised (lexicographic
      sort of names) so we don't enumerate (a, b) AND (b, a) for
      commutative transforms; non-commutative transforms (``sub``,
      ``div``, ``ratio_log``) still get both directions because the
      pair tuple is ordered, but they are emitted as two separate
      candidates with the natural column order each.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        cols = [c for c in cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]

    if include_unary:
        for c in cols:
            for tname in UNARY_TRANSFORMS:
                yield (c,), tname
            if include_trig_on_bounded:
                arr = X[c].to_numpy()
                if _is_bounded_column(arr):
                    for tname in TRIG_BOUNDED_TRANSFORMS:
                        yield (c,), tname

    if include_binary:
        # Commutative transforms: emit only one ordering per pair.
        # Non-commutative: emit both orderings.
        commutative = {"add", "mul", "max", "min", "abs_diff"}
        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i == j:
                    continue
                for tname in BINARY_TRANSFORMS:
                    if tname in commutative and i > j:
                        continue
                    yield (col_i, col_j), tname


# ---------------------------------------------------------------------------
# Materialisation + scoring.
# ---------------------------------------------------------------------------


def _materialise_candidate(
    X: pd.DataFrame, src_cols: tuple[str, ...], tname: str,
) -> Optional[np.ndarray]:
    """Compute ONE candidate column as a float64 ndarray. NaN/Inf scrubbed."""
    try:
        if len(src_cols) == 1:
            x = X[src_cols[0]].to_numpy()
            fn = UNARY_TRANSFORMS.get(tname) or TRIG_BOUNDED_TRANSFORMS.get(tname)
            if fn is None:
                return None
            out = fn(x)
        elif len(src_cols) == 2:
            a = X[src_cols[0]].to_numpy()
            b = X[src_cols[1]].to_numpy()
            fn_b = BINARY_TRANSFORMS.get(tname)
            if fn_b is None:
                return None
            out = fn_b(a, b)
        else:
            return None
    except Exception as exc:
        logger.warning(
            "mi_greedy: materialising %r(%r) raised %r; skipping.",
            tname, src_cols, exc,
        )
        return None
    out = np.asarray(out, dtype=np.float64)
    return np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def generate_mi_greedy_features(
    X: pd.DataFrame,
    candidates: Iterable[tuple[tuple[str, ...], str]],
) -> tuple[pd.DataFrame, list[tuple[tuple[str, ...], str]]]:
    """Materialise the engineered columns for the given candidates.

    Returns
    -------
    engineered : DataFrame
        Same index as ``X``. Columns named via ``engineered_name_unary`` /
        ``engineered_name_binary``. Constant or all-NaN candidates are
        silently dropped (zero MI contribution).
    parsed : list of (src_cols, tname)
        Aligned with the engineered DataFrame's columns; lets the caller
        rebuild recipes without re-parsing the names.
    """
    out_cols: dict[str, np.ndarray] = {}
    parsed: list[tuple[tuple[str, ...], str]] = []
    seen_names: set[str] = set()
    for src_cols, tname in candidates:
        if len(src_cols) == 1:
            name = engineered_name_unary(src_cols[0], tname)
        elif len(src_cols) == 2:
            name = engineered_name_binary(src_cols[0], src_cols[1], tname)
        else:
            continue
        if name in seen_names:
            continue
        arr = _materialise_candidate(X, src_cols, tname)
        if arr is None:
            continue
        # Drop constant or near-constant columns -- they carry zero MI and
        # clutter the score table.
        if float(arr.std()) <= 1e-12:
            continue
        out_cols[name] = arr
        parsed.append((src_cols, tname))
        seen_names.add(name)
    return pd.DataFrame(out_cols, index=X.index), parsed


def _greedy_score_and_select(
    raw_X: pd.DataFrame,
    engineered: pd.DataFrame,
    parsed: list[tuple[tuple[str, ...], str]],
    y: np.ndarray,
    *,
    top_k: int,
    min_uplift: float,
    min_abs_mi_frac: float,
    nbins: int,
    reject_sink: Optional[Callable[..., None]] = None,
) -> tuple[list[str], pd.DataFrame, list[tuple[tuple[str, ...], str]]]:
    """Score engineered columns vs the BETTER of the per-source raw baseline
    MIs, then apply the two-gate selection. Returns the winning column names,
    the full scores DataFrame, and the parsed list aligned with the winners.

    ``reject_sink`` (optional) is a callable invoked once per candidate the
    abs-MAD floor (``marginal_uplift_floor`` gate) kills -- i.e. a candidate
    that passed the uplift gate but missed ``abs_floor``. Pure instrumentation:
    it only RECORDS the already-computed ``engineered_mi`` vs ``abs_floor``; it
    changes NO selection decision. See the FE rejection ledger (W6).
    """
    from ._orthogonal_univariate_fe import _mi_classif_batch
    if engineered.empty:
        return [], pd.DataFrame(columns=[
            "engineered_col", "transform", "source_cols",
            "baseline_mi", "engineered_mi", "uplift",
        ]), []

    y_arr = (
        np.asarray(y).astype(np.int64)
        if not np.issubdtype(np.asarray(y).dtype, np.integer)
        else np.asarray(y, dtype=np.int64)
    )
    raw_mi = _mi_classif_batch(raw_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    raw_mi_map = dict(zip(list(raw_X.columns), raw_mi.tolist()))
    eng_mi = _mi_classif_batch(engineered.to_numpy(dtype=np.float64), y_arr, nbins=nbins)

    rows = []
    parsed_per_eng: list[tuple[tuple[str, ...], str]] = []
    for j, eng_name in enumerate(engineered.columns):
        src_cols, tname = parsed[j]
        baselines = [float(raw_mi_map.get(c, 0.0)) for c in src_cols]
        baseline = max(baselines) if baselines else 0.0
        emi = float(eng_mi[j])
        uplift = emi / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "transform": tname,
            "source_cols": tuple(src_cols),
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
        parsed_per_eng.append((src_cols, tname))

    scores = pd.DataFrame(rows).sort_values("uplift", ascending=False).reset_index(drop=True)

    # Two-gate selection. Use max(raw baseline, engineered_mi) as the absolute
    # floor reference so pure-interaction targets (XOR-like ratio) where ALL
    # raw baselines are noise-floor don't admit noise candidates via tiny-
    # baseline uplift inflation. Mirrors the cross-basis pair stage in
    # _orthogonal_univariate_fe.hybrid_orth_mi_pair_fe.
    max_raw = float(scores["baseline_mi"].max()) if not scores.empty else 0.0
    max_eng = float(scores["engineered_mi"].max()) if not scores.empty else 0.0
    legacy_floor = float(min_abs_mi_frac) * max(max_raw, max_eng)
    # Layer 27 (2026-05-31) noise-aware floor: when ALL raw + engineered
    # baselines sit in the noise band (typical all-noise frame), the
    # legacy ``frac * max(...)`` reference is itself noise -- any FP
    # candidate clears it. The mean+3*std reference is statistical:
    # columns drawn from the same noise distribution exceed it only on
    # the extreme tail, knocking the per-slot FP rate below 5%. On
    # real-signal frames the max baseline is many std above the mean, so
    # the legacy floor dominates and selection is unchanged.
    pool = np.concatenate([
        scores["baseline_mi"].to_numpy(),
        scores["engineered_mi"].to_numpy(),
    ]) if not scores.empty else np.array([])
    if pool.size >= 4:
        med = float(np.median(pool))
        mad = float(np.median(np.abs(pool - med)))
        # 1.4826 * MAD ~= std for a normal distribution; median-based
        # stats are robust to a few legitimate-signal outliers dragging
        # the mean up (which would lift the noise floor above true
        # signals -- false-negative regression observed at Layer 25
        # seed=13 when mean+3*std was used).
        noise_floor = med + 3.5 * 1.4826 * mad
    else:
        noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor)
    qualified = scores[
        (scores["uplift"] >= float(min_uplift))
        & (scores["engineered_mi"] >= abs_floor)
    ]
    # W6 abs-MAD floor instrumentation (pure-record; no decision change):
    # record every candidate that CLEARED the uplift gate but missed the
    # absolute floor -- the ``marginal_uplift_floor`` gate kill the session
    # previously had to diagnose by hand. Selection is byte-identical with or
    # without the sink.
    if reject_sink is not None and not scores.empty:
        _killed = scores[
            (scores["uplift"] >= float(min_uplift))
            & (scores["engineered_mi"] < abs_floor)
        ]
        for _row in _killed.itertuples(index=False):
            try:
                reject_sink(
                    gate="marginal_uplift_floor",
                    candidate=str(_row.engineered_col),
                    operands=tuple(_row.source_cols),
                    operator=str(_row.transform),
                    observed=float(_row.engineered_mi),
                    threshold=float(abs_floor),
                    reason="mi_greedy abs-MAD floor: engineered_mi below med+k*MAD noise floor",
                )
            except Exception:
                pass
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    # parsed list aligned with the winners (in winner order).
    name_to_parsed = dict(zip(list(engineered.columns), parsed_per_eng))
    parsed_winners = [name_to_parsed[n] for n in keep]
    return keep, scores, parsed_winners


def greedy_mi_fe_construct(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    seed_cols_count: int = 5,
    top_k: int = 5,
    include_unary: bool = True,
    include_binary: bool = True,
    include_trig_on_bounded: bool = True,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    reject_sink: Optional[Callable[..., None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end MI-greedy feature constructor.

    Pipeline:
      1. Pick the top-N source columns by raw MI(x; y) (``seed_cols_count``).
         Restricts the candidate explosion: with C transforms and N source
         columns we enumerate ~N unary + N*(N-1) binary candidates per
         non-commutative transform, capped well under O(p^2 * C).
      2. Stream candidates via ``iter_candidates`` over those seed columns.
      3. Materialise every candidate into a working DataFrame.
      4. Batch MI score; apply the two-gate selection.
      5. Append winners to X; return (X_augmented, scores).

    Engineered column naming: ``f"{transform}({col})"`` for unary,
    ``f"({col_i}__{transform}__{col_j})"`` for binary.
    """
    from ._orthogonal_univariate_fe import _mi_classif_batch

    # 1. Seed pool.
    candidates_pool = [
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    if not candidates_pool:
        return X.copy(), pd.DataFrame(columns=[
            "engineered_col", "transform", "source_cols",
            "baseline_mi", "engineered_mi", "uplift",
        ])
    y_arr = (
        np.asarray(y).astype(np.int64)
        if not np.issubdtype(np.asarray(y).dtype, np.integer)
        else np.asarray(y, dtype=np.int64)
    )
    raw_arr = X[candidates_pool].to_numpy(dtype=np.float64)
    raw_mi = _mi_classif_batch(raw_arr, y_arr, nbins=nbins)
    order = np.argsort(-raw_mi)
    seed_cols = [candidates_pool[i] for i in order[: int(seed_cols_count)]]

    # 2. Enumerate candidates.
    candidates = list(iter_candidates(
        X, cols=seed_cols,
        include_unary=include_unary,
        include_binary=include_binary,
        include_trig_on_bounded=include_trig_on_bounded,
    ))

    # 3-4. Materialise + score + select.
    engineered, parsed = generate_mi_greedy_features(X, candidates)
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=[
            "engineered_col", "transform", "source_cols",
            "baseline_mi", "engineered_mi", "uplift",
        ])
    raw_X_for_baseline = X[seed_cols]
    keep, scores, _parsed_winners = _greedy_score_and_select(
        raw_X_for_baseline, engineered, parsed, y_arr,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
        reject_sink=reject_sink,
    )

    # 5. Append winners.
    if keep:
        X_aug = pd.concat([X, engineered[keep]], axis=1)
    else:
        X_aug = X.copy()
    return X_aug, scores


# ---------------------------------------------------------------------------
# Recipe-aware wrapper for MRMR.fit auto-wiring.
# ---------------------------------------------------------------------------


def greedy_mi_fe_construct_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    seed_cols_count: int = 5,
    top_k: int = 5,
    include_unary: bool = True,
    include_binary: bool = True,
    include_trig_on_bounded: bool = True,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """Same as :func:`greedy_mi_fe_construct` but additionally returns a list of
    ``EngineeredRecipe`` objects (one per appended column) so MRMR.transform
    can replay each column on test data without re-running the MI ranking
    AND without referencing y.

    Returns
    -------
    (X_augmented, scores, recipes)
    """
    from .engineered_recipes import build_mi_greedy_transform_recipe

    X_aug, scores = greedy_mi_fe_construct(
        X, y,
        cols=cols, seed_cols_count=seed_cols_count, top_k=top_k,
        include_unary=include_unary, include_binary=include_binary,
        include_trig_on_bounded=include_trig_on_bounded,
        min_uplift=min_uplift, min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
        reject_sink=reject_sink,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    for name in appended:
        # Parse the name back to (src_cols, transform). The naming is stable
        # so this is deterministic for any column whose source name doesn't
        # itself contain a ``__<transform>__`` substring (rare; we don't
        # silently overwrite user column names).
        parsed_bin = _parse_binary_name(name)
        parsed_un = _parse_unary_name(name)
        if parsed_bin is not None:
            tname, col_i, col_j = parsed_bin
            recipes.append(build_mi_greedy_transform_recipe(
                name=name, transform=tname, src_names=(col_i, col_j),
            ))
        elif parsed_un is not None:
            tname, col = parsed_un
            recipes.append(build_mi_greedy_transform_recipe(
                name=name, transform=tname, src_names=(col,),
            ))
        else:
            logger.warning(
                "greedy_mi_fe_construct_with_recipes: cannot parse engineered "
                "column %r back to (transform, source); skipping recipe.",
                name,
            )
    return X_aug, scores, recipes


# ---------------------------------------------------------------------------
# Recipe replay helper -- imported by engineered_recipes._apply_mi_greedy_transform.
# Defined here so the transform registry lookup lives next to the registry
# itself (single source of truth) instead of being duplicated inside the
# recipes module.
# ---------------------------------------------------------------------------


def apply_mi_greedy_transform(
    transform: str, src_values: list[np.ndarray],
) -> np.ndarray:
    """Replay a single MI-greedy transform. Used by ``_apply_mi_greedy_transform``
    in ``engineered_recipes.py``. Centralises the transform-name -> callable
    lookup so a single edit to ``UNARY_TRANSFORMS`` / ``BINARY_TRANSFORMS``
    propagates to BOTH fit-time materialisation and transform-time replay.
    """
    if len(src_values) == 1:
        fn = UNARY_TRANSFORMS.get(transform) or TRIG_BOUNDED_TRANSFORMS.get(transform)
        if fn is None:
            raise KeyError(
                f"apply_mi_greedy_transform: unknown unary transform "
                f"{transform!r}; known: {sorted(UNARY_TRANSFORMS) + sorted(TRIG_BOUNDED_TRANSFORMS)}"
            )
        out = fn(src_values[0])
    elif len(src_values) == 2:
        fn_b = BINARY_TRANSFORMS.get(transform)
        if fn_b is None:
            raise KeyError(
                f"apply_mi_greedy_transform: unknown binary transform "
                f"{transform!r}; known: {sorted(BINARY_TRANSFORMS)}"
            )
        out = fn_b(src_values[0], src_values[1])
    else:
        raise ValueError(
            f"apply_mi_greedy_transform: src_values length {len(src_values)} "
            f"unsupported (only unary / binary registered)."
        )
    out = np.asarray(out, dtype=np.float64)
    return np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
