"""Recipe-application layer for the Haar wavelet basis FE: :func:`generate_wavelet_features`,
:func:`build_orth_wavelet_recipe`, :func:`_apply_orth_wavelet`, :func:`hybrid_wavelet_fe_with_recipes`.

Carved out of ``_wavelet_basis_fe.py`` to keep it under the 1k LOC ceiling; the MI-scoring kernels
(``_dyadic_haar_leg``, ``_select_wavelet_legs``, ``_heldout_incremental_mi*``) and the module's tuning
constants stay in the parent, imported back here.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from mlframe.utils.log_throttle import log_throttle

from ._wavelet_basis_fe import (
    _WAVELET_MAX_LEGS,
    _WAVELET_MAX_SCALE,
    _WAVELET_MIN_INCR_MI,
    _WAVELET_SCALE_SIGMA,
    _WAVELET_SMOOTH_COMPLEMENT_RATIO,
    _dyadic_haar_leg,
    _heldout_incremental_mi,
    _heldout_incremental_mi_from_prep,
    _heldout_incremental_mi_prep,
    _select_wavelet_legs,
)

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)


def generate_wavelet_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    y: Optional[np.ndarray] = None,
    max_scale: int = _WAVELET_MAX_SCALE,
    max_legs: int = _WAVELET_MAX_LEGS,
    scale_sigma: float = _WAVELET_SCALE_SIGMA,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
    feature_dtype: npt.DTypeLike = np.float32,
    max_cols: Optional[int] = None,
) -> tuple[pd.DataFrame, dict]:
    """For each numeric column, held-out-select a small dyadic Haar leg set and
    emit the legs, returning the columns alongside the per-column fit meta needed
    to build leak-safe recipes.

    Parameters
    ----------
    X : DataFrame
        Source frame. Only numeric columns are processed; non-numeric skipped.
    cols : sequence of column names, optional
        Columns to expand. None = all numeric columns.
    y : array-like, optional
        Target. Consulted ONLY by the held-out scale-selection (which legs carry
        held-out MI). Never read for the emitted column VALUE, so the recipe
        replay stays leakage-free / y-independent.
    max_scale : int
        Finest dyadic scale j (default 3 -> scales 0..3, <= 15 candidate legs).
    max_legs : int
        Hard cap on emitted legs per column after selection (candidate control).
    max_cols : int, optional
        2026-07-09 fix: cap on how many columns run the EXPENSIVE held-out scale-selection
        (``_select_wavelet_legs``, which internally calls ``_binned_mi`` per candidate leg).
        Profiled as the second-largest default-ON pre-FE cost on a wide-p fit (~26% of the
        pre-categorize wall at p=420), scaling with column count regardless of row count.
        ``None`` (default) = unlimited, byte-identical legacy behaviour. Unlike the Fourier
        extra-basis cap, columns beyond this cap get NO wavelet legs at all (there is no cheap
        fallback basis for wavelets the way Fourier has a fixed grid) - set only when the wide
        wall-time cost is a bigger concern than wavelet recall on the excluded columns.

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns named
            ``"{col}__haar_j{j}k{k}"`` (the leg ``psi_{j,k}``).
        meta : dict mapping each emitted column name to the recipe metadata
            ``{"src": col, "j": int, "k": int, "lo": float, "span": float}``.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if y is None:
        # Scale-selection is supervised; with no y there is nothing to validate.
        return pd.DataFrame(index=X.index), {}
    if dedup_collinear_sources:
        from ._orthogonal_univariate_fe import _dedup_collinear_source_cols
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    y_arr = np.asarray(y).ravel()
    if y_arr.size != len(X):
        return pd.DataFrame(index=X.index), {}
    out_cols: dict = {}
    meta: dict = {}
    from ._fe_deadline import fe_deadline_passed

    for _col_idx, col in enumerate(cols):
        if max_cols is not None and _col_idx >= int(max_cols):
            break
        # Optional-enrichment wall-clock budget: stop the per-column held-out scale-selection scan once
        # MRMR.fit's deadline passes; return whatever legs were engineered so far. No-op without a budget
        # (mirrors the orth-univariate/pair-cross/extra-basis generators' internal deadline check).
        if fe_deadline_passed():
            break
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            continue
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            continue
        if not finite_mask.all():
            # A wavelet basis over a NaN column is unsound: the nanmean-imputed wavelet becomes a missingness proxy that displaces the genuine
            # missingness-FE columns, and the recipe replay does not impute (transform() emits all-NaN). Skip; the missingness signal belongs to the
            # dedicated missingness-FE family.
            continue
        xf = x[np.isfinite(x)]
        lo = float(xf.min())
        hi = float(xf.max())
        span = max(hi - lo, 1e-12)
        try:
            legs = _select_wavelet_legs(
                x, y_arr, lo, span,
                max_scale=max_scale, max_legs=max_legs, scale_sigma=scale_sigma,
                return_arrays=True,
            )
        except Exception as exc:
            log_throttle(
                logger,
                "wavelet_basis_fe_scale_select_failed",
                logging.WARNING,
                "generate_wavelet_features: scale-select on col=%r raised %r; "
                "skipping wavelet for that column.", col, exc,
            )
            continue
        for j, k, leg_arr in legs:
            # Reuse the array _select_wavelet_legs already built to rank this survivor instead of rebuilding
            # it from scratch; a plain dtype cast (when feature_dtype differs from the selection's default
            # float32) is cheaper than a fresh zeros_like + two boolean-mask assigns, and exact - the leg
            # only ever holds {-1, 0, +1}, bit-identically representable in any float dtype.
            leg = leg_arr if leg_arr.dtype == np.dtype(feature_dtype) else leg_arr.astype(feature_dtype)
            if float(np.std(leg)) <= 1e-12:
                continue
            name = f"{col}__haar_j{j}k{k}"
            out_cols[name] = leg
            meta[name] = {
                "src": col, "j": int(j), "k": int(k),
                "lo": float(lo), "span": float(span),
            }
    return pd.DataFrame(out_cols, index=X.index), meta


def build_orth_wavelet_recipe(
    *, name: str, src_name: str, j: int, k: int, lo: float, span: float,
) -> "EngineeredRecipe":
    """Frozen recipe for one Haar wavelet basis column ``psi_{j,k}(z)`` where
    ``z = clip((X[src_name] - lo) / span, 0, 1)`` with the dyadic ``(j, k)`` and
    ``(lo, span)`` fixed at fit time.

    Replay is closed-form in the source column alone - no y reference captured,
    so ``transform`` is leakage-free by construction. Mirrors
    ``build_orth_spline_recipe`` (store basis params + ``lo``/``span``, replay a
    closed-form basis function of x)."""
    from .engineered_recipes import EngineeredRecipe
    return EngineeredRecipe(
        name=name,
        kind="orth_wavelet",
        src_names=(str(src_name),),
        extra={
            "j": int(j), "k": int(k),
            "lo": float(lo), "span": float(span),
        },
    )


def _apply_orth_wavelet(recipe, X) -> np.ndarray:
    """Replay one Haar wavelet basis column from the stored ``(j, k, lo, span)``
    - a pure function of the source column (no y). Mirrors ``_apply_orth_spline``.
    """
    from .engineered_recipes import _extract_column
    if len(recipe.src_names) != 1:
        raise ValueError(f"orth_wavelet recipe '{recipe.name}' must have exactly 1 " f"src_names; got {len(recipe.src_names)}")
    for key in ("j", "k", "lo", "span"):
        if key not in recipe.extra:
            raise KeyError(f"orth_wavelet recipe '{recipe.name}' missing '{key}' in extra. " f"Re-fit MRMR to regenerate.")
    name = recipe.src_names[0]
    j = int(recipe.extra["j"])
    k = int(recipe.extra["k"])
    lo = float(recipe.extra["lo"])
    span = max(float(recipe.extra["span"]), 1e-12)
    vals = np.asarray(_extract_column(X, name), dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.all():
        fill = float(np.nanmean(vals[finite])) if finite.any() else 0.0
        vals = np.where(finite, vals, fill)
    z = np.clip((vals - lo) / span, 0.0, 1.0)
    return _dyadic_haar_leg(z, j, k)


def hybrid_wavelet_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    max_scale: int = _WAVELET_MAX_SCALE,
    max_legs: int = _WAVELET_MAX_LEGS,
    scale_sigma: float = _WAVELET_SCALE_SIGMA,
    top_k: int = 8,
    min_incr_mi: float = _WAVELET_MIN_INCR_MI,
    smooth_complement_ratio: float = _WAVELET_SMOOTH_COMPLEMENT_RATIO,
    nbins: int = 10,
    feature_dtype: npt.DTypeLike = np.float32,
    max_cols: Optional[int] = None,
    **_legacy_ignored: object,
) -> tuple[pd.DataFrame, list, list, pd.DataFrame]:
    """Haar wavelet basis FE + held-out-incremental-MI selection, returning
    leak-safe recipes. Returns ``(X_augmented, appended_names, recipes, scores)``.

    Four self-limiting bounds keep the candidate count small AND make the operator
    complementary (it adds legs only where a localized contrast genuinely sharpens
    y beyond a SMOOTH refinement of raw x, and stays silent on smooth / noise):

    1. :func:`generate_wavelet_features` already held-out-scale-selected a small
       dyadic leg set per column (the noise-aware held-out MAD floor over the
       candidate legs' held-out MIs + the ``max_legs`` cap), so pure noise emits
       NO leg to score here.
    2. Each surviving leg must clear the held-out INCREMENTAL MI floor
       (``leg_incr >= min_incr_mi`` on the ``%3`` slice): the joint
       ``MI(y; [bin(x), leg])`` must beat ``MI(y; bin(x))`` by an absolute margin
       - the leg sharpens y BEYOND what the coarse raw column already says.
    3. COMPLEMENTARITY GUARD: the leg's incremental MI must also exceed
       ``smooth_complement_ratio`` x the SMOOTH-refinement gain (what finer
       location-only binning of raw x adds over the same coarse baseline). On a
       SMOOTH (sin / monotone) column the smooth refinement dominates -> the leg is
       rejected (the global Fourier basis owns that regime, complementarity); on a
       LOCALIZED step / contrast the leg dominates -> admitted.
    4. ``top_k`` caps the survivors per fit.

    Why the incremental gate, NOT the naive leg-MI-vs-raw-MI uplift the spline /
    Fourier path uses: a localized target ``y`` is a FUNCTION of x in a sub-window,
    so binned raw x already scores HIGH marginal MI and a single leg's marginal MI
    sits BELOW it -> uplift < 1 -> the genuine localized leg is wrongly dropped
    (the same trap the monotone hinge hit, but here for a non-monotone leg). The
    incremental MI conditions on raw x and so measures exactly the localized value
    the wavelet adds. A Haar leg is NON-monotone -> it is MI-VISIBLE (unlike the
    monotone hinge / isotonic, which need a held-out LINEAR-usability gate), so the
    statistic here is MI-based, just conditioned on raw x.

    ``scores`` is the full per-leg ranking (incr_mi, smooth_gain, passed flag;
    winners + rejects).
    """
    engineered, meta = generate_wavelet_features(
        X, cols=cols, y=y,
        max_scale=max_scale, max_legs=max_legs, scale_sigma=scale_sigma,
        feature_dtype=feature_dtype, max_cols=max_cols,
    )
    _empty_cols = [
        "engineered_col", "source_col", "incr_mi", "smooth_gain", "passed",
    ]
    if engineered.empty:
        return X, [], [], pd.DataFrame(columns=_empty_cols)
    # y -> discrete class codes for the binned joint-MI gate; bin continuous y.
    y_arr = np.asarray(y).ravel()
    if not np.issubdtype(y_arr.dtype, np.integer) or np.unique(y_arr).size > 20:
        try:
            y_codes = pd.qcut(
                pd.Series(y_arr), q=min(nbins, max(2, np.unique(y_arr).size)),
                labels=False, duplicates="drop",
            ).to_numpy()
            y_codes = np.where(np.isfinite(y_codes), y_codes, 0).astype(np.int64)
        except Exception as e:
            logger.debug("y quantile-binning failed, falling back to all-zero codes: %s", e)
            y_codes = np.zeros(y_arr.size, dtype=np.int64)
    else:
        y_codes = y_arr.astype(np.int64)
    rows = []
    # Most of _heldout_incremental_mi's work (x_src extraction/fill, xc, base_mi, the 8-shuffle permutation-
    # null baseline, xc_fine/fine_mi/smooth_gain) depends only on (src, y_codes, nbins), not on the leg - up
    # to _WAVELET_MAX_LEGS legs per source column used to redo it identically. Group by src and cache the prep.
    _prep_cache: dict = {}
    for name in engineered.columns:
        m = meta.get(name, {})
        src = str(m.get("src", name.split("__", 1)[0]))
        leg_vals = engineered[name].to_numpy()
        if src in X.columns and pd.api.types.is_numeric_dtype(X[src]):
            if src not in _prep_cache:
                x_src = np.asarray(X[src].to_numpy(), dtype=np.float64)
                finite = np.isfinite(x_src)
                if not finite.all():
                    x_src = np.where(
                        finite, x_src,
                        float(np.nanmean(x_src[finite])) if finite.any() else 0.0,
                    )
                _prep_cache[src] = _heldout_incremental_mi_prep(x_src, y_codes, nbins=nbins)
            incr, smooth_gain = _heldout_incremental_mi_from_prep(_prep_cache[src], leg_vals)
        else:
            # src isn't a real numeric column in X (rare/defensive): each leg scores against ITS OWN values,
            # exactly as the pre-grouping code did, so nothing is cacheable across legs in this branch.
            x_src_fallback = engineered[name].to_numpy(dtype=np.float64)
            incr, smooth_gain = _heldout_incremental_mi(x_src_fallback, leg_vals, y_codes, nbins=nbins)
        # Two-condition admission: (a) absolute incremental floor, (b) the leg
        # beats the smooth-refinement competitor (complementarity guard).
        passed = bool((incr >= float(min_incr_mi)) and (incr >= float(smooth_complement_ratio) * max(smooth_gain, 0.0)))
        rows.append(
            {
                "engineered_col": name,
                "source_col": src,
                "incr_mi": float(incr),
                "smooth_gain": float(smooth_gain),
                "passed": passed,
            }
        )
    scores = (
        pd.DataFrame(rows)
        .sort_values(
            "incr_mi",
            ascending=False,
        )
        .reset_index(drop=True)
    )
    qualified = scores[scores["passed"]]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    recipes = []
    for name in keep:
        if name not in meta:
            continue
        m = meta[name]
        recipes.append(build_orth_wavelet_recipe(
            name=name, src_name=str(m["src"]),
            j=int(m["j"]), k=int(m["k"]),
            lo=float(m["lo"]), span=float(m["span"]),
        ))
    return X_aug, keep, recipes, scores
