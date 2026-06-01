"""Layer 95 PART A (2026-06-01): PERIODIC / MODULAR decomposition FE.

Extends Layer 90 (``_numeric_decompose_fe``: rounding + digit extraction) with
the missing ``x mod period`` transform for CYCLIC signals -- hour-of-day,
day-of-week, day-of-month, calendar / sensor cycles. Layer 32 (Fourier)
captures the *spectrum* of a column; this layer captures periodicity that lives
in the RESIDUE: a target that is a function of ``x mod 24`` (e.g. "load peaks
every day at the same hour") is invisible to any monotone transform of raw
``x`` (the magnitude keeps growing while the signal repeats) and is only
partially captured by a finite Fourier truncation.

Three transforms per ``(col, period)``:

* ``mod``     -- ``x mod period`` -- the raw residue. Recovers a target that is
  piecewise-constant in the residue (bucketed cycle).
* ``modsin``  -- ``sin(2*pi*(x mod period)/period)`` -- phase, sine component.
* ``modcos``  -- ``cos(2*pi*(x mod period)/period)`` -- phase, cosine component.

The sin/cos pair gives CYCLIC CONTINUITY: the raw residue has a discontinuity
at the period boundary (``period - eps`` and ``0`` are maximally far apart in
value but adjacent in phase), which a linear / smooth model mis-reads as a huge
jump. The sin/cos encoding maps the residue onto the unit circle so phase 0 and
phase ``period - eps`` are neighbours -- a smoothly-cyclic target (one whose
mean varies sinusoidally with phase) is then a LINEAR function of the (sin, cos)
pair, which raw ``mod`` cannot represent.

The IT enhancement (auto-period detection + gate)
--------------------------------------------------
``hybrid_modular_fe`` reuses Layer 90's Layer-62 bootstrap-stable MI gate
verbatim (``score_decompose_by_bootstrap_mi``). Candidate columns whose MI lower
confidence bound does not clear the raw column's noise band are dropped, so a
non-periodic column (or a wrong period) emits nothing. ``generate_modular_features``
tries a ladder of common periods; the gate keeps only those with a genuine MI
uplift -- this IS the auto-period detection (the correct period's residue
carries the signal; wrong periods scramble it into noise and fall below the
gate).

Recipe replay
-------------
One recipe kind ``modular`` with ``extra = {period: float, op: str}`` where
``op in {"mod", "sin", "cos"}``. Replay is pure arithmetic on the single source
column -- no y reference, leakage-free by construction, train/test exact.

NOT wired into ``MRMR.fit`` by default -- opt-in via ``fe_modular_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "MODULAR_PREFIX",
    "DEFAULT_PERIODS",
    "engineered_name_modular",
    "apply_modular",
    "generate_modular_features",
    "score_modular_by_bootstrap_mi",
    "hybrid_modular_fe",
    "hybrid_modular_fe_with_recipes",
    "build_modular_recipe",
]

MODULAR_PREFIX = "mod"
DEFAULT_PERIODS = (7, 12, 24, 30, 365)

# op token -> name fragment. ``mod`` -> ``mod``, ``sin`` -> ``modsin``,
# ``cos`` -> ``modcos`` (per the design spec).
_OP_FRAGMENT = {"mod": "mod", "sin": "modsin", "cos": "modcos"}
_VALID_OPS = ("mod", "sin", "cos")


def _fmt_period(period: float) -> str:
    """Stable, name-safe token for a period (``24`` -> ``"24"``, ``0.5`` ->
    ``"0p5"``)."""
    p = float(period)
    if p == int(p):
        return str(int(p))
    return repr(p).replace("-", "m").replace(".", "p")


def engineered_name_modular(col: str, period: float, op: str) -> str:
    """Canonical engineered column name for a modular feature.

    Uses the Layer-90 ``{src}__{suffix}`` convention so the L62 bootstrap-MI
    scorer (which maps a column back to its source via the ``__`` prefix) wires
    in unchanged. ``op='mod'`` -> ``"{col}__mod_{period}"``; ``op='sin'`` ->
    ``"{col}__modsin_{period}"``; ``op='cos'`` -> ``"{col}__modcos_{period}"``.
    """
    if op not in _OP_FRAGMENT:
        raise ValueError(f"modular op must be one of {_VALID_OPS}; got {op!r}")
    return f"{col}__{_OP_FRAGMENT[op]}_{_fmt_period(period)}"


def apply_modular(x: np.ndarray, period: float, op: str) -> np.ndarray:
    """Replay one modular transform on a numeric column.

    * ``op='mod'`` -> ``x mod period`` (numpy ``np.mod`` follows the divisor's
      sign so the residue is always in ``[0, period)`` for ``period > 0``).
    * ``op='sin'`` -> ``sin(2*pi*(x mod period)/period)``.
    * ``op='cos'`` -> ``cos(2*pi*(x mod period)/period)``.

    Output float64. NaN / inf rows scrub to 0 (matching the rest of the FE
    pipeline's ``nan_to_num`` policy) so downstream MI binning / model fit never
    sees NaN.
    """
    p = float(period)
    if p <= 0.0:
        raise ValueError(f"modular period must be > 0; got {p!r}")
    if op not in _VALID_OPS:
        raise ValueError(f"modular op must be one of {_VALID_OPS}; got {op!r}")
    arr = np.asarray(x, dtype=np.float64)
    residue = np.mod(arr, p)
    if op == "mod":
        out = residue
    else:
        phase = (2.0 * np.pi / p) * residue
        out = np.sin(phase) if op == "sin" else np.cos(phase)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _numeric_cols(X: pd.DataFrame, cols: Optional[Sequence[str]]) -> list[str]:
    candidates = list(cols) if cols is not None else list(X.columns)
    return [
        c for c in candidates
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]


def generate_modular_features(
    X: pd.DataFrame,
    num_cols: Optional[Sequence[str]] = None,
    periods: Sequence[float] = DEFAULT_PERIODS,
) -> pd.DataFrame:
    """For each ``(num_col, period)`` emit the residue ``mod`` plus its ``sin``
    / ``cos`` phase encoding.

    Returns a DataFrame (same row index as ``X``) of the engineered columns
    only; column names follow :func:`engineered_name_modular`. Pure function of
    ``X`` -- no y reference, so the generator is leakage-free.
    """
    cols = _numeric_cols(X, num_cols)
    periods = tuple(float(p) for p in periods if float(p) > 0.0)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        x = X[c].to_numpy()
        for p in periods:
            for op in _VALID_OPS:
                out[engineered_name_modular(c, p, op)] = apply_modular(x, p, op)
    return pd.DataFrame(out, index=X.index)


def score_modular_by_bootstrap_mi(
    raw_X: pd.DataFrame,
    eng_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
) -> pd.DataFrame:
    """Gate each modular feature by Layer 62 bootstrap-stable MI.

    Thin wrapper over
    :func:`_orthogonal_bootstrap_mi_fe.score_features_by_bootstrap_mi` -- the
    SAME lower-CB ranking primitive Layer 90 uses, applied to the modular
    candidate family. Each engineered column ``"{src}__mod*_{period}"`` maps
    back to its source via the ``__`` prefix (the L62 scorer's contract), so the
    per-replicate raw MI of ``src`` is the baseline a modular feature must clear.

    Returns the L62 scores frame sorted by ``uplift_lcb`` descending.
    """
    from ._orthogonal_bootstrap_mi_fe import score_features_by_bootstrap_mi

    return score_features_by_bootstrap_mi(
        raw_X, eng_X, y,
        n_boot=n_boot, sample_fraction=sample_fraction,
        seed=seed, nbins=nbins,
    )


def _parse_modular_name(name: str):
    """``"{src}__modsin_24"`` -> ('sin', src, 24.0). Returns None on a name that
    does not match the modular convention. The op fragment must be matched
    longest-first so ``modsin`` / ``modcos`` win over the bare ``mod`` prefix."""
    if "__" not in name:
        return None
    src, suffix = name.split("__", 1)
    # Longest fragment first: modsin / modcos before mod.
    for op, frag in (("sin", "modsin"), ("cos", "modcos"), ("mod", "mod")):
        prefix = frag + "_"
        if suffix.startswith(prefix):
            tok = suffix[len(prefix):]
            try:
                period = float(tok.replace("m", "-").replace("p", "."))
            except ValueError:
                return None
            return (op, src, period)
    return None


def hybrid_modular_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    periods: Sequence[float] = DEFAULT_PERIODS,
    top_k: int = 6,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
    min_uplift_lcb: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Modular-decomposition FE pipeline with a bootstrap-stable MI gate.

    1. Generate the ``mod`` / ``sin`` / ``cos`` candidate columns for every
       ``(col, period)``.
    2. Score every candidate by Layer 62 bootstrap-stable MI uplift (lower CB).
    3. Keep candidates whose ``uplift_mean >= min_uplift_lcb`` AND whose
       ``engineered_mi_lcb`` clears a MAD noise floor anchored on the RAW
       baseline MI distribution. Take the top-K survivors by ``engineered_mi_lcb``.

    This IS the auto-period detection: the correct period's residue carries the
    signal and survives the gate; wrong periods scramble it into noise and fall
    below the floor. A non-periodic column emits nothing.

    Returns ``(X_augmented, scores)`` -- the gate logic mirrors Layer 90's
    ``hybrid_numeric_decompose_fe`` (uplift_mean relevance + absolute
    engineered_mi_lcb stability, which sidesteps the degenerate uplift-ratio
    variance when the raw source MI sits near zero).
    """
    eng = generate_modular_features(X, num_cols=cols, periods=periods)

    empty_cols = [
        "engineered_col", "source_col",
        "baseline_mi_mean", "baseline_mi_std", "baseline_mi_lcb",
        "engineered_mi_mean", "engineered_mi_std", "engineered_mi_lcb",
        "uplift_mean", "uplift_std", "uplift_lcb",
    ]
    if eng.empty:
        return X.copy(), pd.DataFrame(columns=empty_cols)

    raw_X = X[_numeric_cols(X, cols)]
    scores = score_modular_by_bootstrap_mi(
        raw_X, eng, y,
        n_boot=n_boot, sample_fraction=sample_fraction, seed=seed, nbins=nbins,
    )
    if scores.empty:
        return X.copy(), scores

    # MAD noise floor anchored on the RAW BASELINE MI distribution (the genuine
    # null reference), mirroring Layer 90. Anchoring on the engineered-candidate
    # distribution would be bimodal when several periods/ops of the same column
    # recover the signal, inflating the floor and rejecting genuine survivors.
    base_lcb = np.unique(scores["baseline_mi_lcb"].to_numpy())
    if base_lcb.size >= 4:
        med = float(np.median(base_lcb))
        mad = float(np.median(np.abs(base_lcb - med)))
        noise_floor = med + 3.5 * 1.4826 * mad
    else:
        noise_floor = 2.0 * float(base_lcb.max()) if base_lcb.size else 0.0
    qualified = scores[
        (scores["uplift_mean"] >= float(min_uplift_lcb))
        & (scores["engineered_mi_lcb"] >= noise_floor)
    ]
    winners = qualified.sort_values(
        "engineered_mi_lcb", ascending=False
    ).head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, eng[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_modular_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    periods: Sequence[float] = DEFAULT_PERIODS,
    top_k: int = 6,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
    min_uplift_lcb: float = 1.0,
):
    """Same as :func:`hybrid_modular_fe` but additionally returns a list of
    recipes -- one per appended column -- so ``MRMR.transform`` can recompute
    each modular feature on test data without re-running the bootstrap MI gate.
    Each recipe is pure arithmetic on X (no y), so replay is leakage-free.

    Returns ``(X_augmented, appended, recipes, scores)``.
    """
    X_aug, scores = hybrid_modular_fe(
        X, y,
        cols=cols, periods=periods, top_k=top_k,
        n_boot=n_boot, sample_fraction=sample_fraction,
        seed=seed, nbins=nbins, min_uplift_lcb=min_uplift_lcb,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    for name in appended:
        parsed = _parse_modular_name(name)
        if parsed is None:
            logger.warning(
                "hybrid_modular_fe_with_recipes: cannot parse op/period from "
                "column name %r; skipping recipe build.", name,
            )
            continue
        op, src, period = parsed
        recipes.append(build_modular_recipe(
            name=name, src_name=src, period=float(period), op=op,
        ))
    return X_aug, appended, recipes, scores


def build_modular_recipe(*, name: str, src_name: str, period: float, op: str):
    """Frozen recipe for one modular transform of ``X[src_name]``. Replay is
    pure arithmetic on the source column -- no y reference, no fitted state, so
    transform() is leakage-free and train/test exact."""
    from .engineered_recipes import EngineeredRecipe

    p = float(period)
    if p <= 0.0:
        raise ValueError(f"modular period must be > 0; got {p!r}")
    if op not in _VALID_OPS:
        raise ValueError(f"modular op must be one of {_VALID_OPS}; got {op!r}")
    return EngineeredRecipe(
        name=name,
        kind="modular",
        src_names=(str(src_name),),
        extra={"period": p, "op": str(op)},
    )
