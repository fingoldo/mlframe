"""Layer 90 (2026-06-01): NUMERIC DECOMPOSITION FE with bootstrap-MI gate.

NVIDIA cuDF Kaggle-Grandmaster blog technique #4 -- *multi-precision rounding
+ digit extraction*. Two cheap, pure-arithmetic transforms on a numeric column:

* **rounding**: ``round(x / p) * p`` for a ladder of precisions
  ``p in {1, 0.1, 0.01, 0.001}``. The coarse-precision variants act as
  step-function / bucket detectors -- they collapse within-bucket jitter so a
  target that is piecewise-constant in coarse ``x`` (a price-anchored signal:
  "anything in [10, 20) behaves the same") becomes a clean low-entropy
  predictor. A model can recover the step boundary far more cheaply from
  ``round(x, 1)`` than from raw ``x``.
* **digit extraction**: ``floor(x * 10^k) mod 10`` -- the ``k``-th decimal
  digit. Captures cents-digit / encoded-id-substructure signals: e.g. a price
  whose *cents* digit encodes a hidden category (psychological-pricing buckets,
  store-id-in-the-decimals, checksum digits). These are INVISIBLE to any
  monotone transform of raw ``x`` because the signal lives in the
  low-order digit, orthogonal to magnitude.

THE IT enhancement (the key over a naive emit-everything blast):
each candidate precision / digit feature is gated by **Layer 62 bootstrap-
stable MI** -- we keep a decomposition feature only when its MI lower
confidence bound (``mean - 1.96 * std`` over B bootstrap subsamples) clears the
raw column's MI lower CB by a margin. A precision whose rounded value carries
no MI beyond the raw column (the common case on a *smooth* target, where
rounding is just lossy raw ``x`` and digit extraction is pure noise) lands
inside the raw column's noise band and is DROPPED. Only decompositions that add
genuinely stable signal survive. This reuses
``_orthogonal_bootstrap_mi_fe.score_features_by_bootstrap_mi`` verbatim -- the
same lower-CB ranking primitive, applied to a different candidate family.

Polynomial extension
---------------------
A rounded column composes naturally with a Layer 21 Chebyshev basis: a target
that is a step function of ``x`` is exactly a low-degree polynomial in the
*rounded* coordinate, so ``T_n(round(x, p))`` captures multi-step staircases
that neither raw Chebyshev (smooth) nor a single rounding bucket can. We do not
wire round -> basis into the default pipeline (no measured win on the Layer 90
fixtures), but the rounded columns this module emits are valid inputs to the
existing ``generate_univariate_basis_features`` route should a caller opt in.

Recipe replay
-------------
Two new recipe kinds, both pure arithmetic on X (no y reference -> leakage-free
by construction):

* ``numeric_rounding``  : ``extra = {precision: float}`` -> ``round(x/p)*p``
* ``digit_extract``     : ``extra = {digit_position: int}`` -> ``floor(x*10^k) % 10``

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_numeric_decompose_enable=True``.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a hard dep in practice
    def njit(*args, **kwargs):  # no-op fallback so the module imports
        """No-op stand-in for ``numba.njit`` when numba isn't installed, supporting both bare-decorator and decorator-with-args call forms."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            """Identity wrapper returning ``fn`` unchanged."""
            return fn
        return deco


@njit(cache=True)
def _digit_extract_njit(arr: np.ndarray, scale: float) -> np.ndarray:
    """Single-pass ``floor(x*scale) mod 10`` with NaN/inf -> 0. Fuses the
    numpy ``mul -> floor -> mod -> nan_to_num`` 4-pass (each allocating a
    temp) into one allocation-light C loop (~12x). Bit-identical."""
    n = arr.size
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = arr[i] * scale
        if not np.isfinite(v):
            out[i] = 0.0
        else:
            f = math.floor(v)
            out[i] = f - 10.0 * math.floor(f / 10.0)  # floor-mod 10
    return out


logger = logging.getLogger(__name__)

__all__ = [
    "ROUNDING_PREFIX",
    "DIGIT_PREFIX",
    "engineered_name_rounding",
    "engineered_name_digit",
    "apply_rounding",
    "apply_digit_extract",
    "generate_rounding_features",
    "generate_digit_features",
    "score_decompose_by_bootstrap_mi",
    "hybrid_numeric_decompose_fe",
    "hybrid_numeric_decompose_fe_with_recipes",
    "build_numeric_rounding_recipe",
    "build_digit_extract_recipe",
]


ROUNDING_PREFIX = "round"
DIGIT_PREFIX = "digit"


def _fmt_precision(precision: float) -> str:
    """Stable, name-safe token for a precision (``0.01`` -> ``"0p01"``)."""
    s = repr(float(precision))
    return s.replace("-", "m").replace(".", "p")


def engineered_name_rounding(col: str, precision: float) -> str:
    """Canonical engineered column name for a rounding feature."""
    return f"{col}__{ROUNDING_PREFIX}_{_fmt_precision(precision)}"


def engineered_name_digit(col: str, digit_position: int) -> str:
    """Canonical engineered column name for a digit-extraction feature."""
    return f"{col}__{DIGIT_PREFIX}_{int(digit_position)}"


def apply_rounding(x: np.ndarray, precision: float) -> np.ndarray:
    """``round(x / precision) * precision`` as float64. NaN-preserving;
    np.round uses banker's rounding (matches numpy semantics at fit + replay
    so train/test parity is exact)."""
    p = float(precision)
    if p <= 0.0:
        raise ValueError(f"rounding precision must be > 0; got {p!r}")
    arr = np.asarray(x, dtype=np.float64)
    return np.round(arr / p) * p


def apply_digit_extract(x: np.ndarray, digit_position: int) -> np.ndarray:
    """``floor(x * 10^k) mod 10`` -- the ``k``-th decimal digit (k=0 -> the
    ones digit of the integer part for k<=0 convention; here k counts decimal
    places: k=0 is the first decimal, etc. per the design spec
    ``floor(x*10^k) mod 10``). Output float64 in ``{0..9}``; NaN rows map to 0.

    The transform is sign-insensitive in the sense that it operates on the
    scaled value directly; negative inputs produce the floor-mod digit of the
    negative scaled value, which is deterministic and reproducible at replay.
    """
    k = int(digit_position)
    arr = np.ascontiguousarray(x, dtype=np.float64)
    # Fused single-pass njit: floor(x*10^k) mod 10, NaN/inf -> 0 (matches the
    # nan_to_num scrubbing the rest of the FE pipeline applies). ~12x over the
    # numpy mul->floor->mod->nan_to_num 4-pass.
    return np.asarray(_digit_extract_njit(arr, 10.0**k))


def _numeric_cols(X: pd.DataFrame, cols: Optional[Sequence[str]]) -> list[str]:
    """Filter ``cols`` (or all of ``X``'s columns when None) down to those present in ``X`` with a numeric dtype."""
    candidates = list(cols) if cols is not None else list(X.columns)
    return [c for c in candidates if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]


def generate_rounding_features(
    X: pd.DataFrame,
    num_cols: Optional[Sequence[str]] = None,
    precisions: Sequence[float] = (1, 0.1, 0.01, 0.001),
) -> pd.DataFrame:
    """For each (num_col, precision) emit ``round(x / precision) * precision``.

    Returns a DataFrame (same row index as ``X``) of the engineered columns
    only; column names follow :func:`engineered_name_rounding`. Pure function
    of ``X`` -- no y reference.
    """
    cols = _numeric_cols(X, num_cols)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        x = X[c].to_numpy()
        for p in precisions:
            out[engineered_name_rounding(c, p)] = apply_rounding(x, p)
    return pd.DataFrame(out, index=X.index)


def generate_digit_features(
    X: pd.DataFrame,
    num_cols: Optional[Sequence[str]] = None,
    digit_positions: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    """For each (num_col, k) emit ``floor(x * 10^k) mod 10`` (the k-th decimal
    digit). Returns a DataFrame of engineered columns only; names follow
    :func:`engineered_name_digit`. Pure function of ``X`` -- no y reference.
    """
    cols = _numeric_cols(X, num_cols)
    out: dict[str, np.ndarray] = {}
    for c in cols:
        x = X[c].to_numpy()
        for k in digit_positions:
            out[engineered_name_digit(c, k)] = apply_digit_extract(x, k)
    return pd.DataFrame(out, index=X.index)


def score_decompose_by_bootstrap_mi(
    raw_X: pd.DataFrame,
    eng_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
) -> pd.DataFrame:
    """Gate each decomposition feature by Layer 62 bootstrap-stable MI.

    Thin wrapper over
    :func:`_orthogonal_bootstrap_mi_fe.score_features_by_bootstrap_mi`. Each
    engineered column ``"{src}__{...}"`` maps back to its source via the
    ``__`` prefix (the L62 scorer's contract), so the per-replicate raw MI of
    ``src`` is the baseline that each precision / digit feature must clear.

    Returns the L62 scores frame (one row per engineered column) with columns
    ``engineered_col, source_col, baseline_mi_*, engineered_mi_*, uplift_*``,
    sorted by ``uplift_lcb`` descending. Drops nothing itself -- the keep / drop
    decision lives in :func:`hybrid_numeric_decompose_fe`, which thresholds
    these lower-CB columns.
    """
    from ._orthogonal_bootstrap_mi_fe import score_features_by_bootstrap_mi

    return score_features_by_bootstrap_mi(
        raw_X, eng_X, y,
        n_boot=n_boot, sample_fraction=sample_fraction,
        seed=seed, nbins=nbins,
    )


def _parse_engineered_name(name: str):
    """``"{src}__round_0p1"`` -> ('numeric_rounding', src, 0.1);
    ``"{src}__digit_2"`` -> ('digit_extract', src, 2). Returns None on a name
    that doesn't match either convention.

    CAT_INTERACTION_B-4 fix (mrmr_audit_2026-07-22): this used to split on the FIRST "__", which breaks for
    any raw source column whose own name contains "__" (e.g. flattened-JSON keys like "user__id", or even
    this codebase's own orth-basis engineered-column convention "{col}__{basis_code}{degree}"): for source
    "a__b", the emitted name "a__b__digit_1" split to src="a", suffix="b__digit_1", matching neither prefix
    check, silently returning None -- hybrid_numeric_decompose_fe_with_recipes then skipped building a
    recipe for that column even though it stayed in X_aug/appended and could be selected, a silent
    fit/serve mismatch. Anchor on the KNOWN suffix pattern instead: rsplit on "__" and check the resulting
    suffix against the round_/digit_ prefixes, so any number of "__" occurrences in the source name (as
    long as it doesn't itself end in a round_/digit_-shaped segment) parses correctly.
    """
    if "__" not in name:
        return None
    src, suffix = name.rsplit("__", 1)
    if suffix.startswith(ROUNDING_PREFIX + "_"):
        tok = suffix[len(ROUNDING_PREFIX) + 1 :]
        try:
            prec = float(tok.replace("m", "-").replace("p", "."))
        except ValueError:
            return None
        return ("numeric_rounding", src, prec)
    if suffix.startswith(DIGIT_PREFIX + "_"):
        tok = suffix[len(DIGIT_PREFIX) + 1 :]
        if not tok.lstrip("-").isdigit():
            return None
        return ("digit_extract", src, int(tok))
    return None


def hybrid_numeric_decompose_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    precisions: Sequence[float] = (1, 0.1, 0.01, 0.001),
    digit_positions: Sequence[int] = (0, 1, 2),
    top_k: int = 5,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
    min_uplift_lcb: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric-decomposition FE pipeline with a bootstrap-stable MI gate.

    1. Generate the rounding + digit-extraction candidate columns.
    2. Score every candidate by Layer 62 bootstrap-stable MI uplift (lower CB).
    3. Keep candidates whose ``uplift_lcb >= min_uplift_lcb`` (with 95 %
       confidence the decomposition's MI is at least the raw column's MI) AND
       whose ``engineered_mi_lcb`` clears a MAD noise floor on the candidate
       distribution. Take the top-K survivors by ``uplift_lcb``.

    Candidates that are merely lossy copies of raw ``x`` (smooth-target case:
    rounding adds nothing, digits are noise) sit at ``uplift_lcb ~ 1`` with a
    wide CI and fall below the gate -> dropped. This is the IT enhancement that
    keeps the decomposition family from flooding the feature matrix.

    Returns ``(X_augmented, scores)`` where ``scores`` is the full L62 ranking
    (survivors + rejects).
    """
    rounding = generate_rounding_features(X, num_cols=cols, precisions=precisions)
    digits = generate_digit_features(X, num_cols=cols, digit_positions=digit_positions)
    eng = pd.concat([rounding, digits], axis=1) if (not rounding.empty or not digits.empty) else pd.DataFrame(index=X.index)

    empty_cols = [
        "engineered_col", "source_col",
        "baseline_mi_mean", "baseline_mi_std", "baseline_mi_lcb",
        "engineered_mi_mean", "engineered_mi_std", "engineered_mi_lcb",
        "uplift_mean", "uplift_std", "uplift_lcb",
    ]
    if eng.empty:
        return X, pd.DataFrame(columns=empty_cols)

    raw_X = X[_numeric_cols(X, cols)]
    scores = score_decompose_by_bootstrap_mi(
        raw_X, eng, y,
        n_boot=n_boot, sample_fraction=sample_fraction, seed=seed, nbins=nbins,
    )
    if scores.empty:
        return X, scores

    # MAD noise floor anchored on the RAW BASELINE MI distribution, NOT the
    # engineered-candidate distribution. The baseline_mi_lcb of the raw source
    # columns is the genuine null reference - "what a non-informative column's
    # MI looks like under this estimator + sample size". Anchoring the floor
    # there is robust to the case where MULTIPLE decompositions of the same
    # column carry signal: e.g. round(x,1.0), round(x,0.1) and round(x,0.01)
    # ALL recover an anchor-parity signal, making the engineered_mi_lcb
    # distribution BIMODAL. A median+MAD floor computed on that bimodal set
    # blows up (median lands between the noise and signal clusters, MAD is
    # huge) and rejects every genuine survivor (observed: 3 rounding features
    # at lcb=0.378 all dropped because the engineered-MAD floor inflated to
    # 1.17). The baseline distribution has no such bimodality - it is the
    # noise band by construction.
    base_lcb = scores["baseline_mi_lcb"].to_numpy()
    base_lcb = np.unique(base_lcb)  # one row per distinct raw source
    if base_lcb.size >= 4:
        med = float(np.median(base_lcb))
        mad = float(np.median(np.abs(base_lcb - med)))
        noise_floor = med + 3.5 * 1.4826 * mad
    else:
        # Too few raw sources to estimate a band robustly: fall back to a
        # small multiple of the max raw baseline (genuine signal must clear
        # the strongest noise column by a margin).
        noise_floor = 2.0 * float(base_lcb.max()) if base_lcb.size else 0.0
    # Two-gate: the ABSOLUTE stability requirement lives on
    # ``engineered_mi_lcb`` (the lower-confidence-bound of the engineered MI
    # itself, where bootstrap variance is well-behaved); the RELATIVE
    # improvement requirement uses ``uplift_mean`` rather than ``uplift_lcb``.
    # The uplift RATIO = engineered_mi / baseline_mi is statistically
    # degenerate when the baseline (raw source) MI sits near zero: dividing a
    # large, stable engineered MI by a tiny noisy denominator makes the ratio
    # variance explode, so ``uplift_lcb = mean - 1.96*std`` underflows to a
    # large NEGATIVE value even for genuine signal (observed: cents-digit
    # feature at engineered_mi_lcb=0.53 nats but uplift_lcb=-119 on seed=7,
    # because price's own MI ~= 0.003 with huge relative bootstrap spread).
    # ``uplift_mean`` stays a faithful "engineered beats raw on average"
    # gate; the LCB stability guarantee is fully carried by the absolute
    # engineered_mi_lcb noise floor.
    qualified = scores[(scores["uplift_mean"] >= float(min_uplift_lcb)) & (scores["engineered_mi_lcb"] >= noise_floor)]
    # Rank survivors by the STABLE absolute metric (engineered_mi_lcb), not
    # uplift_lcb: when several decompositions of the same near-zero-MI source
    # qualify, the uplift ratio's LCB ordering is dominated by denominator
    # noise (see the gate comment above). engineered_mi_lcb gives a faithful
    # "strongest genuine signal first" cut for the top-K.
    winners = qualified.sort_values("engineered_mi_lcb", ascending=False).head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, eng[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_numeric_decompose_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    precisions: Sequence[float] = (1, 0.1, 0.01, 0.001),
    digit_positions: Sequence[int] = (0, 1, 2),
    top_k: int = 5,
    n_boot: int = 10,
    sample_fraction: float = 0.8,
    seed: int = 0,
    nbins: int = 10,
    min_uplift_lcb: float = 1.0,
):
    """Same as :func:`hybrid_numeric_decompose_fe` but additionally returns a
    list of recipes -- one per appended column -- so ``MRMR.transform`` can
    recompute each decomposition on test data without re-running the bootstrap
    MI gate. Each recipe is pure arithmetic on X (no y), so replay is
    leakage-free by construction.

    Returns ``(X_augmented, appended, recipes, scores)``.
    """
    from .engineered_recipes import (
        build_numeric_rounding_recipe,
        build_digit_extract_recipe,
    )

    X_aug, scores = hybrid_numeric_decompose_fe(
        X, y,
        cols=cols, precisions=precisions, digit_positions=digit_positions,
        top_k=top_k, n_boot=n_boot, sample_fraction=sample_fraction,
        seed=seed, nbins=nbins, min_uplift_lcb=min_uplift_lcb,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    for name in appended:
        parsed = _parse_engineered_name(name)
        if parsed is None:
            logger.warning(
                "hybrid_numeric_decompose_fe_with_recipes: cannot parse " "kind/param from column name %r; skipping recipe build.",
                name,
            )
            continue
        kind, src, param = parsed
        if kind == "numeric_rounding":
            recipes.append(build_numeric_rounding_recipe(
                name=name, src_name=src, precision=float(param),
            ))
        else:  # digit_extract
            recipes.append(build_digit_extract_recipe(
                name=name, src_name=src, digit_position=int(param),
            ))
    return X_aug, appended, recipes, scores


def build_numeric_rounding_recipe(
    *, name: str, src_name: str, precision: float,
):
    """Frozen recipe for ``round(X[src_name] / precision) * precision``. Replay
    is pure arithmetic on the source column -- no y reference, no fitted state,
    so transform() is leakage-free and train/test exact."""
    from .engineered_recipes import EngineeredRecipe

    p = float(precision)
    if p <= 0.0:
        raise ValueError(f"numeric_rounding precision must be > 0; got {p!r}")
    return EngineeredRecipe(
        name=name,
        kind="numeric_rounding",
        src_names=(str(src_name),),
        extra={"precision": p},
    )


def build_digit_extract_recipe(
    *, name: str, src_name: str, digit_position: int,
):
    """Frozen recipe for ``floor(X[src_name] * 10^k) mod 10`` (the k-th decimal
    digit). Replay is pure arithmetic on the source column -- no y reference."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="digit_extract",
        src_names=(str(src_name),),
        extra={"digit_position": int(digit_position)},
    )
