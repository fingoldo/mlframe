"""Layer 95 PART A: PERIODIC / MODULAR decomposition FE.

Extends Layer 90 (``_numeric_decompose_fe``: rounding + digit extraction) with
the missing ``x mod period`` transform for CYCLIC signals - hour-of-day,
day-of-week, day-of-month, calendar / sensor cycles. Layer 32 (Fourier)
captures the *spectrum* of a column; this layer captures periodicity that lives
in the RESIDUE: a target that is a function of ``x mod 24`` (e.g. "load peaks
every day at the same hour") is invisible to any monotone transform of raw
``x`` (the magnitude keeps growing while the signal repeats) and is only
partially captured by a finite Fourier truncation.

Three transforms per ``(col, period)``:

* ``mod``  - ``x mod period`` - the raw residue. Recovers a target that is
  piecewise-constant in the residue (bucketed cycle).
* ``modsin``  - ``sin(2*pi*(x mod period)/period)`` - phase, sine component.
* ``modcos``  - ``cos(2*pi*(x mod period)/period)`` - phase, cosine component.

The sin/cos pair gives CYCLIC CONTINUITY: the raw residue has a discontinuity
at the period boundary (``period - eps`` and ``0`` are maximally far apart in
value but adjacent in phase), which a linear / smooth model mis-reads as a huge
jump. The sin/cos encoding maps the residue onto the unit circle so phase 0 and
phase ``period - eps`` are neighbours - a smoothly-cyclic target (one whose
mean varies sinusoidally with phase) is then a LINEAR function of the (sin, cos)
pair, which raw ``mod`` cannot represent.

The IT enhancement (auto-period detection + gate)
--------------------------------------------------
``hybrid_modular_fe`` reuses Layer 90's Layer-62 bootstrap-stable MI gate
verbatim (``score_decompose_by_bootstrap_mi``). Candidate columns whose MI lower
confidence bound does not clear the raw column's noise band are dropped, so a
non-periodic column (or a wrong period) emits nothing. ``generate_modular_features``
tries a ladder of common periods; the gate keeps only those with a genuine MI
uplift - this IS the auto-period detection (the correct period's residue
carries the signal; wrong periods scramble it into noise and fall below the
gate).

Recipe replay
-------------
One recipe kind ``modular`` with ``extra = {period: float, op: str}`` where
``op in {"mod", "sin", "cos"}``. Replay is pure arithmetic on the single source
column - no y reference, leakage-free by construction, train/test exact.

Wired into ``MRMR.fit`` by default; disable with ``fe_modular_enable=False``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional
from collections.abc import Sequence

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

from mlframe.utils.log_throttle import log_throttle

from ._numeric_decompose_fe import _numeric_cols, score_decompose_by_bootstrap_mi as score_modular_by_bootstrap_mi
from ._orthogonal_bootstrap_mi_fe import baseline_lcb_by_bootstrap_mi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from .engineered_recipes import EngineeredRecipe

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a hard dep in practice
    from mlframe._numba_fallback import njit


# op codes for the fused kernel: 0=mod, 1=sin, 2=cos
@njit(cache=True)
def _modular_njit(arr: np.ndarray, p: float, op_code: int) -> np.ndarray:
    """Fused single-pass modular transform: residue ``x mod p`` (op 0), or
    ``sin``/``cos`` of its phase (ops 1/2), with NaN/inf -> 0. Replaces the
    numpy mod -> phase -> sin/cos -> nan_to_num multi-pass (~2.65x). ``r =
    v - p*floor(v/p)`` reproduces ``np.mod(v, p)`` for ``p > 0`` (residue in
    ``[0, p)``); matches the numpy form to <1e-9 (mod) / <1e-6 (trig), well
    within the MI-binning + model-fit tolerance the values feed into."""
    n = arr.size
    out = np.empty(n, dtype=np.float64)  # Intentional float64: numba kernel body, dtype fixed at compile time (crit helper is not callable under njit)
    k = 2.0 * math.pi / p
    for i in range(n):
        v = arr[i]
        if not np.isfinite(v):
            out[i] = 0.0
            continue
        r = v - p * math.floor(v / p)
        if op_code == 0:
            val = r
        elif op_code == 1:
            val = math.sin(k * r)
        else:
            val = math.cos(k * r)
        out[i] = val if np.isfinite(val) else 0.0
    return out


@njit(cache=True)
def _modular_all_ops_njit(arr: np.ndarray, p: float):
    """Fused single-pass kernel emitting all three modular ops (mod / sin / cos) for one ``(col, period)``.

    The residue ``r = v - p*floor(v/p)`` and the phase ``k*r`` are computed ONCE per element and reused for all three
    outputs, instead of three separate passes (one per :func:`apply_modular` call) that each recompute the residue and
    re-walk the array. Bit-identical to three back-to-back ``_modular_njit`` calls (same residue formula, same NaN/inf
    -> 0 scrub, same ``sin``/``cos`` of the same phase). Returns ``(mod, sin, cos)`` double arrays."""
    n = arr.size
    out_mod = np.empty(n, dtype=np.float64)  # Intentional float64: numba kernel body, dtype fixed at compile time (crit helper is not callable under njit)
    out_sin = np.empty(n, dtype=np.float64)  # Intentional float64: numba kernel body, dtype fixed at compile time (crit helper is not callable under njit)
    out_cos = np.empty(n, dtype=np.float64)  # Intentional float64: numba kernel body, dtype fixed at compile time (crit helper is not callable under njit)
    k = 2.0 * math.pi / p
    for i in range(n):
        v = arr[i]
        if not np.isfinite(v):
            out_mod[i] = 0.0
            out_sin[i] = 0.0
            out_cos[i] = 0.0
            continue
        r = v - p * math.floor(v / p)
        out_mod[i] = r if np.isfinite(r) else 0.0
        kr = k * r
        s = math.sin(kr)
        c = math.cos(kr)
        out_sin[i] = s if np.isfinite(s) else 0.0
        out_cos[i] = c if np.isfinite(c) else 0.0
    return out_mod, out_sin, out_cos


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
_OP_FRAGMENT = {"mod": "mod", "sin": "modsin", "cos": "modcos", "parity": "par"}
_VALID_OPS = ("mod", "sin", "cos", "parity")


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

    Output at the fit's float dtype (``_crit_np_dtype``), the width the fit generated the column at. NaN / inf rows scrub to 0 (matching the rest of the FE
    pipeline's ``nan_to_num`` policy) so downstream MI binning / model fit never
    sees NaN.
    """
    p = float(period)
    if p <= 0.0:
        raise ValueError(f"modular period must be > 0; got {p!r}")
    if op not in _VALID_OPS:
        raise ValueError(f"modular op must be one of {_VALID_OPS}; got {op!r}")
    # Intentional float64: input of the residue v - p*floor(v/p); a float32 timestamp (~1.6e9) has resolution 128, so mod 7 would be noise.
    arr = np.ascontiguousarray(x, dtype=np.float64)
    out_dtype = _crit_np_dtype()
    if op == "parity":
        # Square wave: floor(x / period) mod 2. NaN / inf scrub to 0 like the other ops.
        square: np.ndarray = np.nan_to_num(np.floor(arr / p) % 2.0, nan=0.0, posinf=0.0, neginf=0.0).astype(out_dtype, copy=False)
        return square
    op_code = 0 if op == "mod" else (1 if op == "sin" else 2)
    return np.asarray(_modular_njit(arr, p, op_code)).astype(out_dtype, copy=False)


@njit(cache=True)
def _modular_all_ops_into(arr: np.ndarray, p: float, out_mod: np.ndarray, out_sin: np.ndarray, out_cos: np.ndarray) -> None:
    """:func:`_modular_all_ops_njit` written into caller-allocated outputs of any float dtype (computed at full precision, stored at the
    output's width), so a float32 candidate pool is filled without a full-width intermediate per column."""
    n = arr.size
    k = 2.0 * math.pi / p
    for i in range(n):
        v = float(arr[i])
        if not np.isfinite(v):
            out_mod[i] = 0.0
            out_sin[i] = 0.0
            out_cos[i] = 0.0
            continue
        r = v - p * math.floor(v / p)
        ph = k * r
        s_ = math.sin(ph)
        c_ = math.cos(ph)
        out_mod[i] = r if np.isfinite(r) else 0.0
        out_sin[i] = s_ if np.isfinite(s_) else 0.0
        out_cos[i] = c_ if np.isfinite(c_) else 0.0


def generate_modular_features(
    X: pd.DataFrame,
    num_cols: Optional[Sequence[str]] = None,
    periods: Sequence[float] = DEFAULT_PERIODS,
    dtype: "DTypeLike | None" = None,
) -> pd.DataFrame:
    """For each ``(num_col, period)`` emit the residue ``mod`` plus its ``sin``
    / ``cos`` phase encoding.

    ``dtype`` is the storage width of the engineered columns; ``None`` takes the fit's float dtype (``_crit_np_dtype``), which recipe
    replay (:func:`apply_modular`) also returns, so fit and transform columns agree. The residue itself is computed at full precision.

    Returns a DataFrame (same row index as ``X``) of the engineered columns
    only; column names follow :func:`engineered_name_modular`. Pure function of
    ``X`` - no y reference, so the generator is leakage-free.
    """
    cols = _numeric_cols(X, num_cols)
    periods = tuple(float(p) for p in periods if float(p) > 0.0)
    out: dict[str, np.ndarray] = {}
    n = len(X)
    out_dtype = np.dtype(_crit_np_dtype() if dtype is None else dtype)
    # MLFRAME_MODULAR_DROP_COPIES=1 leaves out the ``mod`` candidate that equals its source column. Default off (bench-attempt-rejected): on the conjunction pillar the changed
    # candidate pool let the greedy pick a different, equivalent-looking set of pair fragments around binary A (A + cell3 in six runs without it, B + digit fragments of the decoys
    # with it), so the dedup needs the selection to be stable under a changed pool first.
    import os

    drop_copies = os.environ.get("MLFRAME_MODULAR_DROP_COPIES", "0").strip().lower() in ("1", "true", "on", "yes")
    for c in cols:
        # Intentional float64: input of the residue v - p*floor(v/p); a float32 timestamp (~1.6e9) has resolution 128, so mod 7 would be noise.
        arr = np.ascontiguousarray(X[c].to_numpy(), dtype=np.float64)
        for p in periods:
            # Fused single pass: residue computed once, all three ops emitted (vs three apply_modular passes), written at the output width.
            mod_v, sin_v, cos_v = np.empty(n, dtype=out_dtype), np.empty(n, dtype=out_dtype), np.empty(n, dtype=out_dtype)
            _modular_all_ops_into(arr, p, mod_v, sin_v, cos_v)
            # A column that already lies in [0, period) has its own value as residue: the ``mod`` candidate would be an exact copy of the raw column (a binary column and period 7
            # gave ``A__mod_7`` == ``A``), which the selection could then carry beside, or instead of, the column itself.
            if not (drop_copies and np.array_equal(mod_v, arr.astype(out_dtype, copy=False))):
                out[engineered_name_modular(c, p, "mod")] = mod_v
            out[engineered_name_modular(c, p, "sin")] = sin_v
            out[engineered_name_modular(c, p, "cos")] = cos_v
    return pd.DataFrame(out, index=X.index)


def _parse_modular_name(name: str):
    """``"{src}__modsin_24"`` -> ('sin', src, 24.0). Returns None on a name that
    does not match the modular convention. The op fragment must be matched
    longest-first so ``modsin`` / ``modcos`` win over the bare ``mod`` prefix."""
    if "__" not in name:
        return None
    src, suffix = name.split("__", 1)
    # Longest fragment first: modsin / modcos before mod.
    for op, frag in (("sin", "modsin"), ("cos", "modcos"), ("mod", "mod"), ("parity", "par")):
        prefix = frag + "_"
        if suffix.startswith(prefix):
            tok = suffix[len(prefix) :]
            try:
                period = float(tok.replace("m", "-").replace("p", "."))
            except ValueError as _exc:
                logger.debug("_parse_modular_name fell back to its default after %s", type(_exc).__name__)
                return None
            return (op, src, period)
    return None


_BASELINE_BLOCK_COLUMNS = 32  # source columns whose raw bootstrap baseline is scored together
_MODULAR_CHUNK_BYTES = 256 * 1024 * 1024  # candidate-matrix budget of one source-column chunk (3 ops x periods x float width x copies per source column)


def _gate_first_halving_on_memory_error(X, y, src_cols, chunk, periods, top_k, n_boot, sample_fraction, seed, nbins, min_uplift_lcb):
    """:func:`_hybrid_modular_fe_gate_first`, retried with half the chunk when a candidate block cannot be allocated."""
    try:
        gated: tuple[Any, Any] = _hybrid_modular_fe_gate_first(X, y, src_cols, max(chunk, 1), periods, top_k, n_boot, sample_fraction, seed, nbins, min_uplift_lcb)
        return gated
    except MemoryError:
        if chunk <= 1:
            raise
    return _gate_first_halving_on_memory_error(X, y, src_cols, max(1, chunk // 2), periods, top_k, n_boot, sample_fraction, seed, nbins, min_uplift_lcb)


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

    Returns ``(X_augmented, scores)`` - the gate logic mirrors Layer 90's
    ``hybrid_numeric_decompose_fe`` (uplift_mean relevance + absolute
    engineered_mi_lcb stability, which sidesteps the degenerate uplift-ratio
    variance when the raw source MI sits near zero).
    """
    src_cols = _numeric_cols(X, cols)
    n_periods = len([p for p in periods if float(p) > 0.0])
    # Per source column: 3 * n_periods candidates at the fit's float width, plus the scorer's copy and bootstrap gather of them.
    _w = np.dtype(_crit_np_dtype()).itemsize
    per_src_bytes = len(X) * max(1, n_periods) * 3 * 3 * _w
    chunk = len(src_cols)
    if chunk > 1:
        from ._feature_engineering_mem_budget import _can_hoist_shared_buffer

        while chunk > 1 and not _can_hoist_shared_buffer(chunk * per_src_bytes, n_workers=1)[0]:
            chunk = max(1, chunk // 2)
    # The chunk is also capped by a fixed candidate-matrix size: whether a 400 MB block can be allocated depends on what else the machine holds, and a failed allocation dropped the modular columns
    # from the fit (the selection then depended on free memory). Chunking does not change the scores.
    chunk = max(1, min(chunk, _MODULAR_CHUNK_BYTES // max(per_src_bytes, 1)))
    if _gate_first_enabled() and src_cols:
        gated_first: tuple[Any, Any] = _gate_first_halving_on_memory_error(X, y, src_cols, chunk, periods, top_k, n_boot, sample_fraction, seed, nbins, min_uplift_lcb)
        return gated_first
    if 0 < chunk < len(src_cols):
        streamed: tuple[Any, Any] = _hybrid_modular_fe_streamed(X, y, src_cols, chunk, periods, top_k, n_boot, sample_fraction, seed, nbins, min_uplift_lcb)
        return streamed
    eng = generate_modular_features(X, num_cols=cols, periods=periods)

    empty_cols = [
        "engineered_col",
        "source_col",
        "baseline_mi_mean",
        "baseline_mi_std",
        "baseline_mi_lcb",
        "engineered_mi_mean",
        "engineered_mi_std",
        "engineered_mi_lcb",
        "uplift_mean",
        "uplift_std",
        "uplift_lcb",
    ]
    if eng.empty:
        return X, pd.DataFrame(columns=empty_cols)

    raw_X = X[_numeric_cols(X, cols)]
    scores = score_modular_by_bootstrap_mi(
        raw_X,
        eng,
        y,
        n_boot=n_boot,
        sample_fraction=sample_fraction,
        seed=seed,
        nbins=nbins,
    )
    if scores.empty:
        return X, scores

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
    qualified = scores[(scores["uplift_mean"] >= float(min_uplift_lcb)) & (scores["engineered_mi_lcb"] >= noise_floor)]
    # Family-wise permutation gate over EVERY scored candidate (not just the ones that cleared the LCB floor): the maximum over the whole family sets the
    # threshold, so the chance that any noise (column, period, op) survives stays at alpha however many were scored.
    from ._family_maxt_gate import family_maxt_pass

    _names = scores["engineered_col"].tolist()
    _fam_ok = family_maxt_pass(eng[_names], y, n_bins=nbins, seed=seed)
    _ok_names = {n for n, ok in zip(_names, _fam_ok) if ok}
    qualified = qualified[qualified["engineered_col"].isin(_ok_names)]
    winners = qualified.sort_values("engineered_mi_lcb", ascending=False).head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, eng[keep]], axis=1) if keep else X.copy(deep=False)
    return X_aug, scores


def _modular_noise_floor(scores: pd.DataFrame) -> float:
    """MAD noise floor anchored on the raw baseline MI distribution (see :func:`hybrid_modular_fe`)."""
    base_lcb = np.unique(scores["baseline_mi_lcb"].to_numpy())
    if base_lcb.size >= 4:
        med = float(np.median(base_lcb))
        mad = float(np.median(np.abs(base_lcb - med)))
        return med + 3.5 * 1.4826 * mad
    return 2.0 * float(base_lcb.max()) if base_lcb.size else 0.0


def _gate_first_enabled() -> bool:
    """``MLFRAME_MODULAR_GATE_FIRST=0`` selects the score-every-candidate-first pipeline (the bootstrap scores all ``3 x periods x columns`` candidates, then the family test)."""
    import os

    return os.environ.get("MLFRAME_MODULAR_GATE_FIRST", "1").strip().lower() in ("1", "true", "on", "yes")


def _hybrid_modular_fe_gate_first(X, y, src_cols, chunk, periods, top_k, n_boot, sample_fraction, seed, nbins, min_uplift_lcb):
    """:func:`hybrid_modular_fe` with the family-wise permutation test BEFORE the bootstrap scoring.

    A candidate is kept only if it clears the bootstrap gate AND the family test, and the family test needs only the candidate's own quantile codes. Running it first leaves the bootstrap
    (10 resamples x raw + candidate MI over every column, about a quarter of a 253-column fit) to the few candidates that pass it; on a frame without a periodic column that is none.
    The noise floor of the bootstrap gate is anchored on the raw baseline MI of every source column, which depends on the column and the seeded resamples only
    (``baseline_lcb_by_bootstrap_mi``), so it is computed without scoring any candidate. Survivors are regenerated and scored with the same seeded resamples, so a survivor's scores equal the
    score-first pipeline's up to the all-distinct tie guard of the resample gather, which looks at the matrix being scored.
    """
    from ._family_maxt_gate import _family_codes, family_maxt_pass_codes

    if not any(float(p) > 0.0 for p in periods):
        return X.copy(deep=False), pd.DataFrame()
    codes: list = []
    nxs: list = []
    names: list = []
    base_parts: list = []
    kept_sources: list = []
    source_of: dict = {}
    periods_pos = [float(p) for p in periods if float(p) > 0.0]
    for c0 in range(0, len(src_cols), chunk):
        cs = list(src_cols[c0 : c0 + chunk])
        eng = generate_modular_features(X, num_cols=cs, periods=periods)
        if eng.empty:
            continue
        cc, nx = _family_codes(eng, nbins)
        codes.append(cc)
        nxs.append(nx)
        names.extend(str(c) for c in eng.columns)
        for c in cs:
            for p in periods_pos:
                for op in ("mod", "sin", "cos"):
                    source_of[engineered_name_modular(c, p, op)] = str(c)
        kept_sources.extend(cs)
        del eng
    if not names or not kept_sources:
        return X.copy(deep=False), pd.DataFrame()
    # The raw baseline depends on the column and the seeded resamples only, so it is scored in blocks of source columns of its own (not one block per candidate chunk, which can be a single column
    # at a million rows): a wide block keeps every thread busy and the resamples and their tables are prepared once. Per-column values do not depend on the block.
    for b0 in range(0, len(kept_sources), _BASELINE_BLOCK_COLUMNS):
        block = kept_sources[b0 : b0 + _BASELINE_BLOCK_COLUMNS]
        base_parts.append(pd.DataFrame({"baseline_mi_lcb": baseline_lcb_by_bootstrap_mi(X[block], y, n_boot=n_boot, sample_fraction=sample_fraction, seed=seed, nbins=nbins)}))
    base = pd.concat(base_parts, ignore_index=True)
    noise_floor = _modular_noise_floor(base)
    cols_all = codes[0] if len(codes) == 1 else np.concatenate(codes, axis=0)
    del codes
    fam_ok = family_maxt_pass_codes(cols_all, np.concatenate(nxs), y, seed=seed)
    ok_names = [n for n, ok in zip(names, fam_ok) if ok]
    if not ok_names:
        return X.copy(deep=False), base
    sources = sorted({source_of[n] for n in ok_names if n in source_of})
    eng_ok = generate_modular_features(X, num_cols=[c for c in src_cols if str(c) in sources], periods=periods)[ok_names]
    scores = score_modular_by_bootstrap_mi(X[[c for c in src_cols if str(c) in sources]], eng_ok, y, n_boot=n_boot, sample_fraction=sample_fraction, seed=seed, nbins=nbins)
    if scores.empty:
        return X.copy(deep=False), base
    qualified = scores[(scores["uplift_mean"] >= float(min_uplift_lcb)) & (scores["engineered_mi_lcb"] >= noise_floor)]
    keep = list(qualified.sort_values("engineered_mi_lcb", ascending=False).head(int(top_k))["engineered_col"])
    if not keep:
        return X.copy(deep=False), scores
    return pd.concat([X, eng_ok[keep]], axis=1), scores


def _hybrid_modular_fe_streamed(X, y, src_cols, chunk, periods, top_k, n_boot, sample_fraction, seed, nbins, min_uplift_lcb):
    """:func:`hybrid_modular_fe` with the candidates generated and scored ``chunk`` source columns at a time.

    The whole candidate matrix (3 ops x periods per source column) does not fit the RAM budget: at 50000 rows x 253 columns it is
    1.5 GB before the scorer's copies. Each candidate's bootstrap score depends only on its own column and its source (the bootstrap rows
    are drawn from the same seeded stream for every chunk), so scoring chunk by chunk gives the same scores. Only the int16 family-gate
    codes of each chunk are kept; the family-wise gate then runs once over every candidate and the winners are regenerated.
    """
    from ._family_maxt_gate import _family_codes, family_maxt_pass_codes

    parts, codes, nxs, names = [], [], [], []
    for c0 in range(0, len(src_cols), chunk):
        cs = list(src_cols[c0 : c0 + chunk])
        eng = generate_modular_features(X, num_cols=cs, periods=periods)
        if eng.empty:
            continue
        sc = score_modular_by_bootstrap_mi(X[cs], eng, y, n_boot=n_boot, sample_fraction=sample_fraction, seed=seed, nbins=nbins)
        if sc.empty:
            continue
        nm = sc["engineered_col"].tolist()
        cc, nx = _family_codes(eng[nm], nbins)
        parts.append(sc)
        codes.append(cc)
        nxs.append(nx)
        names.extend(nm)
        del eng
    if not parts:
        return X, pd.DataFrame()
    scores = pd.concat(parts, ignore_index=True).sort_values("uplift_lcb", ascending=False, kind="mergesort").reset_index(drop=True)
    noise_floor = _modular_noise_floor(scores)
    qualified = scores[(scores["uplift_mean"] >= float(min_uplift_lcb)) & (scores["engineered_mi_lcb"] >= noise_floor)]
    cols_all = codes[0] if len(codes) == 1 else np.concatenate(codes, axis=0)
    del codes
    fam_ok = family_maxt_pass_codes(cols_all, np.concatenate(nxs), y, seed=seed)
    ok_names = {n for n, ok in zip(names, fam_ok) if ok}
    qualified = qualified[qualified["engineered_col"].isin(ok_names)]
    winners = qualified.sort_values("engineered_mi_lcb", ascending=False).head(int(top_k))
    keep = list(winners["engineered_col"])
    if not keep:
        return X.copy(deep=False), scores
    keep_src = sorted({str(winners.loc[winners["engineered_col"] == k, "source_col"].iloc[0]) for k in keep})
    eng_keep = generate_modular_features(X, num_cols=[c for c in src_cols if str(c) in keep_src], periods=periods)[keep]
    return pd.concat([X, eng_keep], axis=1), scores


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
) -> tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]:
    """Same as :func:`hybrid_modular_fe` but additionally returns a list of
    recipes - one per appended column - so ``MRMR.transform`` can recompute
    each modular feature on test data without re-running the bootstrap MI gate.
    Each recipe is pure arithmetic on X (no y), so replay is leakage-free.

    Returns ``(X_augmented, appended, recipes, scores)``.
    """
    X_aug, scores = hybrid_modular_fe(
        X,
        y,
        cols=cols,
        periods=periods,
        top_k=top_k,
        n_boot=n_boot,
        sample_fraction=sample_fraction,
        seed=seed,
        nbins=nbins,
        min_uplift_lcb=min_uplift_lcb,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    for name in appended:
        parsed = _parse_modular_name(name)
        if parsed is None:
            log_throttle(
                logger,
                "periodic_fe_cannot_parse_column",
                logging.WARNING,
                "hybrid_modular_fe_with_recipes: cannot parse op/period from " "column name %r; skipping recipe build.",
                name,
            )
            continue
        op, src, period = parsed
        recipes.append(
            build_modular_recipe(
                name=name,
                src_name=src,
                period=float(period),
                op=op,
            )
        )
    return X_aug, appended, recipes, scores


def build_modular_recipe(*, name: str, src_name: str, period: float, op: str) -> EngineeredRecipe:
    """Frozen recipe for one modular transform of ``X[src_name]``. Replay is
    pure arithmetic on the source column - no y reference, no fitted state, so
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
