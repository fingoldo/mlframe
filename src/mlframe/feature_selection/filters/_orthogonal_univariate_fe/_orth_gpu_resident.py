"""Matrix-native GPU-resident univariate orth-basis build + MI scoring.

Carved out of ``_orthogonal_univariate_fe/__init__.py`` (2026-07-09 monolith split, LOC budget).
Mirrors the host ``generate_univariate_basis_features`` + ``score_features_by_mi_uplift`` pipeline
but builds the candidate matrix ON the device and scores its plug-in MI resident (no H2D for the
engineered matrix). Re-exported from the parent package facade so ``_gpu_build_and_score_univariate``
and ``_raise_if_vram_insufficient`` keep their original import path for existing callers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..hermite_fe import _POLY_BASES
from ._orth_dedup import _dedup_collinear_source_cols
from ._orth_extra_basis_fe import _is_int_as_cat_axis


def _raise_if_vram_insufficient(n_rows: int, k_cols: int) -> None:
    """Raise ``RuntimeError`` when the resident (raw + engineered + scratch) matrices would not comfortably fit the
    current device's FREE VRAM, so the caller falls back to the host/CPU path BEFORE a monolithic multi-GB upload can
    exhaust cupy's pinned host buffer and poison the context. Reuses the FE VRAM governor's per-host fraction so the
    threshold matches the rest of the GPU FE path. A query failure is non-fatal (assume OK; the try/except OOM fallback
    remains the backstop). Multi-GPU: the ``_fe_gpu_batch`` executor already spreads candidate blocks across all visible
    devices with the same governor; this gate only guards the single-device resident builder's whole-matrix upload."""
    try:
        import cupy as cp

        free, _total = cp.cuda.runtime.memGetInfo()
    except Exception:
        return
    try:
        from .._gpu_resident_fe import _gpu_k_chunk_vram_fraction

        frac = _gpu_k_chunk_vram_fraction(int(n_rows))
    except Exception:
        frac = 0.5
    # raw (n x k) + engineered (n x k*degrees, ~2 degrees) + MI scratch; ~4x the raw matrix is a safe floor estimate.
    needed = int(n_rows) * max(1, int(k_cols)) * 8 * 4
    budget = int(free * float(frac))
    if needed > budget:
        raise RuntimeError(
            f"resident univariate FE needs ~{needed / 2**30:.1f}GB VRAM but only ~{budget / 2**30:.1f}GB "
            f"(free {free / 2**30:.1f}GB x {frac:.2f}) is available; routing to the host/CPU FE path."
        )


def _gpu_build_and_score_univariate(X, cols, degrees, basis, y, nbins):
    """MATRIX-NATIVE (Piece 3, gated): build the univariate orth-basis candidate matrix ON the device
    (_gpu_evaluate_basis_column) and score its plug-in MI RESIDENT (_plugin_mi_classif_batch_cuda_resident,
    no H2D) -- mirroring generate_univariate_basis_features + score_features_by_mi_uplift. Routing/dedup/
    skip rules match the host builder exactly (only the per-(col,basis,degree) eval + the MI move to the
    GPU). Returns ``(eng_matrix_cupy, names, scores_df)`` or ``(None, [], empty_scores)`` when no candidate.
    Raises on GPU failure so the caller falls back to the host path (never a correctness regression)."""
    import cupy as cp
    from ..hermite_fe import _plugin_mi_classif_batch_cuda_resident
    from ..hermite_fe._hermite_robust import _robust_axis_enabled
    from .._fe_deadline import fe_deadline_passed
    # Lazy import (avoids a circular import: the parent package's ``__init__`` imports THIS module).
    from . import _BASIS_CODE, basis_route_by_moments, basis_route_by_signal

    _cols_auto = cols is None
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cols = _dedup_collinear_source_cols(X, list(cols), corr_threshold=0.999)
    # Proactive free-VRAM gate. This builder assembles the resident raw + engineered (n_rows x k) matrices on the device;
    # on a large frame that is several GB staged through cupy's pinned HOST buffer. If the host itself is out of RAM
    # (paging) the pinned alloc fails, poisons the CUDA context, and every subsequent per-column op OOMs. Estimate the
    # device footprint and BAIL EARLY (RuntimeError -> the caller's documented host fallback, which now degrades each
    # basis eval to CPU) when it will not comfortably fit free VRAM, so we never trigger the poisoning allocation.
    _n_rows = len(X)
    _k_num = sum(1 for c in cols if pd.api.types.is_numeric_dtype(X[c]))
    _raise_if_vram_insufficient(_n_rows, _k_num)
    _empty = pd.DataFrame(columns=["engineered_col", "source_col", "baseline_mi", "engineered_mi", "uplift"])
    ra = _robust_axis_enabled()
    _ya = np.asarray(y)
    y_arr = np.asarray(_ya, dtype=np.int64) if np.issubdtype(_ya.dtype, np.integer) else _ya.astype(np.int64)
    # y is a FIT-CONSTANT re-uploaded on every orth-family call (univariate-decide / pair-cross / triplet /
    # quadruplet / meta-scorer / adaptive-arity each re-run this builder over the SAME X/y). Route through the
    # resident operand cache so it is uploaded ONCE per fit (selection-equivalent: same int64 labels).
    from .._fe_resident_operands import resident_operand, resident_code_operand, assemble_resident_matrix
    y_gpu = resident_code_operand(y_arr, "orth_uni_y")
    # y min/max is a fit-constant -> compute ONCE and reuse for both the raw-MI and the eng-MI resident
    # calls below, instead of each recomputing it (cp.min/max + scalar D2H). Bit-identical (y is invariant).
    _ymm = cp.asnumpy(cp.stack((cp.min(y_gpu), cp.max(y_gpu))))
    _ymin = int(_ymm[0]); _ncls = int(_ymm[1]) - _ymin + 1
    # Baseline RAW-column MI (resident), for the uplift denominator. Built here but SCORED BELOW in ONE
    # resident MI call stacked with eng_mat -- per-column MI is independent, so the values are identical to
    # two separate calls while issuing one launch set instead of two (raw_mi feeds raw_mi_map at the rows
    # loop, which runs after eng_mat exists, so deferring the score is safe).
    raw_cols = [c for c in cols if pd.api.types.is_numeric_dtype(X[c])]
    raw_mi_map: dict = {}
    raw_mat = None
    if raw_cols:
        # raw_mat is the base raw feature-column matrix, a FIT-CONSTANT re-built+re-uploaded per orth-family
        # call over the SAME X. DEVICE-ASSEMBLE it from its per-column resident operands: each raw column is
        # already uploaded once by the basis builders, so stacking the resident columns content-hits the cache
        # and the whole (n, k) matrix never crosses H2D (vs the prior single whole-matrix upload, a distinct
        # blob that never deduped). Column j is X[raw_cols[j]] verbatim -> same bytes -> selection-equivalent.
        from .._fe_usability_signal import _crit_np_dtype
        _rm_dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
        raw_mat = assemble_resident_matrix(
            X[raw_cols].to_numpy(dtype=_rm_dt), raw_cols, ("orth_raw_mat", tuple(raw_cols)), dtype=_rm_dt,
        )
    code = _BASIS_CODE
    # Routing + skips run on the HOST (cheap njit / moment fingerprint), mirroring the host builder;
    # only the heavy per-(col,basis,degree) eval + the MI move to the GPU -- and the eval is BATCHED.
    # First pass: apply the cheap host skip rules to pick candidate columns + their operand arrays.
    cand_cols: list = []
    cand_x: list = []
    for col in cols:
        if fe_deadline_passed():
            break
        from .._fe_usability_signal import _crit_np_dtype
        x = np.asarray(X[col].to_numpy(), dtype=_crit_np_dtype())  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
        if _cols_auto and _is_int_as_cat_axis(x):
            continue
        if not np.isfinite(x).all():
            continue
        cand_cols.append(col)
        cand_x.append(np.ascontiguousarray(x))
    if not cand_x:
        return None, [], _empty
    from .._gpu_resident_fe import (  # type: ignore[attr-defined]  # dynamically re-exported via globals()
        _gpu_evaluate_basis_matrix, fe_gpu_routing_enabled, _gpu_route_bases_batched,
    )
    # GPU ROUTING (opt-in, default OFF): decide every candidate column's basis on the device at once,
    # mirroring the per-column host basis_route_by_signal. Falls back to the host router per column on any
    # GPU failure or where the device router returned None (degenerate). The top-level guards (y usable,
    # n>=30) match basis_route_by_signal's host fallback conditions, applied once since y/n are shared.
    _gpu_routed = None
    _Mr = None  # resident (n, n_cand) operand matrix uploaded ONCE for routing, reused for the basis-MI build
    if basis == "auto" and y is not None and fe_gpu_routing_enabled():
        _yc = np.asarray(_ya, dtype=np.float64).ravel()
        if _yc.size == cand_x[0].size and cand_x[0].size >= 30 and np.isfinite(_yc).all() and float(np.std(_yc)) >= 1e-12:
            try:
                # _Mr is the candidate-column matrix -- each column is a RAW X base column (cand_x[j] =
                # X[cand_cols[j]] verbatim). DEVICE-ASSEMBLE it from the per-column resident operands so it
                # never crosses H2D as a whole (n, n_cand) blob: each raw column is already resident under the
                # shared ("xbasis_op", col) role, so stacking the resident columns content-hits the cache.
                # Column j is the raw column verbatim -> same bytes -> selection-equivalent; a name/shape
                # mismatch or cupy fault falls back to the whole-matrix upload.
                _Mr = assemble_resident_matrix(
                    np.column_stack(cand_x), cand_cols, ("orth_Mr", tuple(cand_cols)), dtype=np.float64,
                )
                # _yc is the FIT-CONSTANT routing target, re-uploaded per orth-family call (a fresh cp.asarray
                # each time). Route through the content-keyed resident cache so it uploads ONCE per fit and
                # every later orth call reuses the resident copy. Read-only f64 -> selection-equivalent.
                _yc_gpu = resident_operand(_yc, "orth_route_y", dtype=np.float64)
                _gpu_routed = _gpu_route_bases_batched(
                    cp, _Mr, _yc_gpu, list(_POLY_BASES), tuple(degrees), robust_axis=ra,
                )
            except Exception:
                _gpu_routed = None
                _Mr = None
    used_x: list = []
    used_bases: list = []
    used_src: list = []
    used_idx: list = []  # index into cand_x of each survivor, so a resident _Mr can be reused by slice
    for _i, col in enumerate(cand_cols):
        x = cand_x[_i]
        if basis == "auto":
            if _gpu_routed is not None and _gpu_routed[_i] is not None:
                chosen = _gpu_routed[_i]
            else:
                chosen = basis_route_by_signal(x, _ya, degrees=degrees) if y is not None else basis_route_by_moments(x)
        else:
            chosen = basis
        if chosen not in _POLY_BASES:
            continue
        used_x.append(x)
        used_bases.append(chosen)
        used_src.append(col)
        used_idx.append(_i)
    if not used_x:
        return None, [], _empty
    # ONE H2D of the (n, n_used) operand matrix, then ONE vectorised preprocess+Clenshaw per (basis, robust)
    # group/degree. When GPU routing already uploaded the candidate matrix, REUSE it (device slice, no second
    # H2D) -- in residency mode the operands are already on the GPU; re-uploading would be a redundant copy.
    if _Mr is not None:
        M = _Mr if used_idx == list(range(_Mr.shape[1])) else _Mr[:, used_idx]
    else:
        # NOTE (wave-10 audit): ``used_idx`` indexes into ``cand_x``/``cand_cols`` (the host-skip-filtered
        # candidate set), NOT into ``raw_mat``'s ``raw_cols`` ordering -- ``cand_cols`` drops non-finite /
        # int-as-cat columns that ``raw_cols`` does not, so the two lists can diverge and a literal
        # ``raw_mat[:, used_idx]`` reuse would silently pick the WRONG columns whenever any column was
        # dropped. What IS always true is that every ``used_src[j]`` name is the SAME raw column
        # ``X[used_src[j]]`` verbatim that ``raw_mat`` (when built) sources from -- so instead of an
        # index-based reuse, assemble THIS matrix from its own per-column resident operands keyed by NAME
        # (``used_src``), at the SAME dtype (float64) ``_Mr`` uses for its own per-column uploads (matching
        # the ``_Mr is not None`` branch above so both take the same downstream dtype). ``raw_mat``'s OWN
        # per-column uploads are at ``_rm_dt`` (``_crit_np_dtype()``, float32 by DEFAULT under
        # ``MLFRAME_CRIT_DTYPE_RELAXED``, float64 only when that is disabled) -- so this content-hits
        # ``raw_mat``'s cached columns whenever ``_rm_dt`` happens to be float64 (the non-default strict
        # mode), and ALWAYS content-hits ``_Mr``'s cached columns (same float64 dtype) or ANY other
        # same-fit caller later requesting the same column at float64 -- without ever re-uploading the
        # whole (n, n_used) matrix as one un-deduped blob (the prior behaviour here).
        M = assemble_resident_matrix(
            np.column_stack(used_x), used_src, ("orth_used_M", tuple(used_src)), dtype=np.float64,
        )
    eng_mat, meta = _gpu_evaluate_basis_matrix(cp, M, used_bases, list(degrees), robust_axis=ra)
    if eng_mat is None:
        return None, [], _empty
    names = [f"{used_src[_ci]}__{code.get(_b, _b)}{_d}" for (_ci, _b, _d) in meta]
    # D1 (2026-06-22): the TRUE source per emitted name is ``used_src[_ci]`` -- carry it
    # directly rather than re-parsing the name via ``split("__", 1)[0]`` (which mis-stems a
    # one-hot source ``"city__NY"`` and collapses the uplift denominator to the 1e-12 floor).
    name_src = [used_src[_ci] for (_ci, _b, _d) in meta]
    # ONE resident MI over [raw_mat | eng_mat] stacked -- per-column independent, so raw_mi/eng_mi are
    # bit-identical to two separate calls, with one launch set instead of two. Both are already device-
    # resident, so this stays H2D-free (do NOT route through the host-input batcher, which would round-trip).
    if raw_mat is not None:
        _stacked = cp.concatenate((raw_mat, eng_mat.astype(cp.float64, copy=False)), axis=1)
        _all_mi = _plugin_mi_classif_batch_cuda_resident(_stacked, y_gpu, nbins, y_min=_ymin, n_classes=_ncls)
        _rk = int(raw_mat.shape[1])
        raw_mi_map = dict(zip(raw_cols, [float(v) for v in _all_mi[:_rk]]))
        eng_mi = _all_mi[_rk:]
    else:
        eng_mi = _plugin_mi_classif_batch_cuda_resident(eng_mat, y_gpu, nbins, y_min=_ymin, n_classes=_ncls)
    rows = []
    for j, nm in enumerate(names):
        src = name_src[j]
        base = float(raw_mi_map.get(src, 0.0))
        emi = float(eng_mi[j])
        rows.append({
            "engineered_col": nm, "source_col": src,
            "baseline_mi": base, "engineered_mi": emi, "uplift": emi / (base + 1e-12),
        })
    scores = pd.DataFrame(rows).sort_values("uplift", ascending=False).reset_index(drop=True)
    return eng_mat, names, scores
