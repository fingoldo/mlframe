"""njit candidate-column materialise kernels (serial + parallel twins), the binary
op-code registry, narrow-code-dtype selection, and the per-host serial-vs-parallel
kernel_tuning_cache spec + dispatch predicate for the FE pair-search."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# OPT-A (2026-06-07): per-host serial-vs-parallel crossover for the FE materialise /
# searchsorted kernels on the serial-main-thread path (no hardcoded threshold).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

# NJIT MATERIALISATION (2026-06-06). The chunk's candidate columns were filled by a PYTHON loop
# (``out[:, col] = bin_func(a, b)`` per candidate) -- GIL-held -> the remaining single-core bottleneck after the
# discretize/MI kernels were batched. ``_materialise_chunk_njit`` fills the WHOLE chunk in ONE ``nogil`` njit call,
# so the joblib pair-search threads (``backend="threading"``) run it concurrently across cores -- parallelism
# governed by ``n_jobs``. NOTE: it is deliberately NOT ``parallel=True`` -- a numba prange here would nest
# numba-parallel inside the joblib threads and deadlock the threading layer (0% CPU, idle threads). BIT-IDENTICAL to the
# numpy bin_funcs: every op is computed in float32 (the buffer + transformed_vars dtype), and the Python-float
# constants (``1e-9``, ``1.0``) are weak/value-based so the numpy expressions stay float32 too -> same hardware
# float32 ops, same per-element nan_to_num(nan=0, +-inf=0). Only the elementary closed ops (the default ``minimal``
# preset + the non-symmetric medium ops) are coded; a chunk containing any other op (hypot / maximal-preset
# specials) falls back to the per-candidate numpy path for bit-safety.
_NJIT_BINARY_OP_CODES: dict = {
    "mul": 0, "add": 1, "sub": 2, "div": 3, "max": 4, "min": 5,
    "abs_diff": 6, "signed": 7, "ratio_abs": 8,
}


def _njit_binary_op_codes(binary_transformations) -> "np.ndarray | None":
    """Return an int8 op-code per binary-op name (in registry order), or None if ANY op is not njit-coded
    (the caller then uses the per-candidate numpy path -- bit-safe for hypot / scipy.special / comparison ops)."""
    codes = []
    for name in binary_transformations:
        c = _NJIT_BINARY_OP_CODES.get(name)
        if c is None:
            return None
        codes.append(c)
    return np.asarray(codes, dtype=np.int8)


def _narrow_code_dtype(n_bins: int, requested_dtype) -> object:
    """OPT-B (2026-06-07): pick the NARROWEST signed-int dtype that holds the FE discretiser's
    ordinal codes ``[0, n_bins)`` -- ``int8`` when ``n_bins <= 127`` (the single-column
    ``discretize_array`` ALREADY defaults to ``int8``), else ``int16`` (``n_bins <= 32767``),
    else the requested dtype (or int32). The codes are non-negative ordinals, so a narrower
    storage width is VALUE-IDENTICAL -- only the bytes-per-element of the ``disc_2d`` code matrix
    shrink (1-2 vs the 4 of the ``quantization_dtype=int32`` default). That directly cuts the DRAM
    traffic of the two memory-bound hotspots that gather ``disc_2d[:, k]`` column-by-column out of
    a row-major buffer: the ``_searchsorted_2d_right`` code WRITE and the
    ``batch_mi_with_noise_gate`` per-column histogram READ (and it quarters the chunk-disc_2d
    allocation -- on the scene 2407x64152 chunk that is 147 MiB int8 vs 589 MiB int32).

    BIT-IDENTICAL MI: inside ``batch_mi_with_noise_gate`` the per-column histogram ``counts`` and
    the dense ``classes_dense`` / ``joint_counts`` accumulators are typed by the kernel's OWN
    ``dtype`` argument (>= int32), INDEPENDENTLY of ``disc_2d.dtype`` -- the kernel only READS
    ``disc_2d[r, k]`` as a non-negative index (``counts[col[r]] += 1``), which is width-agnostic.
    Every GPU twin re-casts ``disc_2d`` to int32 on H2D upload, so the device path is unaffected.
    ``requested_dtype`` is honoured only when the bins genuinely exceed int16 (never on the FE path
    where ``quantization_nbins`` ~10); we only ever NARROW from the int32 default, never widen."""
    nb = int(n_bins)
    if nb <= 127:
        return np.int8
    if nb <= 32767:
        return np.int16
    return requested_dtype if requested_dtype is not None else np.int32


@njit(nogil=True, cache=True, error_model="numpy")
def _materialise_chunk_njit(tv, a_cols, b_cols, op_codes, out):
    """Fill ``out[:, k] = op_k(tv[:, a_cols[k]], tv[:, b_cols[k]])`` for all K candidates, then nan_to_num each value
    to 0. float32 throughout -> bit-identical to the numpy bin_funcs (see header). SERIAL + ``nogil=True`` ON PURPOSE:
    check_prospective_fe_pairs runs under joblib ``backend="threading"`` (one thread per pair-chunk), so a numba
    ``parallel=True`` prange here would be numba-parallel NESTED inside Python threads -> the threading layer
    deadlocks (observed: 0% CPU, 28 idle threads). With ``nogil`` the joblib threads run THIS kernel concurrently
    across cores -- the parallelism is governed by ``n_jobs`` (consistent with the rest of the pair-search), with no
    nested-parallel hazard."""
    n = tv.shape[0]
    K = op_codes.shape[0]
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for k in range(K):
        ai = a_cols[k]
        bi = b_cols[k]
        op = op_codes[k]
        for r in range(n):
            a = tv[r, ai]
            b = tv[r, bi]
            if op == 0:  # mul
                v = a * b
            elif op == 1:  # add
                v = a + b
            elif op == 2:  # sub
                v = a - b
            elif op == 3:  # div = _safe_div (2026-06-13 form): EXACT x/y for y != 0, eps floor only on an
                # exact-zero denominator (the prior x/(y+sign(y)*eps+eps) perturbed every positive denom by 2*eps,
                # diverging from the canonical _safe_div). float64-promoted then cast to float32 to match _safe_div's
                # float64 division exactly (an all-float32 div would differ in the last bit).
                v = np.float32(np.float64(a) / (np.float64(b) if b != zero else 1e-9))
            elif op == 4:  # max = np.maximum (nan-propagating)
                if a != a or b != b:
                    v = a + b  # nan + anything -> nan (matches np.maximum nan propagation)
                else:
                    v = a if a > b else b
            elif op == 5:  # min = np.minimum (nan-propagating)
                if a != a or b != b:
                    v = a + b
                else:
                    v = a if a < b else b
            elif op == 6:  # abs_diff = |a - b|
                v = abs(a - b)
            elif op == 7:  # signed = sign(a)*|b|. np.sign(nan)=nan and np.abs(nan)=nan -> propagate nan.
                if a != a or b != b:
                    v = a + b
                else:
                    sgn = zero if a == zero else (one if a > zero else -one)
                    v = sgn * abs(b)
            else:  # op == 8: ratio_abs = a/(|b|+1). numpy/numba promote the float64 ``1.0`` literal ->
                # compute in float64 then cast, matching the numpy expression's last bit.
                v = np.float32(np.float64(a) / (np.float64(abs(b)) + 1.0))
            # np.nan_to_num(nan=0, posinf=0, neginf=0)
            if not (v == v and v != np.inf and v != -np.inf):
                v = zero
            out[r, k] = v


@njit(parallel=True, cache=True, error_model="numpy")
def _materialise_chunk_njit_parallel(tv, a_cols, b_cols, op_codes, out):
    """``parallel=True`` (prange over CANDIDATE COLUMNS) twin of ``_materialise_chunk_njit`` --
    BYTE-IDENTICAL output, only the outer ``for k`` candidate loop is a numba ``prange`` so the
    per-candidate float32 op fills spread across cores (OPT-A, 2026-06-07).

    Kept SEPARATE from the serial ``nogil`` kernel (``feedback_keep_all_kernel_versions``): the
    serial variant MUST stay for the joblib ``backend="threading"`` FE path (>=50000 rows) where
    a nested numba ``prange`` deadlocks the threading layer (observed 0% CPU / 28 idle threads --
    documented in the serial kernel). This twin is dispatched ONLY from the SERIAL-MAIN-THREAD FE
    path (``len(X) < 50000`` in ``_mrmr_fe_step`` -> ``check_prospective_fe_pairs`` runs with NO
    joblib threads), mirroring the column-prange already shipped in ``_quantile_edges_2d_njit`` /
    the searchsorted parallel twin.

    BIT-IDENTICAL: each candidate ``k`` reads its own operand columns ``tv[:, a_cols[k]]`` /
    ``tv[:, b_cols[k]]`` and writes ONLY ``out[:, k]`` -- zero cross-column dependence -> result
    independent of thread count. The per-element arithmetic (the float32 ops, the float64-promoted
    div/ratio_abs literals, the nan_to_num) is the IDENTICAL body as the serial kernel.
    """
    n = tv.shape[0]
    K = op_codes.shape[0]
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for k in prange(K):
        ai = a_cols[k]
        bi = b_cols[k]
        op = op_codes[k]
        for r in range(n):
            a = tv[r, ai]
            b = tv[r, bi]
            if op == 0:  # mul
                v = a * b
            elif op == 1:  # add
                v = a + b
            elif op == 2:  # sub
                v = a - b
            elif op == 3:  # div = _safe_div (2026-06-13 form): EXACT x/y for y != 0, eps floor only on exact-zero
                # (matches the serial twin + canonical _safe_div; float64-promoted then cast to float32).
                v = np.float32(np.float64(a) / (np.float64(b) if b != zero else 1e-9))
            elif op == 4:  # max = np.maximum (nan-propagating)
                if a != a or b != b:
                    v = a + b
                else:
                    v = a if a > b else b
            elif op == 5:  # min = np.minimum (nan-propagating)
                if a != a or b != b:
                    v = a + b
                else:
                    v = a if a < b else b
            elif op == 6:  # abs_diff = |a - b|
                v = abs(a - b)
            elif op == 7:  # signed = sign(a)*|b|
                if a != a or b != b:
                    v = a + b
                else:
                    sgn = zero if a == zero else (one if a > zero else -one)
                    v = sgn * abs(b)
            else:  # op == 8: ratio_abs = a/(|b|+1); float64-promoted like the serial twin
                v = np.float32(np.float64(a) / (np.float64(abs(b)) + 1.0))
            if not (v == v and v != np.inf and v != -np.inf):
                v = zero
            out[r, k] = v


# OPT-A SERIAL-vs-PARALLEL CROSSOVER (2026-06-07). On the serial-main-thread FE path
# (``len(X) < 50000`` in ``_mrmr_fe_step`` -> ``check_prospective_fe_pairs`` runs with NO
# joblib threads) the materialise + searchsorted kernels can use their ``parallel=True``
# (column-prange) twins instead of the serial ``nogil`` variants -- spreading the per-column
# work across cores. But ``prange`` has a fixed dispatch/fork-join overhead per call, so for a
# TINY chunk (few candidate columns) the serial kernel is faster. The crossover column-count is
# HARDWARE-dependent (core count, mem bandwidth), so it is resolved per-host via the canonical
# ``kernel_tuning_cache`` (NO hardcoded threshold -- ``feedback_use_kernel_tuning_cache_for_gpu``)
# with a measurement-backed fallback. ``choose(n_cols=K)`` returns "parallel" or "serial".
_FE_PARALLELISM_SWEEP_COLS = [16, 64, 256, 1024, 4096]
_FE_PARALLELISM_SALT = 1


def _make_fe_parallelism_inputs(dims: dict):
    """A (2407-ish rows x ``n_cols``) float32 operand table + index/op arrays mirroring the
    materialise kernel's call shape, so the sweep measures the SERIAL-vs-PARALLEL crossover on
    the real kernel signature."""
    rng = np.random.default_rng(0)
    n_rows = 2048
    K = int(dims["n_cols"])
    n_operands = max(2, min(K, 64))
    tv = rng.standard_normal((n_rows, n_operands)).astype(np.float32)
    a_cols = (rng.integers(0, n_operands, size=K)).astype(np.int64)
    b_cols = (rng.integers(0, n_operands, size=K)).astype(np.int64)
    ops = (rng.integers(0, 9, size=K)).astype(np.int8)
    out = np.empty((n_rows, K), dtype=np.float32)
    return (tv, a_cols, b_cols, ops, out)


def _run_fe_parallelism_sweep() -> list:
    """Real serial-vs-parallel materialise sweep over ``n_cols`` -> "serial" / "parallel"
    regions, fastest EQUIVALENT per band. CPU-only (no GPU axis); the two njit kernels are
    byte-identical so the sweep just times them."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    def _serial(tv, a_cols, b_cols, ops, out):
        """Sweep variant: single-threaded materialise kernel (the sweep's timing reference)."""
        _materialise_chunk_njit(tv, a_cols, b_cols, ops, out)
        return out

    def _parallel(tv, a_cols, b_cols, ops, out):
        """Sweep variant: ``prange``-parallel materialise kernel being benchmarked against ``_serial``."""
        _materialise_chunk_njit_parallel(tv, a_cols, b_cols, ops, out)
        return out

    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        {"serial": _serial, "parallel": _parallel},
        {"n_cols": list(_FE_PARALLELISM_SWEEP_COLS)},
        _make_fe_parallelism_inputs,
        reference="serial", repeats=3, equiv_rtol=0.0, equiv_atol=0.0,
    )


def _fe_parallelism_fallback_choice(n_cols: int) -> str:
    """Pre-sweep heuristic: parallel above a conservative column count where the prange
    fork-join overhead is amortised by the per-column work (each column does n_rows binary
    searches + n_rows float ops); serial below. 256 columns is a deliberately conservative
    floor -- the scene chunks carry thousands of columns, far above it."""
    return "parallel" if int(n_cols) >= 256 else "serial"


_FE_PARALLELISM_SPEC = kernel_tuner(
    kernel_name="mrmr_fe_kernel_parallelism",
    # Both materialise variants participate in the code_version salt (a numerics edit to either
    # invalidates the per-host cache). The searchsorted twins live in the discretization module
    # and share the SAME serial-vs-parallel decision; they are not imported here to avoid an
    # import cycle (this module is imported during discretization's own import graph).
    variant_fns=(_materialise_chunk_njit, _materialise_chunk_njit_parallel),
    tuner=_run_fe_parallelism_sweep,
    axes={"n_cols": list(_FE_PARALLELISM_SWEEP_COLS)},
    fallback=_fe_parallelism_fallback_choice,  # callable (n_cols) -> str
    gpu_capable=False,
    salt=_FE_PARALLELISM_SALT,
    cli_label="mrmr_fe_kernel_parallelism",
)


def _fe_use_parallel_kernels(n_cols: int, serial_main_thread: bool) -> bool:
    """OPT-A dispatch predicate: use the ``parallel=True`` materialise/searchsorted twins iff
    (a) we are on the SERIAL-MAIN-THREAD FE path (no joblib threading nest -- the ONLY place a
    numba prange is safe here) AND (b) the per-host kernel_tuning_cache says the column count is
    above the serial-vs-parallel crossover. On the joblib ``backend="threading"`` path
    (``serial_main_thread`` False) ALWAYS return False -> serial ``nogil`` kernels (nesting a
    prange inside the threading layer deadlocks)."""
    if not serial_main_thread:
        return False
    try:
        return bool(_FE_PARALLELISM_SPEC.choose(n_cols=int(n_cols)) == "parallel")
    except Exception:
        # Cache/pyutilz failure -> heuristic fallback (still gated on serial_main_thread).
        return _fe_parallelism_fallback_choice(int(n_cols)) == "parallel"


@njit(nogil=True, cache=True, error_model="numpy")
def _materialise_extval_njit(param_a, param_b_mat, op_codes, out):
    """Fill ``out[:, e*n_ops + o] = op_o(param_a, param_b_mat[:, e])`` for the external-
    validation MI sweep -- ALL (external_factor x binary_op) candidate columns in ONE
    nogil kernel.

    Unlike ``_materialise_chunk_njit`` this produces the RAW float64 bin_func output with
    NO nan_to_num: the external-validation path discretises the raw values (the per-
    candidate ``discretize_array`` it replaces never scrubbed NaN/inf either --
    ``nanpercentile`` ignores NaN and ``searchsorted`` routes NaN/inf to the rightmost
    bin), so scrubbing here would change the codes. float64 throughout so the arithmetic
    is bit-identical to the numpy bin_funcs (which upcast the float32 ``param_a`` to
    float64 against the float64 ``param_b``): mul/add/sub/max/min are the exact numpy ops
    and ``div`` mirrors the current ``_safe_div`` -- exact ``x/y`` for every nonzero denominator, with the ``1e-9``
    floor substituting only for an exact-zero ``y`` (the 2026-06-13 heavy-tail form, no per-denominator perturbation).

    ``op_codes`` are the ``_NJIT_BINARY_OP_CODES`` (0=mul 1=add 2=sub 3=div 4=max 5=min
    6=abs_diff 7=signed 8=ratio_abs -- ALL registry-coded ops, matching ``_materialise_chunk_njit``;
    a prior version implemented only 0-5 and silently computed 6/7/8 as ``min``);
    callers MUST gate on ``_njit_binary_op_codes(...) is not None`` (returns None for any op
    not in the registry, e.g. hypot / scipy.special) and fall back to the numpy loop otherwise. ``op_codes``
    is in registry (bin_func) order; the inner loop walks it so the column index matches
    the Python ``for ext: for bin_func:`` materialise order exactly.
    """
    n = param_b_mat.shape[0]
    n_ext = param_b_mat.shape[1]
    n_ops = op_codes.shape[0]
    for e in range(n_ext):
        for o in range(n_ops):
            op = op_codes[o]
            col = e * n_ops + o
            for r in range(n):
                a = np.float64(param_a[r])
                b = param_b_mat[r, e]
                if op == 0:  # mul
                    v = a * b
                elif op == 1:  # add
                    v = a + b
                elif op == 2:  # sub
                    v = a - b
                elif op == 3:  # div = _safe_div: exact x/y for y != 0, eps floor only on an exact-zero denominator
                    v = a / (b if b != 0.0 else 1e-9)
                elif op == 4:  # max = np.maximum (nan-propagating)
                    if a != a or b != b:
                        v = a + b
                    else:
                        v = a if a > b else b
                elif op == 5:  # min = np.minimum (nan-propagating)
                    if a != a or b != b:
                        v = a + b
                    else:
                        v = a if a < b else b
                elif op == 6:  # abs_diff = |a - b|
                    v = abs(a - b)
                elif op == 7:  # signed = sign(a)*|b| (nan-propagating)
                    if a != a or b != b:
                        v = a + b
                    else:
                        sgn = 0.0 if a == 0.0 else (1.0 if a > 0.0 else -1.0)
                        v = sgn * abs(b)
                else:  # op == 8: ratio_abs = a/(|b|+1)
                    v = a / (abs(b) + 1.0)
                out[r, col] = v
