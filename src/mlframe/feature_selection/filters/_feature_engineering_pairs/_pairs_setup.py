"""Pre-loop operand setup for ``check_prospective_fe_pairs`` (carved 2026-06-22,
Tier E -- third sibling).

Two verbatim setup blocks lifted out of ``_pairs_core.check_prospective_fe_pairs``
so the parent orchestration drops under the 1k-LOC ceiling:

  * ``_fit_prewarp_and_gate_med`` -- the per-operand learned pre-warp (rank-1 ALS,
    out-of-sample validation) + the per-operand median-gate fit. Returns the four
    fitted-state objects the rest of the pair search consumes.
  * ``_build_operand_table`` -- the unary-materialise loop that fills
    ``transformed_vars`` (in place) + the ``{(var, unary): col}`` index, plus the
    gated GPU-resident operand-table mirror. Returns ``vars_transformations``.

Carved straight: every local read is now an explicit parameter, the lazy
framework imports are passed in, and the in-place ``transformed_vars`` fill is
preserved (the parent passes the same preallocated array). Selection / fitted
state / RNG are byte-for-byte identical to the pre-carve in-function blocks.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermval

from ._pairs_gates import _GATE_MED_UNARY, _PREWARP_UNARY, _gate_med_apply


def _fit_prewarp_and_gate_med(
    *,
    prospective_pairs,
    prewarp_enable,
    prewarp_y,
    prewarp_y_continuous,
    prewarp_basis,
    prewarp_max_degree,
    prewarp_min_val_corr,
    fe_gate_med_enable,
    original_cols,
    _use_subsample,
    _full_n_rows,
    _sample_idx,
    _extval_raw_col,
):
    """Fit the per-operand pre-warp specs + median-gate medians. Returns
    ``(_prewarp_active, _prewarp_spec_by_var, _gate_med_active,
    _gate_med_median_by_var)``.

    CAVEAT: when ``_use_subsample`` is set, both the prewarp ALS coefficients and the gate-med medians are estimated on the
    pair-search SUBSAMPLE (``_sample_idx``), not on full n -- deliberately, so they align with the subsampled operand values
    the pair search scores against. They are therefore subsample estimates and may differ slightly from a full-n fit. This is
    safe by construction: the FROZEN constant is stored on the survivor recipe and replayed closed-form (leak-free), so the
    transform-time column is an exact function of the stored constant regardless of how that constant was estimated."""
    # PER-OPERAND PRE-WARP setup (2026-06-02). When enabled, fit ONE learned
    # 1-D pre-warp per raw operand against the (subsample-aligned) target, and
    # expose it as an extra pseudo-unary named ``_PREWARP_UNARY`` so the existing
    # unary x unary x binary search naturally considers ``binary(prewarp(a),
    # prewarp(b))``, ``binary(prewarp(a), b)`` etc. The fitted spec per var is
    # kept in ``_prewarp_spec_by_var`` for survivor recipe construction; warped
    # values are written into ``transformed_vars`` like any other unary.
    _prewarp_active = bool(prewarp_enable) and prewarp_y is not None
    _prewarp_spec_by_var: dict[int, dict] = {}
    _prewarp_y_eff = None
    if _prewarp_active:
        from ..hermite_fe import apply_operand_prewarp, fit_pair_prewarp_als
        # The ALS reconstruction target: prefer the CONTINUOUS y when supplied (it
        # is the faithful least-squares target; the binned ``classes_y`` codes the
        # target-rebin guard produces are for the MI screen, not for reconstructing
        # a continuous f(a)*g(b)). Fall back to ``classes_y`` codes when no
        # continuous target was threaded (legacy / non-numeric / multi-output y).
        _pw_y_src = prewarp_y_continuous if prewarp_y_continuous is not None else prewarp_y
        _pw_y = np.asarray(_pw_y_src)
        if _use_subsample and _pw_y.shape[0] == _full_n_rows:
            _pw_y = _pw_y[_sample_idx]
        _prewarp_y_eff = np.ascontiguousarray(_pw_y, dtype=np.float64)

        # JOINT per-pair ALS pre-fit. For each prospective pair fit BOTH operand
        # warps together (rank-1 ALS); an independent 1-D fit cannot recover the
        # b-side of a product target whose b-marginal is ~0. First pairing wins
        # for a var shared across pairs (pairs are processed most-prospective-
        # first, so a shared var binds to its strongest interaction). None specs
        # leave the pseudo-unary unregistered for that var.
        # Q8 (2026-06-07): route the prewarp operand extraction through the SHARED
        # ``_extval_raw_col`` memo (the single {var: raw-ndarray} cache) instead of a
        # second un-memoised ``X.iloc[...].values`` per call. The prewarp loop reads each
        # pair's two operands, and a var shared across many prospective pairs was previously
        # re-extracted once PER pair; the shared memo extracts each distinct var ONCE.
        # Bit-identical: ``_extval_raw_col`` performs the IDENTICAL ``.values`` / ``.to_numpy()``
        # extract (same None-on-missing guard) -- only the redundant re-reads are removed.
        _operand_vals = _extval_raw_col

        # OUT-OF-SAMPLE PREWARP VALIDATION (2026-06-03). The ALS prewarp is a
        # SUPERVISED per-operand fit; at small n it overfits noise operands (the
        # in-sample uplift is inflated by the fit AND by the multiple operand/pair
        # comparisons), so a noise-paired warp clears the in-sample uplift gate,
        # gets engineered, and ABSORBS a genuine feature -- the raw column then
        # reads as redundant and is dropped, leaving a noise-diluted feature
        # (measured: a genuine X5 dropped at n=500). Guard: fit the warp on a TRAIN
        # slice and keep it only if its rank-1 reconstruction f(a)*g(b) still tracks
        # y on a HELD-OUT slice. Genuine synergy (incl. zero-marginal XOR) and
        # genuine non-monotone inners generalise; overfit-on-noise collapses on the
        # held-out slice. At large n train ~= full so genuine recovery is untouched.
        # ``fe_pair_prewarp_min_val_corr`` (default 0.08) is the held-out floor; 0.0
        # restores the legacy in-sample-only fit.
        _pw_min_val_corr = float(prewarp_min_val_corr or 0.0)
        _pw_n = int(_prewarp_y_eff.shape[0]) if hasattr(_prewarp_y_eff, "shape") else len(_prewarp_y_eff)
        # Deterministic stride split (no RNG): every 3rd row -> validation (~33%).
        _pw_cv_ok = _pw_min_val_corr > 0.0 and _pw_n >= 60
        if _pw_cv_ok:
            _val_mask = np.arange(_pw_n) % 3 == 0
            _tr_mask = ~_val_mask
            _y_tr = _prewarp_y_eff[_tr_mask]
            _y_val = _prewarp_y_eff[_val_mask] - float(np.mean(_prewarp_y_eff[_val_mask]))

        def _prewarp_generalises(_va_full, _vb_full):
            """True if an ALS warp fit on the train slice still tracks y on the
            held-out slice (rank-1 reconstruction correlation >= floor)."""
            if not _pw_cv_ok:
                return True  # CV disabled / n too small -> accept (legacy behaviour)
            try:
                _a = np.asarray(_va_full, dtype=np.float64).reshape(-1)
                _b = np.asarray(_vb_full, dtype=np.float64).reshape(-1)
                if _a.shape[0] != _pw_n or _b.shape[0] != _pw_n:
                    return True  # length mismatch (subsample edge) -> don't block
                _sa_tr, _sb_tr = fit_pair_prewarp_als(
                    _a[_tr_mask], _b[_tr_mask], _y_tr,
                    basis=prewarp_basis, max_degree=prewarp_max_degree,
                )
                if _sa_tr is None or _sb_tr is None:
                    return False
                _wa = apply_operand_prewarp(_a[_val_mask], _sa_tr)
                _wb = apply_operand_prewarp(_b[_val_mask], _sb_tr)
                _recon = _wa * _wb
                if float(np.std(_recon)) < 1e-12 or float(np.std(_y_val)) < 1e-12:
                    return False
                # measure-experiment-rejected (2026-06-03): a dcor / MI held-out
                # floor (to recover non-monotone XOR prewarps that |corr| might
                # under-credit at small n) gives NO gain. The rank-1 reconstruction
                # f(a)*g(b) is FIT to approximate y, so it is linear-in-y by
                # construction -> |Pearson| is the right measure and is already
                # high for genuine synergy (XOR-sign reconstruction |corr| 0.64-0.75
                # at n=200, far above this 0.08 floor); benched 0/20 cases where
                # |corr|<floor BUT dcor>=0.15 across mul/xor-sign/sq*abs/a*sin(b).
                return abs(float(np.corrcoef(_recon, _y_val)[0, 1])) >= _pw_min_val_corr
            except Exception:
                return True  # validation failure -> fall back to accepting the warp

        # bench-attempt-rejected (2026-06-20): "skip the ALS prewarp fit when the clean-form
        # demotion (commit 3e62742e) would discard it anyway" was investigated and NOT shipped.
        # (1) A per-pair MEMO / cross-chunk shared cache to drop "wasted" refits was a NO-OP:
        # instrumentation showed every distinct (var_a, var_b) index pair is already fit EXACTLY
        # ONCE per FE step (memo never hit a duplicate). The apparent "32 full fits / 8 distinct
        # pairs" signal was a measurement artifact -- distinct var-index pairs collided under a
        # first-2-operand-values signature; there is no redundant fitting to eliminate, so the
        # batched/shared-factorisation lever (B) has no target.
        # (2) A PRE-SCREEN (A) -- skip the fit when the prewarp cannot clear the +5% demotion bar
        # -- needs the clean-form's continuous-y |corr|, but that is the best NON-prewarp ENGINEERED
        # config (unary x unary x binary), i.e. the very search being timed; it is not cheaply
        # computable before the fit. The cheap proxy max(best_single_unary|corr(a)|,|corr(b)|) does
        # NOT separate cleanly: measured prewarp_recon/best_single_unary ratio on the canonical
        # demoted pairs ranged 1.00-1.12 while the genuinely-non-monotone RETAINED case
        # (y=(a**3-2a)*(b**2-b)) ranged 1.05-1.47 -- overlapping bands, so any skip threshold safe
        # for the prewarp_retained pin (>=1.36) would leave most demoted pairs unskipped, and a
        # tighter one would risk pruning the retained case. The whole lstsq/ALS stage is only ~2.8s
        # of a ~128s canonical n=100k fit (~2.2%), and only a fraction is safely skippable, so the
        # bounded, selection-risky win does not clear the bar. Left as the exact per-pair fit.
        for raw_vars_pair, _ in prospective_pairs.keys():
            _va, _vb = raw_vars_pair[0], raw_vars_pair[1]
            if _va in _prewarp_spec_by_var and _vb in _prewarp_spec_by_var:
                continue
            _vals_a = _operand_vals(_va)
            _vals_b = _operand_vals(_vb)
            if _vals_a is None or _vals_b is None:
                continue
            # Reject the warp for this pair if it does not generalise out-of-sample
            # (overfit-on-noise). Leaves the operands unregistered -> the pair search
            # falls back to the library unaries, which do not overfit.
            if not _prewarp_generalises(_vals_a, _vals_b):
                continue
            _sa, _sb = fit_pair_prewarp_als(
                _vals_a, _vals_b, _prewarp_y_eff,
                basis=prewarp_basis, max_degree=prewarp_max_degree,
            )
            if _va not in _prewarp_spec_by_var:
                _prewarp_spec_by_var[_va] = _sa
            if _vb not in _prewarp_spec_by_var:
                _prewarp_spec_by_var[_vb] = _sb

    # PER-OPERAND MEDIAN GATE setup (2026-06-04). When enabled, fit ONE TRAIN
    # median per raw operand (on the subsample-aligned slice, exactly like the
    # operand values the unary search consumes) and expose it as an extra
    # pseudo-unary named ``_GATE_MED_UNARY``. The fit is a single ``np.median``
    # per operand -- no supervision, no held-out validation (a median does not
    # overfit). The fitted float per var is kept in ``_gate_med_median_by_var``
    # for survivor recipe construction; the gated 0/1 column is written into
    # ``transformed_vars`` like any other unary. Operands missing from
    # ``original_cols`` or with no usable variance leave the pseudo-unary
    # unregistered for that var (search falls back to the real unaries).
    _gate_med_active = bool(fe_gate_med_enable)
    _gate_med_median_by_var: dict[int, float] = {}
    if _gate_med_active:
        for raw_vars_pair, _ in prospective_pairs.keys():
            for _gv in raw_vars_pair:
                if _gv in _gate_med_median_by_var:
                    continue
                if _gv not in original_cols:
                    continue
                # Q8: shared {var: raw-ndarray} memo (bit-identical extract).
                _gvals = _extval_raw_col(_gv)
                if _gvals is None:
                    continue
                _gf = np.asarray(_gvals, dtype=np.float64)
                _gmed = float(np.nanmedian(_gf)) if _gf.size else 0.0
                if not np.isfinite(_gmed):
                    continue
                # FE_PAIRS_CORE-3 fix (mrmr_audit_2026-07-22): the comment claimed "Reject no-variance
                # operands (a constant gate is dead)" but the ONLY guard was the isfinite check above,
                # which does not reject a literal constant, non-finite-safe operand (a degenerate/dummy
                # column with a well-defined finite median) -- it registered under gate_med and
                # materialised an all-zero dead column ((x > median) == False everywhere when x is
                # constant), wasting unary+binary+MI-sweep work for every constant column whenever
                # fe_gate_med_enable=True. Add the variance guard the comment already promised.
                _gfinite = _gf[np.isfinite(_gf)]
                if _gfinite.size == 0 or float(np.nanmax(_gfinite) - np.nanmin(_gfinite)) <= 0.0:
                    continue
                _gate_med_median_by_var[_gv] = _gmed
    return _prewarp_active, _prewarp_spec_by_var, _gate_med_active, _gate_med_median_by_var


def _build_operand_table(
    *,
    prospective_pairs,
    transformed_vars,
    _unary_names_eff,
    unary_transformations,
    _extval_raw_col,
    _prewarp_active,
    _prewarp_spec_by_var,
    _gate_med_median_by_var,
    cols,
    verbose,
    logger,
    gpu_compatible_unary_names,
):
    """Materialise the deduped unary operand table into ``transformed_vars`` (in
    place) and build the ``{(var, unary): col}`` index; build the gated
    GPU-resident operand-table mirror. Returns ``vars_transformations``."""
    # Lazy import (parent-resident helper; only needed for the prewarp pseudo-unary column).
    from ..hermite_fe import apply_operand_prewarp
    vars_transformations = {}
    # GPU-RESIDENT OPERAND TABLE (phase 1, gated). Record each successfully-built operand column's
    # (col_idx, raw_vals, unary_name) so a GPU-resident mirror of ``transformed_vars`` can be produced ON
    # the device (the bulk plain-unary columns rebuilt via _unary_apply; prewarp/gate_med/poly copied from
    # the host) -- removing the materialise H2D. Populated only when the gate is on (else stays empty/cheap).
    from .._gpu_resident_fe import _cuda_present
    # fe_gpu_resident_operands_enabled's HOME module is _gpu_resident_materialise (carved there 2026-06-23);
    # _gpu_resident_select only RE-EXPORTS it via a bottom-of-module rebind loop. That re-export is NOT present
    # under a partial-init import cycle: when _gpu_resident_materialise is imported first, _gpu_resident_select's
    # rebind loop runs while _gpu_resident_materialise is mid-top-import (the name not yet defined), so the
    # re-export is silently skipped and importing it from _gpu_resident_select raises ImportError -- which the
    # try/except below then swallows, DISABLING the GPU-resident operand build (a silent residency regression).
    # Import from the HOME module so the name is always present regardless of sibling import order.
    from .._gpu_resident_materialise import fe_gpu_resident_operands_enabled
    _resident_operands_on = False
    try:
        _resident_operands_on = fe_gpu_resident_operands_enabled() and _cuda_present()
    except Exception:
        _resident_operands_on = False
    _operand_col_specs: list | None = [] if _resident_operands_on else None
    i = 0
    for raw_vars_pair, _pair_mi in prospective_pairs.keys():
        for var in raw_vars_pair:
            # Q8 (2026-06-07): SHARED {var: raw-ndarray} memo. This main unary-materialise
            # loop iterates over (pair, var); a var shared across prospective pairs was
            # previously re-extracted via ``X.iloc[...].values`` once per occurrence. Reading
            # the shared ``_extval_raw_col`` memo (same Polars/pandas extract: ``X[:, idx].to_numpy()``
            # / ``X.iloc[:, idx].values``) extracts each distinct var ONCE. Bit-identical raw values.
            #
            # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): ``_extval_raw_col`` now also
            # resolves engineered operands (var not in ``original_cols``) by NAME from the
            # augmented frame, so a step-k>1 ``(eng_i, eng_j)`` pair materialises a real
            # composite candidate. It returns ``None`` only when the var resolves to neither
            # a raw position nor an augmented-frame column (a temp / dropped index); skip
            # silently in that case rather than KeyError out of the whole FE block.
            vals = _extval_raw_col(var)
            if vals is None:
                continue
            for tr_name in _unary_names_eff:
                tr_func = unary_transformations.get(tr_name)
                key = (var, tr_name)
                # Per-operand learned pre-warp: the joint ALS spec was pre-fit
                # above (per pair). When the var has no usable spec (solve failed
                # / non-polynomial basis) the pseudo-unary is simply not
                # registered and the search proceeds with the real unaries only.
                # The fitted spec is stashed for survivor recipe construction
                # (leak-safe replay from coeffs alone).
                if tr_name == _PREWARP_UNARY:
                    if _prewarp_spec_by_var.get(var) is None:
                        continue
                # Median-gate pseudo-unary: skip vars with no fitted median (no
                # variance / not in original_cols). The fitted float is stashed
                # in ``_gate_med_median_by_var`` for survivor recipe construction
                # (leak-safe replay from the stored median alone).
                if tr_name == _GATE_MED_UNARY:
                    if var not in _gate_med_median_by_var:
                        continue
                if key not in vars_transformations:
                    try:
                        if tr_name == _PREWARP_UNARY:
                            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                transformed_vars[:, i] = apply_operand_prewarp(vals, _prewarp_spec_by_var[var])
                        elif tr_name == _GATE_MED_UNARY:
                            transformed_vars[:, i] = _gate_med_apply(vals, _gate_med_median_by_var[var])
                        elif "poly_" in tr_name:
                            # Unbounded hermval tails can overflow; suppress the resulting RuntimeWarnings,
                            # matching every sibling unary branch here (prewarp above, plain-unary below).
                            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                transformed_vars[:, i] = hermval(vals, c=tr_func)
                        else:
                            # WAVE 5 (1/4): if CUDA is available, the
                            # transformation is GPU-compatible, AND the
                            # column is large enough to amortise the H2D
                            # + D2H round trip, run the elementwise op on
                            # GPU via cupy. The numpy-vs-cupy crossover for
                            # this n_samples is resolved per-host via the
                            # shared get_or_tune orchestrator (kernel
                            # "unary_elementwise"), with the old fixed
                            # ~500k-cell breakeven as the measurement-backed
                            # fallback. A "cupy" choice is still gated below
                            # on live CUDA availability + per-op compat.
                            _gpu_used = False
                            from pyutilz.performance.kernel_tuning import array_location

                            from .._unary_elementwise_tuning import unary_elementwise_backend_choice
                            # residency-aware: VRAM-resident input skips H2D, which flips the
                            # numpy/cupy crossover (measured), so pass where ``vals`` lives.
                            _want_gpu = unary_elementwise_backend_choice(int(vals.size), array_location(vals)) == "cupy"
                            if _want_gpu and tr_name in gpu_compatible_unary_names():
                                try:
                                    from pyutilz.core.pythonlib import is_cuda_available
                                    if is_cuda_available():
                                        import cupy as cp
                                        _cp_fn = getattr(cp, tr_name, None)
                                        if _cp_fn is not None:
                                            d_vals = cp.asarray(vals)
                                            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                                d_res = _cp_fn(d_vals)
                                            transformed_vars[:, i] = cp.asnumpy(d_res)
                                            _gpu_used = True
                                except Exception:
                                    _gpu_used = False  # fall through to CPU
                            if not _gpu_used:
                                # Suppress unary-transform NaN/inf RuntimeWarnings
                                # (eg ``overflow in exp``, ``divide by zero in
                                # log``). The downstream nan_to_num + MI-gate
                                # already filter pathological rows; the bare
                                # numpy emit only spams stderr.
                                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                                    transformed_vars[:, i] = tr_func(vals)
                    except Exception as e:
                        # ``np.isnan`` / ``np.isinf`` / ``np.nanmin`` only work on float dtypes. When ``vals`` is object/string (e.g. a polars Utf8 cat column not encoded
                        # before reaching FE), calling them inside the error-log formatter itself raises -- masking the real transformation error and aborting MRMR
                        # entirely. Compute numeric-only diagnostics conditionally.
                        if np.issubdtype(vals.dtype, np.floating):
                            _diag = f", isnan={np.isnan(vals).sum()}, " f"isinf={np.isinf(vals).sum()}, nanmin={np.nanmin(vals)}"
                        else:
                            _diag = f", dtype={vals.dtype} (numeric diagnostics skipped)"
                        logger.error("Error when performing %s on array %s, var=%s: %s%s", tr_name, vals[:5], cols[var], e, _diag)
                    else:
                        vars_transformations[key] = i
                        if _operand_col_specs is not None:
                            # GPU-resident operand table. A PLAIN unary is GPU-built via _unary_apply from the
                            # resident raw (``vals``). PREWARP (R1, 2026-06-21) is GPU-APPLIED on the device
                            # from the resident raw + its stored spec via _gpu_apply_prewarp (no host-column
                            # H2D); the builder falls back to the host copy for any unported basis. gate_med /
                            # hermite-poly remain host-copied (raw_vals=None) -> not yet ported.
                            _is_plain = tr_name != _PREWARP_UNARY and tr_name != _GATE_MED_UNARY and "poly_" not in tr_name
                            _payload = None
                            _raw_for_spec = vals if _is_plain else None
                            if tr_name == _PREWARP_UNARY:
                                _spec = _prewarp_spec_by_var.get(var)
                                if _spec is not None:
                                    _payload = {"kind": "prewarp", "spec": _spec}
                                    _raw_for_spec = vals
                            _operand_col_specs.append((i, _raw_for_spec, tr_name, _payload))
                        i += 1

    # GPU-RESIDENT OPERAND TABLE (phase 1, gated): now that ``transformed_vars`` + ``vars_transformations``
    # are fully built, produce a DEVICE mirror whose bulk plain-unary columns are rebuilt ON the GPU (from
    # the resident raw inputs via _unary_apply) and the fitted/special columns copied from the host, then
    # register it so ``_resident_operand_table`` returns the device array WITH NO H2D (the materialise
    # consumes it transfer-free). The host ``transformed_vars`` is unchanged (the CPU pair-search / discretize
    # readers still use it). Any failure -> skip (the materialise H2Ds the host table as before; never a
    # correctness or availability regression). Any allocated tail columns past the used width (``i`` <
    # n_operands when some (var,tr) raised + were skipped) have no spec -> the builder zero-fills them; the
    # materialise never reads them (operand indices are always < the used width), so their content is moot.
    if _operand_col_specs is not None and len(vars_transformations) > 0:
        try:
            from .._gpu_resident_fe import build_resident_operand_table, register_prebuilt_operand_table  # type: ignore[attr-defined]  # dynamically re-exported via globals()
            # Build a FULL-WIDTH (n, n_operands) device mirror keyed on the SAME ``transformed_vars`` object
            # the materialise / _resolve_col paths pass: GPU-build the plain-unary columns from col_specs,
            # copy every other column (incl. any unused tail) from the host. Registered against the full
            # array so ``_resident_operand_table`` matches it by identity + shape.
            _dev_tv, _n_gpu, _n_cpu = build_resident_operand_table(transformed_vars, _operand_col_specs)
            register_prebuilt_operand_table(transformed_vars, _dev_tv)
            if verbose:
                logger.info(
                    "check_prospective_fe_pairs: GPU-resident operand table built " "(%d GPU-built columns, %d host-copied; materialise H2D skipped).",
                    _n_gpu,
                    _n_cpu,
                )
        except Exception:
            logger.debug("GPU-resident operand-table build failed; falling back to host H2D.", exc_info=True)
    return vars_transformations
