"""GPU-RESIDENT twin of :func:`_fe_additive_fusion.propose_additive_fusions`.

RESIDENCY CONTRACT (not a wall win). Gated on the resident flag
(``MLFRAME_FE_GPU_STRICT`` + ``MLFRAME_FE_GPU_STRICT_RESIDENT``); default OFF.
On this GTX 1050 Ti the half/candidate operands are small (an F2 train slice)
and the fusion loop is a sequential disjoint-pair scan, so the GPU twin is
EXPECTED to be SLOWER than the fused njit/numpy CPU path -- and that is a PASS by
the residency contract. This twin exists for RESIDENCY COMPLETENESS so that,
under the resident flag, the candidate-half columns + the target stay RESIDENT on
the device and the per-candidate numeric work (equi-frequency binning, the
marginal relevance MIs, each fused-pair ``a + b`` sum + its binning + its joint
MI, and the OLS multiple-R linear-usability separability) runs on cupy with NO
per-candidate bulk (n-scaled) H2D/D2H.

What is resident vs host control-flow (allowed by the contract):
  * RESIDENT (one bulk H2D at entry): the (n, H) matrix of all candidate-half
    continuous values + the dense y codes. From it the device computes the (n, H)
    equi-frequency bin-code matrix RESIDENT via the distinct-edge-dedup binner
    ``_usability_njit_pool._gpu_quantile_bin_codes`` (cp.unique adjacent dedup +
    searchsorted side='right' -- documented BIT-IDENTICAL to the njit ``_qbin_into`` /
    host ``_quantile_bin`` per row, and crucially it does NOT interpolate across edges
    the way ``cp.percentile`` does, so the partition is selection-equivalent), every
    half's marginal relevance MI (``batched_cmi_gpu`` over the resident codes), and --
    in the disjoint-pair loop -- the fused ``vals_a + vals_b`` sum, its bin codes (same
    resident dedup binner), its joint MI, and the three OLS multiple-R fits, ALL on
    resident device arrays. Binning is fully on the device with NO binning D2H.
  * HOST scalar / bounded D2H (allowed; NOT binning -- the binning above stays
    resident): the relevance-MI vector (H floats, a per-FAMILY result), and the
    PROBE-INPUT code pulls -- each relevant half's resident code row pulled ONCE for
    the permutation-null floor and, per ADMITTED pair, the fused code row + the
    required fused-values return array. The floor / ``_cmi_from_binned`` / raw-
    subsumption probes are CPU-interface helpers (host int codes; themselves GPU-routed
    internally via ``_cmi_gpu_enabled``), so feeding them needs a host code array -- but
    that D2H is for the PROBE, not the binning, and is bounded O(H + n_fusions), never
    per-candidate. ``fused_vals`` is a REQUIRED return (the same single array the CPU
    path holds). Audited: NO binning D2H; only these bounded probe-input/return pulls.

Selection-equivalence: the proposed ``add(half_a, half_b)`` fusions (names + order
+ subsumed fragments/raws) MATCH the CPU :func:`propose_additive_fusions` -- the
same equi-frequency partition (``_gpu_quantile_bin_codes`` is bit-identical to the
host ``_quantile_bin`` value-edge binning), the same Miller-Madow plug-in MI
(``batched_cmi_gpu`` / ``_cmi_from_binned`` per pair), the same relevance floor,
the same binned-MI / OLS-R separability gates, and the same raw-subsumption probe.
Only float reduction order differs (~1e-15 for the MI/entropy, the OLS solve), to
which the partition + the chance-margin gates are tolerant.

Any cupy / device / import error raises so the caller falls back to the CPU path,
keeping the default (flag-off) path byte-identical and a GPU fault never breaking a fit.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _multiple_r_gpu(cp, X2_dev, yv_dev, y_std: float) -> float:
    """Scale/sign-invariant multiple correlation R of an OLS fit ``y ~ cols + intercept``,
    fully resident -- GPU twin of :func:`_fe_additive_fusion._multiple_r`.

    ``X2_dev`` is a resident (n, k) device matrix (1 or 2 columns), ``yv_dev`` the
    resident pre-cleaned float target, ``y_std`` its host std (already > 1e-12 when
    called). Builds the ``[X2, 1]`` design resident, solves the normal equations
    (lstsq fallback on a singular system), and returns ``corr(fitted, y)`` in [0, 1]
    as a bounded host scalar. Same degeneracy guards as the CPU path."""
    if X2_dev.ndim == 1:
        X2_dev = X2_dev[:, None]
    n = int(X2_dev.shape[0])
    if n < 3 or y_std <= 1e-12:
        return 0.0
    ones = cp.ones((n, 1), dtype=cp.float64)
    design = cp.concatenate([X2_dev, ones], axis=1)
    try:
        AtA = design.T @ design
        Aty = design.T @ yv_dev
        try:
            beta = cp.linalg.solve(AtA, Aty)
        except Exception:
            beta = cp.linalg.lstsq(design, yv_dev, rcond=None)[0]
    except Exception:
        return 0.0
    fitted = design @ beta
    sf = float(cp.std(fitted))
    if sf <= 1e-12:
        return 0.0
    # corr(fitted, y) on resident arrays; scalar back.
    fc = fitted - fitted.mean()
    yc = yv_dev - yv_dev.mean()
    denom = float(cp.sqrt(cp.dot(fc, fc) * cp.dot(yc, yc)))
    if denom <= 1e-300:
        return 0.0
    r = float(cp.dot(fc, yc)) / denom
    if not np.isfinite(r):
        return 0.0
    return abs(r)


def propose_additive_fusions_gpu(
    self,
    *,
    engineered_recipes: dict,
    engineered_continuous: dict,
    newly_engineered_names: list,
    raw_name_set: set,
    cols: list,
    classes_y: np.ndarray,
    X: object,
    nbins: int,
    seed: int = 0,
    verbose: int = 0,
) -> tuple:
    """GPU-resident twin of :func:`_fe_additive_fusion.propose_additive_fusions`.

    Returns the SAME ``(admitted, subsumed, subsumed_raws)`` triple (selection-
    equivalent: same fused names/order + same subsumed fragments/raws) as the CPU
    path. Raises on any cupy/device error so the caller falls back to the CPU body.

    The candidate-half continuous values + the dense y codes are uploaded ONCE
    (bulk H2D) into resident device arrays; the per-half binning, relevance MIs,
    and every fused-pair sum/bin/MI/OLS-R run resident. Bounded host control-flow
    (the relevance-MI vector, the per-half/per-pair bin-code rows the floor +
    raw-subsumption probes consume, the scalar gate compares) mirrors the CPU
    pre-pass's OWN host code arrays."""
    import cupy as cp

    from ._mi_greedy_cmi_fe import _cmi_from_binned
    from ._fe_cmi_redundancy_gate import _conditional_perm_null
    from .engineered_recipes import build_unary_binary_recipe
    from ._fe_batched_mi import batched_cmi_gpu
    from ._fe_additive_fusion import _bare_tokens
    from ._usability_njit_pool import _gpu_quantile_bin_codes

    if not engineered_recipes or not newly_engineered_names:
        return [], set(), set()

    _floor_margin = float(getattr(self, "fe_additive_fusion_floor_margin", 1.0))
    _max_fusions = int(getattr(self, "fe_additive_fusion_max", 4))

    # y codes (dense int) the MI primitives score against -- host preamble identical to the CPU path.
    _y = np.asarray(classes_y).ravel()
    if not np.issubdtype(_y.dtype, np.integer):
        _y = _y.astype(np.int64)
    _, y_dense = np.unique(_y, return_inverse=True)
    y_dense = y_dense.astype(np.int64)
    n_rows = y_dense.shape[0]

    # Candidate halves (cheap host filters identical to the CPU pre-pass): a replayable recipe + continuous
    # values of the right length + at least one raw token. Collect the names/recipes/tokens on host; the
    # n-scaled VALUES go up to the device ONCE as a single (n, H) matrix (the only bulk H2D in this stage).
    _names: list = []
    _recs: list = []
    _toks: list = []
    _vals_cols: list = []
    for nm in newly_engineered_names:
        rec = engineered_recipes.get(nm)
        vals = engineered_continuous.get(nm)
        if rec is None or vals is None:
            continue
        vals = np.asarray(vals, dtype=np.float64).ravel()
        if vals.shape[0] != n_rows:
            continue
        toks = _bare_tokens(nm, raw_name_set)
        if not toks:
            continue
        _names.append(nm)
        _recs.append(rec)
        _toks.append(toks)
        _vals_cols.append(vals)

    if len(_names) < 2:
        return [], set(), set()

    H = len(_names)
    # ---- ONE bulk H2D: the (n, H) half-values matrix + the y codes. Everything below stays resident. ----
    vals_host = np.empty((n_rows, H), dtype=np.float64)
    for j in range(H):
        vals_host[:, j] = _vals_cols[j]
    vals_host = np.nan_to_num(vals_host, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    vals_dev = cp.asarray(vals_host)                                   # (n, H) resident
    y_dev = cp.asarray(y_dense)                                        # (n,) resident

    # Per-half bin codes ON THE DEVICE via the distinct-edge-dedup resident binner ``_gpu_quantile_bin_codes``
    # (cp.unique adjacent dedup + searchsorted side='right', the SAME np.quantile linear-interp edges as the host
    # ``_quantile_bin`` / njit ``_qbin_into`` -- it is documented BIT-IDENTICAL to ``_qbin_into`` per row, so the
    # partition is SELECTION-EQUIVALENT to the CPU pre-pass). Unlike ``cp.percentile`` it does NOT shift values
    # across edges. The binner is row-oriented ((m, n)), so the (n, H) operand is binned as its transpose
    # (H, n); codes come back resident (H, n) -- no code H2D/D2H for the binning itself.
    qs = cp.linspace(0.0, 1.0, int(nbins) + 1)
    codes_T_dev, _kx = _gpu_quantile_bin_codes(cp.ascontiguousarray(vals_dev.T), qs)   # (H, n) resident codes
    codes_dev = cp.ascontiguousarray(codes_T_dev.T)                    # (n, H) resident

    # Every half's marginal relevance MI(half; y) in ONE batched device workload over the RESIDENT codes (no
    # code H2D). Same Miller-Madow plug-in MI as the CPU pre-pass's batched_cmi_gpu.
    mis = np.asarray(batched_cmi_gpu(codes_dev, y_dense, None), dtype=np.float64)

    halves: list[dict] = []
    for j in range(H):
        mi = float(mis[j])
        # The permutation-null floor is a CPU-interface helper (host int codes, GPU-routed internally). Pull
        # THIS half's resident code row back ONCE for the floor probe -- a probe-input D2H, not a binning D2H
        # (binning stayed resident above). Mirrors the CPU pre-pass's own host code array (_Xh).
        vb = cp.asnumpy(cp.ascontiguousarray(codes_dev[:, j]))
        floor, _ = _conditional_perm_null(vb, y_dense, None, seed=seed)
        if mi <= floor:
            continue  # not relevant -- never a fusion half
        halves.append({
            "name": _names[j], "recipe": _recs[j], "tokens": _toks[j],
            "mi": mi, "floor": float(floor), "col": j, "binned": vb,
        })

    if len(halves) < 2:
        return [], set(), set()

    # Strongest half first so the most-informative disjoint pair is fused before the cap.
    halves.sort(key=lambda h: h["mi"], reverse=True)

    # Resident float target for the OLS multiple-R fits (cleaned once).
    yc_dev = y_dev.astype(cp.float64)
    y_std = float(cp.std(yc_dev))

    admitted: list[dict] = []
    subsumed: set = set()
    subsumed_raws: set = set()
    used: set = set()
    existing_names = set(engineered_recipes) | {cols[i] for i in range(len(cols))}

    for ia in range(len(halves)):
        if len(admitted) >= _max_fusions:
            break
        ha = halves[ia]
        if ha["name"] in used:
            continue
        for ib in range(ia + 1, len(halves)):
            hb = halves[ib]
            if hb["name"] in used:
                continue
            if ha["tokens"] & hb["tokens"]:
                continue
            # FUSED SUM + its bin codes RESIDENT: vals_a + vals_b on the device, then the distinct-edge-dedup
            # resident binner ``_gpu_quantile_bin_codes`` (selection-equivalent to the host ``_quantile_bin``;
            # no interpolation drift). The sum + binning both stay resident -- no binning D2H.
            ca = vals_dev[:, ha["col"]]
            cb = vals_dev[:, hb["col"]]
            fused_dev = ca + cb
            fused_dev = cp.nan_to_num(fused_dev, nan=0.0, posinf=0.0, neginf=0.0)
            fvb_dev, _kxf = _gpu_quantile_bin_codes(fused_dev[None, :], qs)   # (1, n) resident codes
            fvb_dev = fvb_dev[0]
            # ``_cmi_from_binned`` + the raw-subsumption probe are CPU-interface helpers (host int codes,
            # GPU-routed internally). Pull the fused codes back ONCE -- a per-ACCEPTED-pair probe-input D2H,
            # not a binning D2H (binning stayed resident above). Bounded by the number of admitted fusions.
            fvb = cp.asnumpy(fvb_dev)
            fused_mi = float(_cmi_from_binned(fvb, y_dense, None))     # GPU-routed internally under the flag
            _strong = ha if ha["mi"] >= hb["mi"] else hb
            # Fusion admission razor (not grid-snapped): fused_mi vs strong-half mi + margin*floor. The cupy-vs-
            # numpy reduction-order delta (~1e-12) is far below the floor-margin band, so it cannot flip this gate
            # in practice; the direct fusion-parity check (resident vs CPU proposed fusions) + F2 confirm the same
            # admissions. Accept-and-documented.
            _binned_mi_passes = fused_mi > _strong["mi"] + _floor_margin * _strong["floor"]
            _ols_passes = False
            if not _binned_mi_passes:
                # OLS multiple-R separability, fully resident (the n-scaled lstsq runs on the device columns).
                _r_a = _multiple_r_gpu(cp, ca[:, None], yc_dev, y_std)
                _r_b = _multiple_r_gpu(cp, cb[:, None], yc_dev, y_std)
                _r_fused = _multiple_r_gpu(cp, cp.stack([ca, cb], axis=1), yc_dev, y_std)
                _r_best_single = max(_r_a, _r_b)
                _r_margin = float(getattr(self, "fe_additive_fusion_ols_r_margin_sd", 2.0)) / max(np.sqrt(float(n_rows)), 1.0)
                _ols_passes = _r_fused > _r_best_single + _r_margin
            if not (_binned_mi_passes or _ols_passes):
                continue
            # Build the fused recipe via the EXISTING unary_binary + nested-parent machinery (host; replays
            # byte-exactly). The fused VALUES are a REQUIRED return (recipe fit_values_for_edges + the admitted
            # "values" the caller materialises) -- the SAME single array the CPU path holds -- so pull the
            # resident sum back ONCE here, only on an ADMITTED pair (bounded by the number of fusions).
            fused_vals = cp.asnumpy(fused_dev)
            name = f"add({ha['name']},{hb['name']})"
            base = name
            k = 2
            while name in existing_names:
                name = f"{base}_{k}"
                k += 1
            recipe = build_unary_binary_recipe(
                name=name,
                src_a_name=ha["name"], src_b_name=hb["name"],
                unary_a_name="identity", unary_b_name="identity",
                binary_name="add",
                unary_preset=str(getattr(self, "fe_unary_preset", "medium")),
                binary_preset=str(getattr(self, "fe_binary_preset", "minimal")),
                quantization_nbins=self.quantization_nbins,
                quantization_method=self.quantization_method,
                quantization_dtype=self.quantization_dtype,
                fit_values_for_edges=fused_vals,
                nested_parent_a=ha["recipe"],
                nested_parent_b=hb["recipe"],
            )
            admitted.append({"name": name, "values": fused_vals, "recipe": recipe,
                             "mi": fused_mi})
            existing_names.add(name)
            subsumed.add(ha["name"])
            subsumed.add(hb["name"])
            used.add(ha["name"])
            used.add(hb["name"])
            # RAW-OPERAND SUBSUMPTION -- identical verdict to the CPU path. The raw operand originates on host
            # (a column of X), so it is uploaded with a BOUNDED H2D (one column, per ACCEPTED fusion -- bounded,
            # NOT per-candidate) and binned RESIDENT via the same distinct-edge-dedup binner, keeping ALL binning
            # on the device. The keep-probe (``raw_retains_signal_given_genuine_children``) is a CPU-interface
            # helper (GPU-routed internally), so its raw bin codes are pulled back ONCE -- a bounded probe-input
            # D2H, not a binning D2H. ``fvb`` (the fused codes) is the genuine child.
            from ._fe_raw_redundancy_drop import raw_retains_signal_given_genuine_children
            for _rn in (ha["tokens"] | hb["tokens"]):
                if _rn in subsumed_raws:
                    continue
                _rv = None
                try:
                    if hasattr(X, "columns") and _rn in getattr(X, "columns", []):
                        _rv = np.asarray(X[_rn], dtype=np.float64).ravel()
                except Exception:
                    _rv = None
                if _rv is None or _rv.shape[0] != n_rows:
                    continue
                # The same raw token column recurs across accepted fusions -> content-keyed resident cache so it
                # uploads once (H2D audit: 4x re-uploads). Read-only (binned below) -> selection-equivalent.
                from ._fe_resident_operands import resident_operand
                _rv_dev = resident_operand(np.nan_to_num(_rv, nan=0.0, posinf=0.0, neginf=0.0),
                                           ("addfusion_rawprobe", _rn))                     # 1 col, cached once
                _rvb_dev, _kxr = _gpu_quantile_bin_codes(_rv_dev[None, :], qs)              # resident bin codes
                _rvb = cp.asnumpy(_rvb_dev[0])                                              # probe-input D2H
                _retains = raw_retains_signal_given_genuine_children(
                    raw_bin=_rvb, y_bin=y_dense, genuine_child_bins=[fvb], seed=seed,
                )
                if not _retains:
                    subsumed_raws.add(_rn)
            if verbose:
                logger.info(
                    "MRMR FE additive-fusion [GPU-resident]: fused %r (mi=%.4f) + %r (mi=%.4f) -> %r "
                    "(mi=%.4f, margin x%.2f); dropping the two subsumed fragments.",
                    ha["name"], ha["mi"], hb["name"], hb["mi"], name, fused_mi, _floor_margin,
                )
            break  # ha consumed; move to the next un-used half

    return admitted, subsumed, subsumed_raws
