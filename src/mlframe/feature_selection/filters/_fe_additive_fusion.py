"""C2 ADDITIVE-FUSION of two unfused, additively-separable engineered halves.

When the FE pair search constructs the two additively-separable halves of a compound
target ``y = f(group_1) + g(group_2) + noise`` -- e.g. ``y = a**2/b + log(c)*sin(d)``
materialised as the SEPARATE engineered features ``div(neg(b),a__p2sin1)`` (the {a,b}
half) and ``mul(log(c),sin(d))`` (the {c,d} half) -- but does NOT fuse them into the
single ``add(...)`` compound, the two fragments survive side by side. The conditional-MI
redundancy gates cannot collapse them (each fragment carries a PRIVATE additive term the
OTHER does not span, so both keep a large CMI) and the downstream model never sees the
fused feature. This is the FUSION-blocked failure mode of the distribution-robustness goal
(``test_f2_single_compound_across_distributions`` heavy_tailed / mixed).

The fix reuses the EXISTING ``unary_binary`` recipe machinery -- NO new recipe kind. A
fused candidate ``add(half_a, half_b)`` (or ``sub`` -- see SIGN-AWARE ALIGNMENT below) is built with
``binary_name in {'add','sub'}``, ``unary_names=('identity','identity')`` and ``nested_parent_a/b`` set to
the two halves' own ``EngineeredRecipe`` objects, so it replays byte-exactly by recursively replaying the
parents (``_recipe_unary_binary.py``). SIGN-AWARE ALIGNMENT: since each half is chosen by SIGN-INVARIANT MI,
a half may arrive sign-flipped; the builder scores BOTH ``ha+hb`` and ``ha-hb`` against y and keeps the
better-aligned one (``binary_name`` records which), so a destructive sum is never materialised. The candidate
is proposed ONLY when two surviving
engineered features (or one engineered + one raw operand) have:
  * DISJOINT raw-token sets (no shared raw operand -- they cover different signal groups);
  * each half RELEVANT (its binned MI clears a marginal-permutation null floor);
  * GENUINE additive separability: the fused ``add`` MI strictly exceeds BOTH halves'
    MI by a margin (so an unrelated pair whose sum carries no joint uplift is never fused).
When admitted, the fused compound is materialised exactly like an escalation survivor and
the two now-subsumed fragment columns are dropped from ``selected_vars`` / the recipe dict
(the fused compound carries both their additive terms, so each fragment's conditional
excess given the fused feature collapses -- the redundancy logic the S5 gate would apply,
realised structurally here because the fragments pre-date the fusion candidate).

Default-ON (``fe_additive_fusion_enable``); self-gates to a no-op when fewer than two
relevant disjoint engineered halves are present (the common case), so it is byte-identical
on every target that does not exhibit the unfused-additive-halves pattern.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Split an engineered name into identifier tokens (``div(neg(b),a__p2sin1)`` ->
# {div, neg, b, a__p2sin1}); the raw operands are the tokens that are raw columns OR
# whose ``base__warp`` prefix is a raw column (the warped-token form the orth-FE /
# prewarp passes emit, e.g. ``a__p2sin1`` -> base ``a``). Mirrors the token recovery in
# ``_fe_raw_redundancy_drop._TOKEN_SPLIT`` so the fusion's raw-coverage agrees with the
# downstream raw-redundancy verdict.
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9_]+")


def _multiple_r(cols_2d: np.ndarray, y: np.ndarray) -> float:
    """Scale/sign-invariant multiple correlation R of an OLS fit ``y ~ cols + intercept``.

    Returns ``corr(fitted, y)`` (the multiple-R), in [0, 1]. OLS absorbs per-column scaling
    and sign, so this is invariant to the operand distributions' scale/sign -- exactly the
    invariance the binned-MI separability gate LACKS under variance imbalance (where the
    dominant half saturates the coarse bins and the weak half's genuine additive lift shows
    almost no binned-MI margin, even though it materially improves the LINEAR usability of
    the compound). NaN-safe; returns 0.0 on a degenerate (constant-y / singular) fit.
    """
    X2 = np.asarray(cols_2d, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64).ravel()
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)
    X2 = np.nan_to_num(X2, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    yv = np.nan_to_num(yv, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if yv.shape[0] < 3 or float(np.std(yv)) <= 1e-12:
        return 0.0
    design = np.column_stack([X2, np.ones(X2.shape[0], dtype=np.float64)])
    try:
        beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    except Exception:
        return 0.0
    fitted = design @ beta
    sf = float(np.std(fitted))
    if sf <= 1e-12:
        return 0.0
    r = float(np.corrcoef(fitted, yv)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return abs(r)


def _bare_tokens(name: str, raw_name_set: set) -> set:
    """Raw-column operands referenced by an engineered (or raw) name -- including warped
    ``base__suffix`` tokens mapped back to their raw base."""
    out: set = set()
    if name in raw_name_set:
        out.add(name)
    for t in _TOKEN_SPLIT.split(name or ""):
        if not t:
            continue
        if t in raw_name_set:
            out.add(t)
        elif "__" in t and t.split("__", 1)[0] in raw_name_set:
            out.add(t.split("__", 1)[0])
    return out


def propose_additive_fusions(
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
    """Propose ``add(half_a, half_b)`` fusions of disjoint, additively-separable engineered halves.

    Returns ``(admitted, subsumed_names)`` where ``admitted`` is a list of
    ``{"name", "values", "recipe"}`` dicts (each a fused compound to materialise exactly
    like an escalation survivor) and ``subsumed_names`` is the set of fragment engineered
    names the caller must drop from ``selected_vars`` / ``engineered_recipes`` (they are
    now carried by the fused compound). Pure / no live state captured (picklable fit).
    """
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _cmi_gpu_enabled, _quantile_bin
    from ._fe_cmi_redundancy_gate import _conditional_perm_null
    from .engineered_recipes import build_unary_binary_recipe

    # GPU-RESIDENT dispatch (residency contract, not a wall win): under the resident flag
    # (MLFRAME_FE_GPU_STRICT + MLFRAME_FE_GPU_STRICT_RESIDENT) keep the candidate-half values + the target
    # resident on the device and run the per-candidate binning, relevance MIs, and each fused-pair sum/bin/MI/
    # OLS-R resident. Selection-equivalent to (NOT byte-identical) this CPU path. Any cupy/device/import error
    # falls through to the CPU body below, so the default (flag-off) path is byte-identical and a GPU fault never
    # breaks a fit. This twin is EXPECTED slower on the small-n / sequential-pair-scan HW -- a PASS by the contract.
    try:
        from ._gpu_strict_fe._entry import fe_gpu_strict_resident_enabled as _fusion_resident_flag_on  # type: ignore
    except Exception:
        _fusion_resident_flag_on = None  # type: ignore
    if _fusion_resident_flag_on is not None and _fusion_resident_flag_on():
        # Import stays broad-guarded (cupy/twin may be absent); the CALL is narrowed to genuine
        # device/linalg faults so a real twin logic/shape bug (ValueError/KeyError/IndexError)
        # propagates to tests instead of silently degrading to CPU as a "device fallback".
        try:
            from ._fe_additive_fusion_gpu_resident import propose_additive_fusions_gpu
            _twin_ready = True
        except Exception:
            _twin_ready = False
        if _twin_ready:
            _dev_errs = []
            try:
                _dev_errs.append(np.linalg.LinAlgError)
                import cupy as _cp  # type: ignore
                _dev_errs.append(_cp.cuda.runtime.CUDARuntimeError)
                _dev_errs.append(_cp.cuda.memory.OutOfMemoryError)
                # FIX4 (2026-06-28): cuSOLVER/cuBLAS faults from cp.linalg.solve/lstsq subclass plain
                # RuntimeError, NOT CUDARuntimeError -> omitting them would crash instead of falling
                # back. getattr so an absent symbol can't break the tuple builder.
                from cupy_backends.cuda.libs import cusolver as _cusolver  # type: ignore
                _dev_errs.append(getattr(_cusolver, "CUSOLVERError", None))
                from cupy_backends.cuda.libs import cublas as _cublas  # type: ignore
                _dev_errs.append(getattr(_cublas, "CUBLASError", None))
            except Exception:
                pass
            _dev_errs = [e for e in _dev_errs if isinstance(e, type) and issubclass(e, BaseException)]
            try:
                return propose_additive_fusions_gpu(
                    self,
                    engineered_recipes=engineered_recipes,
                    engineered_continuous=engineered_continuous,
                    newly_engineered_names=newly_engineered_names,
                    raw_name_set=raw_name_set,
                    cols=cols,
                    classes_y=classes_y,
                    X=X,
                    nbins=nbins,
                    seed=seed,
                    verbose=verbose,
                )
            except tuple(_dev_errs):
                pass  # genuine cupy/device/linalg fault -> CPU path (byte-identical default); logic bugs propagate

    if not engineered_recipes or not newly_engineered_names:
        return [], set(), set()

    # Additive-separability margin multiple (getattr keeps the signature stable). The fused
    # ``add`` MI must exceed the STRONGER half by MORE than this multiple of that half's
    # marginal-permutation floor -- a chance-fluctuation scale, so an unrelated pair whose
    # sum carries no genuine joint uplift (fused MI ~= the stronger half +/- noise) is NOT
    # fused, while a genuine second additive term (the weak half DOES lift the joint MI
    # above the chance scale) is. This separates real additive separability from a spurious
    # sum far more robustly than a flat MI-ratio bar -- the weak half's contribution to the
    # COARSELY-BINNED joint MI is small in absolute terms (the dominant half saturates most
    # bins) yet still well above the per-half chance floor.
    _floor_margin = float(getattr(self, "fe_additive_fusion_floor_margin", 1.0))
    _max_fusions = int(getattr(self, "fe_additive_fusion_max", 4))

    # y codes (dense int) the MI primitives score against.
    _y = np.asarray(classes_y).ravel()
    if not np.issubdtype(_y.dtype, np.integer):
        _y = _y.astype(np.int64)
    _, y_dense = np.unique(_y, return_inverse=True)
    y_dense = y_dense.astype(np.int64)
    n_rows = y_dense.shape[0]

    # Candidate halves: engineered features just admitted that have BOTH a replayable
    # recipe AND continuous values on hand (the fused operand is built on continuous
    # values, exactly as the nested-parent replay reconstructs them at transform time).
    halves: list[dict] = []
    # Pre-pass: collect the candidate halves (cheap host filters + binning), then score their RELEVANCE MI
    # in ONE batched_cmi_gpu workload (y_dense fixed, marginal) instead of a per-half _cmi_from_binned. Same
    # MM plug-in marginal MI -> selection-equivalent; the per-half permutation floor is unchanged.
    _pre: list[tuple] = []
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
        vb = _quantile_bin(vals, nbins=int(nbins))
        _pre.append((nm, rec, vals, toks, vb))
    _mis = None
    try:
        if _cmi_gpu_enabled() and len(_pre) > 1:
            from ._fe_batched_mi import batched_cmi_gpu

            _Xh = np.empty((int(n_rows), len(_pre)), dtype=np.int64)
            for _j, _t in enumerate(_pre):
                _Xh[:, _j] = _t[4]
            _mis = np.asarray(batched_cmi_gpu(_Xh, y_dense, None), dtype=np.float64)
    except Exception:
        _mis = None
    for _j, (nm, rec, vals, toks, vb) in enumerate(_pre):
        mi = float(_mis[_j]) if _mis is not None else float(_cmi_from_binned(vb, y_dense, None))
        floor, _ = _conditional_perm_null(vb, y_dense, None, seed=seed)
        if mi <= floor:
            continue  # not relevant -- never a fusion half
        halves.append({"name": nm, "recipe": rec, "vals": vals, "tokens": toks,
                       "mi": mi, "floor": float(floor), "binned": vb})

    if len(halves) < 2:
        return [], set(), set()

    # Strongest half first so the most-informative disjoint pair is fused before the cap.
    halves.sort(key=lambda h: h["mi"], reverse=True)

    admitted: list[dict] = []
    subsumed: set = set()
    subsumed_raws: set = set()
    used: set = set()  # half names already consumed by an admitted fusion
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
            # DISJOINT raw-token sets -- the two halves cover DIFFERENT signal groups.
            if ha["tokens"] & hb["tokens"]:
                continue
            # SIGN-AWARE ALIGNMENT (2026-07-01). The a/b (and c/d) pair search picks each half by SIGN-INVARIANT
            # MI, so a half can arrive sign-flipped -- e.g. ``div(sqr(a),neg(b))`` = -a**2/b won the a/b pair
            # although the target carries +a**2/b. A blind ``ha + hb`` is then DESTRUCTIVE toward y: on F2
            # ``add(-a**2/b, log(c)sin(d))`` measures |corr(.,y)|=0.03 (garbage) while ``sub`` = -(a**2/b +
            # log(c)sin(d)) = -y measures 0.998. The destructive sum still clears the binned-MI/OLS gate and gets
            # selected, then the retention pass correctly re-attaches the real halves to fix the sign -> the
            # single-step fragmentation regression. Build BOTH the ``add`` and ``sub`` alignment, bin each, and
            # keep the one whose binned MI with y is higher -- the SUM's alignment is NOT sign-invariant even
            # though each half's marginal MI is. Prefer ``add`` unless ``sub`` beats it by more than the stronger
            # half's marginal-permutation floor (the SAME chance-fluctuation scale the admission razor uses
            # below), so a pair already best-aligned as ``add`` stays byte-identical (no spurious binned-MI-noise
            # flips). OLS multiple-R is NOT usable to pick the sign -- it fits FREE per-column coefficients, so
            # ``R(ha,hb) == R(ha,-hb)`` exactly (same column space); it gates admission only, never the sign.
            _strong = ha if ha["mi"] >= hb["mi"] else hb
            _fv_add = np.nan_to_num(ha["vals"] + hb["vals"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            _fv_sub = np.nan_to_num(ha["vals"] - hb["vals"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            _fvb_add = _quantile_bin(_fv_add, nbins=int(nbins))
            _fvb_sub = _quantile_bin(_fv_sub, nbins=int(nbins))
            _mi_add = float(_cmi_from_binned(_fvb_add, y_dense, None))
            _mi_sub = float(_cmi_from_binned(_fvb_sub, y_dense, None))
            if _mi_sub > _mi_add + _floor_margin * _strong["floor"]:
                _binary, fused_vals, fvb, fused_mi = "sub", _fv_sub, _fvb_sub, _mi_sub
            else:
                _binary, fused_vals, fvb, fused_mi = "add", _fv_add, _fvb_add, _mi_add
            # GENUINE ADDITIVE SEPARABILITY: the chosen-alignment fused feature must carry strictly MORE target
            # information than the STRONGER half by more than that half's marginal-perm floor (a chance-
            # fluctuation scale). Two unrelated features whose combination has no genuine joint uplift
            # (fused_mi ~= the stronger half +/- noise) are NOT fused.
            _binned_mi_passes = fused_mi > _strong["mi"] + _floor_margin * _strong["floor"]
            # OLS LINEAR-USABILITY SEPARABILITY (2026-06-24, default-ON, ALONGSIDE the binned-MI
            # gate). Under variance imbalance the coarse binned-MI WRONGLY rejects a genuine
            # fusion: the dominant half saturates most bins, so the fused binned-MI barely exceeds
            # the strong half's even though the second additive term materially improves the
            # compound's LINEAR usability for the downstream model (measured on F2 scaled_1_5:
            # fused binned-MI 0.67 < strong-half 1.06, yet 2-half OLS multiple-R 0.998 > strong
            # half's 0.976). Admit the fusion if EITHER the binned-MI margin OR the OLS-R
            # separability passes. The OLS-R path fits ``y ~ [half_a, half_b] + intercept`` and
            # compares its multiple-R to the best SINGLE-half R (y ~ one half alone); the second
            # half must lift R by a small but real margin (chance-fluctuation scale ~ 1/sqrt(n)),
            # so two unrelated halves whose join adds no linear usability are NOT fused. OLS is
            # scale/sign-invariant, so it is robust to the operand-distribution imbalance the
            # binned-MI path is not.
            _ols_passes = False
            if not _binned_mi_passes:
                _yc = y_dense.astype(np.float64)
                _r_a = _multiple_r(ha["vals"], _yc)
                _r_b = _multiple_r(hb["vals"], _yc)
                _r_fused = _multiple_r(np.column_stack([ha["vals"], hb["vals"]]), _yc)
                _r_best_single = max(_r_a, _r_b)
                # The 2-half fit must beat the best single half by more than a chance-fluctuation
                # margin (~ 1/sqrt(n), the SE scale of a correlation). A genuine second additive
                # term clears it; a noise half does not move multiple-R.
                _r_margin = float(getattr(self, "fe_additive_fusion_ols_r_margin_sd", 2.0)) / max(np.sqrt(float(n_rows)), 1.0)
                _ols_passes = _r_fused > _r_best_single + _r_margin
            if not (_binned_mi_passes or _ols_passes):
                continue
            # Build the fused recipe via the EXISTING unary_binary + nested-parent machinery. The display name
            # and the recipe ``binary_name`` both record the CHOSEN alignment (``add``/``sub``); ``sub`` replays
            # byte-exactly through the identical field-driven machinery (binary_funcs[binary_name]).
            name = f"{_binary}({ha['name']},{hb['name']})"
            base = name
            k = 2
            while name in existing_names:
                name = f"{base}_{k}"
                k += 1
            recipe = build_unary_binary_recipe(
                name=name,
                src_a_name=ha["name"], src_b_name=hb["name"],
                unary_a_name="identity", unary_b_name="identity",
                binary_name=_binary,
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
            # The two fragments are now subsumed by the fused compound (it carries BOTH
            # additive terms): drop them from selection / the recipe dict.
            subsumed.add(ha["name"])
            subsumed.add(hb["name"])
            used.add(ha["name"])
            used.add(hb["name"])
            # RAW-OPERAND SUBSUMPTION (2026-06-24). A raw operand of either half that the
            # fused compound now FULLY captures must drop too -- otherwise it lingers as a
            # redundant single-group fragment (raw ``a`` beside ``add(a**2/b, log(c)sin(d))``).
            # Reuse the production keep-probe (``raw_retains_signal_given_genuine_children``):
            # condition the raw on the fused compound's bin; if it retains NO significant
            # independent residual, it is subsumed -> drop. A raw that carries a genuine
            # PRIVATE term the compound does not span keeps its residual and is KEPT. This is
            # the same n-invariant verdict the post-fit raw-redundancy sweep applies, realised
            # here (the sweep is gated on ``redundancy_policy='drop'``, but a fused-compound's
            # operand is unconditionally redundant once the compound replaces the fragments).
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
                _rvb = _quantile_bin(np.nan_to_num(_rv, nan=0.0, posinf=0.0, neginf=0.0), nbins=int(nbins))
                _retains = raw_retains_signal_given_genuine_children(
                    raw_bin=_rvb, y_bin=y_dense, genuine_child_bins=[fvb], seed=seed,
                )
                if not _retains:
                    subsumed_raws.add(_rn)
            if verbose:
                logger.info(
                    "MRMR FE additive-fusion: fused %r (mi=%.4f) + %r (mi=%.4f) -> %r "
                    "(mi=%.4f, margin x%.2f); dropping the two subsumed fragments.",
                    ha["name"], ha["mi"], hb["name"], hb["mi"], name, fused_mi, _floor_margin,
                )
            break  # ha consumed; move to the next un-used half

    return admitted, subsumed, subsumed_raws
