"""AUTO-ESCALATION of the pair-FE search to the richer SHIPPED bases (2026-06-10, backlog idea B).

When a prospective pair PASSED the pair-MI prescreen (joint-MI ratio gate + order-2 maxT
floor) but the unary/binary operator search admitted NOTHING for it, the legacy behaviour
was a WARNING ("FE produced 0 engineered features despite N pair(s) passing the pair-MI
gate") -- the signal was DETECTED and then silently abandoned. This module escalates
instead: for each such pair it PROPOSES candidates from the two richer shipped basis
families and lets the EXISTING admission gates decide (escalation proposes, gates decide
-- the iron rule):

* SIGNAL-ADAPTIVE ORTHOGONAL-POLY pair warp: the rank-1 ALS per-operand warp
  (``hermite_fe.fit_pair_prewarp_als``) re-run at a HIGHER degree across the four shipped
  polynomial bases (chebyshev / hermite / legendre / laguerre), with the best basis
  selected by held-out rank-1-reconstruction |corr| on a deterministic stride slice --
  catches a poly inner the default degree-4 chebyshev prewarp under-fits or that the
  default prewarp's own held-out gate rejected at its fixed basis.

* ADAPTIVE-FREQUENCY FOURIER / CHIRP pair warp via DEMODULATION: for a multiplicative
  pair signal ``y ~ g(a) * b`` the univariate adaptive-Fourier detector sees nothing
  (``E[y | a] ~ 0`` when b is ~zero-mean); but the DEMODULATED target
  ``t = (y - mean(y)) * zscore(b)`` satisfies ``E[t | a] ~ g(a) * E[zscore(b)^2]``, so
  the SHIPPED held-out-validated multitone detector
  (``_orth_extra_basis_fe._detect_fourier_freqs_for_col``) run on ``(z01(a), t)`` locks
  g's frequency -- e.g. the ``sin(3.7*a)`` INNER frequency no library unary can express.
  The fitted sin/cos mix is stored as a closed-form ``fourier_adaptive`` prewarp spec
  replayed by ``hermite_fe.apply_operand_prewarp`` (a pure function of x -- leak-safe,
  no y at transform time). The CHIRP variant runs the same detector on the shipped
  quadratic-argument warp ``u = sign(z) * z**2`` so a growing-frequency inner is also
  reachable.

GATES (all existing -- escalation only PROPOSES):
  1. held-out validation floors inside the proposers (the shipped detector's >= 0.30
     held-out periodogram floor / the ALS stride-slice reconstruction-|corr| floor);
  2. the Miller-Madow-debiased candidate MI must clear the order-2 maxT permutation
     floor computed over the prospective-pair pool (the same floor that gated the pair);
  3. a marginal-permutation MI floor (``_fe_cmi_redundancy_gate._conditional_perm_null``);
  4. the S5 conditional-MI redundancy gate over {already-admitted engineered survivors}
     UNION {escalation candidates} -- a candidate redundant given the admitted support is
     dropped; verdicts are applied to ESCALATION candidates only (main-path admissions
     are never revoked here).
A pure-noise pair that slipped the prescreen by chance proposes nothing (the detectors
return no validated frequency, the ALS reconstruction fails the held-out floor) or is
rejected by floors 2-4 -- measured 0 admissions on pure-noise controls (see
``tests/feature_selection/test_fe_auto_escalation.py``).

COST: structurally a no-op when every prescreen-surviving pair produced an admitted
engineered column (the common case -- one set-difference). When it fires, the cost is a
handful of ``lstsq`` solves + one detector sweep per escalated pair, bounded by
``fe_escalation_max_pairs``.

Replay/persistence: every admitted candidate carries a standard ``unary_binary``
EngineeredRecipe with ``prewarp`` pseudo-unaries on both sides and the ``mul`` binary, so
``transform()`` replay, pickling and the cross-fold stability vote treat it exactly like
a default-prewarp pair feature. The candidate's fit-time values are computed through the
SAME ``apply_operand_prewarp`` + ``np.multiply`` + ``nan_to_num`` path the recipe replays,
so fit and transform are bit-identical on the same rows.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

__all__ = ["run_fe_auto_escalation"]

# Signal-adaptive poly basis routing: try all four shipped families, best by held-out
# reconstruction |corr|. Chebyshev first (the production prewarp default).
_ESCALATION_POLY_BASES = ("chebyshev", "hermite", "legendre", "laguerre")

# Coarse z-space frequency grids -- VERBATIM the shipped univariate adaptive grids
# (``_orth_extra_basis_fe``): linear axis 0.5..8.0, chirp axis 0.5..24.0.
_ADAPTIVE_F_GRID = tuple(0.5 * k for k in range(1, 17))
_CHIRP_F_GRID = tuple(0.5 * k for k in range(1, 49))

# Identity warp degree for the mate operand (coef [0, 1] -> the basis' affine z map).
_IDENTITY_BASIS = "chebyshev"


def _finite_filled(x: np.ndarray) -> np.ndarray:
    """Copy of ``x`` with non-finite entries replaced by the finite mean (0.0 when no
    finite entries). Used ONLY for detector / ALS FITTING; candidate VALUES are always
    computed from the RAW column through the replay path (warp -> mul -> nan_to_num),
    so fit-time and transform-time values agree bit-for-bit."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.all():
        return x
    fill = float(np.mean(x[finite])) if finite.any() else 0.0
    out = x.copy()
    out[~finite] = fill
    return out


def _identity_prewarp_spec(x: np.ndarray) -> dict | None:
    """Closed-form IDENTITY prewarp spec for the mate operand: chebyshev degree-1 with
    coef [0, 1] evaluates to the basis' affine z-map of x (replayed by
    ``apply_operand_prewarp`` exactly like any learned warp). An affine map of the mate
    keeps the product's MI/correlation structure intact while staying on the standard
    ``prewarp`` recipe path (no new pseudo-unary needed)."""
    from .hermite_fe import _POLY_BASES
    bi = _POLY_BASES[_IDENTITY_BASIS]
    xf = _finite_filled(x)
    if float(np.std(xf)) < 1e-12:
        return None
    _, params = bi["fit"](xf)
    coef = np.zeros(2, dtype=np.float64)
    coef[1] = 1.0
    return {"basis": _IDENTITY_BASIS, "degree": 1, "coef": coef, "preprocess": dict(params)}


def _candidate_values(x_a: np.ndarray, spec_a: dict, x_b: np.ndarray, spec_b: dict) -> np.ndarray | None:
    """Replay-exact candidate column: ``nan_to_num(mul(prewarp_a(x_a), prewarp_b(x_b)))``
    -- the same chain ``_apply_unary_binary`` executes at transform() time."""
    from .hermite_fe import apply_operand_prewarp
    try:
        wa = apply_operand_prewarp(np.asarray(x_a, dtype=np.float64), spec_a)
        wb = apply_operand_prewarp(np.asarray(x_b, dtype=np.float64), spec_b)
    except Exception:
        return None
    out = np.multiply(wa, wb)
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(out)) or float(np.std(out)) < 1e-12:
        return None
    return out


def _propose_poly(x_a, x_b, y_f, *, degree: int, min_val_corr: float):
    """Signal-adaptive orth-poly proposer: rank-1 ALS pair warp per shipped basis,
    held-out stride validation of the rank-1 reconstruction, best basis wins. Returns
    ``(spec_a, spec_b, basis, val_corr)`` or ``None`` (no basis generalises)."""
    from .hermite_fe import apply_operand_prewarp, fit_pair_prewarp_als
    xa = _finite_filled(x_a)
    xb = _finite_filled(x_b)
    n = xa.size
    if n < 60 or float(np.std(xa)) < 1e-12 or float(np.std(xb)) < 1e-12:
        return None
    idx = np.arange(n)
    va = (idx % 3) == 0
    tr = ~va
    y_tr = y_f[tr]
    y_va = y_f[va] - float(np.mean(y_f[va]))
    if float(np.std(y_tr)) < 1e-12 or float(np.std(y_va)) < 1e-12:
        return None
    best_corr = -1.0
    best_basis = None
    for basis in _ESCALATION_POLY_BASES:
        try:
            sa_tr, sb_tr = fit_pair_prewarp_als(xa[tr], xb[tr], y_tr, basis=basis, max_degree=degree)
            if sa_tr is None or sb_tr is None:
                continue
            recon = apply_operand_prewarp(xa[va], sa_tr) * apply_operand_prewarp(xb[va], sb_tr)
            recon = np.nan_to_num(recon, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if float(np.std(recon)) < 1e-12:
                continue
            c = abs(float(np.corrcoef(recon, y_va)[0, 1]))
        except Exception:
            continue
        if np.isfinite(c) and c > best_corr:
            best_corr = c
            best_basis = basis
    if best_basis is None or best_corr < float(min_val_corr):
        return None
    try:
        sa, sb = fit_pair_prewarp_als(xa, xb, y_f, basis=best_basis, max_degree=degree)
    except Exception:
        return None
    if sa is None or sb is None:
        return None
    return sa, sb, best_basis, best_corr


def _fit_fourier_amplitude_spec(axis01: np.ndarray, t: np.ndarray, freqs, preprocess: dict) -> dict | None:
    """Least-squares sin/cos amplitudes of the demodulated target ``t`` at the detected
    frequencies over the fitted axis. Returns the closed-form ``fourier_adaptive``
    prewarp spec (``coef`` packs ``[a_1, b_1, ..., a_K, b_K]``; ``preprocess`` carries
    the axis params + freqs) consumable by ``apply_operand_prewarp``."""
    K = len(freqs)
    if K == 0:
        return None
    D = np.empty((axis01.size, 2 * K), dtype=np.float64)
    for i, f in enumerate(freqs):
        ang = 2.0 * np.pi * float(f) * axis01
        D[:, 2 * i] = np.sin(ang)
        D[:, 2 * i + 1] = np.cos(ang)
    try:
        coef, *_ = np.linalg.lstsq(D, t - float(np.mean(t)), rcond=None)
    except Exception:
        return None
    if coef is None or not np.all(np.isfinite(coef)) or float(np.max(np.abs(coef))) < 1e-12:
        return None
    pp = dict(preprocess)
    pp["freqs"] = [float(f) for f in freqs]
    return {
        "basis": "fourier_adaptive",
        "degree": int(K),
        "coef": np.ascontiguousarray(coef, dtype=np.float64),
        "preprocess": pp,
    }


def _propose_fourier(x_w, x_m, y_f, *, min_val_corr: float, max_freqs: int, chirp: bool = True):
    """Adaptive-frequency Fourier (+ chirp) proposer for the multiplicative pair form
    ``y ~ g(x_w) * x_m`` via DEMODULATION: the shipped held-out multitone detector is run
    on ``(axis(x_w), t = y_c * zscore(x_m))``. Returns a list of fitted warp specs (0-2:
    linear-axis and/or quadratic-chirp-axis), each a ``fourier_adaptive`` prewarp spec."""
    from ._orthogonal_univariate_fe._orth_extra_basis_fe import (
        _chirp_axis,
        _detect_fourier_freqs_for_col,
        _fit_chirp_warp_for_col,
        _fit_fourier_for_col,
        _is_int_as_cat_axis,
    )
    out: list[dict] = []
    xw = _finite_filled(x_w)
    xm = _finite_filled(x_m)
    if _is_int_as_cat_axis(xw):
        # Arbitrary integer label codes carry no real oscillation -- mirror the shipped
        # univariate guard (sin/cos of a region code is spurious periodicity).
        return out
    std_m = float(np.std(xm))
    if std_m < 1e-12 or float(np.std(xw)) < 1e-12:
        return out
    z_m = (xm - float(np.mean(xm))) / std_m
    y_c = y_f - float(np.mean(y_f))
    if float(np.std(y_c)) < 1e-12:
        return out
    t = y_c * z_m
    # Linear axis (shipped robust min-max normalisation).
    lo, span = _fit_fourier_for_col(xw)
    z01 = (xw - float(lo)) / max(float(span), 1e-12)
    freqs = _detect_fourier_freqs_for_col(
        z01, t, f_grid=_ADAPTIVE_F_GRID, min_val_corr=float(min_val_corr),
        min_rows=800, max_freqs=int(max_freqs),
    )
    if freqs:
        spec = _fit_fourier_amplitude_spec(
            z01, t, freqs, {"arg": "linear", "lo": float(lo), "span": float(span)},
        )
        if spec is not None:
            out.append({"kind": "fourier", "spec_w": spec, "freqs": [float(f) for f in freqs]})
    # Quadratic-argument chirp axis (shipped warp): stationary in u for growing-frequency inners.
    if chirp:
        c_mean, c_std, c_lo, c_span = _fit_chirp_warp_for_col(xw)
        if c_span > 1e-12 and c_std > 1e-12:
            u = _chirp_axis(xw, c_mean, c_std, c_lo, c_span)
            if np.all(np.isfinite(u)) and float(np.std(u)) > 1e-12:
                cfreqs = _detect_fourier_freqs_for_col(
                    u, t, f_grid=_CHIRP_F_GRID, min_val_corr=float(min_val_corr),
                    min_rows=800, max_freqs=int(max_freqs),
                )
                if cfreqs:
                    spec = _fit_fourier_amplitude_spec(
                        u, t, cfreqs,
                        {"arg": "quadratic", "mean": float(c_mean), "std": float(c_std),
                         "lo": float(c_lo), "span": float(c_span)},
                    )
                    if spec is not None:
                        out.append({"kind": "chirp", "spec_w": spec, "freqs": [float(f) for f in cfreqs]})
    return out


def _resolve_operand(X, name: str, engineered_continuous: dict | None) -> np.ndarray | None:
    """Continuous values for a RAW column ``name`` from the (possibly augmented) frame.
    Prefers the continuous engineered store (not expected for raw operands, kept for
    symmetry); pandas / polars by-name extraction; ``None`` when unresolvable."""
    if engineered_continuous:
        v = engineered_continuous.get(name)
        if v is not None and np.asarray(v).shape[0] == len(X):
            return np.asarray(v, dtype=np.float64)
    try:
        if hasattr(X, "columns") and name in list(X.columns):
            col = X[name]
            vals = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
            return np.asarray(vals, dtype=np.float64)
    except Exception:
        return None
    return None


def run_fe_auto_escalation(
    self,
    *,
    failed_pairs,
    X,
    cols,
    classes_y,
    pair_maxt_floor: float,
    admitted_pool: dict,
    verbose: int = 0,
):
    """Escalate the FE search to the richer shipped bases for prescreen-surviving pairs
    the unary/binary step admitted NOTHING for. PROPOSES candidates (signal-adaptive
    orth-poly ALS warps + demodulated adaptive-frequency Fourier/chirp warps), then runs
    them through the EXISTING admission gates (order-2 maxT floor on MM-debiased MI,
    marginal-permutation floor, S5 conditional-MI redundancy gate vs the admitted
    engineered support). Returns a list of admitted candidate dicts
    ``{name, values, recipe, mi, kind, pair}`` for the caller to materialise; stamps
    ``self.fe_escalation_info_`` provenance. Never raises (degrades to ``[]``)."""
    from ._fe_cmi_redundancy_gate import _conditional_perm_null, apply_cmi_redundancy_gate
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin
    from .engineered_recipes import build_unary_binary_recipe

    info: dict = {"eligible_pairs": [], "proposed": 0, "admitted": [], "rejected": {},
                  "pair_maxt_floor": float(pair_maxt_floor)}
    self.fe_escalation_info_ = info
    if not failed_pairs:
        return []

    n_rows = int(len(X))
    min_rows = int(getattr(self, "fe_escalation_min_rows", 500))
    if n_rows < min_rows:
        info["skipped"] = f"n_rows={n_rows} < fe_escalation_min_rows={min_rows}"
        return []

    raw_names = set(getattr(self, "feature_names_in_", []) or [])
    eng_cont = getattr(self, "_engineered_continuous_", None)
    max_pairs = int(getattr(self, "fe_escalation_max_pairs", 8))
    min_val_corr = float(getattr(self, "fe_escalation_min_val_corr", 0.15))
    poly_degree = int(getattr(self, "fe_escalation_poly_degree", 6))
    max_freqs = int(getattr(self, "fe_escalation_fourier_max_freqs", 3))
    per_pair_cap = int(getattr(self, "fe_escalation_max_candidates_per_pair", 3))
    seed = int(getattr(self, "random_seed", 0) or 0)
    nbins = int(self.quantization_nbins)

    # Target for the supervised warp fits: the SAME discretised codes the MI sweep and
    # the shipped pair-prewarp fit against (``prewarp_y=classes_y`` convention).
    y_f = np.ascontiguousarray(np.asarray(classes_y), dtype=np.float64)
    y_arr = np.asarray(classes_y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    _, y_dense = np.unique(y_arr, return_inverse=True)
    y_dense = y_dense.astype(np.int64)

    # RAW-RAW pairs only, strongest joint MI first, bounded by the pair budget.
    eligible = []
    for pair, pair_mi in failed_pairs:
        try:
            na, nb = cols[pair[0]], cols[pair[1]]
        except Exception:
            continue
        if na in raw_names and nb in raw_names:
            eligible.append((pair, float(pair_mi), na, nb))
    eligible.sort(key=lambda e: e[1], reverse=True)
    eligible = eligible[:max_pairs]
    info["eligible_pairs"] = [(na, nb) for _, _, na, nb in eligible]
    if not eligible:
        return []

    existing_names = set(cols) | set(admitted_pool)
    candidates: list[dict] = []
    for pair, pair_mi, na, nb in eligible:
        x_a = _resolve_operand(X, na, eng_cont)
        x_b = _resolve_operand(X, nb, eng_cont)
        if x_a is None or x_b is None or x_a.size != n_rows or x_b.size != n_rows:
            continue
        pair_cands: list[dict] = []
        # 1) Signal-adaptive orth-poly ALS warp (higher degree + 4-basis routing).
        poly = _propose_poly(x_a, x_b, y_f, degree=poly_degree, min_val_corr=min_val_corr)
        if poly is not None:
            sa, sb, basis, vcorr = poly
            vals = _candidate_values(x_a, sa, x_b, sb)
            if vals is not None:
                pair_cands.append({
                    "name": f"esc_poly_{basis}_mul({na},{nb})",
                    "values": vals, "spec_a": sa, "spec_b": sb,
                    "src_a": na, "src_b": nb, "kind": f"poly_{basis}",
                    "pair": (na, nb), "val_corr": float(vcorr),
                })
        # 2) Demodulated adaptive-frequency Fourier / chirp, both warp directions.
        for x_w, x_m, nw, nm in ((x_a, x_b, na, nb), (x_b, x_a, nb, na)):
            for prop in _propose_fourier(x_w, x_m, y_f, min_val_corr=min_val_corr,
                                         max_freqs=max_freqs, chirp=True):
                spec_m = _identity_prewarp_spec(x_m)
                if spec_m is None:
                    continue
                vals = _candidate_values(x_w, prop["spec_w"], x_m, spec_m)
                if vals is None:
                    continue
                pair_cands.append({
                    "name": f"esc_{prop['kind']}_mul({nw},{nm})",
                    "values": vals, "spec_a": prop["spec_w"], "spec_b": spec_m,
                    "src_a": nw, "src_b": nm, "kind": prop["kind"],
                    "pair": (na, nb), "freqs": prop["freqs"],
                })
        # Score by the SAME MM-debiased plug-in MI the gates use; cap per pair.
        for c in pair_cands:
            vb = _quantile_bin(np.asarray(c["values"], dtype=np.float64), nbins=nbins)
            c["_binned"] = vb
            c["mi"] = float(_cmi_from_binned(vb, y_dense, None))
        pair_cands.sort(key=lambda c: c["mi"], reverse=True)
        candidates.extend(pair_cands[:max(1, per_pair_cap)])

    info["proposed"] = len(candidates)
    if not candidates:
        return []

    # Deduplicate names defensively (two pairs sharing operands cannot collide on the
    # name template, but an operand name containing "," could).
    seen: set = set()
    for c in candidates:
        base = c["name"]
        k = 2
        while c["name"] in existing_names or c["name"] in seen:
            c["name"] = f"{base}_{k}"
            k += 1
        seen.add(c["name"])

    # GATE 2: order-2 maxT permutation floor (MM-debiased MI scale on BOTH sides --
    # the floor was computed with miller_madow=True, ``_cmi_from_binned`` debiases too).
    # GATE 3: marginal-permutation floor (same primitive the S5 gate's significance leg
    # uses) -- protects the degenerate single-candidate path where the S5 gate would
    # otherwise admit on marginal significance alone.
    survivors: list[dict] = []
    for c in candidates:
        if pair_maxt_floor > 0.0 and c["mi"] < float(pair_maxt_floor):
            info["rejected"][c["name"]] = f"below_maxt_floor (mi={c['mi']:.5f} < {pair_maxt_floor:.5f})"
            continue
        floor_m, _null_mean = _conditional_perm_null(c["_binned"], y_dense, None, seed=seed)
        if c["mi"] <= floor_m:
            info["rejected"][c["name"]] = f"below_marginal_perm_floor (mi={c['mi']:.5f} <= {floor_m:.5f})"
            continue
        survivors.append(c)
    if not survivors:
        if verbose:
            logger.info(
                "MRMR FE auto-escalation: %d candidate(s) proposed for %d pair(s), 0 cleared "
                "the maxT/permutation floors (gates decide; noise control held).",
                info["proposed"], len(eligible),
            )
        return []

    # GATE 4: S5 conditional-MI redundancy gate over admitted support + survivors.
    # Verdicts are applied to ESCALATION candidates only.
    pool: dict = {}
    for nm, (vals, marg) in (admitted_pool or {}).items():
        pool[nm] = (np.asarray(vals, dtype=np.float64), float(marg))
    for c in survivors:
        pool[c["name"]] = (np.asarray(c["values"], dtype=np.float64), c["mi"])
    accepted, _diag = apply_cmi_redundancy_gate(
        pool, y_dense, nbins=nbins,
        retain_frac=float(getattr(self, "fe_engineered_cmi_retain_frac", 0.15)),
        seed=seed, verbose=int(bool(verbose)),
    )
    admitted: list[dict] = []
    for c in survivors:
        if c["name"] not in accepted:
            info["rejected"][c["name"]] = "redundant_under_cmi_gate"
            continue
        recipe = build_unary_binary_recipe(
            name=c["name"],
            src_a_name=c["src_a"], src_b_name=c["src_b"],
            unary_a_name="prewarp", unary_b_name="prewarp",
            binary_name="mul",
            unary_preset=str(getattr(self, "fe_unary_preset", "medium")),
            binary_preset=str(getattr(self, "fe_binary_preset", "minimal")),
            quantization_nbins=self.quantization_nbins,
            quantization_method=self.quantization_method,
            quantization_dtype=self.quantization_dtype,
            fit_values_for_edges=np.asarray(c["values"], dtype=np.float64),
            prewarp_a=c["spec_a"], prewarp_b=c["spec_b"],
        )
        c.pop("_binned", None)
        c["recipe"] = recipe
        admitted.append(c)
        info["admitted"].append(c["name"])
    if verbose and (admitted or info["proposed"]):
        logger.info(
            "MRMR FE auto-escalation: %d pair(s) had 0 admitted engineered features after the "
            "unary/binary search; proposed %d richer-basis candidate(s) (orth-poly ALS x4 bases "
            "+ demodulated adaptive Fourier/chirp), gates admitted %d: %s",
            len(eligible), info["proposed"], len(admitted),
            [f"{c['name']} (mi={c['mi']:.4f})" for c in admitted],
        )
    return admitted
