"""Standalone helpers for the raw-vs-engineered conditional-redundancy drop.

Pure relocation out of ``_fe_raw_redundancy_drop`` to keep that module under the 1000-LOC
ceiling: the standalone probe / keep-rule helper functions plus the module constants they
reference. The main module re-exports every name defined here, so external
``from ..._fe_raw_redundancy_drop import <name>`` importers keep resolving and
``drop_redundant_raw_operands`` still calls these helpers by their bare names.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

# Row cap for the raw-operand redundancy CMI + conditional-permutation null. The perm-null is an
# EXPLICITLY RANDOM null estimate (selection-equivalence, not byte-identical), and the CMI gate already
# subsamples (30k). On the FINAL raw-drop the observed CMI + the ~25-perm within-stratum null run on the
# full n (1M) -- the single largest per-call redundancy hotspot. Estimating both on a strided subsample
# (all of cand / y / z together, so the returned cmi/floor/excess stay MUTUALLY consistent and every
# drop comparison is self-consistent) keeps the decision selection-equivalent while the null cost drops
# ~n/cap. Env-tunable; 0 -> full-n (strict). 100k default (2026-07-03): the perm-null floor's precision
# scales ~1/sqrt(n * nperm), so at 100k x ~25 perms it is already far past the CLT plateau -- 250k -> 100k
# is accuracy-NEUTRAL (not a speed/accuracy trade), and validated selection-equivalent on the adversarial
# user-case-a multi-seed retention pin + the F2 5-profile goal. Measured drop_redundant ~4.1s -> ~3.0s.
try:
    _CMI_NULL_MAX_ROWS = int(os.environ.get("MLFRAME_CMI_NULL_MAX_ROWS", "100000"))
except (ValueError, TypeError):
    _CMI_NULL_MAX_ROWS = 100000


# GATE / BINAGG / ARGMAX pseudo-child name prefixes (2026-06-13). The default-ON
# conditional-gate (``gate_mask__a__b__t...`` / ``gate_select__...``), binned-
# numeric-agg (``binagg_skew(c|qbin(a))``) and row-argmax (``argmax__a__b``)
# families build engineered children that are THRESHOLD / BINNING re-mixes of a raw
# operand. A re-mix of ``a`` (e.g. ``1[a>median]`` or a ``qbin(a)``-strata aggregate)
# PARTIALLY tracks a genuine private LINEAR term ``10*a``, so when a raw's
# conditional excess is computed GIVEN such a pseudo-child its residual collapses
# below the self-retention bar and the raw is WRONGLY judged redundant (the
# raw-self-signal-masking trap -- the same lesson the DPI-trap consumer filter
# encodes for self-transforms). These pseudo-children are therefore EXCLUDED from
# the set of consuming children a raw is conditioned against for the keep/drop
# verdict; only GENUINE elementary composites (ratio / product / poly), which can
# actually SUBSUME the raw, are kept in the conditioning set. Prefix-detected so the
# exclusion is byte-identical to the prior behaviour when no such child exists.
_PSEUDO_CHILD_PREFIXES = ("gate_", "binagg_", "argmax__")

# SELF-RETENTION fraction (keep leg A, 2026-06-10). The raw must retain >= this
# fraction of its OWN marginal debiased excess as CONDITIONAL excess (given the
# combination child) to count as a significant independent residual. A genuine
# private LINEAR term keeps ~6-11% of its marginal given the interaction product;
# a fully-subsumed ``a**2/b`` ratio operand keeps ~0.3-2%. 0.05 sits between with
# >=2x margin on both sides across the validated genuine/subsumed cells.
RAW_SELF_RETAIN_FRAC = 0.05

_MIN_ROWS = 500


def _is_pseudo_remix_child(name: str) -> bool:
    """True iff the engineered name is a conditional-gate / binned-numeric-agg /
    row-argmax pseudo-child -- a threshold/binning re-mix that can MASK (not
    subsume) a raw operand's genuine private signal and so must NOT condition the
    raw's redundancy verdict. Detected by the canonical name prefix family."""
    nm = name or ""
    return any(nm.startswith(p) for p in _PSEUDO_CHILD_PREFIXES)


def raw_retains_signal_given_genuine_children(
    *,
    raw_bin: np.ndarray,
    y_bin: np.ndarray,
    genuine_child_bins: Sequence[np.ndarray],
    self_retain_frac: float = RAW_SELF_RETAIN_FRAC,
    allow_linear_usability: bool = False,
    seed: int = 0,
    genuine_child_bins_dev: Optional[Sequence] = None,
    raw_bin_dev=None,
) -> bool:
    """KEEP-rule probe reused by the PSEUDO-CHILD MASKED-RAW RESCUE (2026-06-13).

    Returns True iff the raw carries a SIGNIFICANT INDEPENDENT RESIDUAL given ONLY
    its GENUINE (non-pseudo) consuming engineered children: its conditional CMI
    clears the within-stratum permutation floor AND its debiased conditional excess
    retains >= ``self_retain_frac`` of its OWN marginal debiased excess. This is the
    exact keep-rule ``drop_redundant_raw_operands`` applies, factored out so the
    masked-raw rescue can decide whether the greedy screen's drop of the raw was a
    genuine subsumption or merely a gate/binagg/argmax pseudo-child MASKING the raw's
    private signal (a re-mix is selected first and DPI-collapses the raw's relevance).
    A genuine private LINEAR term keeps ~50% given its true ratio child; a fully-
    subsumed ``a**2/b`` operand keeps ~0.6% -- the 0.05 bar separates them with >8x
    margin (measured user fixture, n=40000). With NO genuine child (the conditioning
    set is empty) it returns True (cannot prove subsumption -> retention stands)."""
    from ._mi_greedy_cmi_fe import _renumber_joint

    _gc = [g for g in genuine_child_bins if g is not None]
    rb = np.asarray(raw_bin).astype(np.int64).ravel()
    yb = np.asarray(y_bin).astype(np.int64).ravel()
    # DEVICE-BORN candidate residency (same contract as ``drop_redundant_raw_operands``; opt-out
    # MLFRAME_FE_GATE_RESIDENT_CANDS=0). The raw ``rb`` is scored (marginal + conditional) but never
    # ``_renumber_joint``-ed here (the children build ``z_support``), so upload it ONCE as resident int64 codes
    # and thread that handle into both ``_excess_and_floor`` calls -- the candidate never re-crosses H2D at the
    # ``cmi_cand_x`` / ``card_cand_x`` / ``permnull_cand_x`` sites. Host codes on any fault / when off.
    # Prefer an ALREADY-RESIDENT raw candidate code handed in (the caller device-binned it) so the raw never
    # re-crosses H2D; else upload once via the content-keyed cache.
    _rb_cand = raw_bin_dev if raw_bin_dev is not None else rb
    import os as _os
    if raw_bin_dev is None and _os.environ.get("MLFRAME_FE_GATE_RESIDENT_CANDS", "1").strip().lower() in ("1", "true", "on", "yes"):
        try:
            from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
            from ._mi_greedy_cmi_fe import _cmi_gpu_enabled
            if bool(fe_gpu_strict_resident_enabled()) and bool(_cmi_gpu_enabled()):
                from ._fe_resident_operands import resident_code_operand
                _rb_cand = resident_code_operand(rb, "cmi_cand_x")
        except Exception:
            _rb_cand = rb
    _rb_kx = (int(rb.max()) + 1 if getattr(rb, "size", 0) else 1)   # host raw codes -> free cardinality
    _, _, marg_excess = _excess_and_floor(_rb_cand, yb, None, seed=seed, kx=_rb_kx)
    if not _gc:
        return True  # no genuine subsumer survives -> the raw cannot be proven redundant
    # DEVICE-BORN conditioning support when the caller hands resident child codes: join them on device
    # (``_renumber_joint_gpu``, same partition -> selection-identical) so the support never crosses H2D (cmi_z +
    # perm-null order/z_rank). None if any child lacks a resident twin -> host z scored.
    z_support_dev = None
    _zcard = 0   # occupied cardinality of the conditioning support (from whichever join runs) -> kz, no device read
    if genuine_child_bins_dev is not None:
        _gcd = [g for g in genuine_child_bins_dev if g is not None]
        if _gcd and len(_gcd) == len(_gc):
            try:
                from ._mi_greedy_cmi_fe import _renumber_joint_gpu
                z_support_dev, _zcard = _renumber_joint_gpu(*_gcd)
            except Exception:
                z_support_dev = None
    # HOST join only when the device join is unavailable: _excess_and_floor explicitly supports
    # ``z_support=None`` with a resident ``z_support_dev`` (its perm-null / analytic legs D2H the device form
    # only if a genuine host consumer is reached), and the host multi-column ``_renumber_joint`` is a full-n
    # unique SORT per gate call -- a measurable host sink that was paid even when the device join was used.
    z_support = None
    if z_support_dev is None:
        z_support, _zcard = _renumber_joint(*_gc)
    cmi, floor, excess = _excess_and_floor(_rb_cand, yb, z_support, seed=seed, z_support_dev=z_support_dev,
                                           kx=_rb_kx, kz=int(_zcard))
    if (cmi > floor) and (excess >= self_retain_frac * max(0.0, marg_excess)):
        return True
    # LINEAR-USABILITY leg (variant-3): a linearly-usable raw whose conditional CMI collapsed
    # under an excellent NONLINEAR child is still wanted in SIMPLE mode (robust raw set). Gated
    # off in full FE mode where a subsumed monotone operand -- statistically indistinguishable
    # from a genuine linear term -- must still drop (I4b). See the drop-sweep twin leg.
    if not allow_linear_usability:
        return False
    return raw_retains_linear_signal_given_children(rb, yb, _gc, seed=seed)


def _recipe_subexprs(recipe) -> dict:
    """Walk an ``EngineeredRecipe``'s nested-parent operand tree and return a map
    ``{sub_recipe_name -> sub_recipe}`` of EVERY node in the tree (the recipe
    itself plus every nested-engineered parent at any depth).

    NESTED-OPERAND CONSUMER DETECTION (BUG1, 2026-06-12). A step-k composite like
    ``add(div(log(c),reciproc(d)),abs(div(sqr(a),abs(b))))`` carries its operand
    structure as ``EngineeredRecipe`` objects in ``extra['nested_parent_a/b']``
    (the dataclass tree, NOT the str() name). The redundancy verdict must be able
    to condition a raw operand on the CLEAN a-containing sub-expression
    ``div(sqr(a),abs(b))`` = a**2/b (which isolates the raw's full capture), not
    only on the WHOLE fused composite (which mixes a**2/b with log(c)*sin(d) and so
    leaves the raw a spurious residual -> wrongly KEPT). This recovers the
    sub-expressions so each can be replayed to its own continuous values and used
    as a clean conditioning anchor."""
    out: dict = {}
    if recipe is None:
        return out
    stack = [recipe]
    seen_ids = set()
    while stack:
        r = stack.pop()
        if r is None or id(r) in seen_ids:
            continue
        seen_ids.add(id(r))
        nm = getattr(r, "name", None)
        if nm is not None:
            out[nm] = r
        extra = getattr(r, "extra", None) or {}
        for _k in ("nested_parent_a", "nested_parent_b"):
            _p = extra.get(_k)
            if _p is not None:
                stack.append(_p)
    return out


def _subexpr_continuous(recipe, raw_X) -> Optional[np.ndarray]:
    """Replay one (sub-)recipe to its CONTINUOUS values from the raw frame.

    Uses the production ``apply_recipe`` with quantization stripped so the
    sub-expression is reconstructed on continuous values exactly as at fit time.
    Returns ``None`` on any replay failure (the caller then falls back to the
    fused-composite continuous values -- the conservative KEEP direction)."""
    if recipe is None or raw_X is None:
        return None
    try:
        import dataclasses as _dc
        from .engineered_recipes._recipe_dispatch import apply_recipe
        _r = recipe
        if getattr(_r, "quantization", None) is not None:
            _r = _dc.replace(_r, quantization=None)
        vals = np.asarray(apply_recipe(_r, raw_X), dtype=np.float64).ravel()
        vals = np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return vals
    except Exception:
        return None


def _excess_and_floor(cand_bin, y_bin, z_support, *, seed=0, z_support_dev=None, kx=0, kz=0):
    """Return ``(cmi, floor, excess)`` for ``CMI(cand; y | z_support)`` using the
    S5 conditional-permutation null (within-stratum shuffle reproduces the same
    finite-sample bias, so ``excess = max(0, cmi - null_mean)`` is n-invariant).
    ``z_support=None`` -> marginal MI / free-shuffle null (used for the anchor).

    ``z_support_dev`` (optional): a DEVICE-BORN conditioning support (``_renumber_joint_gpu`` of the resident
    conditioning codes). When supplied it is scored RESIDENT -- the CMI reads it via ``_cmi_from_binned_cupy``'s
    resident-z branch and the perm-null derives order/z_rank on device -- so the support never crosses H2D (the
    ``cmi_z`` + order/z_rank uploads). The host ``z_support`` stays the byte-path fallback (and NULL if the
    caller only has the device form)."""
    from ._mi_greedy_cmi_fe import _cmi_from_binned
    from ._fe_cmi_redundancy_gate import _conditional_perm_null

    # Strided-subsample cand / y / z TOGETHER (same rows) so the observed CMI and the perm-null are estimated
    # on the same reduced sample -> the returned (cmi, floor, excess) stay mutually consistent and every drop
    # comparison (cmi>floor, excess>0, cross-candidate relative bar) is self-consistent. See _CMI_NULL_MAX_ROWS.
    _n_full = int(cand_bin.shape[0])
    if _CMI_NULL_MAX_ROWS > 0 and _n_full > _CMI_NULL_MAX_ROWS:
        _st = int(_n_full // _CMI_NULL_MAX_ROWS)
        if _st > 1:
            cand_bin = cand_bin[::_st]
            y_bin = y_bin[::_st]
            z_support = z_support[::_st] if z_support is not None else None
            z_support_dev = z_support_dev[::_st] if z_support_dev is not None else None

    # z handed to the resident CMI scorer: the device-born support when available, else the host support.
    _z_scored = z_support_dev if z_support_dev is not None else z_support

    # bench-attempt-rejected (2026-06-26): computing cmi + the analytic-null df cards from ONE
    # batched_cmi_gpu(return_cards) call (to skip the perm-null's joint_cardinalities via precomp_cards)
    # REGRESSED F2 STRICT (1849 -> 1917, +68). batched_cmi_gpu builds the full shared y/z workload (zc, yzc
    # histograms) which only amortises across MANY candidate columns; for the SINGLE candidate here it is
    # heavier (~11 launches) than _cmi_from_binned (~5) + joint_cardinalities (~4) separately. This path is
    # genuinely per-raw with VARYING conditioning z (base / full-composite / sibling), so it is not
    # batchable with the fixed-y/z primitives -- keep the per-call _cmi_from_binned.
    if z_support is None and z_support_dev is None:
        cmi = float(_cmi_from_binned(cand_bin, y_bin, None, kx=kx))
        floor, null_mean = _conditional_perm_null(cand_bin, y_bin, None, seed=seed)
    else:
        # CONDITIONAL: the observed CMI call already computes the four occupied-cell cards
        # (k_z, k_xz, k_yz, k_xyz) as a byproduct of its fused entropy+nnz pass; hand them to the analytic
        # null via precomp_cards so it skips recomputing the IDENTICAL four joints (joint_cardinalities on
        # GPU / renumber+entropy on CPU). Same occupied-cell definition -> bit-identical df -> selection-
        # identical; removes ~4 histograms per conditional raw on the redundancy gate (both backends).
        # ``_z_scored`` is the device-born support when available (resident -> no cmi_z H2D) else the host one.
        cmi_v, _cards = _cmi_from_binned(cand_bin, y_bin, _z_scored, return_cards=True, kx=kx, kz=kz)
        cmi = float(cmi_v)
        floor, null_mean = _conditional_perm_null(
            cand_bin, y_bin, z_support, seed=seed, precomp_cards=_cards, z_support_dev=z_support_dev)
    return cmi, floor, max(0.0, cmi - null_mean)


# Minimum absolute partial linear correlation for the linear-usability keep-leg: a
# safety floor below the permutation null so a vanishing real residual (a truly
# subsumed operand / pure noise) is never kept even when the perm floor shrinks at
# large n. Tuned so a dominant linear term (s0 in ``y=2*s0-1.3*s1+0.8*s2``) clears it
# with wide margin while an ``a**2/b`` ratio operand (≈0 linear residual) does not.
_LINEAR_RESIDUAL_MIN_PCORR = 0.02


def _rank_transform(a: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties averaged) -> Spearman building block. Maps any
    MONOTONE relationship (linear / exp / log / ordinal) to a linear one, so the
    partial-correlation usability test handles a log-linked count target or an ordinal
    target the same as a plain linear one, while a non-monotone subsumed operand
    (``a`` in even ``a**2/b``) stays ~uncorrelated."""
    a = np.asarray(a, dtype=np.float64).ravel()
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, a.shape[0] + 1, dtype=np.float64)
    # average tied ranks so duplicate bin codes do not bias the correlation.
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0  # mean rank within each tie group (1-based)
    return avg[inv]


def _residualize(target: np.ndarray, design: np.ndarray) -> Optional[np.ndarray]:
    """Return ``target`` minus its OLS projection onto ``[1|design]`` (the linear
    residual), or ``None`` on a degenerate / non-finite fit."""
    try:
        n = target.shape[0]
        X = np.column_stack([np.ones(n, dtype=np.float64), design]) if design.size else np.ones((n, 1))
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        resid = target - X @ coef
        if not np.all(np.isfinite(resid)):
            return None
        return resid
    except Exception:
        return None


def raw_retains_linear_signal_given_children(
    raw_vals: np.ndarray,
    y_vals: np.ndarray,
    child_vals_list: Sequence[np.ndarray],
    *,
    seed: int = 0,
    nperm: int = 32,
    min_pcorr: float = _LINEAR_RESIDUAL_MIN_PCORR,
) -> bool:
    """LINEAR-USABILITY keep-leg (variant-3, 2026-06-20).

    KEEP a raw whose conditional CMI given its engineered children has collapsed but
    which still carries SIGNIFICANT PRIVATE LINEAR signal toward ``y`` the children do
    not LINEARLY reproduce. An MI-only verdict cannot tell a dominant linear term
    info-subsumed by an excellent NONLINEAR predictor (``s0`` captured by
    ``sub(log(s1),cbrt(add(s0,log(s2))))``) from a genuinely subsumed ratio operand
    (``a`` in ``a**2/b``): both have ~0 conditional excess. Linear usability separates
    them -- a nonlinear child is not a linear equivalent of the raw, so ``s0`` retains
    a large partial linear correlation with ``y`` after the children are regressed out,
    while ``a``'s linear residual is ~0 (the ratio child IS ``y`` linearly).

    Returns True iff the partial linear correlation ``corr(raw_resid, y_resid)`` (raw &
    y each residualized on the child design) exceeds BOTH a permutation null floor
    (n-invariant: shuffling the raw residual reproduces the spurious-correlation scale)
    AND ``min_pcorr`` (a finite-sample safety floor). No children -> not applicable
    (the caller's no-subsumer path handles that) -> returns False."""
    _cv = [np.asarray(c, dtype=np.float64).ravel() for c in child_vals_list if c is not None]
    if not _cv:
        return False
    rv = np.asarray(raw_vals, dtype=np.float64).ravel()
    yv = np.asarray(y_vals, dtype=np.float64).ravel()
    n = rv.shape[0]
    if n < _MIN_ROWS or any(c.shape[0] != n for c in _cv) or yv.shape[0] != n:
        return False
    rv = np.nan_to_num(rv, nan=0.0, posinf=0.0, neginf=0.0)
    yv = np.nan_to_num(yv, nan=0.0, posinf=0.0, neginf=0.0)
    # Rank-transform every series -> the partial correlation becomes a partial SPEARMAN,
    # so a monotone-nonlinear target (log-linked count, ordinal) is recovered like a
    # linear one while a non-monotone subsumed operand stays uncorrelated.
    rv = _rank_transform(rv)
    yv = _rank_transform(yv)
    design = np.column_stack([_rank_transform(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)) for c in _cv])
    ry = _residualize(yv, design)
    rx = _residualize(rv, design)
    if ry is None or rx is None:
        return False
    sx, sy = float(np.std(rx)), float(np.std(ry))
    if sx < 1e-12 or sy < 1e-12:
        return False
    pcorr = float(np.corrcoef(rx, ry)[0, 1])
    if not np.isfinite(pcorr) or abs(pcorr) < min_pcorr:
        return False
    # Permutation null: shuffle the raw residual; the 95th-percentile |corr| is the
    # n-invariant spurious-correlation floor the real partial corr must clear.
    # corr(perm, ry) reduces to (perm @ (ry-mean(ry))) / (n*std(rx)*std(ry)): perm is a
    # reordering of rx so its mean/std are loop-invariant, and ry is fixed -- only the
    # cross dot product varies. Hoisting the centred target + constant denominator out
    # of the loop replaces np.corrcoef's per-iteration 2x2 rebuild with one dot product
    # (bit-identical to corrcoef up to FP reduction order, ~1e-17).
    rng = np.random.default_rng(seed)
    ryc = ry - ry.mean()
    denom = n * float(np.std(rx)) * float(np.std(ry))
    null = np.empty(int(nperm), dtype=np.float64)
    for k in range(int(nperm)):
        perm = rng.permutation(rx)
        null[k] = abs(float(perm @ ryc) / denom) if denom > 0.0 else 0.0
    floor_p95 = float(np.percentile(null[np.isfinite(null)], 95)) if np.any(np.isfinite(null)) else 1.0
    return abs(pcorr) > floor_p95


def _heldout_ridge_r2(X: np.ndarray, y: np.ndarray, frac: float = 0.7) -> Optional[float]:
    """Held-out StandardScaler+Ridge R^2 -- the SAME linear probe the I4b downstream no-harm contract measures
    with (first ``frac`` rows train, remainder score; no shuffle, matching the endtoend uplift split). Returns
    None on any degenerate/estimator fault so the caller leaves its decision unchanged."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from mlframe.metrics.core import fast_r2_score
    except Exception:
        return None
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.asarray(y, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        return None
    n = X.shape[0]
    sp = int(n * frac)
    if sp < 50 or (n - sp) < 50 or float(np.std(y[:sp])) < 1e-12:
        return None
    try:
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[:sp], y[:sp])
        return float(fast_r2_score(y[sp:], model.predict(X[sp:])))
    except Exception:
        return None
