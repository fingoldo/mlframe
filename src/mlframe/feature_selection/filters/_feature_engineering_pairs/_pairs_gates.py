"""Acceptance gates + single-best winner selection for the FE pair-search:
the prewarp / median-gate pseudo-unary constants, the marginal-uplift thresholds,
the median-gate closed-form replay, and the one-best-per-pair selector."""
from __future__ import annotations

from typing import Sequence

import numpy as np


# MARGINAL-UPLIFT alternative acceptance for the per-pair engineered winner. The
# primary joint-prevalence gate (``best_mi / pair_mi > fe_min_engineered_mi_prevalence``)
# rejects a genuine signal pair whenever the 2-D JOINT MI is finite-sample-inflated
# ABOVE what any 1-D engineered summary can retain -- and that inflation is severe when
# the target is discretised into many bins (the adaptive ``categorize_dataset`` routinely
# produces 25-30 target bins) while the engineered column is binned to ``quantization_nbins``
# (~10). On the canonical ``y = a**2/b + log(c)*sin(d)`` fixture the genuine ``a**2/b`` pair
# recovers only 0.83-0.97 of its inflated joint and is dropped, leaving a single engineered
# feature, and on the uniform-input variant a cross-pair artefact wins instead of the true
# ``mul(log(c),sin(d))``. A 1-D summary of a 2-D pair structurally cannot retain >~0.90 of
# the inflated joint, so requiring 0.97 asks the impossible for genuine algebraic recovery.
#
# The MARGINAL-UPLIFT criterion is scale-robust: a genuine joint pair's engineered column
# clears the LARGER individual-operand marginal MI by a wide margin (the synergy is real),
# whereas a cross-pair artefact -- whose engineered form mostly recapitulates one operand's
# marginal plus noise -- barely beats it. Measured on both fixtures the genuine pairs sit at
# uplift 1.36-2.32 while the artefacts sit at 1.03-1.44 BUT only ever with a LOW joint-recovery
# ratio; pairing the uplift bar with a relaxed joint floor admits the genuine pairs and keeps
# the artefacts out. This is the SAME notion the prewarp acceptance path and the smart_polynom
# baseline-uplift use ("engineered MI must beat the larger operand marginal"); noise pairs never
# clear it (the upstream pair screen + order-2 maxT floor already removed structureless pairs).
_FE_MARGINAL_UPLIFT_MIN_RATIO: float = 1.30
# Relaxed joint-recovery floor that pairs with the marginal-uplift bar. A 1-D engineered
# summary can recover ~0.82+ of even a heavily over-binned joint when the structure is genuine;
# cross-pair artefacts that DO clear the uplift bar are gated out here (their joint recovery is
# either much lower or -- when high -- comes with a marginal uplift that fails the bar above).
_FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO: float = 0.82
# HW-ROBUST TWO-TIER joint-recovery floor (2026-06-08 regression fix). The single 0.82 floor
# sat a razor-thin 0.006 above a measured CROSS-SIGNAL artefact (``sub(exp(a),invcbrt(c))`` for
# ``y=a**2/b+log(c)*sin(d)`` recovers joint_ratio 0.8141), so a ~1e-3 GPU-vs-CPU MI divergence on
# the user's RTX flipped the gate and admitted a spurious feature the 1050-Ti CPU/GPU paths reject
# -- a HW-DEPENDENT selection divergence (the genuine pairs sit at 0.829 / 0.849, the artefact at
# 0.814, only a 0.015 gap, and the marginal-UPLIFT axis does NOT separate them: the artefact's
# uplift 1.441 sits ABOVE one genuine pair's 1.378). The fix needs a DETERMINISTIC margin wider
# than the cross-HW MI noise. A genuine SAME-SIGNAL pair is one of: (a) high joint recovery
# (>= STRICT floor) on its own, OR (b) a CLEAR-synergy pair (uplift >= the synergy threshold) that
# clears the relaxed base floor. The cross-signal artefact clears NEITHER -- its joint recovery is
# below the strict floor AND its uplift is below the synergy threshold -- so a small MI perturbation
# on either axis cannot flip it. Measured separations: genuine (0,1) uplift 2.32 (synergy branch,
# joint 0.829 vs base 0.82 -> +0.009), genuine (2,3) joint 0.849 (strict branch, vs 0.84 -> +0.009),
# artefact joint 0.814 (< 0.84 by 0.026) AND uplift 1.441 (< 2.0 by 0.56) -- both margins >> 0.006.
_FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO: float = 0.84
_FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT: float = 2.0

# Pseudo-unary name for the per-operand learned pre-warp (2026-06-02). Lives in
# the same namespace as the real unary names (``identity``, ``sqr``, ...) so it
# flows through combination generation, naming, and survivor packing untouched;
# materialisation and recipe construction special-case it via this constant.
_PREWARP_UNARY = "prewarp"

# Reserved (never-colliding, length-3) key under which ``check_prospective_fe_pairs``
# returns the fitted pre-warp specs inside its result dict; real result keys are
# ``raw_vars_pair`` 2-tuples. Lets the caller recover specs across the loky path.
_PREWARP_SPECS_RESULT_KEY = ("__prewarp_specs__", -1, -1)

# MEDIAN-GATE pseudo-unary (2026-06-04). Mirrors ``_PREWARP_UNARY`` exactly: it
# lives in the same namespace as the real unary names (``identity``, ``sqr``, ...)
# so it flows through combination generation, naming, and survivor packing
# untouched; materialisation and recipe construction special-case it via this
# constant. ``gate_med(x) = (x > train_median_x).astype(float)`` -- a STATEFUL
# pseudo-unary whose only fitted state is one float (the TRAIN median of the
# operand). Combined with the existing ``mul`` binary it expresses the
# median-gated operators ``(a > median_a) * b`` (gated_med) and
# ``(a > median_a) & (b > median_b)`` -> via ``mul(gate_med(a), gate_med(b))``
# (thr_and_med). Measured (skew bench) gated_med +0.0355 / thr_and_med +0.0435
# downstream-AUC d_mean vs raw, beating plain products (+0.022/+0.020) and
# threshold-0 gates (+0.009/+0.0001) -- the median adapts the split to each
# operand's distribution. The fit is a single ``np.median`` (no overfit risk,
# no held-out validation needed), the produced column is a closed-form function
# of ``x`` + the stored median, so replay is leak-safe; the gate still passes
# every existing FE prevalence / MI acceptance gate like any engineered feature.
_GATE_MED_UNARY = "gate_med"

# Reserved (never-colliding, length-3) key for the fitted median specs in the
# result dict (loky-parallel recovery path). Distinct from the prewarp key.
_GATE_MED_SPECS_RESULT_KEY = ("__gate_med_specs__", -1, -1)

# Reserved (never-colliding, length-3) key under which ``check_prospective_fe_pairs``
# returns the PER-GATE REJECTION LEDGER records (additive, 2026-06-11). A list of small
# dicts, one per pair the per-pair acceptance gate dropped. Survives the loky-parallel
# path (per-chunk lists merged by the caller). Distinct from the prewarp / gate-med keys.
_FE_REJECTION_RESULT_KEY = ("__fe_rejection_records__", -1, -1)


def _occupied_k(codes: np.ndarray) -> int:
    """Number of OCCUPIED (non-empty) ordinal bins in a 1-D code array -- the same
    ``k = #{bins with count>0}`` :func:`entropy_miller_madow` counts internally
    (backlog #4). Heavy-tailed engineered columns collapse to a few occupied bins,
    so the nominal ``nbins`` over-states the cardinality and over-corrects the MM
    bias term; the occupied count is the cardinality the plug-in MI actually sees."""
    arr = np.asarray(codes)
    if arr.size == 0:
        return 0
    return int(np.unique(arr).size)


def mm_debiased_prevalence_ratio(
    best_mi: float,
    pair_mi: float,
    *,
    k_eng: int,
    k_joint: int,
    k_y: int,
    n: int,
) -> float:
    """Miller-Madow-debiased ``best_mi / pair_mi`` for the FE joint-prevalence gate
    (backlog #1 + #4).

    The numerator ``best_mi`` is the 1-D engineered MI over ``k_eng`` occupied bins;
    the denominator ``pair_mi`` is the 2-D joint MI over ``k_joint`` occupied bins
    (``~nbins^2``). Both are RAW plug-in MIs and both carry the positive bias
    ``(k_x-1)(k_y-1)/2n``, but the JOINT denominator's term is ~``nbins``x larger, so
    the raw ratio is structurally depressed below 1.0 even when the 1-D feature
    captures all the joint information (worst at small/moderate ``n``). We subtract
    the Miller-Madow MI bias term from EACH side (occupied-K per backlog #4) and take
    the corrected ratio.

    DENOMINATOR-POSITIVITY GUARD (the #1 stability risk): the joint bias term can
    exceed the small finite-sample ``pair_mi`` at small ``n`` / high ``k_joint``,
    driving the corrected denominator to <=0 (ratio sign-flips / explodes). When the
    corrected denominator is not safely positive we FALL BACK to the raw plug-in ratio
    -- never admit on a degenerate correction. ``->`` raw ratio as ``n -> inf`` (the
    bias terms vanish), so large-n selection is byte-untouched.
    """
    from ..info_theory import mi_miller_madow_correct

    if pair_mi <= 0.0:
        return 0.0
    raw_ratio = best_mi / pair_mi
    if n <= 0 or k_y <= 1:
        return raw_ratio
    num_mm = mi_miller_madow_correct(best_mi, k_eng, k_y, n)
    den_mm = mi_miller_madow_correct(pair_mi, k_joint, k_y, n)
    # Guard: a non-positive (or vanishing) corrected denominator means the joint
    # bias term swamped the finite-sample joint MI -- the correction is unreliable
    # here, so defer to the raw plug-in ratio (the existing gate behaviour).
    if den_mm <= 1e-9 * max(pair_mi, 1.0):
        return raw_ratio
    # A debiased numerator can go slightly negative for a near-zero engineered MI;
    # clamp at 0 so the ratio cannot turn negative and spuriously "pass" / "fail".
    if num_mm < 0.0:
        num_mm = 0.0
    return num_mm / den_mm


def _gate_med_apply(vals: np.ndarray, median: float) -> np.ndarray:
    """Replay the median-gate pseudo-unary closed-form: ``(x > train_median).astype(float)``.

    NaN inputs (``x > median`` is False for NaN) collapse to 0.0, matching the
    downstream nan_to_num scrubbing. Returns float64 so the binary combine + MI
    discretisation see a continuous-typed 0/1 column (same dtype contract as the
    real unaries). The ``median`` is the TRAIN-fitted float; passing it in keeps
    this function pure (no y, no recompute) -> leak-safe at transform time."""
    return (np.asarray(vals, dtype=np.float64) > float(median)).astype(np.float64)


def _select_single_best(perf: dict, cols_names: Sequence, secondary: dict | None = None,
                        usability: dict | None = None):
    """Pick ONE winning ``config`` from a ``{config: mi}`` mapping.

    Selection key, in priority order:
      1. PRIMARY: maximum ``perf[config]`` -- this MUST be the engineered
         feature's MI WITH THE TARGET. (Regression guard: a prior version
         passed the external-validation score -- MI of the candidate recombined
         with an unrelated third factor -- as the primary key here, so the
         search would discard the true max-target-MI form. e.g. on
         y = log(c)*sin(d) it picked add(log(c),1/d) at MI=0.25 over the true
         mul(log(c),sin(d)) at MI=0.32. Primary MUST be target MI.)
      2. LINEAR-USABILITY (optional tie-break): maximum ``usability[config]`` --
         |corr(continuous engineered values, continuous y)|. Decisive among
         leaders whose target MI is EQUAL: MI is a RANK statistic, blind to
         linear usability, so a raw pair's equivalence class can hold forms with
         IDENTICAL MI but wildly different linear usability -- e.g. on a
         ``y=1.5*a*b`` bilinear target the forms ``mul(a,b)``, ``log(a)+log(b)``
         and ``1/(a**2*b**2)`` are ALL strictly-monotone functions of ``a*b`` so
         their binned MI is bit-identical (0.4561), yet |corr(y)| is 0.76 / 0.61 /
         0.004 respectively. Prefer the linearly-usable leg (``mul``) so a linear
         downstream recovers the magnitude; trees are indifferent (rank-equal).
         (This is exactly the case the 2026-06-03 |corr| experiment MISSED: it
         tested ratio variants (``a2/b == sqr(a)/b``) that ARE linearly equivalent
         within the band, concluding "no gain" -- but DISTINCT monotone warps with
         equal MI and different |corr| are common, the bilinear product being the
         canonical one. The tie-break is gated on EQUAL MI, so it never overrides
         a genuinely higher-MI form -> no regression to the MI-primary contract.)
      3. SECONDARY (optional tie-break): maximum ``secondary[config]`` -- the
         external-validation MI. Decisive among leaders tied on MI AND usability;
         prefers the representation that also generalises against other factors.
      4. deterministic tie-break by the engineered feature name (ascending).

    Used to collapse the leading-features equivalence class (many near-identical
    representations of the same algebraic target) down to a single representative
    per raw pair, restoring the pre-refactor 1-per-pair materialisation. Returns
    ``None`` when ``perf`` is empty.
    """
    if not perf:
        return None
    # Lazy import (parent re-imports this module at its bottom -> avoid a
    # top-level cycle); mirrors the in-function import at the call sites.
    from ..feature_engineering import get_new_feature_name
    _sec = secondary or {}
    _use = usability or {}
    return max(
        perf.items(),
        key=lambda kv: (
            kv[1],
            float(_use.get(kv[0], 0.0)),
            float(_sec.get(kv[0], 0.0)),
            _neg_name_key(get_new_feature_name(fe_tuple=kv[0], cols_names=cols_names)),
        ),
    )[0]


class _neg_name_key:
    """Reverse-ordering wrapper so a HIGHER MI still wins but, on an MI tie, the
    lexicographically-SMALLEST name wins (we negate the name comparison)."""

    __slots__ = ("s",)

    def __init__(self, s: str):
        self.s = s

    def __lt__(self, other: "_neg_name_key") -> bool:
        # Inverted: smaller string sorts as "greater" so max() prefers it.
        return self.s > other.s

    def __eq__(self, other) -> bool:  # pragma: no cover - completeness
        return isinstance(other, _neg_name_key) and self.s == other.s
