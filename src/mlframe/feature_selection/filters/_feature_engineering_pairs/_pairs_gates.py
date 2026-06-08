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


def _gate_med_apply(vals: np.ndarray, median: float) -> np.ndarray:
    """Replay the median-gate pseudo-unary closed-form: ``(x > train_median).astype(float)``.

    NaN inputs (``x > median`` is False for NaN) collapse to 0.0, matching the
    downstream nan_to_num scrubbing. Returns float64 so the binary combine + MI
    discretisation see a continuous-typed 0/1 column (same dtype contract as the
    real unaries). The ``median`` is the TRAIN-fitted float; passing it in keeps
    this function pure (no y, no recompute) -> leak-safe at transform time."""
    return (np.asarray(vals, dtype=np.float64) > float(median)).astype(np.float64)


def _select_single_best(perf: dict, cols_names: Sequence, secondary: dict | None = None):
    """Pick ONE winning ``config`` from a ``{config: mi}`` mapping.

    Selection key, in priority order:
      1. PRIMARY: maximum ``perf[config]`` -- this MUST be the engineered
         feature's MI WITH THE TARGET. (Regression guard: a prior version
         passed the external-validation score -- MI of the candidate recombined
         with an unrelated third factor -- as the primary key here, so the
         search would discard the true max-target-MI form. e.g. on
         y = log(c)*sin(d) it picked add(log(c),1/d) at MI=0.25 over the true
         mul(log(c),sin(d)) at MI=0.32. Primary MUST be target MI.)
      2. SECONDARY (optional tie-break): maximum ``secondary[config]`` -- the
         external-validation MI. Only decisive among leaders whose target MI is
         exactly equal; prefers the representation that also generalises against
         other approved factors.
      3. deterministic tie-break by the engineered feature name (ascending).

    Used to collapse the leading-features equivalence class (many near-identical
    representations of the same algebraic target) down to a single representative
    per raw pair, restoring the pre-refactor 1-per-pair materialisation. Returns
    ``None`` when ``perf`` is empty.
    """
    if not perf:
        return None
    # measure-experiment-rejected (2026-06-03): a |corr| tie-break among the
    # MI-leading equivalence class (to prefer the most linearly-usable algebraic
    # form) was benchmarked and gives NO gain -- the forms within ~5% of the max
    # target MI are syntactic variants of the SAME function (a2/b == sqr(a)/b),
    # so their |corr| with y is identical (0/6 cases differed, OOS-Ridge delta
    # +0.000). Genuinely-different forms (a/b, a2/sqrt|b|) have DIFFERENT MI and
    # fall outside the band. The MI primary key + external-val secondary is right.
    # Lazy import (parent re-imports this module at its bottom -> avoid a
    # top-level cycle); mirrors the in-function import at the call sites.
    from ..feature_engineering import get_new_feature_name
    _sec = secondary or {}
    return max(
        perf.items(),
        key=lambda kv: (
            kv[1],
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
