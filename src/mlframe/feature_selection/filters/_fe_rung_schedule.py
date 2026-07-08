"""Successive-halving / rung-schedule budget for the FE operator search (backlog #16).

Routes the expensive per-pair operator search (``check_prospective_fe_pairs`` --
all unary x binary transforms, CMA-ES / full discretize / prewarp per pair, ~4-50s
per pair) via a CHEAP rung-0 SCREEN so the search goes DEEPER at equal wall-time.

Mechanism (single halving, generalises ``_cat_confirm_bandit``'s budget-allocation idea
from cat-pair permutation confirmation to numeric operator-search budget)
----------------------------------------------------------------------------------
The flat top-K sweep feeds EVERY gate-surviving prospective pair into the expensive
operator search. This module inserts a rung-0 low-fidelity screen FIRST:

  RUNG 0 (zero extra cost): rank the prospective pairs by their JOINT MI ``pair_mi``
    -- the exact monotone-ish proxy the backlog calls for, and ALREADY computed by the
    pair-MI gate (``cached_MIs``), so the screen is FREE. Keep the top ``keep_frac``
    fraction UNION every pair whose ``pair_mi >= rel_floor * max_pair_mi`` (the relative
    floor protects a moderate-MI genuine winner from a too-aggressive fractional cut --
    see the no-drop benchmark below).
  RUNG 1: the unchanged expensive ``check_prospective_fe_pairs`` runs ONLY on the rung-0
    survivors.

Why ``pair_mi`` is a SAFE proxy (the binding correctness check)
--------------------------------------------------------------
The operator search's best engineered MI is bounded by + correlated with the pair's
joint MI: a pair whose 2-D joint MI is near the noise floor cannot yield an engineered
1-D feature with high MI(engineered; y). So a pair the full search would keep as a
survivor has high ``pair_mi`` -- it sits at the TOP of the rung-0 ranking, never near
the cut. Measured (n=5000, p=40, canonical ``a**2/b + log(c)*sin(d)`` fixture + noise,
5 seeds, relaxed-gate WIDE pools of 7-51 pairs):

  * Spearman(pair_mi, full-fidelity survival) = 0.66.
  * the genuine signal pairs (a,b)/(a,c)/(b,c) rank #1-2 by pair_mi in EVERY seed and
    are NEVER dropped by the rung-0 cut; only spurious ``(c, noise)`` over-permissive
    survivors are cut (a denoising bonus, not a loss).
  * speedup vs the flat sweep: 1.7-2.2x at keep_frac=0.5, 3.3-11.2x at keep_frac=0.25.
  * degenerate all-noise pool (max_pair_mi ~ 0.07, no clear top): the relative floor
    keeps EVERY pair -> 1.0x, 0 dropped (no false cut where there is no gradient to
    exploit -- correctness preserved, the rung only saves time when a pair_mi gradient
    exists).

GATES UNTOUCHED. This changes WHERE the operator-search compute goes (only the top
pair_mi pairs get the expensive search), NOT admission: every kept pair still passes
the same engineered-MI / external-validation / stability-vote gates downstream. A pair
the rung-0 screen drops simply never gets the operator search -- exactly as if it had
ranked below the (already-existing) ``fe_synergy_max_pairs`` budget for synergy pairs;
this module GENERALISES that per-pair budget to the whole prospective pool.

Equal-wall deeper search
------------------------
Because rung-0 is free and rung-1 runs on a fraction of the pool, the SAME wall-time
budget can feed a LARGER input pool into rung-0 (raise ``fe_ntop_features`` /
``fe_synergy_max_pairs``) -> a genuine needle ranked outside a hard flat top-K budget
is reached by the cheap rung-0 screen and only then operator-searched. See
``tests/feature_selection/test_fe_rung_schedule.py::test_biz_value_equal_wall_deeper_needle``.

Self-gating
-----------
No-op (byte-identical flat sweep) when the pool has < ``min_pairs`` prospective pairs
(nothing meaningful to halve) or ``keep_frac >= 1.0``. The rung fraction is dispatched
per (n_rows, n_pairs) through ``pyutilz.performance.kernel_tuning.cache`` (the iron
rule -- never hardcode a single fraction across all hardware / data shapes); a
measurement-backed fallback is used until the cache is populated.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ACCURACY-SAFE DEFAULT (2026-06-16). The rung-0 screen ranks gate-survivor pairs by their
# CHEAP joint pair_mi and cuts the bottom (1 - keep_frac). That cut is FUNDAMENTALLY UNSAFE for
# a genuine-but-WEAK interaction pair: a low-marginal second signal (the canonical log(c)*sin(d)
# needle) has a joint pair_mi that is a small FRACTION of a co-present dominant pair (a**2/b), so
# it sits near the bottom of the pair_mi ranking AND below rel_floor * max_pair_mi -> it is cut
# before the operator search and the genuine mul(log(c),sin(d)) never forms. The cheap screen
# cannot distinguish weak-genuine from weak-noise WITHOUT the expensive operator search it is
# trying to avoid, so any fractional cut < 1.0 risks dropping a real winner the flat sweep keeps
# (measured: rung-on drops mul(log(c),sin(d)) at n=4000 AND n=25000 on the canonical fixture).
# Per the project rule that accuracy must never regress for a speed lever, the DEFAULT is now
# no-drop (keep_frac=1.0 -> the screen is a structural no-op, byte-identical to the flat sweep).
# The aggressive fractions below are still measurement-backed (Intel host, 2026-06-10; n=5000,
# p=40, 5 seeds: ~2x at 0.5, 3-11x at 0.25) but are now OPT-IN -- a caller takes the speedup by
# setting ``fe_rung_keep_frac`` (or MLFRAME_FE_RUNG_KEEP_FRAC, or an offline kernel_tuning_cache
# entry) explicitly, accepting the weak-interaction tradeoff on their data. The joint-synergy
# screen of the planned numba FE re-platform is the path to a no-drop screen that is ALSO fast.
_OPTIN_KEEP_FRAC_BY_POOL: tuple[tuple[int, float], ...] = (
    # (min_pairs_in_pool, keep_frac) -- the recommended OPT-IN aggressive fractions; first
    # matching row from the TOP wins. Surfaced for callers / the offline sweep, NOT the default.
    (40, 0.34),  # large pool: signal concentrated at top, keep ~1/3
    (16, 0.50),  # moderate pool: single halving
    (0, 1.00),  # small pool: keep everything (no meaningful screen)
)

# Relative pair_mi floor: ALWAYS keep a pair whose pair_mi >= REL_FLOOR * max_pair_mi,
# regardless of its rank, so a moderate-MI genuine winner is never cut by the fractional
# rung. 0.40 was the binding value in the no-drop sweep: it protects (b,c) at pair_mi
# 0.153 vs max 0.341 (ratio 0.45) while still cutting the (c,noise) spurious survivors
# at pair_mi ~0.06 (ratio 0.17). Lower => more aggressive cut (risk of dropping a real
# winner); higher => safer but less speedup.
_REL_FLOOR_DEFAULT = 0.40

# Below this many prospective pairs the rung screen is a structural no-op (byte-identical
# flat sweep): with a handful of pairs the per-pair operator search is already cheap and
# there is no meaningful top-fraction to keep.
_MIN_PAIRS_DEFAULT = 6


def _fallback_keep_frac(n_pairs: int) -> float:
    """Accuracy-safe default keep-fraction (always 1.0, i.e. no drop) used when no explicit opt-in and no kernel_tuning_cache entry are available."""
    # Accuracy-safe default: no fractional cut (no-drop). The screen only prunes when a caller
    # opts into an aggressive keep_frac explicitly (fe_rung_keep_frac / env / tuned cache).
    return 1.0


def _optin_keep_frac(n_pairs: int) -> float:
    """The recommended aggressive keep_frac for a pool of ``n_pairs`` -- for callers that
    explicitly opt into the rung speedup (and the offline kernel_tuning_cache sweep). NOT used
    as the default; see ``_fallback_keep_frac`` and the module-level rationale."""
    for min_pairs, frac in _OPTIN_KEEP_FRAC_BY_POOL:
        if n_pairs >= min_pairs:
            return frac
    return 1.0


def _dispatch_keep_frac(n_rows: int, n_pairs: int, *, run_auto_tune: bool = False) -> float:
    """Per-host rung keep-fraction for a (n_rows, n_pairs) workload.

    Env override (``MLFRAME_FE_RUNG_KEEP_FRAC``) wins; then the kernel_tuning_cache;
    then the measurement-backed fallback. Mirrors ``dispatch_recursion_backend`` /
    ``_batch_pair_mi_backend_choice`` (the iron rule: never hardcode a single threshold
    across hardware / data shapes -- route through the per-host cache)."""
    forced = os.environ.get("MLFRAME_FE_RUNG_KEEP_FRAC", "").strip()
    if forced:
        try:
            v = float(forced)
            if 0.0 < v <= 1.0:
                return v
        except ValueError:
            pass
    fallback = _fallback_keep_frac(n_pairs)
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        result = KernelTuningCache.load_or_create().get_or_tune(
            "fe_rung_keep_frac",
            dims={"n_rows": int(n_rows), "n_pairs": int(n_pairs)},
            tuner=(lambda: None),  # no online sweep: correctness is floor-guaranteed, the
            # fraction only trades speed; populated offline if desired.
            axes=["n_rows", "n_pairs"],
            fallback={"keep_frac": fallback},
            code_version="rung_v1",
            async_sweep=True,
        )
        if isinstance(result, dict):
            kf = float(result.get("keep_frac", fallback))
            if 0.0 < kf <= 1.0:
                return kf
    except Exception as e:  # pyutilz missing / cache error -> fallback
        logger.debug("fe_rung keep_frac get_or_tune failed: %s", e)
    return fallback


def apply_rung_schedule(
    prospective_pairs: dict,
    *,
    n_rows: int,
    keep_frac: float | None = None,
    rel_floor: float = _REL_FLOOR_DEFAULT,
    min_pairs: int = _MIN_PAIRS_DEFAULT,
    verbose: int = 0,
) -> tuple[dict, dict]:
    """Rung-0 low-fidelity screen over the prospective pairs (backlog #16).

    ``prospective_pairs``: the gate-surviving, already-ranked pair dict from
    ``_mrmr_fe_step`` whose keys are ``(raw_vars_pair, pair_mi)`` 2-tuples (``pair_mi``
    is ``key[1]`` -- the joint MI the gate scored). Returns
    ``(kept_pairs, info)``:

    * ``kept_pairs`` -- a NEW dict (insertion order preserved from the input ranking)
      containing only the rung-0 survivors. The expensive operator search runs on this.
    * ``info`` -- ``{"applied", "n_in", "n_kept", "keep_frac", "rel_floor"}`` for logging
      / tests.

    No-op (returns the input dict unchanged, ``applied=False``) when there are fewer than
    ``min_pairs`` pairs, ``keep_frac >= 1.0``, or the pool has no positive pair_mi.
    """
    n_in = len(prospective_pairs)
    info = {"applied": False, "n_in": n_in, "n_kept": n_in, "keep_frac": 1.0, "rel_floor": rel_floor}
    if n_in < int(min_pairs):
        return prospective_pairs, info

    n_pairs = n_in
    if keep_frac is None:
        keep_frac = _dispatch_keep_frac(int(n_rows), int(n_pairs))
    info["keep_frac"] = float(keep_frac)
    if keep_frac >= 1.0:
        return prospective_pairs, info

    # pair_mi lives in key[1] (the dict is keyed by (raw_vars_pair, pair_mi)).
    def _pm(key):
        """Extract the pair MI (``key[1]``) used to rank/threshold pairs for the rung cut; returns 0.0 for a malformed key."""
        try:
            return float(key[1])
        except (TypeError, IndexError, ValueError):
            return 0.0

    keys = list(prospective_pairs.keys())
    pms = [_pm(k) for k in keys]
    max_pm = max(pms) if pms else 0.0
    if max_pm <= 0.0:
        # No positive joint-MI gradient to exploit (all-zero / XOR-only pool): keep all,
        # never cut blindly. Byte-identical flat sweep.
        return prospective_pairs, info

    ranked = sorted(keys, key=_pm, reverse=True)
    keep_n = max(1, round(n_pairs * keep_frac))
    floor_val = rel_floor * max_pm
    kept_set = set(ranked[:keep_n])
    # Relative-MI floor union: protect every moderate-or-better-MI pair from the cut.
    for k, pm in zip(keys, pms):
        if pm >= floor_val:
            kept_set.add(k)

    if len(kept_set) >= n_in:
        # Floor saved everyone -> no effective cut; report a no-op for clarity.
        info["n_kept"] = n_in
        return prospective_pairs, info

    # Preserve the caller's ranking order (insertion order of the input dict).
    kept = {k: prospective_pairs[k] for k in prospective_pairs if k in kept_set}
    info["applied"] = True
    info["n_kept"] = len(kept)
    if verbose:
        logger.info(
            "MRMR FE rung schedule (#16): rung-0 screen kept %d/%d prospective pairs "
            "(keep_frac=%.2f, rel_floor=%.2f, max_pair_mi=%.4f) -> expensive operator "
            "search runs on the survivors only; gates unchanged.",
            len(kept), n_in, keep_frac, rel_floor, max_pm,
        )
    return kept, info


__all__ = ["apply_rung_schedule", "_dispatch_keep_frac", "_fallback_keep_frac", "_optin_keep_frac"]
