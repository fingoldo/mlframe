"""Signed interaction-information (co-information) ranking & routing for the MRMR FE pair gate (backlog idea #8).

The prospective-pair gate in :mod:`_mrmr_fe_step` admits an engineered pair ``(a, b)`` when its JOINT MI clears
``ind_elems_mi_sum * prevalence`` (and the order-2 maxT floor). That ratio gate conflates two very different
situations under one "the joint beats the marginal sum" test:

  * **genuine synergy** -- ``(a, b)`` JOINTLY carry information about ``y`` that neither marginal does
    (``y = a**2 / b``, ``y = log(c) * sin(d)``, XOR / sign products). Knowing one operand does not let you
    decode ``y`` from the other; the joint is strictly more than the parts.
  * **additive completion** -- ``a`` feeds one independent additive term of ``y`` and ``b`` a DIFFERENT one
    (the user's weak-F2 CROSS-MIX: ``a`` from ``a**2/b`` term + ``c`` from the ``log*sin`` term). The two do
    not interact in ``y`` (``II ~= 0``); the pair's joint MI is high only because it pools two unrelated signals,
    and the per-pair search then fabricates a SPURIOUS cross-mix surrogate (``add(invqubed(a), invsqrt(c))``).

The signed **interaction information** ``II(a;b;y) = I((a,b);y) - I(a;y) - I(b;y)`` is exactly the quantity that
separates them: ``> 0`` genuine synergy, ``~= 0`` additive (no interaction), ``< 0`` redundancy (``a`` and ``b``
carry the SAME signal). Every term is ALREADY computed by the gate -- ``cached_MIs[(a,)]`` / ``cached_MIs[(b,)]``
are the marginals, ``pair_mi`` is the joint -- so II is a near-free signed re-read.

Two iron rules from the backlog (preserved here):

  (a) **Miller-Madow correct all THREE terms before differencing.** Plug-in MI bias ~ ``(Kx-1)(Ky-1)/2n`` grows
      with cardinality, and the JOINT term has ``Kx = nbins_a * nbins_b`` bins -- ~``nbins``x the marginal bias.
      Differencing two un-corrected scales would manufacture a positive II out of pure finite-sample inflation.
      So each term is corrected as ``mi_mm = mi_plugin - (Kx - 1)(Ky - 1) / (2n)`` (the SAME formula the order-1
      maxT floor uses, see :func:`_permutation_null.pooled_permutation_null_gain_floor`), THEN II is the signed
      difference of the corrected terms.

  (b) **Floor positive II via a permutation null.** A deterministic / low-noise SUM (``y = a + c + noise``) still
      produces a *small positive* II (a "completion" synergy: knowing one summand helps decode the other from y),
      so an ``II > 0`` test alone does NOT cleanly reject the additive cross-mix. The permutation null on the MAX
      positive II over the pool (shuffle ``y`` K times, q-quantile of the per-shuffle max II) is the chance ceiling
      for additive-completion + finite-sample II; a genuine multiplicative synergy sits FAR above it, the additive
      cross-mix sits below. :func:`pooled_pair_ii_null_floor` computes it on the SAME estimator scale as the gate.

This module changes only RANKING / ROUTING, never the detection floor (iron rule (d)): the order-2 maxT floor and
the prevalence ratio gate stay as the outer admission guards. II routing then, among the pairs that ALREADY passed,
(i) demotes additive pairs (``II <= floor``) out of the synergy FE search so no cross-mix surrogate is built, and
(ii) tags negative-II pairs for cluster-aggregate / positive-II pairs for product-cross-basis FE.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Routing tags threaded onto a prospective pair by signed interaction information.
ROUTE_SYNERGY = "synergy"  # II > floor: genuine joint signal -> product / cross-basis FE (keep, rank high)
ROUTE_ADDITIVE = "additive"  # 0 <= II <= floor: no interaction -> demote out of the synergy FE search (cross-mix)
ROUTE_REDUNDANT = "redundant"  # II < 0: a and b carry the same signal -> cluster-aggregate / denoise candidate


def _mm_bias(k_x: int, k_y: int, n: int) -> float:
    """Miller-Madow plug-in MI bias ``(Kx - 1)(Ky - 1) / (2n)`` (nats). ``>= 0``; ``-> 0`` as ``n -> inf``."""
    if n <= 0 or k_x < 2 or k_y < 2:
        return 0.0
    return (k_x - 1) * (k_y - 1) / (2.0 * n)


def pair_interaction_information(
    mi_a: float,
    mi_b: float,
    pair_mi: float,
    nbins_a: int,
    nbins_b: int,
    nbins_y: int,
    n: int,
    *,
    miller_madow: bool = True,
    k_a_occupied: int | None = None,
    k_b_occupied: int | None = None,
    k_joint_occupied: int | None = None,
    k_y_occupied: int | None = None,
) -> float:
    """Signed interaction information ``II(a;b;y) = I((a,b);y) - I(a;y) - I(b;y)`` (nats).

    All three MIs are the already-computed plug-in values (``mi_a`` / ``mi_b`` = ``cached_MIs[(a,)]`` /
    ``cached_MIs[(b,)]``, ``pair_mi`` = the joint). When ``miller_madow`` (default) each term is MM-corrected
    on its OWN cardinality before the difference. The MM bias term must be computed on the OCCUPIED (non-empty)
    bin counts, NOT the design cardinality -- this is the same ``k = #{bins with count>0}`` convention
    :func:`info_theory._entropy_kernels.mi_miller_madow_correct` / :func:`entropy_miller_madow` use. The joint
    term is the trap: its DESIGN cardinality is ``nbins_a * nbins_b``, but on a heavy-tailed / sparse column the
    actually-occupied joint cells are far fewer. Subtracting ``(nbins_a*nbins_b - 1)(nbins_y - 1)/(2n)`` then
    OVER-corrects the joint by exactly the empty-cell count, pushing ``pair_mi`` down too far and -- because the
    joint term enters II with a ``+`` sign -- manufacturing a deterministic UPWARD synergy offset on independent
    heavy-tailed pairs (false synergy). Callers with access to the code arrays should pass the occupied counts
    (``k_*_occupied``); when omitted the design cardinality is used as a fallback (correct only for dense joints).
    ``> 0`` synergy, ``~= 0`` additive, ``< 0`` redundancy.
    """
    if miller_madow:
        k_a = int(k_a_occupied) if k_a_occupied is not None else int(nbins_a)
        k_b = int(k_b_occupied) if k_b_occupied is not None else int(nbins_b)
        k_j = int(k_joint_occupied) if k_joint_occupied is not None else int(nbins_a) * int(nbins_b)
        k_y = int(k_y_occupied) if k_y_occupied is not None else int(nbins_y)
        mi_a = mi_a - _mm_bias(k_a, k_y, int(n))
        mi_b = mi_b - _mm_bias(k_b, k_y, int(n))
        pair_mi = pair_mi - _mm_bias(k_j, k_y, int(n))
    return float(pair_mi - mi_a - mi_b)


def pooled_pair_ii_null_floor(
    factors_data: np.ndarray,
    nbins: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    marginal_mi_a: np.ndarray,
    marginal_mi_b: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    *,
    n_permutations: int = 25,
    quantile: float = 0.95,
    miller_madow: bool = True,
    random_seed: int | None = None,
) -> float:
    """Permutation-null floor on the MAX positive interaction information over a prospective-pair pool.

    The chance ceiling for a *positive* II is NOT zero: a deterministic / low-noise additive target
    (``y = a + c + noise``) yields a small positive "completion" II, and finite-sample plug-in bias adds more,
    so an ``II > 0`` test admits the additive cross-mix the routing is meant to demote. This null measures that
    ceiling on the SAME estimator scale: shuffle ``classes_y`` K times (destroying every X-y dependency while
    preserving each column's marginal AND the pool size), and for each shuffle record the MAX MM-corrected II
    over the WHOLE pair pool. The marginal MIs under the shuffled target are recomputed per shuffle from a
    single-pass joint histogram (the joint MI uses the SAME ``a * nbins_b + b`` encoding as
    :func:`batch_pair_mi_prange`, so floor and gated value are on one scale). The ``quantile``-th quantile of
    those K maxes is the floor a genuine multiplicative synergy clears and additive completion does not.

    ``marginal_mi_a`` / ``marginal_mi_b`` are unused under the null (recomputed under each shuffle) but kept in
    the signature so the caller passes the same arrays it gates with -- documents the parallel to the observed
    path. Returns ``0.0`` (no-op floor) on a degenerate pool (n too small, < 2 pairs, single-class target, or no
    permutations requested), so the caller can unconditionally compare ``ii > floor``.

    Perf (cProfile ``bench_interaction_information_routing.py``, n=30000/p=25/K=25): ~1.8 s, dominated by the
    Python-level per-(shuffle, pair) :func:`_marginal_mi_codes` re-score (1.0 s of 1.9 s). Acceptable for the
    OPT-IN router (this whole path is default-off, see ``mrmr.py`` ``fe_ii_routing_enable``); were it ever made
    default-on it should move to a numba ``prange`` kernel mirroring :func:`batch_pair_mi_prange` (which the
    order-2 joint-MI floor already uses) -- left un-numba'd deliberately while opt-in to keep it readable.
    """
    n = int(factors_data.shape[0])
    n_pairs = int(np.asarray(pair_a).shape[0])
    k_y = int(np.asarray(freqs_y).shape[0])
    if n < 8 or n_permutations < 1 or n_pairs < 2 or k_y < 2:
        return 0.0

    pa = np.ascontiguousarray(pair_a, dtype=np.int64)
    pb = np.ascontiguousarray(pair_b, dtype=np.int64)
    nb = np.ascontiguousarray(nbins).astype(np.int64)
    y0 = np.ascontiguousarray(classes_y).astype(np.int64)

    # Per-column invariants: marginal entropy H(x) and bin count are permutation-invariant; only the joint
    # with the shuffled y changes. Precompute the columns referenced by the pool once.
    cols = np.unique(np.concatenate((pa, pb)))
    inv_n = 1.0 / n
    col_codes: dict[int, np.ndarray] = {}
    for c in cols:
        ci = int(c)
        col_codes[ci] = np.ascontiguousarray(factors_data[:, ci]).astype(np.int64)

    rng = np.random.default_rng(random_seed)
    y_perm = y0.copy()
    maxes = np.empty(int(n_permutations), dtype=np.float64)
    for s in range(int(n_permutations)):
        rng.shuffle(y_perm)
        # y marginal counts are invariant under shuffle; entropy/H(y) cancels in MI so we score MI directly.
        best = 0.0
        # Cache per-column marginal MI under THIS shuffle (each column appears in many pairs).
        mi_cache: dict[int, tuple[float, int]] = {}
        for p in range(n_pairs):
            ia, ib = int(pa[p]), int(pb[p])
            ca, cb = col_codes[ia], col_codes[ib]
            nb_a, nb_b = int(nb[ia]), int(nb[ib])
            cached_a = mi_cache.get(ia)
            if cached_a is None:
                cached_a = _marginal_mi_codes(ca, y_perm, nb_a, k_y, inv_n)
                mi_cache[ia] = cached_a
            mi_a, k_a_occ = cached_a
            cached_b = mi_cache.get(ib)
            if cached_b is None:
                cached_b = _marginal_mi_codes(cb, y_perm, nb_b, k_y, inv_n)
                mi_cache[ib] = cached_b
            mi_b, k_b_occ = cached_b
            joint = ca * nb_b + cb
            pair_mi, k_j_occ = _marginal_mi_codes(joint, y_perm, nb_a * nb_b, k_y, inv_n)
            ii = pair_interaction_information(
                mi_a, mi_b, pair_mi, nb_a, nb_b, k_y, n, miller_madow=miller_madow,
                k_a_occupied=k_a_occ, k_b_occupied=k_b_occ, k_joint_occupied=k_j_occ,
            )
            if ii > best:
                best = ii
        maxes[s] = best

    return float(np.quantile(maxes, float(quantile)))


def _marginal_mi_codes(x_codes: np.ndarray, y_codes: np.ndarray, k_x: int, k_y: int, inv_n: float) -> tuple[float, int]:
    """Plug-in MI ``I(X;Y)`` (nats) + OCCUPIED X-bin count from ordinal code arrays via a single joint-histogram pass.

    Mirrors the joint-MI accumulation in :func:`batch_pair_mi_prange` (``jf * log(jf / (px * py))``), so the
    null floor's marginal/joint terms are on the exact same plug-in scale as the gated values. The second return
    value is ``#{X-bins with count>0}`` -- the occupied cardinality the Miller-Madow bias must be computed on (the
    design ``k_x`` over-corrects sparse / heavy-tailed columns, see :func:`pair_interaction_information`).
    """
    joint = x_codes.astype(np.int64) * k_y + y_codes.astype(np.int64)
    counts = np.bincount(joint, minlength=k_x * k_y).astype(np.float64)
    counts = counts.reshape(k_x, k_y)
    px = counts.sum(axis=1) * inv_n
    py = counts.sum(axis=0) * inv_n
    k_x_occupied = int((px > 0).sum())
    pj = counts * inv_n
    mask = pj > 0
    if not mask.any():
        return 0.0, k_x_occupied
    pj_m = pj[mask]
    # broadcast px (rows) and py (cols) over the masked joint cells
    rows, colz = np.nonzero(mask)
    denom = px[rows] * py[colz]
    good = denom > 0
    return float(np.sum(pj_m[good] * np.log(pj_m[good] / denom[good]))), k_x_occupied


def route_prospective_pairs(
    prospective_pairs: dict,
    *,
    cached_MIs: dict,
    nbins: np.ndarray,
    nbins_y: int,
    n: int,
    ii_floor: float,
    synergy_added_idx: set | None = None,
    miller_madow: bool = True,
    verbose: int = 0,
) -> tuple[dict, dict, dict]:
    """Tag every prospective pair by signed interaction information and DEMOTE the additive cross-mix pairs.

    ``prospective_pairs`` is the gate's survivor dict keyed by ``((var_a, var_b), pair_mi)``. For each pair we
    read the already-cached marginals + joint, compute MM-corrected signed II, and assign a route:

      * ``II > ii_floor``           -> ``synergy``   (genuine joint signal; kept, eligible for product/cross-basis)
      * ``ii_floor >= II >= -eps``  -> ``additive``  (no interaction; DEMOTED -- removed from the returned dict so
                                                      the per-pair FE search never builds a cross-mix surrogate)
      * ``II < -eps``               -> ``redundant`` (a,b carry the same signal; kept + tagged for cluster-aggregate)

    Only SYNERGY-ADDED (speculative bootstrap) additive pairs are demoted -- a selected-selected additive pair is
    a legitimately strong pair the user already wants, so it is kept (tagged ``additive``) to preserve the
    create/keep/drop contract; the demotion targets the cross-mix the synergy bootstrap manufactures.

    Returns ``(kept_pairs, routes, ii_values)``: the (possibly trimmed) prospective-pairs dict, a
    ``{(var_a, var_b): route}`` map, and a ``{(var_a, var_b): ii}`` map (both keyed by the raw var tuple) for
    provenance / downstream routing. ``ii_floor <= 0`` or an empty dict makes this a structural no-op (every pair
    kept, every route ``synergy``) so the path stays byte-stable where the floor is not informative.
    """
    routes: dict = {}
    ii_values: dict = {}
    if not prospective_pairs:
        return prospective_pairs, routes, ii_values

    syn = synergy_added_idx or set()
    eps = 1e-9
    kept: dict = {}
    n_demoted = 0
    for key, sort_val in prospective_pairs.items():
        raw_vars_pair, pair_mi = key
        va, vb = raw_vars_pair
        # INFO_THEORY_B-10 fix: a missing marginal-MI cache entry silently
        # substituted 0.0, which INFLATES the interaction-information score (pair_mi - 0 - mi_b) and can
        # mis-route a pair to "synergy" instead of surfacing a real upstream caching defect. Log so the
        # two cases (genuinely-zero marginal vs. missing cache entry) are distinguishable.
        if (va,) not in cached_MIs:
            logger.debug("mrmr: interaction-information routing found no cached marginal MI for column %r; treating as 0.0.", va)
        if (vb,) not in cached_MIs:
            logger.debug("mrmr: interaction-information routing found no cached marginal MI for column %r; treating as 0.0.", vb)
        mi_a = float(cached_MIs.get((va,), 0.0))
        mi_b = float(cached_MIs.get((vb,), 0.0))
        nb_a = int(nbins[va])
        nb_b = int(nbins[vb])
        ii = pair_interaction_information(
            mi_a, mi_b, float(pair_mi), nb_a, nb_b, int(nbins_y), int(n), miller_madow=miller_madow,
        )
        ii_values[raw_vars_pair] = ii
        if ii < -eps:
            route = ROUTE_REDUNDANT
        elif ii > ii_floor:
            route = ROUTE_SYNERGY
        else:
            route = ROUTE_ADDITIVE
        routes[raw_vars_pair] = route

        # Demote additive cross-mix pairs: only the SPECULATIVE (synergy-added) ones, only when the floor is
        # informative (> 0). A selected-selected pair, or any pair when the floor is a no-op, is kept verbatim.
        _is_speculative = bool(syn) and (raw_vars_pair[0] in syn or raw_vars_pair[1] in syn)
        if route == ROUTE_ADDITIVE and _is_speculative and ii_floor > 0.0:
            n_demoted += 1
            continue
        kept[key] = sort_val

    if verbose >= 1 and n_demoted:
        logger.info(
            "MRMR FE II-routing: demoted %d additive (interaction-information <= null floor=%.5f) speculative "
            "cross-mix pair(s) from the FE search; kept %d (synergy/redundant/selected).",
            n_demoted, ii_floor, len(kept),
        )
    return kept, routes, ii_values
