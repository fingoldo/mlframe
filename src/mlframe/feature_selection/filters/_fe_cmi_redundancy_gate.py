"""Conditional-MI redundancy gate for engineered FE candidates (strategy S5).

The PRINCIPLED replacement for the hardcoded ``fe_min_engineered_mi_prevalence``
joint-prevalence ratio. Where the ratio gate asks "does the 1-D engineered
column retain >= X% of its operand-pair's 2-D joint MI?" (a constant the user
must hand-tune per dataset), THIS gate asks the constant-free, information-
theoretic question:

    Does the candidate engineered feature carry information about y that
    SURVIVES conditioning on the engineered features already admitted?

A spurious / redundant engineered column (e.g. a ``sub(exp(a), invcbrt(c))``
whose y-information is already wholly carried by the admitted ``div(sqr(a),
abs(b))`` and ``mul(log(c), sin(d))``) collapses to ~0 conditional MI and is
rejected. A genuine engineered column carrying a PRIVATE interaction term that
no admitted feature holds keeps a large CMI and is admitted.

Validated design (S5; won 10/10 vs four failing approaches across 16
(seed, formula) cells in ``D:/Temp/prevalence_proto.py``). The decisive
conditioning is on the OTHER already-selected ENGINEERED features -- NOT the
candidate's own operands (CMI given own operands is ~0 for EVERY feature incl.
genuine ones -- a data-processing-inequality trap), NOT the raw top-k (operand-
coverage collisions kill real features).

Two legs, BOTH load-bearing:
  1. CMI clears a CONDITIONAL-PERMUTATION floor (the significance bar -- reuses
     the production within-stratum permutation null,
     ``_conditional_permutation.conditional_permutation_test``).
  2. The DEBIASED conditional MI retains >= ``retain_frac`` (TAU, default 0.15)
     of the WEAKEST already-admitted feature's DEBIASED CMI (the relative-gap /
     order-of-magnitude separator that the floor alone misses -- a redundant
     ``sub`` sits a few x above its own permutation floor yet far below every
     genuine engineered feature). TAU is a SCALE-FREE FRACTION of an in-data
     quantity; it is NOT an MI-nats constant.

n-INVARIANCE (the debiasing -- 2026-06-08 hardening)
----------------------------------------------------
The plug-in CMI (even Miller-Madow corrected by ``_cmi_from_binned``) carries a
RESIDUAL positive finite-sample bias of order O(occupied_cells / n). That bias
is LARGER (relatively) for a low-true-CMI redundant feature than for a high-
true-CMI genuine one, so below a few-x10^4 rows the redundant ``sub``'s biased
raw CMI can cross the TAU bar and be WRONGLY ADMITTED (observed on the weaker-
signal F2 at n~=20k: raw cmi 0.040 > raw rel_bar 0.037).

The fix makes the redundancy decision n-INVARIANT by comparing DEBIASED EXCESS
CMI instead of raw CMI. For each candidate we already compute its OWN
conditional-permutation null (within-stratum shuffle of the candidate, which
PRESERVES the candidate's conditional marginal and so reproduces the SAME
finite-sample bias). The null's MEAN is an estimate of that bias; the EXCESS

    cmi_excess = max(0, cmi_obs - null_mean)

cancels the bias at any n (the bias is present in BOTH terms). The TAU relative
bar and the admitted-feature anchors are taken on the EXCESS, not the raw CMI.
A redundant feature's excess collapses to ~0 (its CMI is pure bias/noise given
the admitted support) while a genuine feature keeps a large positive excess ->
clean separation at every n from 1k up. The significance FLOOR leg is unchanged
(it is already a debiased comparison: both ``cmi_obs`` and the floor quantile
carry the same bias). The seed feature's anchor likewise uses a marginal-
permutation-debiased excess so all anchors live on one debiased scale.

STRONG-SIGNIFICANCE ESCAPE (the false-reject fix -- 2026-06-11 hardening)
------------------------------------------------------------------------
The relative bar (leg 2) is a RELATIVE separator anchored to the WEAKEST admitted
feature's excess. When the admitted set is dominated by ONE strong seed, the bar
``TAU * seed_excess`` becomes a large ABSOLUTE CMI threshold, and a genuinely
COMPLEMENTARY feature that is merely WEAKER than that seed (independent of the
admitted support, but adding an order-of-magnitude less information) is dropped
``redundant_below_rel_bar`` -- a FALSE REJECT. Adversarial repro: one strong
driver plus several INDEPENDENT weak drivers (each provably non-redundant -- MI
with the support ~0 -- each clearing its OWN conditional-permutation floor 20x+)
were ALL dropped, costing ~3% R2 on a real gradient-boosting model. "Much weaker
than the strongest selected feature" is NOT "redundant".

The fix adds a strong-significance escape: a candidate whose observed CMI clears
its OWN conditional-permutation floor by at least ``significance_escape_margin``
(default 3x) is admitted even when its excess is below the relative bar. Robust
conditional significance proves the information is genuinely NEW (not in the
admitted support): a truly redundant feature's CMI collapses to ~its floor
(measured ``cmi/floor`` 1.0--1.4 for the spurious cross-signal ``sub``) while a
genuine complementary feature clears it 20--340x, so the escape opens NO
false-ADMIT path (a redundant feature cannot reach 3x its floor; a monotone remap
of an admitted driver has CMI==0 and is caught at the floor itself). The escape
requires ``passes_floor`` first (leg 1), so it only ever loosens leg 2, never
leg 1.

Greedy: seed on the highest-marginal-MI engineered candidate (admitted on its
marginal significance -- nothing to condition on yet), then admit remaining
candidates in MI order subject to the two-leg test, folding each admitted
feature into the conditioning support.

All MI/CMI is computed via the production primitives
(``_cmi_from_binned`` / ``_quantile_bin`` / ``_renumber_joint`` from
``_mi_greedy_cmi_fe``) -- this module does NOT reimplement MI. The function is
pure (no live framework state captured), so a fitted MRMR remains picklable.
"""
from __future__ import annotations

import logging
import zlib
from typing import Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Default TAU (relative-retention fraction). Scale-free fraction of the weakest
# admitted feature's in-data CMI -- robust window measured [0.084, 1.0) across
# 16 (seed, formula) cells; 0.15 sits in the middle with ~2x margin both sides.
DEFAULT_CMI_RETAIN_FRAC = 0.15

# Conditional-permutation floor: number of within-stratum shuffles and the
# null-quantile used as the significance bar. 25 / 0.95 matches the prototype;
# the floor is the cheap leg (the relative-gap leg does the heavy separation).
_CMI_FLOOR_PERMUTATIONS = 25
_CMI_FLOOR_QUANTILE = 0.95

# Strong-significance escape margin for the relative-gap (leg 2) bar. The leg-2
# bar (TAU * weakest-admitted excess) is a RELATIVE separator: it drops a
# candidate whose information is an order of magnitude weaker than the weakest
# admitted feature. Its DESIGN TARGET is the genuinely redundant feature (e.g. a
# cross-signal ``sub`` whose y-information is already carried by the admitted
# pair) -- such a feature sits only a SMALL multiple above its OWN conditional-
# permutation floor (measured ``cmi/floor`` 1.0--1.4 for the spurious ``sub`` vs
# 50--340 for the genuine ``div``/``mul``). But the bar by itself ALSO drops a
# genuinely COMPLEMENTARY feature that is merely WEAKER than a strong incumbent
# (independent of the support, ``cmi/floor`` ~20, real predictive value) -- a
# FALSE REJECT, because "much weaker than the strongest selected feature" is not
# the same as "redundant". The escape fixes this: a candidate whose observed CMI
# clears its OWN conditional-permutation floor by at least this MULTIPLICATIVE
# margin is robustly-significant NEW information that is NOT contained in the
# admitted support (a redundant feature's CMI collapses to ~its floor), so it is
# admitted even when its excess is below the relative bar. 3.0 sits with ~2x
# margin on BOTH sides of the measured gap (redundant <=1.4x, genuine >=20x).
_CMI_SIGNIFICANCE_ESCAPE_MARGIN = 3.0

# Conditioning-support fragmentation cap (chi-squared rule-of-thumb: cells must
# average >= 5 samples for the plug-in CMI to stay reliable). When folding the
# next admitted feature would push the joint support cardinality past
# ``n / _SUPPORT_FRAG_DIVISOR`` the support is FROZEN (the feature is still
# admitted, but later candidates are scored against the previous support so
# their CMI stays measurable). Mirrors ``greedy_cmi_fe_construct``'s frag_cap.
_SUPPORT_FRAG_DIVISOR = 5

# Below this many rows the within-stratum permutation null + the conditional MI
# both become unreliable (strata collapse to <=1 element). Fall back to
# admitting every candidate on its marginal significance.
_MIN_ROWS_FOR_CMI = 500

# COST GUARD (2026-06-11): the greedy gate is O(K^2) in the candidate count K --
# every still-remaining candidate is re-scored against the admitted support in
# EACH greedy round, and each scoring runs a within-stratum permutation null (25
# permutations x a per-stratum Python shuffle). With a wide FE candidate pool
# (the synergy bootstrap / GBM seeder can surface dozens-to-hundreds of survivors)
# the cost blows up unbounded: measured 1.9 s @ 9 cands -> 27 s @ 99 cands, a
# clean 2.0x per doubling of K (pure O(K^2)). The cap PRE-RANKS the pool by
# marginal MI and keeps only the top-M before the greedy. This is SAFE for the
# redundancy decision: (a) the gate already seeds on the highest-marginal-MI
# candidate and admits in MI order, so the top-M-by-marginal-MI prefix is exactly
# the prefix the unbounded gate would process first; (b) a redundant monotone/
# linear remap has the SAME marginal MI as its genuine sibling, so each genuine
# driver's representative is retained -- only deep-tail redundant remaps (which the
# gate would reject anyway) are dropped pre-greedy; (c) genuinely complementary
# drivers are real signal with high marginal MI, so they survive the cap. Default
# 64 sits ~4x above the widest validated pool; raise to admit a larger greedy.
_DEFAULT_MAX_CANDIDATES = 64


def _conditional_perm_null(
    cand_bin: np.ndarray,
    y_bin: np.ndarray,
    z_support: Optional[np.ndarray],
    *,
    n_permutations: int = _CMI_FLOOR_PERMUTATIONS,
    quantile: float = _CMI_FLOOR_QUANTILE,
    seed: int = 0,
    salt: int = 0,
) -> tuple[float, float]:
    """Conditional-permutation null for ``CMI(cand; y | z_support)``.

    ``salt`` mixes a per-candidate value into the RNG (via ``SeedSequence([seed, salt])``) so each
    candidate's null is drawn from an INDEPENDENT stream. Without it every candidate in a greedy round
    reused the identical permutation key sequence, correlating their floors/null-means and making the
    per-candidate significance test (leg 1) not independent. Callers pass a stable per-candidate salt.

    Reuses the production within-stratum permutation infrastructure
    (``conditional_permutation_test``): the candidate column is permuted WITHIN
    each support stratum, preserving the ``cand | support`` distribution, so the
    null measures the CMI a candidate of the SAME conditional marginal would show
    by chance.

    Returns ``(floor, null_mean)``:

      * ``floor``     -- the ``quantile`` of the null distribution; the
                         data-derived SIGNIFICANCE bar (leg 1).
      * ``null_mean`` -- the MEAN of the null distribution; an estimate of the
                         candidate's finite-sample CMI BIAS at this n. Subtracted
                         from the observed CMI to form the n-invariant DEBIASED
                         EXCESS that the relative-gap leg (leg 2) compares (the
                         bias is present in both ``cmi_obs`` and ``null_mean`` and
                         cancels). The within-stratum shuffle reproduces the SAME
                         bias because it leaves ``cand | support`` (hence the
                         occupied-cell structure that drives the bias) intact.

    When ``z_support`` is ``None`` (the seed step, nothing to condition on) the
    null is the MARGINAL-permutation null: ``cand`` is freely shuffled and the
    null statistic is the marginal ``MI(cand_perm; y)``. This debiases the seed
    anchor on the SAME footing as the conditional candidates, so every admitted
    feature's relative-bar anchor is a debiased excess.
    """
    # Sparse, renumber-based plug-in CMI/MI -- the SAME estimator used for the
    # observed value (``_cmi_from_binned``), so the floor / null mean and the
    # point estimate are directly comparable, and the memory stays bounded by n
    # (no dense (K_x, K_y, K_z) contingency allocation when the frozen support's
    # joint cardinality climbs into the thousands).
    from ._mi_greedy_cmi_fe import _cmi_from_binned, cmi_from_binned_fixed_yz, precompute_cmi_yz_terms

    x = np.ascontiguousarray(cand_bin, dtype=np.int64).ravel()
    y = np.ascontiguousarray(y_bin, dtype=np.int64).ravel()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(salt) & 0xFFFFFFFF]))

    if z_support is None or z_support.size == 0:
        # Marginal-permutation null (seed step): free shuffle of the candidate
        # -> null MARGINAL MI(cand; y). Mean estimates the marginal MI bias at
        # this n; the seed's debiased excess = max(0, marginal_mi - this mean).
        nperm = int(n_permutations)
        nulls = np.empty(nperm, dtype=np.float64)
        for i in range(nperm):
            x_perm = x[rng.permutation(x.size)]
            nulls[i] = float(_cmi_from_binned(x_perm, y, None))
        return float(np.quantile(nulls, quantile)), float(np.mean(nulls))

    z = np.ascontiguousarray(z_support, dtype=np.int64).ravel()
    # Group row indices by support stratum once; permute the CANDIDATE column
    # within each stratum (preserves the ``cand | support`` distribution -- the
    # conditional permutation null of Berrett et al. 2020).
    order = np.argsort(z, kind="stable")
    sorted_z = z[order]
    boundaries = np.flatnonzero(np.diff(sorted_z)) + 1
    groups = [g for g in np.split(order, boundaries) if g.size > 1]
    if not groups:
        return 0.0, 0.0
    # y and z are fixed across permutations (only x is reshuffled within strata),
    # so the H(Y,Z) / H(Z) block of the conditional CMI is invariant -- hoist it
    # out of the loop and recompute only the x-dependent xz / xyz terms per perm.
    y_i, z_i, h_yz, h_z, k_yz, k_z, n_f = precompute_cmi_yz_terms(y, z)
    # VECTORISED within-stratum permutation (perf, 2026-06-19). The previous per-stratum Python loop
    # ``for g in groups: x_perm[g] = x[g[rng.permutation(g.size)]]`` issued ONE rng.permutation PER
    # stratum PER perm; at n=100k with a high-cardinality conditioning support that is hundreds of
    # thousands of calls -- measured 697k calls / 14.4s, the single largest steady-state FE hotspot.
    # A uniform within-group shuffle is a single lexsort: draw one random key per row and sort by
    # (stratum, key); since ``sorted_z`` is already sorted, each contiguous stratum block is reordered
    # by its keys alone -> an independent uniform permutation within every stratum in one vectorised
    # pass (size-1 strata are fixed points, exactly as the old ``g.size > 1`` guard left them). This is
    # the SAME conditional permutation null (Berrett et al. 2020) -- only the RNG draw sequence changes,
    # so the floor/mean are unchanged in expectation (verified: identical feature selection on the
    # canonical recovery fit). ``order``/``sorted_z`` were computed once above.
    x_sorted = x[order]
    nulls = np.empty(int(n_permutations), dtype=np.float64)
    for i in range(int(n_permutations)):
        keys = rng.random(x.size)
        within = np.lexsort((keys, sorted_z))  # within each (already-sorted) stratum block: random order
        x_perm = np.empty_like(x)
        x_perm[order] = x_sorted[within]
        nulls[i] = float(cmi_from_binned_fixed_yz(x_perm, y_i, z_i, h_yz, h_z, k_yz, k_z, n_f))
    return float(np.quantile(nulls, quantile)), float(np.mean(nulls))


def apply_cmi_redundancy_gate(
    candidates: dict,
    y_bin: np.ndarray,
    *,
    nbins: int = 10,
    retain_frac: float = DEFAULT_CMI_RETAIN_FRAC,
    n_permutations: int = _CMI_FLOOR_PERMUTATIONS,
    quantile: float = _CMI_FLOOR_QUANTILE,
    significance_escape_margin: float = _CMI_SIGNIFICANCE_ESCAPE_MARGIN,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    seed: int = 0,
    verbose: int = 0,
) -> tuple[set, dict]:
    """Greedy CMI-redundancy gate over the surviving engineered candidate pool.

    Parameters
    ----------
    candidates : dict ``{name -> (continuous_values: np.ndarray, marginal_mi: float)}``
        The engineered columns that already cleared the per-pair acceptance
        machinery (joint / prewarp / marginal-uplift). ``continuous_values`` is
        the full-n float column (NOT pre-binned -- binned here so the CMI codes
        match the production quantile binning the prototype validated).
    y_bin : np.ndarray
        Discretised target codes (the same ``classes_y`` the MI sweep scores
        against). Renumbered to dense 0..K-1 internally.
    nbins : int
        Equi-frequency bins per candidate column.
    retain_frac : float
        TAU -- the scale-free relative-retention fraction (default 0.15).
    n_permutations, quantile : int, float
        Conditional-permutation floor config.
    significance_escape_margin : float
        Strong-significance escape for the relative-gap (leg 2) bar. A candidate
        whose observed CMI clears its OWN conditional-permutation floor by at
        least this MULTIPLICATIVE margin is robustly-significant NEW information
        not contained in the admitted support, so it is admitted even when its
        debiased excess is below the relative bar. Prevents the FALSE REJECT of a
        genuinely complementary but individually-WEAKER feature (whose excess is
        below ``retain_frac * strongest-incumbent`` yet which clears its floor
        20x+). Set ``<= 1`` to disable the escape (pure two-leg behaviour).
    max_candidates : int
        COST GUARD. The greedy gate is O(K^2) in the candidate count (every
        remaining candidate re-scored against the support in each round, each
        scoring running a per-stratum permutation null). When ``len(candidates)``
        exceeds this cap the pool is PRE-RANKED by marginal MI and only the
        top-``max_candidates`` enter the greedy -- bounding the cost to O(M^2)
        regardless of how wide the upstream FE pool grows. Safe for the redundancy
        decision: the gate already admits in marginal-MI order and a redundant
        remap shares its genuine sibling's marginal MI, so every genuine driver's
        representative is retained; only deep-tail redundant remaps (rejected
        anyway) are dropped. ``<= 0`` disables the cap (unbounded greedy). The
        dropped tail is diagnosed with ``reason='dropped_cost_cap'``.
    seed : int
        RNG seed for the conditional-permutation floor (deterministic).
    verbose : int
        >0 emits per-candidate accept/reject diagnostics via the module logger.

    Returns
    -------
    (accepted_names, diagnostics)
        ``accepted_names`` is the set of candidate names to KEEP; everything
        else is dropped as redundant. ``diagnostics`` maps each name to a dict
        with ``accept`` / ``cmi`` (raw observed CMI) / ``cmi_excess`` (debiased
        excess = ``max(0, cmi - null_mean)``, what the relative bar compares) /
        ``floor`` / ``null_mean`` / ``rel_bar`` / ``reason``.

    Degenerate fallback: with <2 candidates, or fewer than ``_MIN_ROWS_FOR_CMI``
    rows, there is nothing to condition on (or the conditional estimate is
    unreliable) -- ACCEPT every candidate on its marginal significance rather
    than rejecting everything.
    """
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin, _renumber_joint

    names = list(candidates.keys())
    diagnostics: dict = {}
    if not names:
        return set(), diagnostics

    y_arr = np.asarray(y_bin)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    _, y_dense = np.unique(y_arr, return_inverse=True)
    y_dense = y_dense.astype(np.int64)
    n_rows = int(y_dense.size)

    # Degenerate: nothing to condition on, or too few rows for a reliable
    # conditional estimate -> admit all on marginal significance.
    if len(names) < 2 or n_rows < _MIN_ROWS_FOR_CMI:
        for nm in names:
            diagnostics[nm] = dict(
                accept=True, cmi=float(candidates[nm][1]),
                cmi_excess=float(candidates[nm][1]), floor=0.0, null_mean=0.0,
                rel_bar=0.0, reason="degenerate_marginal_admit",
            )
        return set(names), diagnostics

    # Bin every candidate once (production quantile binner). Done BEFORE the cost
    # cap so the cap can collapse exact-partition duplicates first (see below).
    cand_bins: dict = {}
    for nm in names:
        vals = np.asarray(candidates[nm][0], dtype=np.float64)
        cand_bins[nm] = _quantile_bin(vals, nbins=nbins)
    marg = {nm: float(candidates[nm][1]) for nm in names}

    # PARTITION DEDUP (2026-06-11): a monotone/linear remap of an admitted feature
    # bins to the IDENTICAL equi-frequency partition (equi-frequency binning is
    # rank-invariant; a decreasing remap yields the reversed-but-identical
    # partition). Such a remap carries IDENTICAL information and IDENTICAL marginal
    # MI, so the greedy would correctly admit exactly one and reject the rest -- but
    # at full per-round permutation-null cost, AND (critically) a plain top-M-by-
    # marginal-MI cap can STARVE a genuine driver: when many tied-MI remaps of the
    # strongest driver crowd out every form of a weaker (but genuine) driver, the
    # weaker driver loses ALL its forms before the greedy even runs. Collapse each
    # exact-partition equivalence class to its best representative (highest marginal
    # MI; ties broken by name for determinism) BEFORE the cap so the cap operates on
    # DISTINCT partitions -- every genuine driver keeps a representative and the
    # redundant siblings never pay the greedy cost. Partition identity is hashed on
    # the canonical (dense-renumbered) bin codes so a reversed-but-identical
    # partition collapses to the same key.
    _partition_rep: dict = {}   # canonical partition key -> representative name
    _dups_collapsed: list[str] = []
    for nm in sorted(names):  # deterministic iteration (name order)
        _, inv = np.unique(cand_bins[nm], return_inverse=True)
        key = inv.astype(np.int64).tobytes()
        rep = _partition_rep.get(key)
        if rep is None:
            _partition_rep[key] = nm
        else:
            # keep the higher-marginal-MI rep; the loser is a collapsed duplicate
            if marg[nm] > marg[rep] or (marg[nm] == marg[rep] and nm < rep):
                _partition_rep[key] = nm
                _dups_collapsed.append(rep)
            else:
                _dups_collapsed.append(nm)
    if _dups_collapsed:
        _keep = set(_partition_rep.values())
        names = [nm for nm in names if nm in _keep]
        for nm in _dups_collapsed:
            diagnostics[nm] = dict(
                accept=False, cmi=float(marg.get(nm, candidates[nm][1])),
                cmi_excess=0.0, floor=0.0, null_mean=0.0, rel_bar=0.0,
                reason="redundant_partition_duplicate",
            )
        if verbose:
            logger.info(
                "CMI-redundancy gate: collapsed %d exact-partition duplicate(s) "
                "(monotone/linear remaps of a kept feature) before the greedy.",
                len(_dups_collapsed),
            )

    # COST GUARD: the greedy below is O(K^2) in the candidate count. When the pool
    # is STILL wide after partition dedup, PRE-RANK by marginal MI and keep only the
    # top-``max_candidates`` before the greedy (bounds cost to O(M^2)). The dropped
    # tail is the lowest-marginal-MI DISTINCT-partition candidates -- genuine drivers
    # are real signal with high marginal MI, so they survive; only deep-tail weak
    # candidates (which the greedy would reject anyway) never pay the per-round
    # permutation-null cost. ``max_candidates <= 0`` disables it.
    _dropped_for_cost: list[str] = []
    if int(max_candidates) > 0 and len(names) > int(max_candidates):
        names_by_marg = sorted(names, key=lambda nm: (float(candidates[nm][1]), nm), reverse=True)
        _dropped_for_cost = names_by_marg[int(max_candidates):]
        names = names_by_marg[: int(max_candidates)]
        for nm in _dropped_for_cost:
            diagnostics[nm] = dict(
                accept=False, cmi=float(candidates[nm][1]),
                cmi_excess=0.0, floor=0.0, null_mean=0.0, rel_bar=0.0,
                reason="dropped_cost_cap",
            )
        if verbose:
            logger.info(
                "CMI-redundancy gate: cost cap -- %d distinct-partition candidate(s) "
                "exceed max_candidates=%d; kept the top %d by marginal MI, dropped %d "
                "low-marginal-MI tail candidate(s) before the O(K^2) greedy.",
                len(_dropped_for_cost) + int(max_candidates), int(max_candidates),
                int(max_candidates), len(_dropped_for_cost),
            )

    accepted: list[str] = []          # admitted candidate names, in selection order
    accepted_bins: list[np.ndarray] = []
    admitted_excess: list[float] = []  # DEBIASED-EXCESS CMI of admitted features (for the rel bar)
    remaining = set(names)
    frag_cap = max(2, n_rows // _SUPPORT_FRAG_DIVISOR)
    z_support: Optional[np.ndarray] = None

    # Seed: highest-marginal-MI candidate, admitted on its marginal significance
    # (nothing to condition on yet). Its DEBIASED-EXCESS marginal MI (marginal MI
    # minus its marginal-permutation null mean) anchors the relative bar -- so the
    # anchor lives on the SAME n-invariant debiased scale as the conditional
    # candidate excesses below (the raw marginal MI would re-introduce the bias the
    # excess removes).
    seed_name = max(remaining, key=lambda nm: marg[nm])
    seed_floor, seed_null_mean = _conditional_perm_null(
        cand_bins[seed_name], y_dense, None,
        n_permutations=n_permutations, quantile=quantile, seed=seed,
        salt=zlib.crc32(seed_name.encode("utf-8")),
    )
    seed_excess = max(0.0, marg[seed_name] - seed_null_mean)
    accepted.append(seed_name)
    accepted_bins.append(cand_bins[seed_name])
    admitted_excess.append(seed_excess)
    remaining.discard(seed_name)
    diagnostics[seed_name] = dict(
        accept=True, cmi=marg[seed_name], cmi_excess=seed_excess,
        floor=seed_floor, null_mean=seed_null_mean, rel_bar=0.0,
        reason="seed_marginal",
    )

    while remaining:
        z_support, _ = _renumber_joint(*accepted_bins)
        # Relative bar is a fraction of the WEAKEST admitted feature's DEBIASED
        # EXCESS CMI -- n-invariant because the finite-sample bias cancels in the
        # excess (cmi_obs - null_mean). A redundant candidate's excess ~ 0; a
        # genuine candidate's excess stays a large positive value.
        rel_bar = float(retain_frac) * min(admitted_excess)
        best_name = None
        best_excess = -1.0
        scored: dict = {}
        for nm in list(remaining):
            cmi = float(_cmi_from_binned(cand_bins[nm], y_dense, z_support))
            floor, null_mean = _conditional_perm_null(
                cand_bins[nm], y_dense, z_support,
                n_permutations=n_permutations, quantile=quantile, seed=seed,
                salt=zlib.crc32(nm.encode("utf-8")),
            )
            cmi_excess = max(0.0, cmi - null_mean)
            scored[nm] = (cmi, cmi_excess)
            passes_floor = cmi > floor                 # leg 1: significance
            # Strong-significance escape: a candidate whose observed CMI clears
            # its OWN conditional-permutation floor by a robust multiplicative
            # margin carries genuinely NEW conditional information that is NOT in
            # the admitted support (a redundant feature's CMI collapses to ~its
            # floor -- measured cmi/floor 1.0--1.4 for the spurious cross-signal
            # vs >=20 for a genuine complementary feature). Such a feature is
            # admitted even if its debiased excess is below the relative bar,
            # which by itself would FALSELY REJECT a complementary-but-weaker
            # feature as "redundant". The escape requires passes_floor first (no
            # division-by-noise on a sub-floor candidate).
            strongly_significant = (
                passes_floor
                and significance_escape_margin > 1.0
                and floor > 0.0
                and cmi >= significance_escape_margin * floor
            )
            passes_rel = cmi_excess >= rel_bar         # leg 2: debiased relative gap
            passes = passes_floor and (passes_rel or strongly_significant)
            if nm not in diagnostics:
                diagnostics[nm] = {}
            diagnostics[nm].update(
                accept=False, cmi=cmi, cmi_excess=cmi_excess, floor=floor,
                null_mean=null_mean, rel_bar=rel_bar,
                reason=("redundant_below_floor" if not passes_floor
                        else "pending" if (passes_rel or strongly_significant)
                        else "redundant_below_rel_bar"),
            )
            if passes and cmi_excess > best_excess:
                best_excess = cmi_excess
                best_name = nm
        if best_name is None:
            # No remaining candidate adds enough NEW information -> stop; the
            # rest are redundant given the admitted engineered support.
            break
        diagnostics[best_name].update(accept=True, reason="admitted_cmi")
        # Fold the winner into the conditioning support, respecting the
        # fragmentation cap (freeze support if folding would shatter the strata).
        new_bin = cand_bins[best_name]
        candidate_support, _ = _renumber_joint(*(accepted_bins + [new_bin]))
        if int(np.unique(candidate_support).size) <= frag_cap:
            accepted_bins.append(new_bin)
        # else: keep accepted_bins frozen; the feature is still admitted.
        accepted.append(best_name)
        admitted_excess.append(best_excess)
        remaining.discard(best_name)

    if verbose:
        for nm in names:
            d = diagnostics.get(nm, {})
            logger.info(
                "CMI-redundancy gate: %s accept=%s cmi=%.4f excess=%.4f "
                "floor=%.4f rel_bar=%.4f (%s)",
                nm, d.get("accept"), d.get("cmi", float("nan")),
                d.get("cmi_excess", float("nan")), d.get("floor", float("nan")),
                d.get("rel_bar", float("nan")), d.get("reason", "-"),
            )
    return set(accepted), diagnostics


__all__ = ["apply_cmi_redundancy_gate", "DEFAULT_CMI_RETAIN_FRAC"]
