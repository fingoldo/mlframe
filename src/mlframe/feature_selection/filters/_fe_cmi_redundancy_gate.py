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

# Significance-null primitives carved to a sibling; re-imported so this facade and apply_cmi_redundancy_gate resolve them unchanged.
from ._fe_cmi_redundancy_null import (  # noqa: F401
    _CMI_FLOOR_PERMUTATIONS,
    _CMI_FLOOR_QUANTILE,
    _CMI_ANALYTIC_NULL_MIN_N_DEFAULT,
    _cmi_analytic_null_min_n,
    _conditional_perm_null,
)

# Default TAU (relative-retention fraction). Scale-free fraction of the weakest
# admitted feature's in-data CMI -- robust window measured [0.084, 1.0) across
# 16 (seed, formula) cells; 0.15 sits in the middle with ~2x margin both sides.
DEFAULT_CMI_RETAIN_FRAC = 0.15

# Conditional-permutation floor: number of within-stratum shuffles and the
# null-quantile used as the significance bar. 25 / 0.95 matches the prototype;
# the floor is the cheap leg (the relative-gap leg does the heavy separation).

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

# Minimum n at which the ANALYTIC chi-square CMI null replaces the within-stratum permutation null
# (see ``_conditional_perm_null``). Distinct from the pair-MI path's ``analytic_null_min_n`` (50k):
# the conditional path's real safe-condition is the per-table cell-occupancy floor (avg expected
# count >= ``_min_expected_cell``, checked at the call site), so the n-only floor only needs to keep
# the chi-square asymptotic itself reliable. 20k sits well above ``_MIN_ROWS_FOR_CMI`` and matches the
# size the default screen subsample (30k) feeds the gate, so the canonical large-n fit engages the
# analytic null where the cells are dense and falls back to permutation where they are sparse. Env-
# tunable (``MLFRAME_CMI_ANALYTIC_NULL_MIN_N``); a future kernel_tuning_cache sweep can refine per host.
import os as _os

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

_Y_DENSE_MEMO: dict = {}


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

    # DEVICE-BORN candidate-code residency (default ON under fe_gpu_strict_resident_enabled; env opt-out
    # MLFRAME_FE_GATE_RESIDENT_CANDS=0). When on, each candidate is quantile-binned ONCE on the device and its
    # int64 codes are KEPT RESIDENT, then routed through the resident-input branches of the round-batched CMI /
    # per-candidate CMI / conditional perm-null so the derived candidate codes never re-cross H2D (the
    # ``cmi_cand_x`` / ``card_cand_x`` / ``permnull_cand_x`` re-uploads the host-code path incurred, plus the
    # ``qbin_x`` float that the host binner used to D2H back). The host int64 copy (``cand_bins``) is the D2H of
    # the SAME resident partition, retained for the host-only sites (partition-dedup hashing, ``_renumber_joint``
    # support building on admit, and the CPU fallbacks) -- byte-identical to the device codes, so selection is
    # unchanged. Falls back per-candidate to the host ``_quantile_bin`` on any cupy fault.
    _gate_resident = False
    if _os.environ.get("MLFRAME_FE_GATE_RESIDENT_CANDS", "1").strip().lower() in ("1", "true", "on", "yes"):
        try:
            from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
            from ._mi_greedy_cmi_fe import _cmi_gpu_enabled
            _gate_resident = bool(fe_gpu_strict_resident_enabled()) and bool(_cmi_gpu_enabled())
        except Exception:
            _gate_resident = False

    y_arr = np.asarray(y_bin)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    # Content-memoised dense renumber: y_bin is a fit-constant handed to every gate call, and the dense
    # renumber is a full-n np.unique sort each time. Same pattern as _coerce_y_classes / infer_classification
    # (bounded FIFO; a COPY is returned into the local so a mutating consumer cannot poison the cache).
    _yk = None
    try:
        _yk = (y_arr.shape, str(y_arr.dtype), hash(y_arr.tobytes()))
        _yhit = _Y_DENSE_MEMO.get(_yk)
    except Exception:
        _yhit = None
    if _yhit is not None:
        y_dense = _yhit.copy()
    else:
        _, y_dense = np.unique(y_arr, return_inverse=True)
        y_dense = y_dense.astype(np.int64)
        if _yk is not None:
            if len(_Y_DENSE_MEMO) > 8:
                _Y_DENSE_MEMO.pop(next(iter(_Y_DENSE_MEMO)))
            _Y_DENSE_MEMO[_yk] = y_dense.copy()
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
    # ``cand_bins`` holds the HOST int64 codes (source of truth for the host-only sites); ``cand_bins_dev``
    # holds the RESIDENT cupy int64 codes fed to the three device-scoring paths (empty when the resident path is
    # off). Under residency each candidate is device-binned ONCE and the host copy is the D2H of that SAME
    # resident partition (byte-identical), so no separate host ``np.quantile`` runs and the two forms cannot
    # diverge; a per-candidate cupy fault falls back to the host ``_quantile_bin`` (no resident code -> that
    # candidate's device sites re-upload the host code, exactly as before this change).
    cand_bins: dict = {}
    cand_bins_dev: dict = {}
    for nm in names:
        vals = np.asarray(candidates[nm][0], dtype=np.float64)
        _dev = None
        if _gate_resident and np.isfinite(vals).all():
            try:
                from ._mi_greedy_cmi_fe import _quantile_bin_gpu_resident
                _dev = _quantile_bin_gpu_resident(vals, nbins)
            except Exception:
                _dev = None
        if _dev is not None:
            import cupy as _cp
            cand_bins_dev[nm] = _dev
            cand_bins[nm] = _cp.asnumpy(_dev).astype(np.int64)  # host copy = D2H of the SAME resident partition
        else:
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
    _partition_rep: dict = {}  # canonical partition key -> representative name
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
                "CMI-redundancy gate: collapsed %d exact-partition duplicate(s) " "(monotone/linear remaps of a kept feature) before the greedy.",
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
        _dropped_for_cost = names_by_marg[int(max_candidates) :]
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

    accepted: list[str] = []  # admitted candidate names, in selection order
    accepted_bins: list[np.ndarray] = []
    # Parallel list of the RESIDENT (device) code for each admitted feature (None where that feature fell back to
    # host binning). When every admitted feature has a resident code the round conditioning support Z is built
    # DEVICE-BORN (``_renumber_joint_gpu``) and threaded RESIDENT through the round-batched CMI, the per-candidate
    # CMI, AND the conditional perm-null (device order/z_rank), so Z never crosses H2D (the ``cmi_z`` +
    # order/z_rank uploads). A mixed / host-fallback round keeps the host ``_renumber_joint`` support.
    accepted_bins_dev: list = []
    admitted_excess: list[float] = []  # DEBIASED-EXCESS CMI of admitted features (for the rel bar)
    remaining = set(names)
    frag_cap = max(2, n_rows // _SUPPORT_FRAG_DIVISOR)
    z_support: Optional[np.ndarray] = None
    # Carries the joint support built by a round's OWN frag-cap admission check (candidate_support,
    # below) forward to the next round's z_support build, when that join is reused verbatim (see the
    # top of the ``while remaining`` loop).
    _pending_z_support: Optional[np.ndarray] = None
    _pending_z_card: Optional[int] = None

    # Seed: highest-marginal-MI candidate, admitted on its marginal significance
    # (nothing to condition on yet). Its DEBIASED-EXCESS marginal MI (marginal MI
    # minus its marginal-permutation null mean) anchors the relative bar -- so the
    # anchor lives on the SAME n-invariant debiased scale as the conditional
    # candidate excesses below (the raw marginal MI would re-introduce the bias the
    # excess removes).
    # Deterministic, hash-INDEPENDENT seed selection. ``remaining`` is a SET, so a bare
    # ``max(remaining, key=marg)`` over candidates with EQUAL marginal MI returns whichever
    # element the set happened to iterate first -- PYTHONHASHSEED-randomised for string names,
    # so the seed (and therefore which equal-MI form anchors the support and which redundant
    # siblings are dropped) flipped across processes. This was the root cause of the non-
    # deterministic F2 ``heavy_tailed`` fusion flake: two RANK-EQUAL a/b forms (the raw (a,b)
    # ratio ``mul(sqr(a),reciproc(b))`` vs a prewarp form ``div(abs(b),a__p2sin1)`` over an
    # engineered/warped operand, both at marginal MI 0.81220) tie for the seed; the raw form
    # fuses cleanly into the target compound while the prewarp form does not, so the per-process
    # set ordering decided whether the compound was recovered (~40% of seeds failed).
    #
    # MI is a RANK statistic and cannot separate the two (their bins are rank-equivalent), so the
    # tie is resolved by a STRUCTURAL preference: among equal-MI candidates prefer the SIMPLEST
    # representative -- fewest WARPED operand tokens (``base__warp`` prewarp/engineered operands,
    # counted by ``__``), then fewest total operator tokens, then ascending name. This is the
    # project's "prefer the simpler/raw member of an MI-equivalence class" rule (raw operands carry
    # no fitted warp -> lower overfit risk AND they remain fusable by the downstream additive-fusion
    # step, which a prewarp-operand form is not), made deterministic. ``_tie_key`` orders so that
    # ``min`` picks the most-preferred candidate; ``-marg[nm]`` keeps marginal MI the primary key.
    def _tie_key(nm: str) -> tuple:
        """Orders MI-tied candidate names so ``min`` picks the highest-marginal-MI, structurally simplest
        (fewest warped operands, fewest operators, then lexical) representative of the equivalence class."""
        return (-marg[nm], nm.count("__"), nm.count("("), nm)

    seed_name = min(remaining, key=_tie_key)
    # Seed marginal perm-null from the RESIDENT candidate code when available (device-born), else host.
    seed_floor, seed_null_mean = _conditional_perm_null(
        cand_bins_dev.get(seed_name, cand_bins[seed_name]) if cand_bins_dev else cand_bins[seed_name],
        y_dense, None,
        n_permutations=n_permutations, quantile=quantile, seed=seed,
        salt=zlib.crc32(seed_name.encode("utf-8")),
    )
    seed_excess = max(0.0, marg[seed_name] - seed_null_mean)
    accepted.append(seed_name)
    accepted_bins.append(cand_bins[seed_name])
    accepted_bins_dev.append(cand_bins_dev.get(seed_name) if cand_bins_dev else None)
    admitted_excess.append(seed_excess)
    remaining.discard(seed_name)
    diagnostics[seed_name] = dict(
        accept=True, cmi=marg[seed_name], cmi_excess=seed_excess,
        floor=seed_floor, null_mean=seed_null_mean, rel_bar=0.0,
        reason="seed_marginal",
    )

    while remaining:
        # DEVICE-BORN conditioning support Z: when every admitted feature has a resident code, join them ON the
        # device (``_renumber_joint_gpu``) so Z never crosses H2D -- threaded RESIDENT through the round-batched
        # CMI, the per-candidate CMI, and the conditional perm-null (which derives order/z_rank on device). The
        # device join yields a different dense-id numbering than the host njit factorize but the SAME partition
        # -> the same CMI, so admit/reject is selection-identical. Mixed / host-fallback rounds keep the host
        # join. The host ``z_support`` is still built when NOT device-born (or as a rare perm-null fallback).
        # A round's frag-cap admission check (candidate_support, below) joins accepted_bins + the
        # winning candidate -- EXACTLY the join this round needs here once that candidate is folded in
        # (accepted_bins now equals that same set). Reuse it instead of a second _renumber_joint over
        # the identical columns; consumed-and-cleared unconditionally so a frag-cap-frozen round (which
        # left accepted_bins unchanged) or the very first round never carries a stale/absent value in.
        _prev_z_support, _prev_z_card = _pending_z_support, _pending_z_card
        _pending_z_support, _pending_z_card = None, None
        z_support_dev = None
        if _gate_resident and accepted_bins_dev and all(_b is not None for _b in accepted_bins_dev):
            try:
                from ._mi_greedy_cmi_fe import _renumber_joint_gpu
                z_support_dev, _ = _renumber_joint_gpu(*accepted_bins_dev)
            except Exception:
                z_support_dev = None
        if z_support_dev is None:
            if _prev_z_support is not None:
                z_support, _z_card = _prev_z_support, _prev_z_card
            else:
                z_support, _z_card = _renumber_joint(*accepted_bins)
        else:
            z_support = None  # host support built lazily only if a host consumer needs it this round
            _z_card = None
        # z handed to the DEVICE scorers (round-batched CMI + per-candidate CMI): the resident support when
        # device-born, else the host support -- both accepted by the cupy resident-input branches.
        _z_scored = z_support_dev if z_support_dev is not None else z_support
        # Relative bar is a fraction of the WEAKEST admitted feature's DEBIASED
        # EXCESS CMI -- n-invariant because the finite-sample bias cancels in the
        # excess (cmi_obs - null_mean). A redundant candidate's excess ~ 0; a
        # genuine candidate's excess stays a large positive value.
        rel_bar = float(retain_frac) * min(admitted_excess)
        best_name = None
        best_excess = -1.0
        scored: dict = {}
        # ROUND-LEVEL BATCHED CMI + ANALYTIC FLOOR/df (launch-reduction): within a greedy round z_support is
        # FIXED, so EVERY remaining candidate's CMI(x; y | z) AND its analytic-null cardinalities
        # (k_z/k_xz/k_yz/k_xyz -> df) are computed in ONE batched_cmi_gpu(return_cards) workload instead of a
        # per-candidate _cmi_from_binned + joint_cardinalities_cupy. The occupied-cell counts equal the
        # per-candidate path's (same definition) -> df, hence floor = chi2.ppf(q,df)/2N and null_mean =
        # df/2N, are BIT-IDENTICAL to _conditional_perm_null's analytic branch (the SAME applicability gate:
        # n >= analytic-min-n, df > 0, n/k_xyz >= min-expected-cell). Candidates where the analytic null does
        # NOT apply fall back to the per-candidate _conditional_perm_null (permutation path) below. Gated
        # under STRICT/CMI_GPU (default OFF -> per-candidate CPU, byte-identical).
        _round_cmi: dict = {}
        _round_floor: dict = {}
        _round_cards: dict = {}  # nm -> (k_z, k_xz, k_yz, k_xyz) from the batched return_cards workload
        # Stable, hash-independent iteration order: ``remaining`` is a set of name strings, so a
        # bare ``list(remaining)`` is PYTHONHASHSEED-randomised and the per-round winner tie-break
        # (``cmi_excess > best_excess``, first-wins) would flip across processes on equal-excess
        # candidates. Order by the SAME structural-preference key as the seed (simplest/raw form
        # first) so the winner among equal-excess candidates is reproducible AND consistent with
        # the seed's preference.
        _rem_list = sorted(remaining, key=_tie_key)
        try:
            from ._mi_greedy_cmi_fe import _cmi_gpu_enabled
            if _cmi_gpu_enabled() and len(_rem_list) > 1:
                from ._fe_batched_mi import batched_cmi_gpu
                # RESIDENT candidate-matrix build: when every round candidate has a device-resident code
                # (device-born binning), stack them into a RESIDENT (n, K) cupy matrix so the derived codes
                # never re-cross H2D into ``batched_cmi_gpu`` (which accepts a resident ``x_cols``). Mixed /
                # host-fallback rounds keep the host int64 build (byte-identical codes -> same CMI).
                if cand_bins_dev and all(_nm in cand_bins_dev for _nm in _rem_list):
                    import cupy as _cp
                    _Xc = _cp.empty((y_dense.shape[0], len(_rem_list)), dtype=_cp.int64)
                    for _j, _nm in enumerate(_rem_list):
                        _Xc[:, _j] = cand_bins_dev[_nm]
                else:
                    _Xc = np.empty((y_dense.shape[0], len(_rem_list)), dtype=np.int64)
                    for _j, _nm in enumerate(_rem_list):
                        _Xc[:, _j] = cand_bins[_nm]
                _cmis, _kz, _kxz, _kyz, _kxyz = batched_cmi_gpu(_Xc, y_dense, _z_scored, return_cards=True)
                _cmis = np.asarray(_cmis, dtype=np.float64)
                _round_cmi = {_nm: float(_cmis[_j]) for _j, _nm in enumerate(_rem_list)}
                # cards for EVERY candidate -> pass to the per-candidate permutation-null fallback so it never
                # recomputes joint_cardinalities_cupy (z_support is conditional here, so cards apply).
                if _z_scored is not None:
                    _kxz_a = np.asarray(_kxz)
                    _kxyz_a = np.asarray(_kxyz)
                    _round_cards = {_nm: (int(_kz), int(_kxz_a[_j]), int(_kyz), int(_kxyz_a[_j])) for _j, _nm in enumerate(_rem_list)}
                # analytic floor/null for all candidates from the batched cards (matches _conditional_perm_null)
                try:
                    from ._analytic_mi_null import _HAVE_CHI2, _chi2, _min_expected_cell, analytic_null_enabled
                    if _HAVE_CHI2 and analytic_null_enabled() and n_rows >= _cmi_analytic_null_min_n():
                        _nf = float(max(1, n_rows)); _mincell = _min_expected_cell()
                        for _j, _nm in enumerate(_rem_list):
                            _df = int(_kxyz[_j]) + int(_kz) - int(_kxz[_j]) - int(_kyz)
                            _cells = max(1, int(_kxyz[_j]))
                            if _df > 0 and (_nf / float(_cells)) >= _mincell:
                                _flr = float(_chi2.ppf(float(quantile), _df)) / (2.0 * _nf)
                                _round_floor[_nm] = (_flr if _flr > 0.0 else 0.0, _df / (2.0 * _nf))
                except Exception:
                    _round_floor = {}
        except Exception:
            _round_cmi = {}
            _round_floor = {}
        # z_support is FIXED within the round, so read its occupied cardinality ONCE here (one D2H) and pass it
        # to every per-candidate CMI fallback as kz -- otherwise _cmi_from_binned_cupy re-reads int(dz.max()) per
        # candidate. Candidate codes are nbins-binned, so kx=nbins is a safe (empty-bin) upper bound with no read.
        _zcard = 0
        if _z_card is not None:
            _zcard = int(_z_card)
        elif _z_scored is not None:
            try:
                _zcard = (int(_z_scored.max()) + 1) if getattr(_z_scored, "size", 0) else 0
            except Exception:
                _zcard = 0
        for nm in _rem_list:
            # Prefer the RESIDENT candidate code for the per-candidate CMI + perm-null fallbacks (both dispatch
            # to the cupy resident-input branch: ``_cmi_from_binned`` -> ``_cmi_from_binned_cupy(isinstance
            # cp.ndarray)``; ``conditional_perm_null_gpu(isinstance cp.ndarray)``), so a candidate that misses the
            # round-batched result is scored from its resident codes without a re-upload. Host code otherwise.
            _cb = cand_bins_dev.get(nm) if cand_bins_dev else None
            if _cb is None:
                _cb = cand_bins[nm]
            cmi = _round_cmi[nm] if nm in _round_cmi else float(_cmi_from_binned(_cb, y_dense, _z_scored, kx=int(nbins), kz=int(_zcard)))
            if nm in _round_floor:
                floor, null_mean = _round_floor[nm]
            else:
                floor, null_mean = _conditional_perm_null(
                    _cb, y_dense, z_support,
                    n_permutations=n_permutations, quantile=quantile, seed=seed,
                    salt=zlib.crc32(nm.encode("utf-8")),
                    precomp_cards=_round_cards.get(nm), z_support_dev=z_support_dev,
                )
            cmi_excess = max(0.0, cmi - null_mean)
            scored[nm] = (cmi, cmi_excess)
            passes_floor = cmi > floor  # leg 1: significance
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
            strongly_significant = passes_floor and significance_escape_margin > 1.0 and floor > 0.0 and cmi >= significance_escape_margin * floor
            passes_rel = cmi_excess >= rel_bar  # leg 2: debiased relative gap
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
        candidate_support, _cand_card = _renumber_joint(*[*accepted_bins, new_bin])
        if _cand_card <= frag_cap:
            accepted_bins.append(new_bin)
            # Keep the resident-code list in lockstep so the NEXT round's device-born support join includes
            # this winner (None where it fell back to host binning -> that round takes the host support).
            accepted_bins_dev.append(cand_bins_dev.get(best_name) if cand_bins_dev else None)
            # accepted_bins now equals exactly the columns just joined above -> carry the join forward
            # for the next round's z_support build (see the top of this loop).
            _pending_z_support, _pending_z_card = candidate_support, _cand_card
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
