"""AUTO-ESCALATION of the pair-FE search to the richer SHIPPED bases (2026-06-10, backlog idea B).

When a prospective pair PASSED the pair-MI prescreen (joint-MI ratio gate + order-2 maxT
floor) but the unary/binary operator search admitted NOTHING for it, the legacy behaviour
was a WARNING ("FE produced 0 engineered features despite N pair(s) passing the pair-MI
gate") - the signal was DETECTED and then silently abandoned. This module escalates
instead: for each such pair it PROPOSES candidates from the two richer shipped basis
families and lets the EXISTING admission gates decide (escalation proposes, gates decide
-- the iron rule):

* SIGNAL-ADAPTIVE ORTHOGONAL-POLY pair warp: the rank-1 ALS per-operand warp
  (``hermite_fe.fit_pair_prewarp_als``) re-run at a HIGHER degree across the four shipped
  polynomial bases (chebyshev / hermite / legendre / laguerre), with the best basis
  selected by held-out rank-1-reconstruction |corr| on a deterministic stride slice -
  catches a poly inner the default degree-4 chebyshev prewarp under-fits or that the
  default prewarp's own held-out gate rejected at its fixed basis.

* ADAPTIVE-FREQUENCY FOURIER / CHIRP pair warp via DEMODULATION: for a multiplicative
  pair signal ``y ~ g(a) * b`` the univariate adaptive-Fourier detector sees nothing
  (``E[y | a] ~ 0`` when b is ~zero-mean); but the DEMODULATED target
  ``t = (y - mean(y)) * zscore(b)`` satisfies ``E[t | a] ~ g(a) * E[zscore(b)^2]``, so
  the SHIPPED held-out-validated multitone detector
  (``_orth_extra_basis_fe._detect_fourier_freqs_for_col``) run on ``(z01(a), t)`` locks
  g's frequency - e.g. the ``sin(3.7*a)`` INNER frequency no library unary can express.
  The fitted sin/cos mix is stored as a closed-form ``fourier_adaptive`` prewarp spec
  replayed by ``hermite_fe.apply_operand_prewarp`` (a pure function of x - leak-safe,
  no y at transform time). The CHIRP variant runs the same detector on the shipped
  quadratic-argument warp ``u = sign(z) * z**2`` so a growing-frequency inner is also
  reachable.

GATES (all existing - escalation only PROPOSES):
  1. held-out validation floors inside the proposers (the shipped detector's >= 0.30
     held-out periodogram floor / the ALS stride-slice reconstruction-|corr| floor);
  2. the Miller-Madow-debiased candidate MI must clear the order-2 maxT permutation
     floor computed over the prospective-pair pool (the same floor that gated the pair);
  3. a marginal-permutation MI floor (``_fe_cmi_redundancy_gate._conditional_perm_null``);
  4. the S5 conditional-MI redundancy gate over {already-admitted engineered survivors}
     UNION {escalation candidates} - a candidate redundant given the admitted support is
     dropped; verdicts are applied to ESCALATION candidates only (main-path admissions
     are never revoked here).
A pure-noise pair that slipped the prescreen by chance proposes nothing (the detectors
return no validated frequency, the ALS reconstruction fails the held-out floor) or is
rejected by floors 2-4 - measured 0 admissions on pure-noise controls (see
``tests/feature_selection/test_fe_auto_escalation.py``).

COST: structurally a no-op when every prescreen-surviving pair produced an admitted
engineered column (the common case - one set-difference). When it fires, the cost is a
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
from typing import Any, Sequence

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from ._fe_auto_escalation_proposers import (  # noqa: F401  -- carved sibling, re-exported
    _ADAPTIVE_F_GRID,
    _CHIRP_F_GRID,
    _ESCALATION_POLY_BASES,
    _IDENTITY_BASIS,
    _candidate_values,
    _finite_filled,
    _fit_fourier_amplitude_spec,
    _identity_prewarp_spec,
    _propose_fourier,
    _propose_poly,
)

__all__ = ["run_fe_auto_escalation", "find_underdelivering_pairs"]

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
    except Exception as exc:
        # See _candidate_values's matching fix above.
        logger.debug("_resolve_operand: column extraction failed for %r; skipping: %r", name, exc)
        return None
    return None


def find_underdelivering_pairs(
    self: Any,
    *,
    prospective_pairs: Any,
    prospective_additions: dict,
    X: Any,
    cols: Sequence[str],
    classes_y: Any,
    done: Any,
    max_rows: int = 20000,
    n_permutations: int = 8,
) -> list[tuple]:
    """UNDERDELIVERY trigger for the auto-escalation: pairs whose
    unary/binary search DID admit a column, but whose best admitted capture leaves
    SIGNIFICANT conditional pair signal on the table.

    Why a leftover-CMI test and not an MI-ratio bar: the prescreen ``pair_mi`` is a
    2-D joint MI over the (possibly adaptive) operand codes and structurally
    UNDER-estimates the pair information, so ``best_admitted_mi / pair_mi`` does not
    separate a weak envelope capture from a genuine one (measured on the
    ``y=sin(3.7*a)*b`` fixture: the junk ``mul(sin(a),qubed(b))`` envelope capture
    scores ratio 1.20 - ABOVE the genuine He3 capture's own scale). The leftover
    conditional MI ``CMI(joint(a,b) codes; y | best admitted column's codes)`` is the
    exact quantity of interest: ~bias when the capture is complete (He3 fixture),
    large when the library form only caught an envelope of the detected signal (the
    sin fixture, where most of ``sin(3.7a)*b`` lies beyond ``sin(a)*b**3``).

    TRIGGER (three legs): the leftover CMI must clear (1) the conditional-permutation
    null's quantile floor (same-bias null: the pair codes are permuted WITHIN
    admitted-code strata), (2) a small debiased-excess bar relative to the captured MI
    (``fe_escalation_underdelivery_excess_frac``, default 0.05) so a floor-grazing
    fluctuation cannot fire it, and (3) a DISCRETISATION-RESIDUAL control: even a
    functionally COMPLETE capture leaves leftover CMI in the 2-D joint, because its
    own ``nbins`` quantile code is coarse (within-bin variation of the captured value
    still predicts y) - so the joint's leftover must exceed
    ``fe_escalation_underdelivery_self_ratio`` (default 3.0) times the capture's OWN
    finer-binning refinement ``CMI(capture @ 2*nbins; y | capture @ nbins)``. A
    complete capture refines itself about as much as the joint refines it (measured:
    He3 perfect capture ratio 0.70, F2 a**2/b capture 0.83, F2 log*sin capture 2.44),
    while an envelope junk capture cannot (sin-fixture ``mul(sin(a),qubed(b))``
    measures 14.6 - most of ``sin(3.7a)*b`` lies beyond any binning of the envelope).
    A FALSE trigger is safe - escalation only PROPOSES and every candidate still
    faces the full admission gates (incl. the S5 CMI gate conditioned on the pair's
    own admitted column) - so the trigger is tuned cheap, not razor-sharp: all
    arrays are stride-subsampled to ``max_rows`` and the null uses
    ``n_permutations=8`` (the real gates re-verify at full rigor / full n).

    Returns ``[(pair_idx_tuple, pair_mi), ...]`` ready to append to the escalation's
    ``failed_pairs`` argument. Never raises (skips a pair on any internal hiccup)."""
    from ._fe_cmi_redundancy_gate import _conditional_perm_null
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin

    out: list = []
    n = len(X)
    if n <= 0:
        return out
    step = max(1, int(np.ceil(n / float(max_rows))))
    sl = slice(None, None, step)
    y_arr = np.asarray(classes_y)[sl]
    _, y_dense = np.unique(y_arr, return_inverse=True)
    y_dense = y_dense.astype(np.int64)
    if np.unique(y_dense).size < 2:
        return out
    nbq = int(self.quantization_nbins)
    # feature_names_in_ is an ndarray (sklearn convention); "or []" on it would test its truthiness and raise
    # ("truth value of an array... is ambiguous") once it holds more than one element - getattr's own default
    # already covers the missing-attribute case, so no "or" fallback is needed.
    raw_names = set(getattr(self, "feature_names_in_", []))
    eng_cont = getattr(self, "_engineered_continuous_", None)
    seed = int(getattr(self, "random_seed", 0) or 0)
    excess_frac = float(getattr(self, "fe_escalation_underdelivery_excess_frac", 0.05))
    self_ratio = float(getattr(self, "fe_escalation_underdelivery_self_ratio", 3.0))
    # Per-column quantile-bin memo: a popular raw column recurs across many prospective pairs
    # (this loop iterates the FULL prospective_pairs set), and its O(n log n) quantile-bin was
    # being recomputed once PER PAIR it participates in. Lazily built, keyed by column name,
    # reused across every pair referencing that column within this call.
    _qbin_memo: dict = {}

    def _qbin_cached(name, x):
        """Return column ``name``'s quantile-bin encoding, computing and memoizing it on first use."""
        c = _qbin_memo.get(name)
        if c is None:
            c = _quantile_bin(x[sl], nbins=nbq, host_only=True).astype(np.int64)
            _qbin_memo[name] = c
        return c

    for _k in prospective_pairs:
        try:
            pair, pair_mi = _k[0], float(_k[1])
            if pair in done:
                continue
            v = prospective_additions.get(pair)
            # Zero-admission pairs are the PRIMARY trigger's job; here we only look at
            # pairs that admitted something (values matrix + names present).
            if not v or not v[0] or v[1] is None or not v[2]:
                continue
            na, nb = cols[pair[0]], cols[pair[1]]
            if na not in raw_names or nb not in raw_names:
                continue
            x_a = _resolve_operand(X, na, eng_cont)
            x_b = _resolve_operand(X, nb, eng_cont)
            if x_a is None or x_b is None or x_a.size != n or x_b.size != n:
                continue
            ca = _qbin_cached(na, x_a)
            cb = _qbin_cached(nb, x_b)
            joint = ca * (int(cb.max()) + 1) + cb
            _, joint = np.unique(joint, return_inverse=True)
            joint = joint.astype(np.int64)
            # Conditioning support: the SINGLE best admitted column (by marginal MI on
            # the same subsample). Conditioning on one column keeps the null's strata
            # populated; a multi-column capture that is only jointly complete merely
            # causes a false trigger, which the downstream S5 gate (conditioned on ALL
            # admitted columns) absorbs.
            tvals, ncols_names = v[1], v[2]
            best_mi, best_codes, best_vals = -1.0, None, None
            for j in range(min(len(ncols_names), int(tvals.shape[1]))):
                vj = tvals[sl, j]
                if not hasattr(vj, "dev"):
                    vj = np.asarray(vj, dtype=np.float64)
                cj = _codes_of(vj, nbq, _quantile_bin)
                mij = float(_cmi_from_binned(cj, y_dense, None))
                if mij > best_mi:
                    best_mi, best_codes, best_vals = mij, cj, vj
            if best_codes is None or best_mi <= 0.0:
                continue
            assert best_vals is not None  # set together with best_codes above
            leftover = float(_cmi_from_binned(joint, y_dense, best_codes))
            floor, null_mean = _conditional_perm_null(
                joint, y_dense, best_codes,
                n_permutations=int(n_permutations),
                seed=seed + 1009 * int(pair[0]) + int(pair[1]),
            )
            if leftover <= floor or (leftover - null_mean) < excess_frac * best_mi:
                continue
            # Leg 3 - discretisation-residual control (see docstring): the capture's
            # OWN finer-binning refinement bounds the leftover a COMPLETE capture shows.
            cap_fine = _codes_of(best_vals, 2 * nbq, _quantile_bin)
            leftover_self = max(0.0, float(_cmi_from_binned(cap_fine, y_dense, best_codes)))
            if leftover > self_ratio * leftover_self:
                out.append((pair, pair_mi))
        except Exception as e:  # pragma: no cover - trigger must never break the FE step
            logger.debug("swallowed exception in _fe_auto_escalation.py: %s", e)
            continue
    return out


def _codes_of(values, nbins: int, quantile_bin) -> np.ndarray:
    """Host int64 equi-frequency codes of ``values``. A device-backed column is binned ON the device and only the narrow int8 codes are copied back (an
    eighth of the float64 column); a host column takes the host binner."""
    if hasattr(values, "dev") and nbins <= 127:
        try:
            import cupy as cp

            from ._mi_greedy_cmi_fe_binning import _quantile_bin_device

            x = values.dev
            if bool(cp.isfinite(x).all()):
                return np.asarray(cp.asnumpy(_quantile_bin_device(x, nbins).astype(cp.int8))).astype(np.int64)
        except Exception as e:
            logger.debug("device binning of the escalation capture failed, binning on the host: %s", e)
    return np.array(quantile_bin(np.asarray(values, dtype=np.float64), nbins=nbins, host_only=True), dtype=np.int64)


def _slice_admitted_pool(admitted_pool: dict, idx, classes_y_sub, nbins: int) -> dict:
    """Restrict the admitted-support pool ``{name: (values, marginal_mi)}`` to the escalation row subsample ``idx``.

    The marginals are re-estimated on the subsample with the estimator the escalation survivors use (``_quantile_bin`` +
    ``_cmi_from_binned`` against the densified subsample target). The caller computed them on the full frame, and plug-in MI is biased
    upward by roughly (k_x - 1)(k_y - 1) / 2n: kept as-is next to subsample-estimated survivor MIs in one redundancy-gate pool, they make
    the admitted support look weaker than the candidates and set the gate's bar from the wrong scale, in the direction that admits more.
    """
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin

    y_sub = np.asarray(classes_y_sub)
    if not np.issubdtype(y_sub.dtype, np.integer):
        y_sub = y_sub.astype(np.int64)
    _, y_dense = np.unique(y_sub, return_inverse=True)
    y_dense = y_dense.astype(np.int64)
    out: dict = {}
    for name, (values, _full_frame_marginal) in admitted_pool.items():
        sliced = np.asarray(values)[idx]
        out[name] = (sliced, float(_cmi_from_binned(_quantile_bin(np.asarray(sliced, dtype=np.float64), nbins=nbins, host_only=True), y_dense, None)))
    return out


def _skip_for_nominal_target(self, info: dict) -> bool:
    """Record a skip and return True when the fit target is nominal multiclass labels.

    The proposers fit Pearson-validated warps against the target rank, which for nominal class labels is the arbitrary label order, so a
    multiclass target admitted order-dependent columns (4-class make_classification: esc_poly of two informative raws, later fused over raws).
    """
    if not bool(getattr(self, "_fe_escalation_nominal_target_", False)):
        return False
    info["skipped"] = "nominal multiclass target: label order is not an ordinal signal"
    return True


def run_fe_auto_escalation(
    self: Any,
    *,
    failed_pairs: Any,
    X: Any,
    cols: Sequence[str],
    classes_y: Any,
    pair_maxt_floor: float,
    admitted_pool: dict,
    verbose: int = 0,
    capture_vals: dict | None = None,
    rescue_pairs: set | None = None,
) -> list[dict]:
    """Escalate the FE search to the richer shipped bases for prescreen-surviving pairs
    the unary/binary step admitted NOTHING for (or, via the UNDERDELIVERY trigger,
    admitted only a partial capture for). PROPOSES candidates (signal-adaptive
    orth-poly ALS warps + demodulated adaptive-frequency Fourier/chirp warps), then runs
    them through the EXISTING admission gates (order-2 maxT floor on MM-debiased MI,
    marginal-permutation floor, S5 conditional-MI redundancy gate vs the admitted
    engineered support).

    ``capture_vals`` (optional): per-pair matrix of the pair's ALREADY-ADMITTED
    engineered column values (full n). When present for a pair, the proposers fit the
    RESIDUAL of the supervised target after removing each capture column's BINNED
    CONDITIONAL MEAN (nonparametric - removes ANY function of the capture at bin
    resolution, crucially including the rank-vs-raw monotone remap a linear residual
    leaves behind), so they hunt for the MISSING part of the signal: a candidate that
    merely re-expresses the existing capture finds ~no residual correlation and dies
    at the proposers' held-out floors - measured on the He3(a)*b fixture, where the
    default prewarp capture's leftover triggers underdelivery but the full-target /
    lstsq-residual re-fits both produced a +0.0008-held-out-R^2 remap candidate (the
    S5 gate's train-side conditional MI admitted it); the binned-mean residual kills
    it at the proposal stage while the sin(3.7a)*b inner frequency - genuinely
    ABSENT from its envelope capture - survives residualisation and is recovered.

    Returns a list of admitted candidate dicts ``{name, values, recipe, mi, kind,
    pair}`` for the caller to materialise; stamps ``self.fe_escalation_info_``
    provenance. Never raises (degrades to ``[]``)."""
    from ._fe_cmi_redundancy_gate import apply_cmi_redundancy_gate
    from ._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin

    info: dict = {"eligible_pairs": [], "proposed": 0, "admitted": [], "rejected": {}, "pair_maxt_floor": float(pair_maxt_floor)}
    self.fe_escalation_info_ = info
    # Accumulated per-call history (one entry per FE step that ran escalation), so a
    # late no-op step does not erase the provenance of an earlier admitting step.
    if not isinstance(getattr(self, "fe_escalation_history_", None), list):
        self.fe_escalation_history_ = []
    self.fe_escalation_history_.append(info)
    if not failed_pairs or _skip_for_nominal_target(self, info):
        return []

    n_rows = len(X)
    min_rows = int(getattr(self, "fe_escalation_min_rows", 500))
    if n_rows < min_rows:
        info["skipped"] = f"n_rows={n_rows} < fe_escalation_min_rows={min_rows}"
        return []

    raw_names = set(getattr(self, "feature_names_in_", []))
    eng_cont = getattr(self, "_engineered_continuous_", None)
    max_pairs = int(getattr(self, "fe_escalation_max_pairs", 8))
    min_val_corr = float(getattr(self, "fe_escalation_min_val_corr", 0.15))
    poly_degree = int(getattr(self, "fe_escalation_poly_degree", 6))
    max_freqs = int(getattr(self, "fe_escalation_fourier_max_freqs", 3))
    per_pair_cap = int(getattr(self, "fe_escalation_max_candidates_per_pair", 3))
    seed = int(getattr(self, "random_seed", 0) or 0)
    nbins = int(self.quantization_nbins)

    # SUBSAMPLED DECISION. The escalation proposers (orth-poly ALS warp fit +
    # adaptive Fourier/chirp periodogram DETECTION) are the dominant active orth-FE CPU cost
    # and ran on the FULL frame. Decide on the SAME row-subsample the rest of FE uses
    # (fe_check_pairs_subsample_n + random_seed), then rebuild each ADMITTED candidate's
    # ``values`` at full n via its closed-form recipe before returning (output-safe). Operands,
    # target, residualisation captures and the admitted-support pool are all subsampled in
    # lockstep so the gates decide on a consistent slice. Default off -> full-data decision.
    _X_full = X
    _esc_ss_n = int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0)
    _esc_do_sub = isinstance(_esc_ss_n, int) and 0 < _esc_ss_n < n_rows
    _y_rank_eff = getattr(self, "_fe_escalation_y_rank_", None)
    if _esc_do_sub:
        _esc_idx = np.sort(np.random.default_rng(seed).choice(n_rows, size=int(_esc_ss_n), replace=False))
        X = X.iloc[_esc_idx].reset_index(drop=True) if hasattr(X, "iloc") else np.asarray(X)[_esc_idx]
        classes_y = np.asarray(classes_y)[_esc_idx]
        if capture_vals:
            capture_vals = {k: np.asarray(v)[_esc_idx] for k, v in capture_vals.items()}
        if admitted_pool:
            admitted_pool = _slice_admitted_pool(admitted_pool, _esc_idx, classes_y, nbins)
        if _y_rank_eff is not None and np.asarray(_y_rank_eff).shape[0] == n_rows:
            _y_rank_eff = np.asarray(_y_rank_eff)[_esc_idx]
        n_rows = int(_esc_ss_n)

    # Target for the supervised warp fits. PREFER the rank-transformed raw y stashed
    # by ``_fit_impl`` (``_fe_escalation_y_rank_``): the FE step's ``classes_y`` are
    # LABEL codes from the internal target quantisation - NOT guaranteed ordinal /
    # monotone in y (measured 37 unordered codes on a heavy-tailed regression y) -
    # which silently destroys a Pearson-validated ALS / periodogram fit (held-out
    # corr 0.42 on the genuine (c,d) term with rank-y vs ~0 with the label codes).
    # The rank is monotone-equivalent to y and heavy-tail-robust; fall back to the
    # codes when the stash is unavailable (multi-output / non-numeric y).
    _y_rank = _y_rank_eff
    if _y_rank is not None and np.asarray(_y_rank).shape[0] == n_rows:
        y_f = np.ascontiguousarray(_y_rank, dtype=np.float64)
    else:
        y_f = np.ascontiguousarray(np.asarray(classes_y), dtype=np.float64)
    y_arr = np.asarray(classes_y)
    if not np.issubdtype(y_arr.dtype, np.integer):
        y_arr = y_arr.astype(np.int64)
    _, y_dense = np.unique(y_arr, return_inverse=True)
    y_dense = y_dense.astype(np.int64)

    # RAW-RAW pairs only, bounded by the pair budget. A prevalence-failed-synergy rescue
    # pair (a genuine SMOOTH ratio interaction the raw-MI ratio under-rates) has LOW raw
    # joint MI BY CONSTRUCTION, so a plain joint-MI sort buries it below the zero-admission
    # cross-mix pairs and the ``max_pairs`` cap drops it (measured on F2: the genuine (a,b)
    # at joint MI 0.028 was squeezed out by 6 higher-MI cross pairs) - rescue pairs get a
    # RESERVED half of the budget so this can't happen.
    #
    # HALF-RESERVED, not absolute priority: an earlier version sorted rescue pairs strictly
    # ahead of everything else, which starves the OTHER direction - when many noise-driven
    # cross pairs spuriously clear the prevalence-failed-synergy gate (var-vs-noise-column
    # combos landing just below the raw-MI ratio threshold by chance) they can fill the
    # entire budget and squeeze out a zero-admission pair with a far higher joint MI than
    # any of them (measured on the sin(3.7*a)*b fixture: pair_mi=1.42, 8/8 higher than every
    # rescue candidate's 0.19-0.56, still dropped because 8 rescue pairs alone filled
    # max_pairs=8). Reserving only half the budget for rescue keeps the F2-style low-MI
    # genuine rescue guaranteed a claim while a high-MI zero-admission pair can never be
    # fully starved by an arbitrarily large rescue count.
    _rescue = {tuple(p) for p in (rescue_pairs or set())}
    eligible_all = []
    for pair, pair_mi in failed_pairs:
        try:
            na, nb = cols[pair[0]], cols[pair[1]]
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            continue
        if na in raw_names and nb in raw_names:
            eligible_all.append((pair, float(pair_mi), na, nb))
    eligible_all.sort(key=lambda e: e[1], reverse=True)
    _rescue_cap = max(1, max_pairs // 2)
    eligible: list = []
    _others: list = []
    _rescue_taken = 0
    _split_rescue_eligible(eligible_all, _rescue, _rescue_taken, _rescue_cap, eligible, _others)
    eligible.extend(_others[: max_pairs - len(eligible)])
    eligible.sort(key=lambda e: (tuple(e[0]) in _rescue, e[1]), reverse=True)
    eligible = eligible[:max_pairs]
    info["eligible_pairs"] = [(na, nb) for _, _, na, nb in eligible]
    # Raw cols-space index tuples of the processed pairs - the caller's per-fit
    # dedup ledger key (stable across FE steps: engineered columns append at the end).
    info["eligible_idx"] = [tuple(pair) for pair, _, _, _ in eligible]
    if not eligible:
        return []

    existing_names = set(cols) | set(admitted_pool)
    candidates: list[dict] = []
    for pair, _pair_mi, na, nb in eligible:
        x_a = _resolve_operand(X, na, eng_cont)
        x_b = _resolve_operand(X, nb, eng_cont)
        if x_a is None or x_b is None or x_a.size != n_rows or x_b.size != n_rows:
            continue
        pair_cands: list[dict] = []
        # RESIDUAL fitting target for UNDERDELIVERY-triggered pairs (see docstring):
        # remove EVERYTHING a function of the pair's already-admitted capture can
        # explain, so the proposers hunt for the MISSING part only. The removal is the
        # per-column BINNED CONDITIONAL MEAN (not lstsq): the supervised target is
        # rank-y while the capture is raw-valued, so a LINEAR residual leaves the
        # monotone remap of the capture itself in the residual and a proposer happily
        # "recovers" that deterministic function of the existing capture (measured on
        # He3(a)*b: laguerre val_corr 0.72 on the lstsq residual, +0.0008 held-out R^2
        # - pure remap; the binned-mean removal kills it while the sin(3.7a)*b inner
        # frequency, genuinely absent from its envelope capture, survives).
        # Zero-admission pairs (no capture) fit the full target as before.
        y_pair = y_f
        _cv = (capture_vals or {}).get(tuple(pair))
        y_pair = _pairwise_control_matrix(_cv, n_rows, y_f, y_pair)
        # 1) Signal-adaptive orth-poly ALS warp (higher degree + 4-basis routing).
        poly = _propose_poly(
            x_a, x_b, y_pair, degree=poly_degree, min_val_corr=min_val_corr,
            pairness_margin=float(getattr(self, "fe_escalation_pairness_margin", 1.15)),
        )
        _poly_candidate_values(poly, x_a, x_b, pair_cands, na, nb)
        # 2) Demodulated adaptive-frequency Fourier / chirp, both warp directions.
        _propose_fourier_both_warps(x_a, x_b, na, nb, y_pair, min_val_corr, max_freqs, pair_cands)
        # Score by the SAME MM-debiased plug-in MI the gates use; cap per pair.
        for c in pair_cands:
            vb = _quantile_bin(np.asarray(c["values"], dtype=np.float64), nbins=nbins, host_only=True)
            c["_binned"] = vb
            c["mi"] = float(_cmi_from_binned(vb, y_dense, None))
        pair_cands.sort(key=lambda c: c["mi"], reverse=True)
        candidates.extend(pair_cands[: max(1, per_pair_cap)])

    info["proposed"] = len(candidates)
    if not candidates:
        return []

    # Deduplicate names defensively (two pairs sharing operands cannot collide on the
    # name template, but an operand name containing "," could).
    seen: set = set()
    _uniquify_candidate_names(candidates, existing_names, seen)

    # GATE 2: order-2 maxT permutation floor (MM-debiased MI scale on BOTH sides -
    # the floor was computed with miller_madow=True, ``_cmi_from_binned`` debiases too).
    # GATE 3: marginal-permutation floor (same primitive the S5 gate's significance leg
    # uses) - protects the degenerate single-candidate path where the S5 gate would
    # otherwise admit on marginal significance alone.
    survivors: list[dict] = []
    _apply_pair_maxt_floor(candidates, pair_maxt_floor, info, y_dense, seed, survivors)
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
    _admit_cmi_survivors(self, survivors, accepted, info, admitted)
    # FULL-n OUTPUT: when the DECISION ran on a subsample, the candidate ``values`` are
    # subsample-length - rebuild each admitted candidate's column on the full X via its
    # closed-form recipe so the caller materialises the full-n column (output equals a
    # full-data fit given the same admitted set). A candidate whose full replay fails is
    # dropped (it would otherwise inject a wrong-length column).
    #
    # SELECTION-EQUIVALENCE NOTE (P1-5/P1-6): the orth-poly proposers gate on subsample values computed via
    # polyeval_dispatch at the SMALL subsample n (njit/Horner), while this replay rebuilds at full n where
    # the dispatch may pick the CUDA recurrence - which differs from njit-Horner by ~1e-12 for cheb/leg/herme
    # (see _gpu_resident_fe P2-2 note; laguerre is forward on both). So a near-FLOOR esc-poly admit decided on
    # Horner values ships a column whose binned MI can differ by that ~1e-12. This is far below the gate's
    # effective resolution (min_val_corr / pairness_margin), and escalation admits ~nothing at the canonical
    # fit anyway (the interleaved A/B above records the same eligible pairs + 0 proposed); the decide->replay
    # set is unchanged. Pinning one polyeval backend across decide+replay is a FUTURE change, unneeded here.
    admitted = _rebuild_admitted_substitutes(_esc_do_sub, admitted, _X_full)
    if verbose and (admitted or info["proposed"]):
        logger.info(
            "MRMR FE auto-escalation: %d pair(s) had 0 admitted engineered features after the "
            "unary/binary search; proposed %d richer-basis candidate(s) (orth-poly ALS x4 bases "
            "+ demodulated adaptive Fourier/chirp), gates admitted %d: %s",
            len(eligible), info["proposed"], len(admitted),
            [f"{c['name']} (mi={c['mi']:.4f})" for c in admitted],
        )
    return admitted


def _uniquify_candidate_names(candidates, existing_names, seen):
    """Rename the candidates so their names are unique."""
    for c in candidates:
        base = c["name"]
        k = 2
        while c["name"] in existing_names or c["name"] in seen:
            c["name"] = f"{base}_{k}"
            k += 1
        seen.add(c["name"])


def _admit_cmi_survivors(self, survivors, accepted, info, admitted):
    """Build the recipes of the survivors the CMI gate accepted and admit them."""
    from mlframe.feature_selection.filters.engineered_recipes import build_unary_binary_recipe

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


def _split_rescue_eligible(eligible_all, _rescue, _rescue_taken, _rescue_cap, eligible, _others):
    """Keep the rescued candidates within the rescue cap."""
    for _cand in eligible_all:
        if tuple(_cand[0]) in _rescue and _rescue_taken < _rescue_cap:
            eligible.append(_cand)
            _rescue_taken += 1
        else:
            _others.append(_cand)


def _poly_candidate_values(poly, x_a, x_b, pair_cands, na, nb):
    """Evaluate the polynomial candidate of a pair when one was proposed."""
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


def _pairwise_control_matrix(_cv, n_rows, y_f, y_pair):
    """Build the low-dimensional control matrix from the conditioning values."""
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin

    if _cv is not None:
        try:
            _A = np.asarray(_cv, dtype=np.float64).reshape(n_rows, -1)[:, :8]
            _r = np.asarray(y_f, dtype=np.float64).copy()
            _nb_res = int(min(32, max(8, n_rows // 64)))
            for _j in range(_A.shape[1]):
                _cb = _quantile_bin(np.nan_to_num(_A[:, _j], nan=0.0, posinf=0.0, neginf=0.0), nbins=_nb_res, host_only=True).astype(np.int64)
                _cnt = np.maximum(np.bincount(_cb, minlength=int(_cb.max()) + 1), 1)
                _means = np.bincount(_cb, weights=_r, minlength=int(_cb.max()) + 1) / _cnt
                _r = _r - _means[_cb]
            if float(np.std(_r)) > 1e-9:
                y_pair = _r
        except Exception:
            # No silent swallow: a failure here means we fall back to the FULL target instead of the
            # residual, which defeats residualisation (the proposer re-proposes already-captured signal).
            logger.debug("fe-escalation residualisation failed; using full target", exc_info=True)
    return y_pair


def _propose_fourier_both_warps(x_a, x_b, na, nb, y_pair, min_val_corr, max_freqs, pair_cands):
    """Propose demodulated Fourier / chirp candidates for both warp directions."""
    for x_w, x_m, nw, nm in ((x_a, x_b, na, nb), (x_b, x_a, nb, na)):
        for prop in _propose_fourier(x_w, x_m, y_pair, min_val_corr=min_val_corr, max_freqs=max_freqs, chirp=True):
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


def _apply_pair_maxt_floor(candidates, pair_maxt_floor, info, y_dense, seed, survivors):
    """Reject the candidates below the pair max-T floor."""
    from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import _conditional_perm_null

    for c in candidates:
        if pair_maxt_floor > 0.0 and c["mi"] < float(pair_maxt_floor):
            info["rejected"][c["name"]] = f"below_maxt_floor (mi={c['mi']:.5f} < {pair_maxt_floor:.5f})"
            continue
        floor_m, _null_mean = _conditional_perm_null(c["_binned"], y_dense, None, seed=seed)
        if c["mi"] <= floor_m:
            info["rejected"][c["name"]] = f"below_marginal_perm_floor (mi={c['mi']:.5f} <= {floor_m:.5f})"
            continue
        survivors.append(c)


def _rebuild_admitted_substitutes(_esc_do_sub, admitted, _X_full):
    """Rebuild the admitted candidates through the recipe replay."""
    if _esc_do_sub and admitted:
        from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
        _rebuilt: list[dict] = []
        for c in admitted:
            try:
                c["values"] = np.asarray(apply_recipe(c["recipe"], _X_full), dtype=np.float64)
                _rebuilt.append(c)
            except Exception:  # noqa: PERF203 - per-iteration fault isolation is intentional, not a hoisting candidate
                log_throttle(
                    logger,
                    "fe_auto_escalation_replay_failed",
                    logging.WARNING,
                    "MRMR FE auto-escalation: full-n replay failed for %r; dropping.",
                    c.get("name"),
                )
        admitted = _rebuilt
    return admitted
