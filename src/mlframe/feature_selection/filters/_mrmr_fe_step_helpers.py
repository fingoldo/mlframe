"""Self-contained FE-step sub-blocks carved out of ``_mrmr_fe_step._run_fe_step``.

Two blocks with explicit inputs/outputs (no shared-local threading with the surrounding orchestrator beyond their declared parameters):

* ``compute_pair_maxt_floor`` -- the order-2 Westfall-Young permutation-null joint-MI floor over the whole candidate-pair pool. Returns ``(floor, per_pair_mm_bias)``: the float floor applied as an extra gate in both the XOR and uplift branches of the pair loop, plus (when ``fe_mm_debias_prevalence``) the per-pair Miller-Madow joint-MI bias map subtracted from the observed ``pair_mi`` at the floor comparison so the floor stays on the same debiased scale as the prevalence ratio gate (IRON RULE: consistent debias on both sides).
* ``run_cluster_aggregate_emission`` -- the opt-in clustered-feature aggregation emission block (runs once on the first FE step; returns the updated ``(data, cols, nbins, X, selected_vars, n_recommended_features)`` tuple and stamps the fitted summary onto ``self``).

Both take the ``MRMR`` instance as ``self`` so they read the frozen ``fe_*`` / ``cluster_aggregate_*`` config off it exactly as the inlined blocks did. The heavy kernels stay in their own modules (``_permutation_null``, ``_cluster_aggregate``) and are lazy-imported in-body to avoid import cycles.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def apply_synergy_bootstrap(
    self,
    *,
    num_fs_steps: int,
    data,
    cols,
    target_indices,
    categorical_vars,
    numeric_vars_to_consider: set,
    non_numeric_idx: set,
    verbose,
):
    """Synergy bootstrap: widen the FE pair pool with unselected raw numeric columns so zero-marginal synergy pairs get joint-MI screened.

    Pure-synergy interactions (a*d, sign products, log(c)*sin(d)) have ~zero marginal MI per factor, so neither factor is screened-in and the pair never reaches the prospective-pair
    screen -- even though that screen ALREADY keeps a zero-individual-MI pair whose JOINT MI is positive (the canonical XOR branch). The fix is purely to widen the POOL: when the raw
    numeric feature count is within ``fe_synergy_screen_max_features``, add the UNSELECTED raw numeric columns so the all-pairs joint-MI sweep screens the synergy pairs. Two cost/quality
    guards live downstream: (1) synergy pairs (>=1 bootstrap-added operand) must clear the STRICTER ``fe_synergy_min_prevalence`` uplift bar (rejects finite-sample-bias noise pairs), and
    (2) the surviving synergy pairs are budget-capped to ``fe_synergy_max_pairs`` by joint MI. Runs only on the FIRST FE step.

    Gated by a MIN-ROWS guard (``fe_synergy_min_rows``: a 2-D joint-MI estimate is finite-sample-bias-dominated at tiny n, admitting pure-NOISE pairs) and an N-AWARE COST GATE
    (``fe_synergy_max_sweep_cost`` ~ n*p^2: the O(p^2*n) sweep grows super-linearly in n; default 5e8 fires on the measured wins but SKIPS large-n blow-ups; set to inf to disable).

    bench-rejected (2026-06-03) -- GAP-3 "poly-feature synergy re-entry": feeding ENGINEERED univariate poly features (He2(a) etc.) back into this bootstrap pool so a synergy like
    y=He2(a)*b surfaces was benchmarked and REJECTED. Pool entry is NOT the blocker -- the raw (a,b) pair already reaches the prospective-pair screen as the top-joint-MI pair, and the
    per-pair unary search builds He2(a)*b from it. poly x raw classification is already recovered 6/6 via the default-on prewarp operand; the genuine miss is poly x raw regression (1/6),
    blocked simultaneously by four noise-control gates, and forcing recovery would loosen the gates ``test_biz_val_mrmr_default_filtering.py`` pins for a fragile 2/6. Don't re-add it.

    Returns ``(numeric_vars_to_consider, synergy_added_idx)`` -- the (possibly widened) pool and the set of bootstrap-added operand indices (empty when the bootstrap did not fire).
    """
    synergy_cap = int(getattr(self, "fe_synergy_screen_max_features", 0) or 0)
    synergy_added_idx: set = set()
    # Reset per call so the exhaustive-active flag never leaks across FE steps / fits.
    self._fe_synergy_exhaustive_active_ = False
    synergy_min_rows = int(getattr(self, "fe_synergy_min_rows", 300) or 0)
    n_rows_for_synergy = int(data.shape[0]) if hasattr(data, "shape") else 0
    synergy_max_sweep_cost = float(getattr(self, "fe_synergy_max_sweep_cost", 5e8) or float("inf"))
    if synergy_cap > 0 and num_fs_steps == 0 and n_rows_for_synergy >= synergy_min_rows:
        _raw_names = set(getattr(self, "feature_names_in_", []) or [])
        _target_idx_set = {int(t) for t in np.atleast_1d(target_indices)}
        _cat_set = set(categorical_vars)
        _raw_numeric_idx = {
            i for i, nm in enumerate(cols)
            if nm in _raw_names and i not in _target_idx_set and i not in _cat_set
        }
        _raw_numeric_idx -= non_numeric_idx
        if self.factors_to_use is not None:
            _raw_numeric_idx &= set(self.factors_to_use)
        if self.factors_names_to_use is not None:
            _raw_numeric_idx &= {cols.index(n) for n in self.factors_names_to_use if n in cols}
        _n_raw = len(_raw_numeric_idx)
        # SECOND FUNNEL STAGE -- GPU-EXHAUSTIVE SYNERGY SWEEP (2026-06-19). The pre-rank below is an O(p)
        # propensity score; it PROVABLY cannot recover a perfectly-balanced (L=0) interaction whose every
        # univariate higher moment vs y is zero (balanced XOR / sign product). Only the EXHAUSTIVE C(p,2)
        # joint-MI sweep recovers such a pair. When ``fe_synergy_exhaustive`` is "force"/True AND a CUDA GPU is
        # available AND the predicted wall-time (measured CUDA pairs/s via kernel_tuning_cache, ~5e4 fallback) is
        # under ``fe_synergy_exhaustive_max_seconds`` (default 180s), bypass the cap + pre-rank + sweep-cost gate
        # and sweep ALL raw numeric columns. "auto" (default) keeps the pre-rank + capped behaviour below.
        _exhaustive_on = False
        try:
            from ._fe_synergy_exhaustive import decide_exhaustive_sweep

            _exhaustive_on, _exh_reason = decide_exhaustive_sweep(
                self, n_samples=n_rows_for_synergy, n_raw=_n_raw, verbose=verbose,
            )
            if verbose:
                logger.info("MRMR FE synergy exhaustive: %s", _exh_reason)
        except Exception as _exh_e:
            if verbose:
                logger.info("MRMR FE synergy exhaustive decision degraded (%s: %s); using pre-rank path.",
                            type(_exh_e).__name__, _exh_e)
        # WIDE-FRAME PRE-RANK (2026-06-19). Above the cap the bootstrap historically SKIPPED entirely, so a
        # zero-marginal interaction on a wide frame (p >> cap) was engineered as NOTHING. Marginal MI cannot
        # pick the surviving cap columns (the operands have ~0 marginal MI by construction -- the whole reason
        # the bootstrap exists). Instead rank by an interaction-propensity score |corr(x^2,y)|+|corr(x,y^2)|
        # (higher moments leak even when the linear marginal is flat) and keep the top ``synergy_cap`` so the
        # exhaustive O(cap^2) sweep runs on the columns most likely to carry interaction signal. Bench
        # (2026-06-18, test_fe_interaction_prerank): recovers the planted operands into the top-250 at recall
        # ~0.88 (L=0.1) vs marginal-MI 0.68 / random 0.12, at O(p*n) ~5s for p=10k. IRREDUCIBLE: a perfectly
        # balanced zero-higher-moment interaction (L=0) is invisible to any O(p) score and still needs the full
        # exhaustive sweep -- the pre-rank does not claim it. Default ON (fe_synergy_prerank); set False to
        # restore the legacy skip-past-cap behaviour. The ranking uses the discretised ``data`` codes (a
        # monotone transform preserves the higher-moment structure; bench-confirmed code-path recall holds).
        _prerank_on = bool(getattr(self, "fe_synergy_prerank", True))
        if _exhaustive_on:
            # FULL exhaustive sweep selected: treat EVERY raw numeric column as in-pool. Lift the cap to _n_raw
            # (no pre-rank trimming) and disable the n*p^2 cost gate so the all-pairs joint-MI sweep runs over all
            # columns -- the only path that recovers a balanced (L=0) interaction. The exhaustive CUDA kernel is
            # the dispatch target downstream (_step_core force_backend="cuda").
            synergy_cap = _n_raw
            synergy_max_sweep_cost = float("inf")
            self._fe_synergy_exhaustive_active_ = True
        elif _prerank_on and _n_raw > synergy_cap and (
            n_rows_for_synergy * (synergy_cap ** 2) > synergy_max_sweep_cost
        ):
            # COST-GATE FIRST (2026-06-19, critique #4). After the pre-rank keeps exactly ``synergy_cap``
            # columns the sweep cost is n*cap^2; if THAT already exceeds the budget the sweep below will be
            # skipped regardless -- so do NOT pay the O(p*n) pre-rank (which also risks OOM on a very wide
            # frame at large n). Skip straight to the legacy log path.
            if verbose:
                logger.info(
                    "MRMR FE synergy bootstrap: %d raw cols > cap %d, but the post-pre-rank sweep cost "
                    "n*cap^2=%.2g already exceeds fe_synergy_max_sweep_cost=%.2g; skipping the pre-rank AND "
                    "the sweep (raise fe_synergy_max_sweep_cost to force it).",
                    _n_raw, synergy_cap, float(n_rows_for_synergy * (synergy_cap ** 2)), synergy_max_sweep_cost,
                )
        elif _prerank_on and _n_raw > synergy_cap:
            try:
                from ._fe_interaction_prerank import top_k_by_interaction_propensity

                _ty = int(np.atleast_1d(target_indices)[0])
                _y_codes = np.asarray(data)[:, _ty]
                # Thread MRMR's own time budget so the "auto" criterion can size-gate the high-recall
                # gbm-fused pre-rank (max_runtime_mins * 60); None => the pre-rank's soft default budget.
                _mins = getattr(self, "max_runtime_mins", None)
                try:
                    _budget_s = float(_mins) * 60.0 if (_mins is not None and float(_mins) > 0) else None
                except (TypeError, ValueError):
                    _budget_s = None
                _kept = top_k_by_interaction_propensity(
                    np.asarray(data), _y_codes, _raw_numeric_idx, top_k=synergy_cap,
                    budget_seconds=_budget_s,
                )
                if _kept:
                    _raw_numeric_idx = set(_kept)
                    _n_raw = len(_raw_numeric_idx)
                    if verbose:
                        logger.info(
                            "MRMR FE synergy bootstrap: wide frame (>%d raw numeric cols); interaction-propensity "
                            "pre-rank kept the top %d by |corr(x^2,y)|+|corr(x,y^2)| (marginal MI cannot rank "
                            "zero-marginal operands). A perfectly-balanced interaction is irreducible to this "
                            "O(p) score; set fe_synergy_prerank=False to restore the legacy skip-past-cap.",
                            synergy_cap, _n_raw,
                        )
            except Exception as _e:  # correctness over the optimisation -- fall back to the legacy skip
                if verbose:
                    logger.info("MRMR FE synergy pre-rank degraded (%s: %s); using legacy skip-past-cap.",
                                type(_e).__name__, _e)
        _sweep_cost = n_rows_for_synergy * (_n_raw ** 2)
        if 0 < _n_raw <= synergy_cap and _sweep_cost <= synergy_max_sweep_cost:
            _added = _raw_numeric_idx - numeric_vars_to_consider
            if _added:
                synergy_added_idx = set(_added)
                numeric_vars_to_consider = numeric_vars_to_consider | _raw_numeric_idx
                if verbose:
                    logger.info(
                        "MRMR FE synergy bootstrap: augmented pair pool with %d unselected raw "
                        "numeric columns (%d raw <= cap %d) so zero-marginal synergy pairs "
                        "(a*d / sign products / log*sin) get joint-MI screened.",
                        len(_added), _n_raw, synergy_cap,
                    )
        elif 0 < _n_raw <= synergy_cap and _sweep_cost > synergy_max_sweep_cost and verbose:
            logger.info(
                "MRMR FE synergy bootstrap: %d raw numeric cols <= cap %d but sweep cost n*p^2=%.2g exceeds "
                "fe_synergy_max_sweep_cost=%.2g (O(p^2*n) blow-up on large n); skipping the all-pairs sweep. "
                "Raise fe_synergy_max_sweep_cost to force it.",
                _n_raw, synergy_cap, float(_sweep_cost), synergy_max_sweep_cost,
            )
        elif _n_raw > synergy_cap and verbose:
            logger.info(
                "MRMR FE synergy bootstrap: %d raw numeric columns > cap %d; skipping the "
                "all-pairs synergy sweep (keeping the selected-only pool). Raise "
                "fe_synergy_screen_max_features to enable it on this frame.",
                _n_raw, synergy_cap,
            )
    return numeric_vars_to_consider, synergy_added_idx


def apply_surrogate_gbm_seeder(
    self: Any,
    *,
    num_fs_steps: int,
    data: Any,
    nbins: Any,
    cols: Sequence[str],
    categorical_vars: Any,
    target_indices: Any,
    classes_y: Any,
    freqs_y: Any,
    numeric_vars_to_consider: set,
    non_numeric_idx: set,
    verbose: int,
) -> tuple[set, list]:
    """Surrogate-GBM split-co-occurrence interaction seeder (backlog #6) + its order-3 maxT rail (#7).

    Fits one shallow LightGBM on the discretised matrix, walks root-to-leaf paths, and tallies
    depth-discounted split-gain co-occurrence to propose interaction PAIRS + TRIPLES whose operands
    have ~0 univariate MI (so the univariate ``seed_count`` never reaches them). The proposer
    SELF-GATES on a permuted-y OOF comparison (pure noise -> no seeds). Seeded PAIRS are MERGED into
    ``numeric_vars_to_consider`` so their joint MI is screened by the existing pair pipeline (the
    seed_count bypass). Seeded TRIPLES are gated by the order-3 Westfall-Young maxT permutation-null
    floor (``batch_triple_mi_prange``-based, the #7 rail) and the survivors stamped onto
    ``self._seeded_triplets_`` for the triplet FE stage. Runs only on the FIRST FE step.

    Returns ``(numeric_vars_to_consider, seeded_pairs)`` -- the (possibly widened) operand pool and
    the surviving seeded pair-index tuples (empty when the seeder is off / self-gate fails). The
    seeded triples + raw seeder diagnostics are stamped on ``self`` (``_seeded_pairs_`` /
    ``_seeded_triplets_`` / ``fe_gbm_seeder_info_``). OPT-IN (``fe_gbm_seeder_enable``); self-routes
    OFF below ``fe_gbm_seeder_min_features`` columns where seed_count is not the blocker.
    """
    self._seeded_pairs_ = []
    self._seeded_triplets_ = []
    self._seeded_triplets_names_ = []
    self.fe_gbm_seeder_info_ = {}
    if not bool(getattr(self, "fe_gbm_seeder_enable", False)) or num_fs_steps != 0:
        return numeric_vars_to_consider, []

    # Self-routing cost gate: seed_count is not the blocker on a narrow pool (it already
    # sees every operand), so the LightGBM fit cost is not worth paying there.
    _raw_name_set = set(getattr(self, "feature_names_in_", []) or [])
    _target_idx_set = {int(t) for t in np.atleast_1d(target_indices)}
    _cat_set = set(categorical_vars)
    _raw_numeric_idx = {
        i for i, nm in enumerate(cols)
        if nm in _raw_name_set and i not in _target_idx_set and i not in _cat_set
    }
    _raw_numeric_idx -= set(non_numeric_idx)
    if self.factors_to_use is not None:
        _raw_numeric_idx &= set(self.factors_to_use)
    if self.factors_names_to_use is not None:
        _raw_numeric_idx &= {cols.index(n) for n in self.factors_names_to_use if n in cols}
    _min_feats = int(getattr(self, "fe_gbm_seeder_min_features", 30))
    if len(_raw_numeric_idx) < _min_feats:
        if verbose:
            logger.info(
                "MRMR FE GBM seeder: %d raw numeric cols < fe_gbm_seeder_min_features=%d; "
                "skipping (seed_count already sees every operand on this narrow pool).",
                len(_raw_numeric_idx), _min_feats,
            )
        return numeric_vars_to_consider, []

    try:
        from ._surrogate_interaction_seeder import surrogate_gbm_interaction_seeds

        # The seeder's candidate pool is ALL raw numeric columns (the operands the
        # univariate seed_count ranks among) -- the whole point is to reach the
        # zero-marginal ones the top-N cut drops.
        _cand = sorted(_raw_numeric_idx)
        # The surrogate predicts the DISCRETISED ordinal target ``classes_y`` -- the SAME
        # small-cardinality representation every FE-gate MI is scored against -- so the
        # split co-occurrence is on the exact signal the pair/triple floors gate. The binned
        # target is always a low-cardinality ordinal (a multiclass classification problem for
        # the surrogate), whether the original task was classification or regression.
        seeded_pairs, seeded_triples, info = surrogate_gbm_interaction_seeds(
            data, np.ascontiguousarray(classes_y), _cand,
            is_classification=True,
            top_k_pairs=int(getattr(self, "fe_gbm_seeder_top_k_pairs", 12)),
            top_k_triples=int(getattr(self, "fe_gbm_seeder_top_k_triples", 8)),
            n_estimators=int(getattr(self, "fe_gbm_seeder_n_estimators", 150)),
            max_depth=int(getattr(self, "fe_gbm_seeder_max_depth", 4)),
            self_gate_margin=float(getattr(self, "fe_gbm_seeder_self_gate_margin", 0.0)),
            self_gate_reps=int(getattr(self, "fe_gbm_seeder_self_gate_reps", 3)),
            self_gate_min_z=float(getattr(self, "fe_gbm_seeder_self_gate_min_z", 2.0)),
            random_seed=int(getattr(self, "random_seed", 0) or 0),
        )
        self.fe_gbm_seeder_info_ = {k: v for k, v in info.items()
                                    if k in ("oof_real", "oof_perm", "self_gate_z", "gated", "n_pairs", "n_triples")}
        # The seeder fit failed entirely (no surrogate) -> nothing to do. ``gated`` is the PAIR
        # self-gate; TRIPLES are emitted regardless (the order-3 floor gates them), so we do NOT
        # early-return on ``gated=False`` -- only when there is no co-occurrence output at all.
        if not seeded_pairs and not seeded_triples:
            return numeric_vars_to_consider, []

        # --- ORDER-3 maxT FLOOR (#7) on the seeded triples (the mandatory rail + binding
        # noise guard for 3-way, since the OOF self-gate is blind to hard 3-way signal) ---
        _kept_triples = _gate_seeded_triples_order3(
            self, seeded_triples, data=data, nbins=nbins, classes_y=classes_y,
            freqs_y=freqs_y, verbose=verbose,
        )
        self._seeded_pairs_ = list(seeded_pairs)
        self._seeded_triplets_ = list(_kept_triples)
        # Map the surviving triple COLUMN INDICES -> column NAMES for the triplet FE stage
        # (which operates on the original X DataFrame). Only RAW-column triples (all three
        # legs map to a name) are forwarded; an index out of range is skipped defensively.
        _ncols = len(cols)
        _named = []
        for (a, b, c) in _kept_triples:
            if 0 <= a < _ncols and 0 <= b < _ncols and 0 <= c < _ncols:
                _named.append((cols[a], cols[b], cols[c]))
        self._seeded_triplets_names_ = _named

        # Merge seeded-pair operands into the pool so their JOINT MI gets screened by the
        # existing pair pipeline (bypassing the univariate seed_count that dropped them).
        # ``seeded_pairs`` is non-empty ONLY when the OOF pair self-gate passed; the order-2
        # maxT floor in that pipeline is the outer guard on the merged pairs.
        _new_ops = set()
        for a, b in seeded_pairs:
            _new_ops.add(int(a)); _new_ops.add(int(b))
        _added = _new_ops - set(numeric_vars_to_consider)
        if _added:
            numeric_vars_to_consider = set(numeric_vars_to_consider) | _new_ops
        if verbose and (_added or _kept_triples):
            logger.info(
                "MRMR FE GBM seeder: OOF %.4f vs permuted %.4f (z=%.2f, pair-gate=%s); merged %d "
                "co-occurrence-seeded operand(s) into the pair pool, kept %d order-3-floored "
                "triple(s) for the triplet FE -- recovering zero-marginal interactions the "
                "univariate seed_count misses.",
                info.get("oof_real", float("nan")), info.get("oof_perm", float("nan")),
                info.get("self_gate_z", float("nan")), info.get("gated", False),
                len(_added), len(_kept_triples),
            )
        return numeric_vars_to_consider, list(seeded_pairs)
    except Exception:
        logger.warning(
            "MRMR FE GBM seeder failed; continuing without seeded interactions.",
            exc_info=True,
        )
        return numeric_vars_to_consider, []


def _gate_seeded_triples_order3(
    self, seeded_triples, *, data, nbins, classes_y, freqs_y, verbose,
) -> list:
    """Apply the order-3 Westfall-Young maxT permutation-null floor (#7) to the seeded triples.

    Computes the floor ONCE over the proposed-triple pool (smaller proposer family => less
    punishing, still bounds the chance-max 3-way joint MI), scores each seeded triple's observed
    3-way joint MI with the SAME ``batch_triple_mi_prange`` estimator, and keeps only triples whose
    joint MI clears the floor. SELF-GATING: below ``fe_triple_maxt_min_triples`` candidate triples
    (or ``fe_triple_maxt_null_permutations=0``) the floor is 0.0 (no-op => all seeded triples kept).
    """
    if not seeded_triples:
        return []
    _perms = int(getattr(self, "fe_triple_maxt_null_permutations", 25) or 0)
    _min_triples = int(getattr(self, "fe_triple_maxt_min_triples", 4))
    if _perms <= 0 or len(seeded_triples) < _min_triples:
        return list(seeded_triples)
    try:
        from ._permutation_null import pooled_triple_permutation_null_joint_mi_floor
        from .info_theory import batch_triple_mi_prange

        _ta = np.fromiter((t[0] for t in seeded_triples), dtype=np.int64, count=len(seeded_triples))
        _tb = np.fromiter((t[1] for t in seeded_triples), dtype=np.int64, count=len(seeded_triples))
        _tc = np.fromiter((t[2] for t in seeded_triples), dtype=np.int64, count=len(seeded_triples))
        _nb = np.ascontiguousarray(nbins)
        _fy = np.ascontiguousarray(freqs_y, dtype=np.float64)
        _cy = np.ascontiguousarray(classes_y)
        _floor = pooled_triple_permutation_null_joint_mi_floor(
            data, _nb, _ta, _tb, _tc, _cy, _fy,
            n_permutations=_perms,
            quantile=float(getattr(self, "fe_triple_maxt_null_quantile", 0.95)),
            random_seed=getattr(self, "random_seed", None),
        )
        _obs = batch_triple_mi_prange(data, _ta, _tb, _tc, _nb, _cy, _fy)
        _kept = [seeded_triples[i] for i in range(len(seeded_triples)) if float(_obs[i]) >= _floor]
        if verbose >= 1:
            logger.info(
                "MRMR FE: order-3 maxT permutation-null joint-MI floor=%.5f over %d seeded triple(s) "
                "(q=%.2f, K=%d) -- %d/%d clear it (rejects best-of-pool chance-max noise triples).",
                _floor, len(seeded_triples), float(getattr(self, "fe_triple_maxt_null_quantile", 0.95)),
                _perms, len(_kept), len(seeded_triples),
            )
        return _kept
    except Exception:
        logger.warning(
            "MRMR FE: order-3 maxT floor on seeded triples failed; keeping un-floored triples.",
            exc_info=True,
        )
        return list(seeded_triples)


def compute_pair_maxt_floor(
    self,
    *,
    numeric_vars_to_consider,
    n_pairs: int,
    data,
    nbins,
    classes_y,
    freqs_y,
    verbose,
) -> tuple[float, dict]:
    """Order-2 Westfall-Young permutation-null joint-MI floor over the candidate-pair pool.

    The pair-gating loop ranks O(p^2) candidate pairs by JOINT MI(x_i, x_j; y); at high p the MAX joint MI over PURE-NOISE pairs is a positive order statistic that grows with the
    pool size -- the same best-of-p selection bias the order-1 screening floor rejects, now at order 2. The per-pair prevalence gates are PER-PAIR; they do NOT account for the
    max-over-pool selection, so a wide noise matrix still surfaces "synergistic-looking" noise pairs. Compute the floor ONCE over the WHOLE candidate pool: shuffle the discretised
    target K times, take the per-shuffle MAX joint MI via the SAME batched plug-in estimator the screen scores ``pair_mi`` with, floor at the q-th quantile. SELF-GATING: below
    ``fe_pair_maxt_min_pairs`` candidate pairs the floor is 0.0 (no-op => byte-identical narrow pools). ``fe_pair_maxt_null_permutations=0`` disables.
    """
    _pair_maxt_floor = 0.0
    # MM-DEBIAS (2026-06-09, backlog #1 IRON RULE): per-pair Miller-Madow joint-MI bias
    # (sorted-index tuple -> bias). Subtracted from BOTH the floor's per-shuffle joint MIs
    # (inside the null kernel) AND the observed ``pair_mi`` at the gate-floor comparison,
    # so debiasing the prevalence ratio does NOT weaken this outer best-of-pool guard.
    _pair_mm_bias: dict = {}
    _mm_debias = bool(getattr(self, "fe_mm_debias_prevalence", False))
    _pair_maxt_perms = int(getattr(self, "fe_pair_maxt_null_permutations", 25) or 0)
    if _pair_maxt_perms > 0 and len(numeric_vars_to_consider) >= 2 and n_pairs >= int(getattr(self, "fe_pair_maxt_min_pairs", 30)):
        try:
            from ._permutation_null import pooled_pair_permutation_null_joint_mi_floor, pairwise_mm_joint_bias

            _maxt_pairs = list(combinations(numeric_vars_to_consider, 2))
            _maxt_pa = np.fromiter((p[0] for p in _maxt_pairs), dtype=np.int64, count=len(_maxt_pairs))
            _maxt_pb = np.fromiter((p[1] for p in _maxt_pairs), dtype=np.int64, count=len(_maxt_pairs))
            # Per-pair MM joint-MI bias (permutation-invariant) is needed BOTH at the gate (mapped into
            # _pair_mm_bias below) AND, when mm-debias, by the floor itself; compute it once up front so the
            # resident-GPU floor can subtract the IDENTICAL per-pair term the CPU floor does (consistent debias).
            _bias_vec = None
            if _mm_debias:
                _k_y = int(np.asarray(freqs_y).shape[0])
                _bias_vec = pairwise_mm_joint_bias(data, _maxt_pa, _maxt_pb, nbins, _k_y)
            # Resident-GPU branch (selection-equivalent device twin: same pooled-MAX construction, device-born
            # shuffles, host-owned quantile). DEFAULT ON under the resident FE path / opt-out
            # MLFRAME_FE_PAIR_MAXT_PERM_NULL_GPU=0; returns None on any cupy fault -> exact CPU njit floor below.
            _pair_maxt_floor = None
            try:
                from ._permutation_null_pair_resident import (
                    pair_maxt_perm_null_gpu_enabled,
                    pooled_pair_permutation_null_joint_mi_floor_cupy,
                    trip_pair_maxt_gpu_circuit_breaker,
                )

                if pair_maxt_perm_null_gpu_enabled(int(data.shape[0]), len(_maxt_pa)):
                    _pair_maxt_floor = pooled_pair_permutation_null_joint_mi_floor_cupy(
                        factors_data=data,
                        pair_a=_maxt_pa,
                        pair_b=_maxt_pb,
                        nbins=nbins,
                        classes_y=classes_y,
                        freqs_y=freqs_y,
                        n_permutations=_pair_maxt_perms,
                        quantile=float(getattr(self, "fe_pair_maxt_null_quantile", 0.95)),
                        mm_debias=_mm_debias,
                        mm_bias=_bias_vec,
                        random_seed=getattr(self, "random_seed", None),
                    )
            except Exception:
                # A silent bare fallback here made a ~1h CPU floor invisible (the resident cupy path faults
                # on WDDM-TDR / cudaErrorLaunchFailure on a 4 GB GTX 1050 Ti). WARN once and TRIP the process
                # circuit breaker so every later pair-maxT floor skips the poisoned GPU context immediately
                # and goes straight to the exact CPU njit floor below (no futile per-call re-fault).
                logger.warning(
                    "MRMR FE: resident-GPU order-2 maxT permutation-null floor faulted; tripping the GPU "
                    "circuit breaker and falling back to the CPU njit floor for the rest of this process.",
                    exc_info=True,
                )
                try:
                    trip_pair_maxt_gpu_circuit_breaker()
                except Exception:
                    pass
                _pair_maxt_floor = None
            if _pair_maxt_floor is None:
                _pair_maxt_floor = pooled_pair_permutation_null_joint_mi_floor(
                    factors_data=data,
                    nbins=nbins,
                    pair_a=_maxt_pa,
                    pair_b=_maxt_pb,
                    classes_y=classes_y,
                    freqs_y=freqs_y,
                    n_permutations=_pair_maxt_perms,
                    quantile=float(getattr(self, "fe_pair_maxt_null_quantile", 0.95)),
                    random_seed=getattr(self, "random_seed", None),
                    mm_debias=_mm_debias,
                )
            if _mm_debias and _bias_vec is not None:
                for _pi, _pr in enumerate(_maxt_pairs):
                    _pair_mm_bias[tuple(sorted(_pr))] = float(_bias_vec[_pi])
            if _pair_maxt_floor != 0.0 and verbose >= 1:
                logger.info(
                    "MRMR FE: order-2 maxT permutation-null joint-MI floor=%.5f over %d candidate "
                    "pairs (q=%.2f, K=%d, mm_debias=%s) - rejects best-of-p chance-max noise pairs.",
                    _pair_maxt_floor, n_pairs, float(getattr(self, "fe_pair_maxt_null_quantile", 0.95)),
                    _pair_maxt_perms, _mm_debias,
                )
        except Exception:
            logger.warning(
                "MRMR FE: order-2 maxT permutation-null floor failed; continuing without it.",
                exc_info=True,
            )
            _pair_maxt_floor = 0.0
            _pair_mm_bias = {}
    return _pair_maxt_floor, _pair_mm_bias


def run_cluster_aggregate_emission(
    self,
    *,
    data,
    cols,
    nbins,
    X,
    target_indices,
    categorical_vars,
    cached_MIs,
    engineered_recipes,
    selected_vars,
    n_recommended_features: int,
    num_fs_steps: int,
    _is_polars_input: bool,
    verbose,
):
    """Opt-in clustered-feature aggregation: denoise correlated "reflection" clusters into aggregate columns.

    Runs once on the first FE step (clusters only raw ``feature_names_in_`` columns, which don't change across FE steps). Guarded so a failure never aborts fit. Returns the updated
    ``(data, cols, nbins, X, selected_vars, n_recommended_features)`` tuple; on success also appends to ``self._cluster_aggregate_removals_`` and ``self.cluster_aggregate_``.
    """
    if getattr(self, "cluster_aggregate_enable", False) and num_fs_steps == 0:
        try:
            from ._cluster_aggregate import run_cluster_aggregate_step

            data, cols, nbins, X, _ca_added, _ca_removed, _ca_indices, _ca_summary = run_cluster_aggregate_step(
                data=data, cols=cols, nbins=nbins, X=X, target_indices=target_indices,
                feature_names_in_=list(self.feature_names_in_), categorical_idx=categorical_vars,
                cached_MIs=cached_MIs, engineered_recipes=engineered_recipes,
                quantization_nbins=self.quantization_nbins, quantization_method=self.quantization_method,
                quantization_dtype=self.quantization_dtype,
                methods=tuple(self.cluster_aggregate_methods),
                mi_prevalence=self.cluster_aggregate_mi_prevalence,
                min_member_relevance=self.cluster_aggregate_min_member_relevance,
                min_cluster_size=self.cluster_aggregate_min_cluster_size,
                max_cluster_size=self.cluster_aggregate_max_cluster_size,
                corr_threshold=self.cluster_aggregate_corr_threshold,
                homogeneity_tau=self.cluster_aggregate_homogeneity_tau,
                max_candidates=self.cluster_aggregate_max_candidates,
                mode=self.cluster_aggregate_mode, is_polars_input=_is_polars_input,
                verbose=verbose, dtype=self.dtype,
            )
            n_recommended_features += int(_ca_added)
            if _ca_indices:
                # The aggregate already passed the supervised MI gate (beats best member), so select it
                # directly -- don't rely on a re-screen (with the default fe_max_steps=1 the loop breaks
                # before re-screening). Remap routes the engineered name into _engineered_recipes_.
                _sv = list(selected_vars) if not isinstance(selected_vars, list) else selected_vars
                selected_vars = _sv + [i for i in _ca_indices if i not in _sv]
            if _ca_removed:  # replace mode: consumed in _fit_impl before the cols->original remap
                self._cluster_aggregate_removals_ = list(getattr(self, "_cluster_aggregate_removals_", [])) + list(_ca_removed)
            if _ca_summary:
                # Fitted summary -> surfaced into meta_info["feature_selection_report"]["cluster_aggregate"].
                self.cluster_aggregate_ = list(getattr(self, "cluster_aggregate_", []) or []) + _ca_summary
                logger.info(
                    "MRMR cluster_aggregate (%s): built %d denoised aggregate(s) from correlated clusters: %s",
                    self.cluster_aggregate_mode, _ca_added,
                    [f"{r['name']} (method={r['method']}, k={len(r['members'])}, +MI {r['mi_gain']:.4f})" for r in _ca_summary],
                )
        except Exception:
            logger.warning("cluster_aggregate step failed; continuing without it.", exc_info=True)
    return data, cols, nbins, X, selected_vars, n_recommended_features


def apply_interaction_information_routing(
    self: Any,
    *,
    prospective_pairs: dict,
    cached_MIs: Any,
    nbins: Any,
    freqs_y: Any,
    classes_y: Any,
    data: Any,
    synergy_added_idx: Any,
    verbose: int,
) -> dict:
    """Signed interaction-information routing over the prospective pairs (backlog idea #8).

    Computes the order-2 permutation-null floor on the MAX positive interaction information over the SAME
    prospective-pair pool, then tags every pair by signed MM-corrected ``II(a;b;y) = I((a,b);y) - I(a;y) - I(b;y)``
    and DEMOTES the additive cross-mix (``II <= floor``) speculative pairs out of the FE search so no spurious
    cross-mix surrogate is built. Positive II -> synergy (product/cross-basis), negative II -> redundant
    (cluster-aggregate). Changes routing/ranking only; the order-2 maxT floor + ratio gate remain the detection
    guards. SELF-GATING: below ``fe_ii_routing_min_pairs`` candidate pairs (or floor==0.0 / disabled) every pair
    is kept (byte-stable). Stamps ``self.fe_interaction_routes_`` / ``self.fe_interaction_ii_`` for provenance.

    Returns the (possibly trimmed) ``prospective_pairs`` dict.
    """
    if not bool(getattr(self, "fe_ii_routing_enable", True)) or not prospective_pairs:
        return prospective_pairs
    _perms = int(getattr(self, "fe_ii_routing_null_permutations", 25) or 0)
    _min_pairs = int(getattr(self, "fe_ii_routing_min_pairs", 30))
    if _perms <= 0 or len(prospective_pairs) < _min_pairs:
        return prospective_pairs
    try:
        from ._interaction_information import pooled_pair_ii_null_floor, route_prospective_pairs

        _keys = list(prospective_pairs.keys())
        _pa = np.fromiter((k[0][0] for k in _keys), dtype=np.int64, count=len(_keys))
        _pb = np.fromiter((k[0][1] for k in _keys), dtype=np.int64, count=len(_keys))
        _n = int(data.shape[0]) if hasattr(data, "shape") else 0
        _nbins_y = int(np.asarray(freqs_y).shape[0])
        _ii_floor = pooled_pair_ii_null_floor(
            factors_data=data,
            nbins=nbins,
            pair_a=_pa,
            pair_b=_pb,
            marginal_mi_a=np.zeros(len(_keys)),
            marginal_mi_b=np.zeros(len(_keys)),
            classes_y=classes_y,
            freqs_y=freqs_y,
            n_permutations=_perms,
            quantile=float(getattr(self, "fe_ii_routing_null_quantile", 0.95)),
            miller_madow=True,
            random_seed=getattr(self, "random_seed", None),
        )
        kept, routes, ii_values = route_prospective_pairs(
            prospective_pairs,
            cached_MIs=cached_MIs,
            nbins=nbins,
            nbins_y=_nbins_y,
            n=_n,
            ii_floor=_ii_floor,
            synergy_added_idx=synergy_added_idx,
            miller_madow=True,
            verbose=verbose,
        )
        # Provenance: routes/II keyed by the raw (var_a, var_b) tuple, plus the null floor used.
        self.fe_interaction_routes_ = routes
        self.fe_interaction_ii_ = ii_values
        self.fe_interaction_ii_floor_ = float(_ii_floor)
        if _ii_floor > 0.0 and verbose >= 1:
            logger.info(
                "MRMR FE: interaction-information null floor=%.5f over %d prospective pairs (q=%.2f, K=%d); "
                "routed synergy/additive/redundant, demoted additive speculative cross-mix pairs.",
                _ii_floor, len(_keys), float(getattr(self, "fe_ii_routing_null_quantile", 0.95)), _perms,
            )
        return kept
    except Exception:
        logger.warning(
            "MRMR FE: interaction-information routing failed; continuing with the un-routed prospective pairs.",
            exc_info=True,
        )
        return prospective_pairs


def log_fe_summary(
    *,
    prospective_pairs,
    prospective_additions,
    n_recommended_features: int,
    fe_min_pair_mi_prevalence,
    fe_min_engineered_mi_prevalence,
    fe_min_nonzero_confidence,
    fe_good_to_best_feature_mi_threshold,
    verbose,
    fe_acceptance: str = "conditional_mi",
) -> None:
    """Log the per-step FE summary (pairs considered / with additions / total engineered + gate thresholds).

    Surfaces WHY FE added 0 features when the operator configured it explicitly: a prod log showed 88 min of Hermite Optuna yielding 0 engineered cols with no visible explanation.
    The summary reports n_pairs_considered (pairs screened), n_pairs_with_additions (pairs that produced ANY recipe), and n_engineered_features (recipes that survived all gates); at
    ``verbose >= 1`` with 0 additions it also names the likely too-tight knob (often ``fe_min_engineered_mi_prevalence``). Pure logging, no state mutation.
    """
    try:
        _n_pairs_considered = int(len(prospective_pairs))
    except Exception:
        _n_pairs_considered = -1
    try:
        _n_pairs_with_additions = sum(
            1 for v in prospective_additions.values()
            if v[0]  # this_pair_features non-empty
        )
    except Exception:
        _n_pairs_with_additions = -1
    if verbose >= 1:
        logger.info(
            "FE summary: %d pair(s) considered, %d produced engineered cols, "
            "n_total_engineered=%d. Gate thresholds: "
            "fe_min_pair_mi_prevalence=%.3f, "
            "fe_min_engineered_mi_prevalence=%.3f, "
            "fe_min_nonzero_confidence=%.3f, "
            "fe_good_to_best_feature_mi_threshold=%.3f.",
            _n_pairs_considered, _n_pairs_with_additions,
            n_recommended_features,
            float(fe_min_pair_mi_prevalence),
            float(fe_min_engineered_mi_prevalence),
            float(fe_min_nonzero_confidence),
            float(fe_good_to_best_feature_mi_threshold),
        )
        if n_recommended_features == 0 and _n_pairs_considered > 0:
            # Suggest a value STRICTLY BELOW the value actually in effect -- the
            # old text hardcoded "lower to 0.90", which is a no-op once 0.90 IS
            # the default (regression: it literally told the operator to set the
            # knob to the value it was already at). Derive the suggestion from the
            # live threshold (5% under it, floored away from 0) so it is always
            # actionable regardless of how the knob was configured.
            _cur_eng = float(fe_min_engineered_mi_prevalence)
            _suggest_eng = round(max(0.50, _cur_eng * 0.95), 3)
            _cur_pair = float(fe_min_pair_mi_prevalence)
            _suggest_pair = round(max(1.0, 1.0 + (_cur_pair - 1.0) * 0.5), 3)
            logger.warning(
                "FE produced 0 engineered features despite %d pair(s) passing the "
                "pair-MI gate. Likely cause: the fe_min_engineered_mi_prevalence=%.3f "
                "threshold is tight relative to the pair-level MI. Try lowering it to "
                "%.3f (5%% under the value currently in effect), or widen the pool with "
                "fe_min_pair_mi_prevalence=%.3f. The principled fix is to leave "
                "fe_acceptance='conditional_mi' (the default S5 conditional-MI "
                "redundancy gate, currently '%s'), which admits genuine engineered "
                "summaries of real interactions WITHOUT a hand-tuned prevalence "
                "constant; the prevalence ratios above are only the cheap upstream "
                "pre-screen.",
                _n_pairs_considered,
                _cur_eng,
                _suggest_eng,
                _suggest_pair,
                str(fe_acceptance),
            )
