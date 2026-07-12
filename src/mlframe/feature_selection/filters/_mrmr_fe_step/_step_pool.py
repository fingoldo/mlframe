"""FE operand-pool construction stage of ``MRMR._run_fe_step``.

Carved verbatim from ``_step_core.py`` to keep that module under the 1k-LOC ceiling. ``build_fe_operand_pool``
builds ``numeric_vars_to_consider`` from ``selected_vars`` (the fe_ntop / categorical / non-numeric /
factors_to_use restrictions), runs the synergy bootstrap, the gate-operand re-classification, and the
surrogate-GBM / gradient-interaction seeders, and carries the standing bench-rejected rationale for the
unbuilt RFF (#9) and Apriori-lattice (#10) proposers. Selection is byte-for-byte identical to the inline
block. Returns ``(numeric_vars_to_consider, _synergy_added_idx)``.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

from .._mrmr_fe_step_helpers import (
    apply_surrogate_gbm_seeder,
    apply_synergy_bootstrap,
)
from .._gradient_interaction_seeder import propose_gradient_interaction_pairs
from ._helpers import _non_numeric_column_indices


def build_fe_operand_pool(
    self,
    *,
    selected_vars,
    categorical_vars,
    cols, X, data, nbins,
    target_indices,
    classes_y, freqs_y,
    cached_MIs,
    num_fs_steps,
    verbose,
):
    """Construct the FE operand pool ``numeric_vars_to_consider`` and the synergy-added index set."""

    # bench-attempt-rejected (2026-06-11, FS backlog #6 surrogate-GBM split-co-occurrence 3-way seeder + #7 order-3 maxT floor):
    # Hypothesis was that a shallow GBM's gain-weighted root->leaf path co-occurrence could propose pure zero-marginal
    # 3-way synergy triples directly into this FE pool, bypassing the blind univariate top-N seed below. Decisive Stage-1
    # measurement (n=4000, p=200 iid, y=sign(x7*x42*x113)+0.3*noise; ALL univariate |corr| <= 0.026 and all
    # pairwise-product |corr| <= 0.026) shows the needle {7,42,113} NEVER surfaces:
    #   baseline150 / 1000r / extra_trees / colsample0.1 / extra+colsample : needle_rank=None, top-triple recall 0/3.
    #   best case (depth5 extra_trees 1000r): rank 18778/32947, score 12 vs top ~150 -- still 0/3.
    # Positive control (same walker) DOES rank a detectable pairwise XOR 0/1640 and a 3-way-with-pairwise-leak needle
    # 0/766, proving the null is the pure-3way's greedy-invisibility, not a code bug. Tree-path co-occurrence inherits
    # the greedy blindness it claimed to bypass (consistent with beam-search-useless-for-MRMR: conditional-MI is not
    # greedy-trappable). The order-3 Westfall-Young maxT floor (#7) is the mandatory rail for a 3-way proposer, but with
    # no working proposer it ships nothing -- both backlog items rejected/not-built together. A detect-without-enumerate
    # screen (#9 RFF) is the open non-greedy path to pure 3-way synergy.
    if self.fe_ntop_features:
        numeric_vars_to_consider = selected_vars[: self.fe_ntop_features]
    else:
        numeric_vars_to_consider = selected_vars

    numeric_vars_to_consider = set(numeric_vars_to_consider) - set(categorical_vars)

    # Drop any operand whose source column is not numeric in X, even if it was not flagged in ``categorical_vars``:
    # the polynomial / Hermite pair search applies numeric basis transforms (np.isfinite, z-score) that raise on a
    # string / categorical column (e.g. ``"ZZZ_UNSEEN"`` / ``"NA"``). Mirrors the per-stage numeric guards the
    # triplet / quadruplet stages already apply (_mrmr_fit_impl.py is_numeric_dtype filters).
    _non_numeric_idx = _non_numeric_column_indices(X, cols)
    if _non_numeric_idx:
        numeric_vars_to_consider = numeric_vars_to_consider - _non_numeric_idx

    # Honor factors_to_use / factors_names_to_use in the FE step too; intersect the FE pool with the user's
    # restriction so the contract matches the screening step.
    if self.factors_to_use is not None:
        numeric_vars_to_consider = numeric_vars_to_consider & set(self.factors_to_use)
    if self.factors_names_to_use is not None:
        # name -> index map built once (O(F)) instead of an ``in`` test + ``.index()`` rescan of
        # ``cols`` per name (O(F) each) -- turns the O(K*F) lookup below into O(K+F).
        _cols_idx = {nm: i for i, nm in enumerate(cols)}
        allowed = {idx for n in self.factors_names_to_use if (idx := _cols_idx.get(n)) is not None}
        numeric_vars_to_consider = numeric_vars_to_consider & allowed

    numeric_vars_to_consider, _synergy_added_idx = apply_synergy_bootstrap(
        self,
        num_fs_steps=num_fs_steps,
        data=data,
        cols=cols,
        target_indices=target_indices,
        categorical_vars=categorical_vars,
        numeric_vars_to_consider=numeric_vars_to_consider,
        non_numeric_idx=_non_numeric_idx,
        verbose=verbose,
    )

    # GATE-OPERAND RE-CLASSIFICATION (2026-06-13). A selected conditional_gate / row_argmax feature is
    # built FROM one or more raw columns (its recipe ``src_names``); the high-MI gate column is selected
    # by the greedy screen AHEAD of its raw source, so the raw source is dropped from ``selected_vars``
    # (redundant given the gate) and re-enters the FE pool only via the synergy bootstrap above -- which
    # tags it ``synergy_added`` and forces every pair over it onto the STRICTER ``fe_synergy_min_prevalence``
    # bar. On CASE1 ``y=a**2/b+log(c)*sin(d)`` that demoted the clean elementary (c,d) pair (joint MI 0.176
    # vs the 0.231 synergy bar) and SUPPRESSED ``mul(log(c),sin(d))`` (MI 0.314) -- the gate EVICTED rather
    # than out-competed it. Reclassify a gate's raw operands as REGULARLY-selected (drop from synergy_added)
    # so their elementary pair competes on the lenient bar and the clean form can WIN where it beats the gate
    # composite. The gate column itself stays selected + pairable (CASE2's warped (c,d) gate capture intact).
    # Byte-identical when no gate fired (``_gate_raw_operands_`` empty). Same lesson as the gate_med fix.
    _gate_raw_names = getattr(self, "_gate_raw_operands_", None)
    if _gate_raw_names and _synergy_added_idx:
        _gate_raw_idx = {i for i in _synergy_added_idx if cols[i] in _gate_raw_names}
        if _gate_raw_idx:
            _synergy_added_idx = _synergy_added_idx - _gate_raw_idx
            if verbose:
                logger.info(
                    "MRMR FE: reclassified %d gate-source raw operand(s) %s from synergy-bootstrap to "
                    "regularly-selected so their elementary pairs use the lenient prevalence bar.",
                    len(_gate_raw_idx), sorted(cols[i] for i in _gate_raw_idx),
                )

    # SURROGATE-GBM SPLIT-CO-OCCURRENCE SEEDER (2026-06-09, backlog #6) + its mandatory
    # order-3 maxT rail (#7). Fits one shallow LightGBM on the discretised matrix, walks
    # root-to-leaf paths, and tallies depth-discounted split-gain co-occurrence to propose
    # interaction PAIRS + TRIPLES whose operands have ~0 univariate MI -- the zero-marginal
    # needles (``y = sign(x_a*x_b*x_c)``) the univariate seed_count never reaches. Seeded PAIR
    # operands are MERGED into ``numeric_vars_to_consider`` so their joint MI is screened by the
    # existing pair pipeline below (bypassing seed_count); seeded TRIPLES are order-3-floored and
    # stamped onto ``self._seeded_triplets_`` for the triplet FE stage. SELF-GATES on a permuted-y
    # OOF comparison (pure noise -> no seeds -> pool not polluted); the order-2/order-3 maxT floors
    # then gate every emitted candidate. OPT-IN (``fe_gbm_seeder_enable``); self-routes OFF below
    # ``fe_gbm_seeder_min_features`` where seed_count is not the blocker. No-op otherwise.
    numeric_vars_to_consider, _gbm_seeded_pairs = apply_surrogate_gbm_seeder(
        self,
        num_fs_steps=num_fs_steps,
        data=data,
        nbins=nbins,
        cols=cols,
        categorical_vars=categorical_vars,
        target_indices=target_indices,
        classes_y=classes_y,
        freqs_y=freqs_y,
        numeric_vars_to_consider=numeric_vars_to_consider,
        non_numeric_idx=_non_numeric_idx,
        verbose=verbose,
    )

    # GRADIENT-INTERACTION (MIXED SECOND PARTIALS) SEEDER (2026-06-10, backlog #21). Fits one
    # smooth differentiable RFF+ridge surrogate on a row sample, estimates E[(d2f/dxa dxb)^2] per
    # pair (a large mixed partial is the calculus definition of a non-additive interaction), and
    # proposes the operands of high-energy pairs into the SAME pool as #6 so their joint MI is
    # screened by the pipeline below. Targets SMOOTH/ROTATED interactions (a*b saddles, sin(a)*b).
    # Carries its OWN self-gate: an OOF-R2 vs permuted-y check (no learning -> 0 proposals), a
    # GAM additive-residual baseline (additive targets -> ~0 mixed partials), and a permutation
    # null on the max mixed-partial energy. OPT-IN (``fe_gradient_interaction_enable``, default
    # OFF -- see the module's bench-reject note: on the prescribed sin(x5)*x31 fixture the gradient
    # detector ranks the saddle #1 but the #6 GBM seeder does NOT under-rank it, so the two are
    # equally good rather than complementary there, and the full self-gated proposer is ~8 s at
    # n=2000/p=60). Routed OFF outside its size regime by ``_route_gradient_seeder``. No-op otherwise.
    numeric_vars_to_consider, _gradient_seeded_idx = propose_gradient_interaction_pairs(
        self,
        num_fs_steps=num_fs_steps,
        data=data,
        # The smooth surrogate REQUIRES the RAW CONTINUOUS values: ``data`` here is the
        # discretised (few-bin int) matrix every FE-gate MI is scored on, and a piecewise-constant
        # ~5-level step function has ~zero mixed partials regardless of the underlying smooth
        # interaction. ``X`` is the raw float frame (indexed by ``cols``); pass it so the RFF fit
        # can resolve the smooth saddle. (This is the architectural reason the detector is niche --
        # unlike the joint-MI sweep / GBM seeder, it cannot operate on the discretised pool.)
        X_continuous=X,
        cols=cols,
        target_indices=target_indices,
        categorical_vars=categorical_vars,
        numeric_vars_to_consider=numeric_vars_to_consider,
        non_numeric_idx=_non_numeric_idx,
        verbose=verbose,
    )

    # bench-rejected (2026-06-09) -- backlog #9 "RFF / random-projection interaction pre-screen
    # (detect-without-enumerate)": a sibling proposer to the GBM seeder above that would draw R
    # random SPARSE (support 2-4 cols) hyperplane projections phi_r(x)=cos(w_r^T x_std + b_r), score
    # each phi_r's MI vs y, and promote supports whose best-of-trials MI uplift over the additive
    # baseline clears a permutation-null maxT floor -- meant to recover zero-marginal/smooth
    # interaction supports the all-pairs synergy bootstrap SKIPS when p > fe_synergy_screen_max_features.
    # Prototyped + benchmarked end-to-end (D:/Temp/rff_proto*.py, rff_coverage.py, rff_vs_allpairs.py,
    # rff_smooth_check.py, rff_largep_birthday.py; n=2000, p=500-3000). The MECHANISM is sound -- when
    # the needle support is COVERED, best-of-T RFF ranks it #0 with uplift 0.146 vs a null floor 0.0145
    # (10x), 0 noise false-promote -- but it loses in EVERY regime against the direct joint-MI sweep:
    #   1) COVERAGE is the binding constraint, not enumeration. A SPECIFIC localized k=2 needle in p=500
    #      has hit-probability R / C(p,2): R=2000 -> 1.6%, R=16000 -> 12%, R=124750 (== full all-pairs) ->
    #      63%; reaching 0.95 recall of ONE needle needs ~374k random draws. Random sparse supports cannot
    #      guarantee recall of a specific needle without ~enumerating all pairs.
    #   2) The BIRTHDAY-PARADOX rationale is wrong for sparse planted needles: with K=30 planted needles at
    #      p=3000 (C(p,2)=4.5M), R=20000 x T=6 = 120k evals (65s) DREW 0/30 needles -> recall 0/30. Hitting
    #      a fixed K-needle set needs draws ~ C(p,2)/K (~150k here), not R << p^2; the birthday paradox is
    #      about collisions AMONG draws, not hitting a fixed small target set.
    #   3) COST: one RFF best-of-T (T=8) support evaluation is 17x the cost of one direct joint-MI pair
    #      evaluation (4.1ms vs 0.24ms). 0.95-recall of one needle ~1533s of RFF vs ~30s for the full
    #      deterministic C(500,2) joint-MI sweep.
    #   4) NO EXCLUSIVE SWEET SPOT: the 2-D joint-MI estimator is a nonparametric (x_i, x_j, y) contingency
    #      MI, so it recovers the SMOOTH sin(x1*x2) needle (rank #0/2000, jMI 1.29) exactly as well as the
    #      zero-marginal product x3*x400 (rank #0/2000, jMI 1.51). The "smooth/rotated interactions trees
    #      miss" argument applies to the GBM seeder's AXIS-ALIGNED trees (#6/#21), NOT to the joint-MI sweep
    #      that #9 was meant to replace.
    # RE-CONFIRMED (2026-06-11, cycle-10) on the 3-WAY trees-miss case. Cycle-5 proved the #6 GBM
    # seeder is GREEDY-BLIND to pure 3-way synergy (sign(x7*x42*x113)+0.3noise, n=4000/p=200, needle
    # rank 0/3), so RFF was re-attempted as the last open NON-GREEDY path. Decisive A/B/C
    # (D:/Temp/cycle10_rff_results.md): (A) 2-way x3*x400 p=500 R=4000 -> support recall 0->0 (needle
    # NEVER drawn, P(hit)~1.9%); (B) 3-way p=200 R=8000 sparse-3 -> support recall 0->0 (needle NEVER
    # drawn, P(hit)~0.36%); (C) pure-noise -> 0 promoted (floor holds). When the needle support is
    # INJECTED, the uplift DOES score it (3-way covered uplift 0.0224 > additive baseline 0.0077 and >
    # all 2-subsets ~0) -- so the scorer is NOT greedy-blind to 3-way; the wall is purely COVERAGE.
    # 0.95-recall of ONE needle needs ~623k draws (2-way) / ~6.56M (3-way) == ~85x the full
    # deterministic enumeration cost in BOTH regimes. Pure 3-way stays UNREACHED: trees are
    # greedy-blind, RFF is coverage-blind; only direct order-3 enumeration (raise the cap) recovers it.
    # The correct fix for "needle missed when p > fe_synergy_screen_max_features" is to RAISE/relax that cap
    # (and its fe_synergy_max_sweep_cost n*p^2 budget): the direct all-pairs joint-MI bootstrap recovers
    # both the zero-marginal and the smooth needle DETERMINISTICALLY and ~17x cheaper at p<=1000 (full sweep
    # ~30s at p=500, ~120s at p=1000). RFF random-support prescreen would only conceivably help at p >~ 5000
    # where C(p,2) joint-MI is itself infeasible -- but even there it is coverage-bound (test #2 above) and
    # never recovers a SPECIFIC needle, so it is not implemented. Do NOT re-attempt as a default or opt-in
    # proposer without a fundamentally different (non-random-support) coverage scheme.

    # bench-rejected (2026-06-09) -- backlog #10 "Conditional-MI complementarity growth (Apriori lattice)":
    # grow triples from the order-2 SURVIVOR frontier (the prospective pairs below) by testing only third
    # columns c maximising the conditional-MI uplift I((a,b,c);y) - I((a,b);y) (reusing batch_triple_mi_prange
    # for scale-consistency with the #7 order-3 maxT floor), keeping triples above that floor; meant to catch a
    # 3rd operand that matters ONLY given (a,b) -- pure higher-order synergy the univariate triplet seed_count
    # misses. Prototyped + benchmarked end-to-end (D:/Temp/cmi_lattice_proto{,2,3,4,5}.py, cmi_lattice_cost.py;
    # n=3000, p=40, the backlog fixture). REJECTED: the Apriori ANTI-MONOTONE premise is FALSE for the exact
    # signal class #10 targets, and the part that DOES work is already shipped as #6.
    #   1) NOT GREEDY-TRAPPABLE (the binding wall). The backlog fixture y=sign(x1*x2)*x3>0 is, mathematically, a
    #      PURE 3-way sign XOR (verified 100% agreement with sign(x0)*sign(x1)*sign(x2)>0), NOT a "detectable
    #      2-way (x1,x2) + conditional x3" as the backlog text claims. For a pure k-way interaction ALL its
    #      (k-1)-way sub-tuples have joint MI AT THE NOISE FLOOR: the needle pair ranks ~39th of C(40,2) (jMI
    #      0.0002-0.0107 across nbins 2..8, indistinguishable from the noise-pair max), so it NEVER enters the
    #      top-N order-2 frontier -> the needle triple is NEVER grown (frontier in {6,8,12} all miss it). This is
    #      the same conditional-MI-is-not-greedy-trappable wall that killed beam-search for MRMR -- Apriori
    #      pruning ("grow only from surviving (k-1)-tuples") structurally CANNOT reach a pure higher-order
    #      synergy whose every lower-order projection is null.
    #   2) The GROWTH KERNEL itself is sound but moot. Given a base pair holding 2 of the 3 needle legs, the
    #      CMI uplift ranks the true 3rd operand RANK-0 every time (6/6, uplift 0.21-0.61 >> noise) -- but that
    #      base pair is exactly what the frontier never contains (point 1).
    #   3) NOISE ADMISSION on the one fixture where the lattice CAN grow the needle (a marginal-signal base x0
    #      plus a conditional-only x2, y=(x0>0)^((x1>0)&(x2>0))): the lattice grows (0,1,2) as the #1 triple
    #      (jMI 0.4495) BUT the order-3 maxT floor (0.0287) admits 31/31 grown noise triples, ALL sharing the
    #      marginal-signal leg x0 (a (0,noise,noise) triple inherits x0's MI and clears the floor). This is not
    #      a lattice-specific defect (#6's seeded triples admit 8/8 on the same fixture) -- it is the downstream
    #      per-triplet uplift gate's job, NOT the maxT floor's, so the lattice adds NO admission advantage.
    #   4) FULLY REDUNDANT with the shipped #6 GBM seeder. On the pure-3-way-XOR fixture the GBM
    #      split-co-occurrence proposer recovers the SAME needle {x1,x2,x3} as its RANK-0 triple (z=5.46),
    #      order-3-floored -- because path co-occurrence conditions a zero-marginal operand on its co-splitter
    #      WITHOUT requiring a detectable (k-1)-tuple, so it is immune to the anti-monotone wall that blocks #10.
    #      #10's whole purpose (reach a 3-way the univariate seeder misses) is already delivered by #6.
    # The lattice's ONLY edge is cost (19.5 ms growth vs 1641 ms for the GBM fit+self-gate at n=3000/p=40) -- but
    # that merely buys a proposer that misses its own target signal class. Do NOT re-implement as the backlog
    # specifies. A non-Apriori conditional-growth scheme (e.g. seed triples from the GBM's co-occurrence pool,
    # THEN refine the 3rd operand by CMI uplift) could compose with #6, but the recovery is already complete via
    # #6 alone, so there is no measured gap to close.

    return numeric_vars_to_consider, _synergy_added_idx
