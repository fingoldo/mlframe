"""Honest re-validation + trust diagnostics for the SHAP-proxied selector.

The proxy ranks subsets cheaply but is a *biased* estimator of a retrained model's quality (it
attributes the full model restricted to S, not a model retrained on S). Three guards close that gap:

  - ``proxy_trust_guard``: on a sample of anchor subsets, score BOTH the cheap proxy and an honest
    retrain, then report Spearman/Kendall rank-correlation + recall@k of proxy-top vs honest-top.
    Converts "trust me" into measured proxy fidelity on the user's own data; warns below a floor.
  - ``revalidate_top_n``: honestly retrains each proxy-top-N candidate and evaluates it on a holdout
    DISJOINT from the SHAP/objective data (avoids winner's curse over millions of combos). Returns
    the best / most stability-penalised subset, plus a same-size random-subset baseline for context.
  - ``importance_topk_ablation``: gating check that proxy search beats plain SHAP-importance-top-k
    (Aleksey Pichugin's concern: does this reduce to feature importance + a wrapper?).

Honest scoring uses sklearn metrics (not the hot path); rank correlation uses scipy. Lower = better
loss everywhere, to match the proxy objective.
"""
from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_selection._shap_proxy_objective import coalition_margin_T, proxy_loss, resolve_metric
from mlframe.feature_selection._shap_proxy_revalidate._shap_proxy_sampling import (
    _sample_anchor_subsets, _softmax_weights, _zipf_card_probs,
)
from mlframe.feature_selection._shap_proxy_revalidate._shap_proxy_loss import (
    HonestLossCache, _expand, _honest_loss, _open_disk_cache,
    _parallel_honest_losses, _permutation_importance_ranking, _try_cap_n_estimators,
)
from mlframe.feature_selection._shap_proxy_revalidate._shap_proxy_refine import (
    _ucb_auto_slack, _ucb_stop_remaining_cannot_win,
    active_learning_revalidate, importance_topk_ablation, revalidate_top_n, within_cluster_refine,
)

logger = logging.getLogger(__name__)


_FIDELITY_FLOOR_UNSET = object()


def proxy_trust_guard(
    phi, base, y_search, model_template, X_search, X_holdout, y_holdout,
    *, classification, metric=None, n_anchors=30, rng=None, min_card=1, max_card=None,
    fidelity_floor=0.5, n_jobs=-1, unit_to_members=None, cache=None, n_estimators_cap=None,
    unit_f_scores=None, anchor_uniform_tail_frac=0.2, cardinality_dist="uniform", zipf_alpha=1.0,
    fidelity_weights=(0.6, 0.4), trustworthy_metric="proxy_fidelity_score",
    spearman_floor=_FIDELITY_FLOOR_UNSET, inner_n_jobs_cap=False, disk_cache_dir=None,
):
    """Measure proxy-vs-honest rank fidelity on anchor subsets. Returns a report dict.

    Anchors are sampled in proxy (unit) space; honest losses retrain on the expanded member columns,
    so this measures the most decision-relevant question: does the cheap unit-proxy rank subsets the
    way honestly-retrained real-feature models do?

    ``n_estimators_cap`` reduces the per-anchor booster size; the trust report only consumes RANKS
    (Spearman / Kendall / recall@k) of anchor losses, so a capped booster gives a fast, faithful
    fidelity signal. The corrector data (proxy / honest pairs) IS persisted on the report and used by
    a downstream regression-based bias-corrector that consumes the absolute honest values, so when the
    cap is enabled the corrector trains on capped values; the corrector's regression learns the
    proxy->honest_capped mapping, which is still a valid rank-preserving signal for re-ranking
    candidates. Leave as ``None`` (default) to preserve legacy absolute-value semantics on the
    corrector training pairs.

    ``fidelity_weights`` (iter17, default ``(0.6, 0.4)``): weights ``(w_spearman, w_recall)`` for the
    composite ``proxy_fidelity_score = w_spearman * spearman + w_recall * recall_at_k``. Both Spearman
    and recall@k live on ``[0, 1]`` for non-degenerate inputs (Spearman is clipped on the gate side
    only; the raw field can be negative on a broken proxy and the gate then correctly trips because
    the composite drops below the floor). The composite is the trust-guard's headline metric; raw
    ``spearman`` / ``kendall`` / ``recall_at_k`` remain as diagnostic fields. Iter16 shipped a
    symmetric (0.5, 0.5) default for lack of evidence; iter17 calibrated the default by correlating
    each component independently with downstream selector RECOVERY across 5 regimes (additive high-SNR,
    redundancy-heavy, interaction-heavy, xor-interaction, noise-heavy). Result: ``corr(spearman,
    recovery_rate) = 0.93`` vs ``corr(recall@k, recovery_rate) = 0.55``. Spearman tracks the proxy's
    whole-ranking quality which actually predicts whether downstream candidate ranking finds the
    informatives; recall@k is bounded above (small anchor top-k overlap stays high even on
    half-broken proxies) and below (the 1-anchor top-k is trivially 1.0), so it lacks the dynamic
    range to drive the gate. The corr-proportional split (0.629, 0.371) rounds to (0.6, 0.4); the
    rounded value is the registered default. Iter15 still motivates the composite over raw Spearman:
    Zipf at alpha=0.25 dropped spearman 0.969->0.956 but lifted recall@k 0.833->1.0, so the composite
    went 0.901->0.978 -- a real win the raw-Spearman gate had masked. Iter17's (0.6, 0.4) preserves
    that win (the composite stays above the floor) while letting Spearman dominate the gate decision
    in the calibration-supported direction.

    ``trustworthy_metric`` (iter16, default ``'proxy_fidelity_score'``): which scalar gates the
    ``trustworthy`` boolean. ``'proxy_fidelity_score'`` is the new composite (default). ``'spearman'``
    preserves pre-iter16 semantics for callers that pinned the floor against the raw Spearman scale.

    ``fidelity_floor`` (iter18, default ``0.5``): below this value the gate trips and ``trustworthy``
    is ``False``. Interpreted in the chosen ``trustworthy_metric`` scale; both metrics live on [0, 1].
    Iter18 recalibrated the default from 0.6 to 0.5 after iter17 flipped the gate from raw Spearman to
    the composite ``proxy_fidelity_score = 0.6*spearman + 0.4*recall@k``: the legacy 0.6 was set
    against the raw-Spearman scale and is too conservative on the composite scale, flagging the
    ``interaction_heavy`` regime (recovery 6/8 = 75%, a real partial success) as LOW. The new floor
    is the lowest composite of any regime that hits ``recovery_rate >= 0.7`` across the same 5-regime
    bench used for the weights calibration:

      regime              spearman  recall@k  composite  recovery  recovery_rate  gate@0.5
      additive_highSNR     0.9533    0.8333    0.9053     8/8        1.000          PASS
      redundancy_heavy     0.8839    0.8333    0.8637     8/8        1.000          PASS
      interaction_heavy    0.5640    0.5000    0.5384     6/8        0.750          PASS
      noise_heavy          0.9506    0.8333    0.9036     7/8        0.875          PASS
      xor_interaction      0.3459    0.6667    0.4742     2/6        0.333          FAIL

    ``recovery_rate >= 0.7`` PASS group min composite = 0.5384 (interaction_heavy); the only regime
    with ``recovery_rate < 0.5`` (xor) sits at 0.4742. A floor at 0.5 separates the two groups
    cleanly (0.039 margin to the PASS floor, 0.026 margin above the FAIL ceiling). See
    ``_benchmarks/calib_iter18_fidelity_floor.py`` for reproducible measurement.

    ``spearman_floor`` (DEPRECATED iter18 alias for ``fidelity_floor``): kept for backwards-compat
    with callers that hard-coded the iter15 kwarg name. Emits a ``DeprecationWarning`` when supplied.
    Passing BOTH ``fidelity_floor`` and ``spearman_floor`` raises ``ValueError``.

    ``unit_f_scores``: optional length-``phi.shape[1]`` float vector of per-unit marginal-strength
    weights (e.g. ANOVA F-scores aggregated from the prefilter's cached stage-A scores). When supplied,
    anchor columns are drawn by softmax(unit_f_scores) instead of uniform-at-random, with a small
    uniform tail (``anchor_uniform_tail_frac``, default 20%) for tail-of-distribution coverage. The
    rationale: on wide data with a heavy noise tail, uniform anchors are dominated by noise columns,
    so proxy-vs-honest spread reflects sample noise rather than fidelity. Stratifying by F-score
    spends the same anchor budget on subsets where the proxy is actually being asked to rank
    informative-mix-vs-noise-mix subsets, lifting the measured Spearman without changing the anchor
    count. None (default) keeps the legacy uniform sampler. Non-finite entries (-inf for constant /
    degenerate columns) sink to the noise-floor probability via the softmax.

    ``cardinality_dist`` (iter15+iter16): how anchor cardinality ``k`` is drawn over
    ``[min_card, max_card]``. The MODULE-LEVEL default of this kwarg is still ``'uniform'`` (so direct
    callers of ``proxy_trust_guard`` get legacy behaviour); the FACADE-LEVEL default
    (``ShapProxiedFS.trust_guard_cardinality_dist``) is ``'zipf'`` with ``zipf_alpha=0.25`` after the
    iter16 composite-fidelity re-evaluation showed Zipf alpha=0.25 lifts ``proxy_fidelity_score`` from
    0.779 to 0.834 on the iter14 width=6000 regime (raw Spearman dips 0.891->0.834 but recall@k jumps
    0.667->0.833). ``'zipf'`` uses ``P(k) ∝ k^(-zipf_alpha)`` (small-k concentration). Higher alpha
    over-compresses to small-k extremes where proxy and honest agree trivially; ``alpha=0`` degenerates
    Zipf to uniform."""
    from scipy.stats import kendalltau, spearmanr

    # iter18: ``spearman_floor`` is a deprecated alias of ``fidelity_floor`` (renamed because the
    # gate has been the composite ``proxy_fidelity_score`` since iter16; the legacy name was a
    # misnomer). Resolve here so the rest of the body only deals with ``fidelity_floor``.
    if spearman_floor is not _FIDELITY_FLOOR_UNSET:
        import warnings
        if fidelity_floor != 0.5:
            # Both supplied -- ambiguous; refuse rather than silently picking one.
            raise ValueError(
                "proxy_trust_guard: pass either `fidelity_floor` (new name) or `spearman_floor` "
                "(deprecated alias), not both.")
        warnings.warn(
            "`spearman_floor` is deprecated since iter18; use `fidelity_floor` (same semantics). "
            "The kwarg name was inherited from the iter15 raw-Spearman gate but the gate has been "
            "the composite `proxy_fidelity_score` since iter16.",
            DeprecationWarning, stacklevel=2,
        )
        fidelity_floor = spearman_floor

    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    f = phi.shape[1]
    weights = None
    if unit_f_scores is not None:
        weights = np.asarray(unit_f_scores, dtype=np.float64)
        if weights.shape[0] != f:
            logger.warning(
                "ShapProxiedFS trust-guard: unit_f_scores length %d != phi.shape[1] %d; "
                "falling back to uniform anchor sampling.", int(weights.shape[0]), int(f))
            weights = None
    anchors = _sample_anchor_subsets(f, n_anchors, rng, min_card, max_card,
                                     weights=weights, uniform_tail_frac=anchor_uniform_tail_frac,
                                     cardinality_dist=cardinality_dist, zipf_alpha=zipf_alpha)

    from mlframe.feature_selection._shap_proxy_calibrate import subset_redundancy_many

    tid = ("trust_cap", int(n_estimators_cap)) if n_estimators_cap is not None else None
    # Transpose phi once so each anchor's coalition gather is contiguous (phi is row-major
    # (n_samples, n_units); phi[:, idx] is a strided column gather, ~4x slower than phi_T[idx] at
    # tall n -- same layout win as the _Evaluator seed margins). n_anchors gathers per fit.
    _phi_T = np.ascontiguousarray(phi.T)
    proxy_losses = [proxy_loss(coalition_margin_T(_phi_T, base, idx), y_search, metric) for idx in anchors]
    # iter80: open the cross-process disk cache once (None when disabled). The cache short-circuits
    # the per-anchor xgboost fit whenever (X_search, y_search, X_holdout, y_holdout, expanded cols,
    # template params, cap) was retrained by a prior fit -- the standard ShapProxiedFS hyperparam
    # sweep / ablation pattern. Open here (not per-anchor) so the LRU evictor sees the whole batch.
    disk_cache = _open_disk_cache(disk_cache_dir)
    honest_losses = _parallel_honest_losses(
        [(_expand(idx, unit_to_members), None) for idx in anchors], model_template, X_search, y_search,
        X_holdout, y_holdout, classification, metric, n_jobs, cache=cache,
        n_estimators_cap=n_estimators_cap, template_id=tid, inner_n_jobs_cap=inner_n_jobs_cap,
        disk_cache=disk_cache)
    cards = np.array([len(a) for a in anchors], dtype=np.float64)
    # Reuse the contiguous _phi_T built above for the coalition margins instead of letting
    # subset_redundancy_many re-transpose phi (one fewer O(n_samples*n_units) copy per trust_guard call).
    redunds = subset_redundancy_many(phi, anchors, phi_T=_phi_T)
    proxy_losses = np.asarray(proxy_losses)
    honest_losses = np.asarray(honest_losses)
    ok = np.isfinite(proxy_losses) & np.isfinite(honest_losses)
    proxy_losses, honest_losses, cards, redunds = proxy_losses[ok], honest_losses[ok], cards[ok], redunds[ok]

    sp = float(spearmanr(proxy_losses, honest_losses).statistic) if len(proxy_losses) > 2 else float("nan")
    kt = float(kendalltau(proxy_losses, honest_losses).statistic) if len(proxy_losses) > 2 else float("nan")
    # recall@k: do the proxy's best-k anchors overlap the honest best-k?
    k = max(1, len(proxy_losses) // 5)
    proxy_best = set(np.argsort(proxy_losses)[:k].tolist())
    honest_best = set(np.argsort(honest_losses)[:k].tolist())
    recall = len(proxy_best & honest_best) / k if k else float("nan")

    # Composite fidelity: weighted convex combination of Spearman and recall@k. Both live on [0, 1]
    # for non-degenerate inputs; Spearman is clipped to [0, 1] for the composite so a broken-proxy
    # negative Spearman doesn't get "credited" by a high recall@k (the recall@k of a 1-anchor top-k
    # is trivially 1.0). The raw spearman field is kept unchanged for diagnostics.
    w_sp, w_rc = float(fidelity_weights[0]), float(fidelity_weights[1])
    total = w_sp + w_rc
    if total <= 0:
        raise ValueError(f"fidelity_weights must sum to a positive value, got {fidelity_weights!r}")
    w_sp, w_rc = w_sp / total, w_rc / total
    sp_pos = max(0.0, sp) if np.isfinite(sp) else 0.0
    rc_pos = recall if np.isfinite(recall) else 0.0
    fidelity = float(w_sp * sp_pos + w_rc * rc_pos)
    gate_metric_name = str(trustworthy_metric).lower()
    if gate_metric_name == "spearman":
        gate_value = sp
    elif gate_metric_name in ("proxy_fidelity_score", "fidelity", "composite"):
        gate_metric_name = "proxy_fidelity_score"
        gate_value = fidelity
    else:
        raise ValueError(
            f"trustworthy_metric must be 'proxy_fidelity_score' or 'spearman', got {trustworthy_metric!r}")
    trustworthy = np.isfinite(gate_value) and gate_value >= fidelity_floor
    report = dict(n_anchors=int(len(proxy_losses)), spearman=sp, kendall=kt,
                  recall_at_k=recall, k=int(k),
                  # iter18: ``fidelity_floor`` is the canonical key for the gate threshold.
                  # ``spearman_floor`` is kept as a deprecated alias in the report so legacy
                  # downstream consumers that inspect the dict by the old name don't break.
                  fidelity_floor=fidelity_floor, spearman_floor=fidelity_floor,
                  # iter16: composite gate (default) -- raw spearman / recall stay above as diagnostics.
                  proxy_fidelity_score=fidelity,
                  fidelity_weights=(w_sp, w_rc),
                  trustworthy_metric=gate_metric_name,
                  trustworthy=bool(trustworthy),
                  # Anchor sampling mode: 'stratified' when F-score weights were supplied + applied,
                  # else 'uniform' (legacy). Diagnostic so downstream consumers can see when the
                  # F-score-aware prior was active without inspecting kwargs.
                  anchor_sampling=("stratified" if weights is not None else "uniform"),
                  anchor_uniform_tail_frac=float(anchor_uniform_tail_frac) if weights is not None else None,
                  # Cardinality prior: 'zipf' (iter15 default) or 'uniform' (legacy). Recorded so
                  # downstream diagnostics / bench scripts can see which prior generated the anchors.
                  anchor_cardinality_dist=str(cardinality_dist).lower(),
                  anchor_zipf_alpha=float(zipf_alpha) if str(cardinality_dist).lower() == "zipf" else None,
                  # Raw anchor pairs (proxy, honest, cardinality, redundancy) for the bias corrector.
                  _corrector_data=dict(proxy=proxy_losses.tolist(), honest=honest_losses.tolist(),
                                       cards=cards.tolist(), redund=redunds.tolist()))
    if not trustworthy:
        logger.warning(
            "ShapProxiedFS: proxy fidelity LOW (%s=%.3f < floor=%.2f; spearman=%.3f, recall@%d=%.2f). "
            "The SHAP-coalition proxy may mis-rank subsets on this data; treat the result with caution "
            "(consider a smaller exhaustive honest search or more selected features).",
            gate_metric_name, gate_value, fidelity_floor, sp, int(k), recall,
        )
    else:
        logger.info(
            "ShapProxiedFS: proxy fidelity OK (%s=%.3f; spearman=%.3f, kendall=%.3f, recall@%d=%.2f).",
            gate_metric_name, gate_value, sp, kt, int(k), recall,
        )
    return report

# Re-exported for external importers after the sampling / loss / refine carve.
__all__ = [
    "proxy_trust_guard",
    "HonestLossCache",
    "_honest_loss",
    "_open_disk_cache",
    "_parallel_honest_losses",
    "_permutation_importance_ranking",
    "_try_cap_n_estimators",
    "revalidate_top_n",
    "active_learning_revalidate",
    "within_cluster_refine",
    "importance_topk_ablation",
    "_sample_anchor_subsets",
    "_softmax_weights",
    "_zipf_card_probs",
    "_ucb_stop_remaining_cannot_win",
    "_ucb_auto_slack",
]
