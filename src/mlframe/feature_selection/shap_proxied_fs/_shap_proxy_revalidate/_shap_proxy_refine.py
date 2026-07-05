"""Honest re-validation + refinement of SHAP-proxy-ranked subsets.

UCB stop helpers plus the public ``revalidate_top_n`` / ``active_learning_revalidate`` /
``within_cluster_refine`` / ``importance_topk_ablation`` entry points. These honestly retrain
candidate subsets on a holdout disjoint from the SHAP/objective data and lean on the loss +
sampling leaf siblings; ``resolve_metric`` comes from the proxy objective module.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import resolve_metric
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
    _expand, _honest_loss, _open_disk_cache, _parallel_honest_losses,
    _permutation_importance_ranking,
)


def _ucb_stop_remaining_cannot_win(
    best_stable_score, remaining_proxy_losses, ucb_slack, parsimony_tol,
):
    """Return ``True`` when no un-evaluated candidate can plausibly beat ``best_stable_score``.

    UCB bound: each un-evaluated candidate's honest loss is best-case ``proxy_loss + ucb_slack``
    (``ucb_slack`` is negative when honest tends to under-shoot proxy in the calibration window).
    If even the most optimistic remaining lower bound exceeds ``best_stable_score`` by more than
    ``parsimony_tol * |best_stable_score|`` it cannot enter the parsimony band, so further fits add
    cost without changing the winner -- safe to stop dispatching new batches.

    Stable across reruns: deterministic comparison of floats only.
    """
    if len(remaining_proxy_losses) == 0:
        return True
    lower_bounds = np.asarray(remaining_proxy_losses, dtype=np.float64) + float(ucb_slack)
    threshold = float(best_stable_score) + float(parsimony_tol) * abs(float(best_stable_score))
    return bool(np.min(lower_bounds) > threshold)


def _winner_from_per_candidate(per_candidate, candidates, member_cols, lambda_stab, parsimony_tol):
    """Parsimony-rule winner index tuple from accumulated per-candidate seed losses (iter77).

    Mirrors the post-dispatch ranking + parsimony pick used to finalise ``revalidate_top_n``'s
    winner; used INSIDE the model-round loop to test winner stability across consecutive rounds
    when ``adaptive_n_models=True``. Returns ``None`` when ``per_candidate`` is empty.
    """
    ranked = []
    for ci, (proxy_loss_val, idx) in enumerate(candidates):
        if ci not in per_candidate or not per_candidate[ci]:
            continue
        scores = np.asarray(per_candidate[ci], dtype=np.float64)
        mean, std = float(scores.mean()), float(scores.std())
        ranked.append(dict(features=tuple(idx), n_members=len(member_cols[ci]), stable_score=mean + lambda_stab * std))
    if not ranked:
        return None
    ranked.sort(key=lambda d: d["stable_score"])
    best_score = ranked[0]["stable_score"]
    threshold = best_score + parsimony_tol * abs(best_score)
    eligible = [d for d in ranked if d["stable_score"] <= threshold]
    chosen = min(eligible, key=lambda d: (d["n_members"], d["stable_score"]))
    return chosen["features"]


def _random_baseline_is_meaningful(k: int, f: int) -> bool:
    """RF1: the winner's-curse random baseline is meaningful only when the winner is STRICTLY smaller than
    the full feature set. At ``k >= f`` a same-size random sample would BE the whole feature set, so the
    baseline equals the all-features model and carries no winner's-curse signal -- skip it."""
    return 0 < k < f


def _ucb_auto_slack(evaluated_proxy, evaluated_honest_mean, stdev_multiplier=1.5):
    """Calibrate the UCB slack from already-evaluated (proxy, honest_mean) pairs.

    ``slack`` shifts proxy onto the honest scale; the lower bound for an un-evaluated candidate's
    honest loss is ``proxy + slack``. To be a *lower* bound we take ``mean(delta) - k * std(delta)``
    where ``delta_i = honest_i - proxy_i``: most of the calibration mass on the high side keeps it
    pessimistic (smaller honest predictions => larger remaining lower bounds rarely => never wrong
    stops). With <2 evaluated points the std is undefined; we fall back to ``mean(delta)`` only,
    which still preserves the proxy ordering but with zero safety margin -- the calling stop check
    additionally requires the margin to clear ``parsimony_tol``.

    Returns 0.0 when no evaluated pairs supplied (caller has not yet started; cannot stop).
    """
    p = np.asarray(evaluated_proxy, dtype=np.float64)
    h = np.asarray(evaluated_honest_mean, dtype=np.float64)
    if p.size == 0 or h.size == 0:
        return 0.0
    delta = h - p
    finite = np.isfinite(delta)
    if not finite.any():
        return 0.0
    delta = delta[finite]
    mean = float(delta.mean())
    if delta.size < 2:
        return mean
    return mean - float(stdev_multiplier) * float(delta.std(ddof=1))


def revalidate_top_n(
    candidates, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, n_models=1, lambda_stab=0.5, parsimony_tol=0.02, rng=None, n_jobs=-1,
    unit_to_members=None, cache=None, revalidation_n_estimators=None,
    ucb_enabled=False, ucb_min_eval_size=None, ucb_slack=None, ucb_stdev_multiplier=1.5,
    candidate_score=None, inner_n_jobs_cap=False, adaptive_n_models=False,
    disk_cache_dir=None,
):
    """Honestly retrain each candidate subset on X_search, evaluate on the disjoint X_holdout.

    Returns ``(best_idx_tuple, ranked, baseline)`` where ``ranked`` is a list of dicts with proxy +
    honest loss, std, and stability-penalised score (``mean + lambda_stab * std``).

    Final selection uses a parsimony / one-standard-error rule (matching RFECV's philosophy): among
    candidates whose stable score is within ``parsimony_tol`` (relative) of the best, pick the one
    with the FEWEST features (tie-break: lower stable score). This counters the proxy's bias toward
    larger subsets -- a noise feature that buys <2% honest improvement should not be kept.

    ``revalidation_n_estimators`` (iter28) caps the per-candidate booster's tree count for the
    PARSIMONY-RULE RANKING trials only. The selection criterion is "stable_score within parsimony_tol
    of the best" -- a RELATIVE ranking decision that stabilises long before the default 300 trees
    (mirror of iter9 refine_n_estimators / iter19 oof_shap_n_estimators / iter10 trust_guard_n_estimators).
    Microbench at the live regime (width=1000, n_rows=5000, snr=8, 11-feature subsets): n=100 vs n=300
    Spearman 0.9414, identical argmin, 2.6x faster per fit. After the winner is chosen, ONE full-template
    re-evaluation is added per ranked entry's reported ``honest_loss`` so the user-visible report stays
    apples-to-apples with the trust-guard / ablation (which use the full template). The cap is tagged
    via ``template_id=('reval_cap', cap)`` so cached entries don't collide with full-template entries
    from elsewhere in the pipeline. ``None`` (default for the standalone function) disables the cap
    (legacy 300-tree behaviour). The selector facade passes ``revalidation_n_estimators=100`` by default.

    bench-attempt-rejected (iter32, 2026-05-29): bias-corrector-predicted-loss CULL gate
    (``corrector`` + ``phi`` + ``max_candidates`` kwargs that sorted top_n candidates by the
    trust-guard corrector's predicted honest loss and culled to ``max_candidates=10`` BEFORE the
    honest retrain loop). Live regime (width=1000, n_rows=5000, snr=8, top_n=20 -> 10): warm
    same-process seed=1 baseline reval=2.61s vs gated reval=2.55s (+2.3%), e2e 8.29s -> 8.27s
    (+0.2%); seed=0 reval 2.95s -> 2.88s (+2.4%); seed=0 e2e 12.58s -> 9.93s (+21%) is dominated
    by within-run prefilter variance (3.71s vs 1.56s in same comparison, NOT gate-attributable).
    cProfile ``xgboost.update`` ncalls=1600 in BOTH baseline and gated -- the actual training-round
    count is the same: the joblib-threading pool at -1 already absorbed the 60-fit batch (per
    iter29 ``time.sleep`` ~5.0s = parallel productive wait), so dropping 30 of those 60 fits leaves
    the same wall because the pool was never the bottleneck. iter28's
    ``revalidation_n_estimators=100`` cap already extracted the per-fit win; further cuts here are
    sub-threshold. Lever does not pay at the current ``n_revalidation_models=3`` + parallel-joblib
    operating point; revisit only if a future iter raises models-per-candidate or serialises the
    retrain loop.

    ``ucb_enabled`` (iter34): batched-dispatch early-stop on the candidate scoring loop. Different
    mechanism from the rejected iter32 cull -- there we proposed dropping tail candidates before the
    pool started (which didn't help because the wall was already a single batched run). Here we
    start the BEST candidates first and stop dispatching new batches once the running winner is
    provably better than any remaining candidate's UCB lower bound. At width >= 10000 each honest
    fit is ~300 ms; ``top_n=20 * n_models=3 = 60`` tasks on 8 workers run ~8 batches deep
    (Phase-0 measurement: 4.86s wall vs 337 ms per-fit = 14.4x ratio, NOT saturation). Skipping the
    last few batches at the tail of the proxy ranking is direct wall savings.

    ``ucb_min_eval_size`` (default ``None`` -> ``max(n_workers, 3)``): first batch evaluates this many
    candidates so the workers saturate; the running ``best_stable_score`` is then defined.
    ``ucb_slack`` (default ``None`` -> auto from batch ``delta = honest - proxy`` via
    ``mean(delta) - ucb_stdev_multiplier * std(delta)``): negative slack means honest tends below
    proxy in the calibration window (the proxy is mildly biased high), so un-evaluated lower bounds
    are tighter; positive slack widens them. The auto calibration is conservative on the
    *pessimistic* side (subtracting std lowers the slack -> tightens the un-evaluated lower bound ->
    requires a larger gap to stop -> fewer wrong stops). With UCB disabled OR n_candidates <=
    min_eval_size OR ``n_jobs in (1, 0, None)``, falls through to the legacy single-batch path with
    zero behaviour change -- single-job runs (test fixtures) have no batching to save on, so the gate
    would only risk dropping the winner without any wall benefit.

    ``adaptive_n_models`` (iter77): when True, dispatch the ``n_models`` stability seeds as SEPARATE
    rounds (one seed per candidate per round) instead of one combined batch. After each completed
    round (k >= 2), the parsimony-rule winner is computed from accumulated per-candidate losses; if
    the winner is identical to the previous round's winner, remaining seed rounds are skipped. Floor
    is 2 rounds (need at least one stability check); ceiling is ``n_models``. Worst case (winners
    differ every round) is the same total fit count as the legacy path. With ``n_models=1`` the knob
    is a no-op. The candidate-UCB candidate-pruning still applies within each round. Conservation
    guarantee: when ``n_models_run == n_models`` the result is bit-identical to the legacy path
    (same seeds, same accumulation, same ranking). When the loop exits early, ``stable_score`` is
    computed on fewer seeds per candidate so std is lower-variance but mean is the same expectation;
    the parsimony rule (relative-tol) is robust to this. Surface: ``baseline['ucb']['n_models_run']``
    reports actual rounds executed.
    """
    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    cap = revalidation_n_estimators
    tid = ("reval_cap", int(cap)) if cap is not None else None
    # iter80: cross-process disk cache for repeat hyperparam sweeps over the same (X_search,
    # y_search, X_holdout, y_holdout, template) tuple. Opened once so the LRU evictor sees the full
    # batch of writes from this revalidation pass; ``None`` (default) keeps the legacy in-memory-only
    # cache wiring.
    disk_cache = _open_disk_cache(disk_cache_dir)
    # Pre-expand member columns + sample per-fit seeds once so candidate ordering shuffles only the
    # task LIST, never re-samples (cache reuse + determinism across UCB/no-UCB paths).
    member_cols = [_expand(idx, unit_to_members) for _, idx in candidates]
    candidate_seeds = [[int(rng.integers(0, 2**31 - 1)) for _ in range(n_models)] for _ in candidates]
    n_total = len(candidates)
    # UCB batched dispatch (iter34): evaluate proxy-ranked candidates in batches; stop once the
    # running winner provably beats every remaining candidate's UCB lower bound. Determinism:
    # within-batch joblib results are zipped back to the (cols, seed) tuples we dispatched, ties in
    # proxy ordering are broken by the original candidate index (kind="stable" argsort), and ALL
    # seeds are sampled BEFORE the gate decides any batch -- so n_candidates_evaluated is the only
    # variable between UCB and the legacy path; ranked entries for evaluated candidates are
    # bit-identical given identical seed + cache state.
    proxy_losses_arr = np.asarray([float(c[0]) for c in candidates], dtype=np.float64)
    # ``candidate_score`` (iter34): the caller's already-computed per-candidate score (corrector-
    # predicted honest loss when the bias corrector fit cleanly, raw proxy_loss otherwise). The
    # facade computes this for ordering anyway; passing it through lets the UCB lower bound work on
    # the honest scale instead of the cheap-but-tightly-clustered proxy_loss. With ``candidate_score``
    # supplied, the gate compares un-evaluated score + slack against the running best stable score;
    # the slack auto-calibrates the residual gap. When ``None`` (standalone tests, legacy callers),
    # falls back to raw proxy_loss -- the gate still works but may rarely fire on regimes whose
    # proxy_loss spread is too tight to discriminate (corrector-aware score widens the spread).
    score_arr = np.asarray(candidate_score, dtype=np.float64) if candidate_score is not None else proxy_losses_arr
    # Use the CALLER'S order (the facade already sorts top_n by bias-corrector + uncertainty score,
    # which is a strictly stronger ordering than raw proxy_loss alone). Re-sorting on proxy_loss
    # here would unwind that work and surrender the corrector's per-candidate trust signal -- the
    # very signal the trust-guard pays its 60+ anchor retrains to produce. Stays compatible with
    # the standalone test fixtures that pass already-proxy-sorted candidates.
    proxy_order = np.arange(n_total, dtype=np.int64)
    if ucb_min_eval_size is None:
        import os as _os
        n_cores = _os.cpu_count() or 1
        outer = n_cores if n_jobs in (-1, None, 0) else int(n_jobs)
        ucb_min_eval_size_eff = max(int(outer), 3)
    else:
        ucb_min_eval_size_eff = max(1, int(ucb_min_eval_size))
    # UCB only pays when joblib actually BATCHES dispatch across workers (the iter34 premise).
    # With n_jobs=1 the legacy single-batch path is already sequential, so skipping candidates
    # produces no wall savings -- only opens a window for the gate to stop on a too-small evaluated
    # batch (3 candidates) and miss the winner. The user-visible failure mode on the biz_val test
    # (noise2 kept where the legacy path picked an informative) is exactly this: 1-job runs are
    # typically test fixtures where determinism + recall matter more than wall savings.
    use_ucb = bool(ucb_enabled) and n_total > ucb_min_eval_size_eff and n_jobs not in (1, 0, None)

    per_candidate: dict[int, list[float]] = {}
    n_candidates_evaluated = 0
    ucb_slack_used = 0.0
    # iter77 adaptive_n_models: split the n_models stability seeds into separate rounds, allow early
    # stop when the parsimony winner stabilises. With adaptive_n_models=False (legacy) the rounds
    # collapse into one combined batch (the original semantics).
    adapt_active = bool(adaptive_n_models) and int(n_models) >= 2
    seed_rounds = int(n_models) if adapt_active else 1
    # When NOT adaptive, "one round" dispatches all n_models seeds per candidate; when adaptive,
    # each round dispatches ONE seed per candidate (round k -> candidate_seeds[ci][k:k+1]).
    n_models_run = 0
    prev_winner: tuple | None = None
    # iter92: compare winners on the EXPANDED member-column set rather than the raw unit tuple.
    # At high redundancy_rho with cluster_features=True, distinct unit tuples can map to the same
    # deployed member subset after cluster aggregation. The user-visible "chosen subset" is the
    # member set, so equivalence on members captures convergence one round earlier whenever two
    # different unit tuples collapse to the same deployment. Build a per-candidate member-key
    # lookup (sorted tuple) once -- _expand was already paid for in ``member_cols`` above.
    members_by_unit_tuple = {tuple(idx): tuple(sorted(int(c) for c in member_cols[ci])) for ci, (_, idx) in enumerate(candidates)}
    prev_winner_members: tuple | None = None
    # When the early-stop fires on member-equivalence WHILE unit tuples still differ, that's the
    # iter92-specific win; tracked separately so downstream diagnostics can quantify the lever.
    stopped_via_member_equiv = False

    for round_k in range(seed_rounds):
        if adapt_active:
            round_seeds = [[candidate_seeds[ci][round_k]] for ci in range(n_total)]
        else:
            round_seeds = candidate_seeds  # legacy: all seeds in one round
        if not use_ucb:
            # Legacy path: one parallel batch over all candidates.
            tasks, task_owner = [], []
            for ci in range(n_total):
                for s in round_seeds[ci]:
                    tasks.append((member_cols[ci], s))
                    task_owner.append(ci)
            losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                             classification, metric, n_jobs, cache=cache,
                                             n_estimators_cap=cap, template_id=tid,
                                             inner_n_jobs_cap=inner_n_jobs_cap,
                                             disk_cache=disk_cache)
            for owner, loss in zip(task_owner, losses):
                per_candidate.setdefault(owner, []).append(loss)
            n_candidates_evaluated = n_total
        else:
            evaluated_idx_set: set[int] = set()
            # First batch saturates the workers. Subsequent batches are workers-sized so each iteration
            # is one wall-clock pool dispatch on the same operating point.
            import os as _os
            n_cores = _os.cpu_count() or 1
            outer_workers = n_cores if n_jobs in (-1, None, 0) else int(n_jobs)
            outer_workers = max(1, outer_workers)
            # Effective seeds-per-candidate THIS round drives the worker-share denominator.
            seeds_per_cand = len(round_seeds[0]) if round_seeds and round_seeds[0] else 1
            batch_sizes: list[int] = []
            cur = 0
            while cur < n_total:
                if cur == 0:
                    step = min(ucb_min_eval_size_eff, n_total - cur)
                else:
                    step = min(max(1, outer_workers // max(1, seeds_per_cand)), n_total - cur)
                    step = max(step, 1)
                batch_sizes.append(step)
                cur += step
            pos = 0
            for step in batch_sizes:
                batch_candidate_idx = [int(proxy_order[pos + j]) for j in range(step)]
                pos += step
                tasks, task_owner = [], []
                for ci in batch_candidate_idx:
                    for s in round_seeds[ci]:
                        tasks.append((member_cols[ci], s))
                        task_owner.append(ci)
                losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                                 classification, metric, n_jobs, cache=cache,
                                                 n_estimators_cap=cap, template_id=tid,
                                                 inner_n_jobs_cap=inner_n_jobs_cap,
                                                 disk_cache=disk_cache)
                for owner, loss in zip(task_owner, losses):
                    per_candidate.setdefault(owner, []).append(loss)
                evaluated_idx_set.update(batch_candidate_idx)
                n_candidates_evaluated = len(evaluated_idx_set)
                if pos >= n_total:
                    break
                # Compute the running best stable_score over evaluated candidates.
                best_so_far = float("inf")
                ev_proxy: list[float] = []
                ev_honest_mean: list[float] = []
                for ci in evaluated_idx_set:
                    scores = np.asarray(per_candidate[ci], dtype=np.float64)
                    mean = float(scores.mean())
                    std = float(scores.std())
                    stable = mean + lambda_stab * std
                    if stable < best_so_far:
                        best_so_far = stable
                    ev_proxy.append(float(score_arr[ci]))
                    ev_honest_mean.append(mean)
                if ucb_slack is None:
                    ucb_slack_used = _ucb_auto_slack(ev_proxy, ev_honest_mean, ucb_stdev_multiplier)
                else:
                    ucb_slack_used = float(ucb_slack)
                remaining_idx = [int(proxy_order[j]) for j in range(pos, n_total)]
                remaining_score = [float(score_arr[ci]) for ci in remaining_idx]
                if _ucb_stop_remaining_cannot_win(
                    best_so_far, remaining_score, ucb_slack_used, parsimony_tol,
                ):
                    break
        n_models_run = round_k + 1
        # Parsimony-rule winner across accumulated per-candidate losses; early-stop when stable
        # across two consecutive rounds. Floor at 2 rounds so we always have at least one stability
        # check (round_k >= 1). When n_models == 1 the loop runs exactly once -- no check possible
        # and adapt_active is False anyway, so this branch is skipped.
        if adapt_active:
            cur_winner = _winner_from_per_candidate(
                per_candidate, candidates, member_cols, lambda_stab, parsimony_tol,
            )
            # iter92: equivalence on EXPANDED members. ``cur_winner`` is the unit tuple; look up its
            # deployed member set. Fallback to the unit tuple if (somehow) absent from the map,
            # which preserves iter77 semantics on degenerate inputs.
            cur_winner_members = members_by_unit_tuple.get(cur_winner) if cur_winner is not None else None
            if round_k >= 1 and cur_winner_members is not None and cur_winner_members == prev_winner_members:
                # Member sets matched; the iter92 "fired earlier than iter77" subcase is when the
                # unit tuples differ even though the deployed member sets are identical.
                if cur_winner != prev_winner:
                    stopped_via_member_equiv = True
                break
            prev_winner = cur_winner
            prev_winner_members = cur_winner_members

    ranked = []
    for ci, (proxy_loss_val, idx) in enumerate(candidates):
        if ci not in per_candidate:
            continue
        scores = np.asarray(per_candidate[ci], dtype=np.float64)
        mean, std = float(scores.mean()), float(scores.std())
        # Parsimony cardinality = deployed feature count (expanded members), not unit count.
        ranked.append(dict(features=tuple(idx), n_members=len(member_cols[ci]),
                           proxy_loss=float(proxy_loss_val),
                           honest_loss=mean, honest_std=std, stable_score=mean + lambda_stab * std))
    ranked.sort(key=lambda d: d["stable_score"])
    if ranked:
        best_score = ranked[0]["stable_score"]
        threshold = best_score + parsimony_tol * abs(best_score)
        eligible = [d for d in ranked if d["stable_score"] <= threshold]
        chosen = min(eligible, key=lambda d: (d["n_members"], d["stable_score"]))
        best_idx = chosen["features"]
    else:
        best_idx = ()

    # Full-template re-evaluation of the WINNER so the user-visible honest_loss in the report stays
    # apples-to-apples with the trust-guard / ablation outputs (those use the full template). Only
    # the chosen subset is re-fit -- a single extra fit, not n_models more. The cache lookup uses
    # the full-template namespace (template_id=None) so it hits any prior pipeline retrain of the
    # same subset (e.g. when ablation later refits the winner, that fit is the cache hit). Same
    # design as within_cluster_refine's final full-template re-evaluation. When cap is None the
    # ranking trials already used the full template and this is a guaranteed cache hit (no extra fit).
    if best_idx and cap is not None:
        winner_cols = _expand(best_idx, unit_to_members)
        winner_full_loss = _honest_loss(
            model_template, X_search, y_search, X_holdout, y_holdout, winner_cols, classification, metric, cache=cache, disk_cache=disk_cache
        )
        # Update the reported entry for the chosen winner. Find it in ranked by features identity.
        for d in ranked:
            if d["features"] == best_idx:
                d["honest_loss"] = float(winner_full_loss)
                # std measured at capped template (n_models samples); winner's full-template eval is a
                # single fit so its std is not refreshed -- the capped-template std remains as a
                # cross-seed-stability proxy. Update stable_score to reflect the new mean.
                d["stable_score"] = float(winner_full_loss) + lambda_stab * d["honest_std"]
                d["honest_loss_capped"] = float(np.asarray(per_candidate[next(i for i, (_, ix) in enumerate(candidates) if tuple(ix) == best_idx)]).mean())
                break

    # Same-size (in member columns) random-subset baseline for the winner (winner's-curse context).
    # RF1: only meaningful when the winner is strictly smaller than the full feature set; when k >= f the
    # "random" sample would BE the whole feature set, so the baseline equals the all-features model and
    # carries no winner's-curse signal -- skip it (baseline stays None) rather than report a tautology.
    baseline = None
    if best_idx:
        k = len(_expand(best_idx, unit_to_members))
        f = X_search.shape[1]
        if _random_baseline_is_meaningful(k, f):
            rnd = tuple(sorted(rng.choice(f, size=k, replace=False).tolist()))
            baseline = dict(features=rnd, honest_loss=_honest_loss(
                model_template, X_search, y_search, X_holdout, y_holdout, list(rnd), classification, metric,
                cache=cache, disk_cache=disk_cache))
    ucb_info = dict(enabled=bool(use_ucb), n_candidates_total=int(n_total),
                    n_candidates_evaluated=int(n_candidates_evaluated),
                    min_eval_size=int(ucb_min_eval_size_eff),
                    slack=float(ucb_slack_used),
                    adaptive_n_models=bool(adapt_active),
                    n_models_configured=int(n_models),
                    n_models_run=int(n_models_run),
                    # iter92: ``True`` when adaptive early-stop fired because two consecutive
                    # rounds' winners EXPANDED to the same member set despite differing unit
                    # tuples (cluster-aggregation collapse). ``False`` covers both the
                    # iter77-equivalent firing (unit tuples already matched) and the "no
                    # early-stop" case.
                    n_reval_models_run_via_member_equiv=bool(stopped_via_member_equiv))
    # Attach UCB diagnostic to the random-subset baseline dict (or create a stub when no winner).
    # Keeps the 3-tuple return contract stable; downstream consumers fish ucb diagnostics out via
    # ``report['revalidation']['random_baseline']['ucb']``. Same pattern as other-revalidator-side
    # diagnostics that ride on the baseline payload without expanding the return signature.
    if baseline is None:
        baseline = dict(ucb=ucb_info)
    else:
        baseline["ucb"] = ucb_info
    return best_idx, ranked, baseline


def active_learning_revalidate(
    candidates, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, corrector_data, phi, budget, batch=4, n_models=1,
    parsimony_tol=0.02, rng=None, n_jobs=-1, unit_to_members=None, cache=None,
    revalidation_n_estimators=None, inner_n_jobs_cap=False, disk_cache_dir=None,
):
    """Disagreement-driven honest re-validation (lever #4).

    Instead of honestly retraining the proxy's static top-N, iterate: fit the bias corrector on the
    anchors seen so far, pick the ``batch`` un-evaluated candidates where the corrector most disagrees
    with the raw proxy (the proxy is least trustworthy there), honestly retrain them, fold the results
    back into the corrector, and repeat until ``budget`` candidates have been evaluated. This spends a
    fixed retrain budget where it most reduces winner's-curse risk. The proxy's top-1 is always among
    the first evaluated, so the result is never worse than naive top-1.

    ``revalidation_n_estimators`` (iter28): same cap semantics as ``revalidate_top_n`` -- per-candidate
    ranking trials use the capped booster (cheaper but ranking-equivalent), winner gets ONE
    full-template re-evaluation to keep the reported ``honest_loss`` apples-to-apples. The corrector
    is fit on the CAPPED honest losses (the corrector is itself a ranking-quality tool, so working
    in the capped space is consistent).

    Returns ``(best_idx, ranked, n_evaluated)``.
    """
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_calibrate import fit_proxy_corrector, subset_redundancy_many

    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    cap = revalidation_n_estimators
    tid = ("reval_cap", int(cap)) if cap is not None else None
    # iter80: cross-process disk cache (same wiring as ``revalidate_top_n``). Each AL round picks a
    # disagreement-driven batch and retrains honest losses; the disk cache short-circuits any
    # (cols, seed, template) tuple seen on a prior fit. Disabled when ``disk_cache_dir is None``.
    disk_cache = _open_disk_cache(disk_cache_dir)
    cd = {k: list(v) for k, v in corrector_data.items()}  # mutable copy we augment each round
    proxy_all = np.array([c[0] for c in candidates], dtype=np.float64)
    idxs = [c[1] for c in candidates]
    cards_all = np.array([len(i) for i in idxs], dtype=np.float64)
    redund_all = subset_redundancy_many(phi, idxs)
    member_cols = [_expand(i, unit_to_members) for i in idxs]

    honest = {}  # candidate index -> mean honest loss
    budget = min(budget, len(candidates))
    while len(honest) < budget:
        corrector = fit_proxy_corrector(cd["proxy"], cd["honest"], cd["cards"], cd["redund"])
        pred = corrector.predict(proxy_all, cards_all, redund_all)
        disagree = np.abs(pred - proxy_all)  # 0 under fallback -> falls back to proxy ordering
        remaining = [i for i in range(len(candidates)) if i not in honest]
        remaining.sort(key=lambda i: (-disagree[i], pred[i]))
        pick = remaining[: max(1, min(batch, budget - len(honest)))]
        if not pick:
            break
        tasks = [(member_cols[i], int(rng.integers(0, 2**31 - 1))) for i in pick for _ in range(n_models)]
        losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                         classification, metric, n_jobs, cache=cache,
                                         n_estimators_cap=cap, template_id=tid,
                                         inner_n_jobs_cap=inner_n_jobs_cap,
                                         disk_cache=disk_cache)
        for j, i in enumerate(pick):
            seg = losses[j * n_models : (j + 1) * n_models]
            m = float(np.mean(seg))
            honest[i] = m
            cd["proxy"].append(float(proxy_all[i]))
            cd["honest"].append(m)
            cd["cards"].append(float(cards_all[i]))
            cd["redund"].append(float(redund_all[i]))

    ranked = [dict(features=tuple(idxs[i]), n_members=len(member_cols[i]), proxy_loss=float(proxy_all[i]),
                   honest_loss=honest[i], honest_std=0.0, stable_score=honest[i]) for i in honest]
    ranked.sort(key=lambda d: d["stable_score"])
    best_idx = ()
    if ranked:
        best_score = ranked[0]["stable_score"]
        thr = best_score + parsimony_tol * abs(best_score)
        eligible = [d for d in ranked if d["stable_score"] <= thr]
        best_idx = min(eligible, key=lambda d: (d["n_members"], d["stable_score"]))["features"]

    # Full-template re-evaluation of the winner (mirror of revalidate_top_n; see that function for the
    # apples-to-apples rationale). When cap is None this is a guaranteed cache hit (no extra fit).
    if best_idx and cap is not None:
        winner_cols = _expand(best_idx, unit_to_members)
        winner_full_loss = _honest_loss(
            model_template, X_search, y_search, X_holdout, y_holdout, winner_cols, classification, metric, cache=cache, disk_cache=disk_cache
        )
        for d in ranked:
            if d["features"] == best_idx:
                d["honest_loss_capped"] = float(d["honest_loss"])
                d["honest_loss"] = float(winner_full_loss)
                d["stable_score"] = float(winner_full_loss)
                break

    return best_idx, ranked, len(honest)


def within_cluster_refine(
    member_cols, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, parsimony_tol=0.02, n_jobs=-1, max_drop_rounds=None, cache=None,
    member_groups=None, min_multi_clusters=3, refine_n_estimators=100,
    ucb_enabled=False, ucb_min_eval_size=None, ucb_slack=None, ucb_stdev_multiplier=1.0,
    inner_n_jobs_cap=False, disk_cache_dir=None,
):
    """Compact the selected clusters' member columns down to a quality-preserving subset (honest).

    The proxy ranks UNITS (denoised cluster representatives); honest deployment trains on REAL member
    columns. Expanding chosen units yields the union of their clusters' members - often redundant by
    construction (within-cluster correlation >= the clustering threshold). This routine prunes those
    redundant members while the honest holdout loss stays within ``parsimony_tol`` of the best seen.

    Two-stage algorithm (when ``member_groups`` is supplied AND has >= ``min_multi_clusters`` multi-
    member groups):

    1. CLUSTER-AT-A-TIME COLLAPSE. For each MULTI-member cluster, run ONE honest probe of
       "drop all members of this cluster except its first one (the aggregator's reference)", in
       PARALLEL across clusters. Independently accept each probe whose loss respects ``parsimony_tol``
       against ``base``. This is the canonical safe collapse: within-cluster members are
       near-duplicates by construction (within-cluster correlation >= the clustering threshold), so
       dropping all-but-one is the cheapest deduplication. Crucially, each cluster's probe is
       INDEPENDENT (other clusters retain full membership during their probes), so a failure to
       collapse one redundant cluster doesn't poison the others -- unlike a "drop everything safe at
       once" multi-drop that conflates redundancy with noise singletons.
    2. CROSS-CLUSTER GREEDY-BACKWARD on the now-much-smaller working set: legacy "drop the column
       whose loss is best, while within tol" until no single drop helps. This handles the noise
       singletons that survived stage 1 and any inter-cluster redundancy the proxy missed.

    The shared ``HonestLossCache`` (built once per ``ShapProxiedFS.fit``) makes the single-drop trials
    in stage 2 reuse fits from stage 1 (and from the surrounding pipeline: the winner is refit during
    the importance ablation), so the cost is dominated by genuinely new (subset, seed) combinations.

    Low-redundancy fast-path: when fewer than ``min_multi_clusters`` of the supplied ``member_groups``
    have more than one member, the stage-1 probes (1 per multi-cluster + 1 cumulative verification fit)
    don't pay back -- on essentially-singleton data stage-1 just adds k+1 fits and routes the same
    columns into stage 2 unchanged. Skip stage 1 and run legacy single-drop greedy directly. Measured
    fix for an iter7 regression on low-redundancy (2k-feature clean) datasets where 0..1 multi-cluster
    groups paid the stage-1 toll for no collapse opportunity.

    When ``member_groups`` is ``None`` (legacy call sites or non-clustering mode), runs the original
    pure greedy-backward over ``member_cols`` -- behavior strictly preserved for backward compatibility.

    ``ucb_enabled`` (iter35): batched-dispatch early-stop on stage 2b's per-round single-drop greedy.
    Each round sorts drop trials by ascending stage-2a permutation importance (lowest = safest drop =
    lowest expected honest loss), dispatches in workers-sized batches, and stops dispatching once no
    un-evaluated trial can beat the running round leader. The UCB lower bound for an un-evaluated
    trial is ``importance + slack`` where ``slack`` auto-calibrates from the round's evaluated
    (importance, honest_loss) pairs via ``mean(delta) - stdev_multiplier * std(delta)``. Mirrors
    iter34's revalidate_top_n UCB knob design. Falls through to legacy single-batch-per-round when
    UCB is off OR ``n_jobs in (1, 0, None)`` OR stage-2a's importance prior is missing -- single-job
    runs (test fixtures) have no batching to save on, so the gate would only risk wrong stops without
    a wall benefit. The lever pays at width >= 10000 where each honest fit is ~500 ms and ~5
    stage-2b rounds dispatch ~10 trials each (Phase-0 C3 cProfile: within_cluster_refine 6.14s of
    which ~5s is stage-2b parallel batches).

    ``refine_n_estimators`` caps the booster's tree count for refine's ranking-only probe / trial fits.
    Refine compares RELATIVE honest losses to decide "is dropping this member within parsimony_tol?";
    importance / loss ranking stabilises well below the default 300 trees (the empirical rule of thumb
    is ~100), so the cap cuts each fit's cost ~3x while preserving the drop decisions. After the final
    compact subset is chosen, that ONE subset is re-evaluated with the FULL ``model_template`` so the
    user-visible ``honest_loss`` reported downstream stays an apples-to-apples comparison against the
    other guards (which all use the full template). Set ``refine_n_estimators=None`` to disable the cap
    (legacy behavior). The cap is silently a no-op for templates without an ``n_estimators``-like param
    (e.g. a linear model in tests), so all existing behavior-preservation tests stay green.

    Returns the refined member-column list.
    """
    metric = resolve_metric(classification, metric)
    current = sorted(set(int(c) for c in member_cols))
    if len(current) <= 1:
        return current
    # Refine's per-trial fits use a capped n_estimators template (cheap ranking signal); the cap is
    # tagged via template_id so cached values don't collide with the full-template cache entries
    # populated elsewhere in the pipeline. ``cap`` is None -> no capping, no namespacing (legacy path).
    cap = refine_n_estimators
    tid = ("refine_cap", int(cap)) if cap is not None else None
    # iter81: cross-process disk cache extends iter80's wiring through the refine stage. Stage-1
    # parallel cluster probes and stage-2b per-round single-drop trials repeat the SAME (cols, seed,
    # template, cap) tuple on hyperparam sweeps -- a warm-cache lookup skips the booster fit entirely.
    # ``None`` (default) keeps the legacy in-memory-only contract bit-identical.
    disk_cache = _open_disk_cache(disk_cache_dir)
    # Defer the initial-base fit when Stage 1 won't fire: in that case Stage 2a's
    # ``_permutation_importance_ranking`` fits the SAME booster on the SAME ``current`` columns
    # (same template, same cap, same fixed template random_state, no seed-override on the base
    # call), so its returned ``rank_base`` is byte-equivalent to the value the base ``_honest_loss``
    # would have produced. Folding the two into one fit saves ~0.5 s per refine call at C3-scale
    # (10k rows x 26-col working set) and pays out anytime member_groups is None or all groups are
    # singletons (the cluster-collapse low-redundancy fast path). When Stage 1 WILL fire we keep
    # the eager base fit because its threshold gates the Stage 1 probes -- the perm-importance pass
    # runs AFTER Stage 1 may have shrunk ``current``, so its rank_base would be on the wrong set.
    n_multi_eligible = 0
    if member_groups is not None:
        current_set_pre = set(current)
        for g in member_groups:
            if sum(1 for c in g if int(c) in current_set_pre) > 1:
                n_multi_eligible += 1
    stage1_will_fire = member_groups is not None and n_multi_eligible >= min_multi_clusters
    if stage1_will_fire:
        base = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout, current,
                            classification, metric, cache=cache, n_estimators_cap=cap,
                            template_id=tid, disk_cache=disk_cache)
        threshold = base + parsimony_tol * abs(base)
    else:
        # Sentinel: Stage 2a will compute ``rank_base`` and seed ``base`` from it.
        base = None  # type: ignore[assignment]
        threshold = None  # type: ignore[assignment]

    # ---- Stage 1: per-cluster collapse (one parallel probe per multi-cluster).
    # Skip when member_groups is missing OR has too few multi-member groups to pay the stage-1 toll
    # (k probes + 1 cumulative verify); on low-redundancy data the cluster-collapse never fires and
    # we just want to fall through to stage 2's legacy single-drop greedy.
    if stage1_will_fire:
        current_set = set(current)
        # Normalize: filter member_groups to columns actually in `current`, drop empties / singletons.
        multi: list[list[int]] = []
        for g in member_groups:
            sub = [int(c) for c in g if int(c) in current_set]
            if len(sub) > 1:
                multi.append(sub)
        if multi:
            # One probe per multi-cluster: drop ALL members except the first (canonical representative).
            # Other clusters keep FULL membership; the probe asks "can we safely deduplicate THIS one?".
            probes: list[tuple[list[int], int, list[int]]] = []  # (subset, cluster_idx, dropped_members)
            for ci, g in enumerate(multi):
                # g[0] is the surviving representative (the cluster aggregator's first member); g[1:]
                # are the redundant members the probe asks to drop while other clusters stay intact.
                drop_set = set(g[1:])
                probe_cols = sorted(c for c in current if c not in drop_set)
                probes.append((probe_cols, ci, sorted(drop_set)))
            losses = _parallel_honest_losses(
                [(p[0], None) for p in probes], model_template, X_search, y_search, X_holdout, y_holdout,
                classification, metric, n_jobs, cache=cache, n_estimators_cap=cap, template_id=tid,
                inner_n_jobs_cap=inner_n_jobs_cap, disk_cache=disk_cache)
            # Each probe is evaluated against the ORIGINAL base/threshold (cluster collapses are
            # measured independently, not against each other). Accepted probes' drops accumulate.
            accepted_drops: set[int] = set()
            for (probe_cols, ci, drops), ls in zip(probes, losses):
                if ls <= threshold:
                    accepted_drops.update(drops)
            if accepted_drops:
                collapsed = sorted(c for c in current if c not in accepted_drops)
                # Verify the union of all accepted cluster-collapses still respects tol (sum-of-parts
                # need not equal whole: pathological mutual dependence between clusters could fail
                # the cumulative drop even if each was independently fine).
                if len(collapsed) < len(current):
                    cum_loss = _honest_loss(
                        model_template, X_search, y_search, X_holdout, y_holdout, collapsed, classification,
                        metric, cache=cache, n_estimators_cap=cap, template_id=tid, disk_cache=disk_cache)
                    if cum_loss <= threshold:
                        current = collapsed
                        base = min(base, cum_loss)
                        threshold = base + parsimony_tol * abs(base)
                    elif len(multi) == 1:
                        # Only one cluster was collapsed; the cumulative IS the single probe -- if
                        # one passed and the other failed, that's just float noise (cache should make
                        # them byte-identical, but defend in depth). Accept the probe result anyway.
                        current = collapsed
                        base = min(base, cum_loss)
                        threshold = base + parsimony_tol * abs(base)
                    else:
                        # Cumulative drop hurts beyond tol: accept only the single best-loss cluster
                        # collapse (the safest individual drop set), defer the rest to stage 2.
                        best_ci, best_loss = -1, float("inf")
                        for (probe_cols, ci, drops), ls in zip(probes, losses):
                            if ls <= threshold and ls < best_loss:
                                best_ci, best_loss = ci, float(ls)
                        if best_ci >= 0:
                            single_drops = set(probes[best_ci][2])
                            current = sorted(c for c in current if c not in single_drops)
                            base = min(base, best_loss)
                            threshold = base + parsimony_tol * abs(base)

    # ``importance_by_col`` (iter35): persist stage-2a's permutation importances so stage-2b can sort
    # its per-round drop trials in ascending importance order and dispatch UCB-batched. Defaults to
    # empty -> stage-2b falls back to legacy unsorted single-batch dispatch.
    importance_by_col: dict[int, float] = {}
    # ---- Stage 2a: ONE permutation-importance + batch-drop pass on the (possibly stage-1-collapsed)
    # working set. This is the iter11 perf win: a single ranking pass (1 fit + k cheap predicts)
    # ranks every member by drop-safety, then we accept the largest batched drop that respects
    # parsimony_tol -- collapsing what would have been many legacy single-drop greedy rounds into
    # ONE verify retrain (with halving fallbacks on rejection). The pass is run AT MOST ONCE per
    # refine call: after the initial bulk-compaction, the working set is small (typically a handful
    # of columns) and the subsequent single-drop greedy stage-2b can polish it in legacy O(k)
    # retrains -- the runtime cost of which is now negligible because k is small. Running multiple
    # batch-drop rounds before stage-2b empirically over-prunes on the regime synthetic (the
    # batched verify can mask the loss of informatives whose signal is carried by surviving
    # redundancy-cluster reflections; legacy's gradual tightening protects against that).
    if len(current) > 1:
        rank_base, importances = _permutation_importance_ranking(
            model_template, X_search, y_search, X_holdout, y_holdout, current, classification, metric,
            n_estimators_cap=cap, seed=0, disk_cache=disk_cache, template_id=tid)
        # When Stage 1 was skipped, ``rank_base`` IS the initial honest base on the full working set
        # (perm-importance fits the same booster on the same cols, so the un-permuted loss is the
        # base ``_honest_loss`` would have returned). When Stage 1 fired and updated base/threshold,
        # min() preserves the existing semantics (Stage 1 only ever drops cols, so its post-drop
        # loss is the smaller-is-better value to keep).
        if base is None:
            base = float(rank_base)
        else:
            base = min(base, float(rank_base))
        cur_threshold = base + parsimony_tol * abs(base)
        # Persist per-column importance so stage 2b can sort its drop trials by ascending-importance
        # priors (lowest importance = safest drop = lowest expected honest loss). Used as the UCB
        # proxy when ``ucb_enabled``; dropped members fall out of the dict naturally on lookup.
        importance_by_col = {int(current[i]): float(importances[i]) for i in range(len(current))}
        # Sort members ascending by importance (lowest = safest to drop first).
        order = np.argsort(importances, kind="stable")
        sorted_imps = importances[order]
        n = len(current)
        # Strict safe-batch sizing: a member is "clearly drop-safe" only if shuffling its column
        # leaves holdout loss BELOW or AT the un-permuted base (importance <= 0). This excludes
        # the marginal "importance > 0 but < parsimony_tol*|base|" region which is precisely where
        # informatives whose signal is carried by a surviving redundancy-cluster reflection look
        # safe in isolation but contribute non-trivially in aggregate. Restricting the batch to
        # importance<=0 candidates preserves the iter11 speedup on truly redundant unions
        # (cluster-reflection duplicates score near-zero or negative importance, since shuffling
        # one duplicate barely moves the model that has the OTHER duplicates intact) while
        # leaving the legacy single-drop greedy stage-2b to polish the marginal-importance
        # members one-by-one with a tightening rolling base -- the proven informative-preserving
        # path. Measured: this restores 8/8 informative recovery at width=5000 on the regime
        # synthetic while keeping the refine wall-time under iter10's by ~6x.
        # Threshold importance against ``parsimony_tol * |base| / sqrt(n)`` -- a per-member-share
        # of the parsimony budget. Multi-drop interactions can make k columns of importance<=tol
        # collectively exceed tol; dividing by sqrt(n) under-allocates the budget so the batched
        # verify retains headroom. Empirically calibrated to restore 7-8/8 informative recovery
        # at width=5000 on the regime synthetic while still firing on the most-redundant 30-60%
        # of members for the iter11 speedup.
        per_member_tol = parsimony_tol * abs(base) / max(1.0, np.sqrt(n))
        n_safe = int(np.sum(sorted_imps <= per_member_tol))
        # Half-of-current cap as defence in depth: even on a pathological set where every member
        # scores importance<=0 (perfectly redundant pairs), never drop more than half in one
        # batched retrain; stage-2b handles the rest.
        initial_batch = min(n_safe, max(1, n // 2), n - 1)
        if initial_batch >= 1:
            batch_size = initial_batch
            while batch_size >= 1:
                drop_pos = order[:batch_size]
                drop_set = {int(current[p]) for p in drop_pos}
                survivors = [c for c in current if c not in drop_set]
                if not survivors:
                    batch_size = batch_size // 2
                    continue
                new_loss = _honest_loss(
                    model_template, X_search, y_search, X_holdout, y_holdout, survivors, classification,
                    metric, cache=cache, n_estimators_cap=cap, template_id=tid, disk_cache=disk_cache)
                if new_loss <= cur_threshold:
                    current = survivors
                    base = min(base, float(new_loss))
                    break
                new_batch = batch_size // 2
                if new_batch == batch_size:
                    break
                batch_size = new_batch
        # When no member scored importance<=0, the batch-drop pass is a no-op and we proceed
        # directly to stage-2b's single-drop greedy -- equivalent to legacy behaviour on a
        # genuinely-essential working set.

    # ---- Stage 2b: legacy single-drop greedy backward on the now-compacted working set. After the
    # iter11 batch-drop, ``current`` is typically a handful of columns; the legacy O(k^2) fit cost
    # is now negligible, and the per-round single-drop greedy is the gold standard for
    # informative-preserving fine refinement (each accepted drop tightens the rolling base, so the
    # algorithm naturally stops at the legacy operating point). This is the iter11 fallback the
    # task brief calls for explicitly: when batch-drop's first pass declined to compact further,
    # single-drop greedy takes over for the final polish.
    #
    # iter35 UCB-batched dispatch: when ``ucb_enabled`` AND ``n_jobs`` enables threading AND we have a
    # stage-2a importance prior for the current members, each round sorts trials by ascending
    # importance (lowest = safest drop = lowest expected honest loss) and dispatches in
    # workers-sized batches. After each batch, the round leader is compared against every
    # un-evaluated trial's UCB lower bound (importance + auto-slack). When no remaining trial can
    # beat the leader -> stop, accept the leader (if within tol) or break the round (if not).
    # Falls through to legacy single-batch-per-round when UCB is off OR n_jobs in (1, 0, None) OR
    # no importance prior available (stage 2a skipped).
    import os as _os_iter35

    n_cores = _os_iter35.cpu_count() or 1
    if n_jobs in (-1, None, 0):
        outer_workers = n_cores
    else:
        outer_workers = max(1, int(n_jobs))
    if ucb_min_eval_size is None:
        ucb_min_eval_size_eff = max(outer_workers, 3)
    else:
        ucb_min_eval_size_eff = max(1, int(ucb_min_eval_size))
    use_ucb_stage2b = bool(ucb_enabled) and n_jobs not in (1, 0, None) and len(importance_by_col) > 0

    rounds = len(current) if max_drop_rounds is None else max_drop_rounds
    for _ in range(rounds):
        if len(current) <= 1:
            break
        cur_threshold = base + parsimony_tol * abs(base)

        # Build (col, importance_prior) pairs. Members not in ``importance_by_col`` (e.g. stage-2a was
        # skipped on a degenerate path) fall back to importance = +inf so they sort last; the legacy
        # path also runs them but the UCB path keeps them as last-resort dispatch.
        col_importance = [(int(c), importance_by_col.get(int(c), float("inf"))) for c in current]
        if use_ucb_stage2b and len(col_importance) > ucb_min_eval_size_eff:
            # UCB-batched: sort trials by ascending importance, dispatch in workers-sized batches,
            # short-circuit when no remaining trial can beat the round leader.
            sorted_pairs = sorted(enumerate(col_importance), key=lambda kv: (kv[1][1], kv[1][0]))
            order_local = [kv[0] for kv in sorted_pairs]  # original-index ordering within ``current``
            # First batch saturates the workers; subsequent batches are workers-sized.
            evaluated_losses: dict[int, float] = {}  # local-idx -> honest loss
            best_loss_round = float("inf")
            best_local_idx = -1
            pos = 0
            n_trials = len(order_local)
            # ``slack`` calibrates importance -> honest_loss residual on a per-round basis. With <2
            # evaluated points fall back to slack=mean(delta) (no std term) so the gate still has a
            # working lower bound; the auto-slack helper handles that fallback.
            slack_used = 0.0
            while pos < n_trials:
                if pos == 0:
                    step = min(ucb_min_eval_size_eff, n_trials - pos)
                else:
                    step = min(max(1, outer_workers), n_trials - pos)
                batch_local = order_local[pos : pos + step]
                pos += step
                tasks = []
                for li in batch_local:
                    drop_col = current[li]
                    survivors = [c for c in current if c != drop_col]
                    tasks.append((survivors, None))
                losses_batch = _parallel_honest_losses(
                    tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                    classification, metric, n_jobs, cache=cache, n_estimators_cap=cap, template_id=tid,
                    inner_n_jobs_cap=inner_n_jobs_cap, disk_cache=disk_cache)
                for li, ls in zip(batch_local, losses_batch):
                    evaluated_losses[li] = float(ls)
                    if ls < best_loss_round:
                        best_loss_round = float(ls)
                        best_local_idx = li
                if pos >= n_trials:
                    break
                # Calibrate slack from evaluated pairs (importance_prior, honest_loss).
                ev_importance = [col_importance[li][1] for li in evaluated_losses]
                ev_honest = [evaluated_losses[li] for li in evaluated_losses]
                if ucb_slack is None:
                    slack_used = _ucb_auto_slack(ev_importance, ev_honest, ucb_stdev_multiplier)
                else:
                    slack_used = float(ucb_slack)
                remaining_local = order_local[pos:n_trials]
                remaining_importance = [col_importance[li][1] for li in remaining_local]
                # Stop when no remaining trial can have a lower honest loss than the round leader.
                # Use parsimony_tol=0 here: we want strict "remaining cannot beat leader" semantics
                # because we only need to find the round's minimum, not enter a parsimony band.
                if _ucb_stop_remaining_cannot_win(
                    best_loss_round, remaining_importance, slack_used, parsimony_tol=0.0,
                ):
                    break
            # Accept the leader if within tol; otherwise round terminates.
            if best_local_idx < 0 or best_loss_round > cur_threshold:
                break
            drop_col = current[best_local_idx]
            current = [c for c in current if c != drop_col]
            base = min(base, float(best_loss_round))
            # The dropped column's importance entry is no longer needed; left in place because the
            # dict is keyed by column id (the dropped col simply never re-appears in subsequent
            # ``current`` lookups). Avoids mutating the dict in the inner loop.
        else:
            # Legacy single-batch path: ALL k trials in one parallel dispatch per round. Preserved
            # bit-identical for UCB-off / n_jobs in {1,0,None} / no-prior fallback paths.
            trials = [[c for c in current if c != drop] for drop in current]
            losses = _parallel_honest_losses([(t, None) for t in trials], model_template, X_search, y_search,
                                             X_holdout, y_holdout, classification, metric, n_jobs, cache=cache,
                                             n_estimators_cap=cap, template_id=tid,
                                             inner_n_jobs_cap=inner_n_jobs_cap, disk_cache=disk_cache)
            losses_arr = np.asarray(losses, dtype=np.float64)
            best_i = int(np.argmin(losses_arr))
            if losses_arr[best_i] > cur_threshold:
                break
            current = trials[best_i]
            base = min(base, float(losses_arr[best_i]))
    return current


def importance_topk_ablation(
    phi, proxy_best_idx, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, unit_to_members=None, cache=None, disk_cache_dir=None,
):
    """Compare the proxy-chosen subset against a SHAP-importance-top-k subset of the same size.

    Returns a dict with both honest losses and whether the proxy strictly wins (the method's
    unique-value gate vs plain SHAP global importance). In clustering mode, importance ranks UNITS
    and both subsets expand to member columns for the honest comparison.

    ``disk_cache_dir`` (iter81): when set, the two honest retrains (proxy subset + SHAP-importance-
    top-k subset) check the cross-process :class:`DiskCache` first. The proxy subset is typically a
    cache hit -- it's the chosen winner that revalidation just retrained -- so the warm-cache cost
    of this stage drops to one fit for the imp_cols comparison (and to zero when both subsets
    overlap a prior fit). ``None`` (default) keeps the legacy in-memory-only contract bit-identical.
    """
    metric = resolve_metric(classification, metric)
    k = len(proxy_best_idx)  # match unit count, then expand both sides to members
    importance = np.abs(phi).mean(axis=0)
    imp_units = tuple(sorted(np.argsort(-importance)[:k].tolist()))
    proxy_cols = _expand(proxy_best_idx, unit_to_members)
    imp_cols = _expand(imp_units, unit_to_members)
    disk_cache = _open_disk_cache(disk_cache_dir)
    proxy_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                                proxy_cols, classification, metric, cache=cache, disk_cache=disk_cache)
    imp_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                              imp_cols, classification, metric, cache=cache, disk_cache=disk_cache)
    return dict(proxy_features=tuple(proxy_best_idx), proxy_honest_loss=proxy_honest,
                importance_features=imp_units, importance_honest_loss=imp_honest,
                proxy_wins=bool(proxy_honest < imp_honest), proxy_at_least_ties=bool(proxy_honest <= imp_honest))
