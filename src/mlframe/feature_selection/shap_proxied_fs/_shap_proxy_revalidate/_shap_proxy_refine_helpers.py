"""Helpers carved out of ``_shap_proxy_refine`` to keep that module under its size budget."""

from __future__ import annotations

from typing import Any

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_revalidate._shap_proxy_loss import (
    _expand, _honest_loss, _parallel_honest_losses,
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
    cost without changing the winner - safe to stop dispatching new batches.

    Stable across reruns: deterministic comparison of floats only.
    """
    if len(remaining_proxy_losses) == 0:
        return True
    lower_bounds = np.asarray(remaining_proxy_losses, dtype=np.float64) + float(ucb_slack)
    threshold = float(best_stable_score) + float(parsimony_tol) * abs(float(best_stable_score))
    return bool(np.min(lower_bounds) > threshold)


def _ucb_auto_slack(evaluated_proxy, evaluated_honest_mean, stdev_multiplier=1.5):
    """Calibrate the UCB slack from already-evaluated (proxy, honest_mean) pairs.

    ``slack`` shifts proxy onto the honest scale; the lower bound for an un-evaluated candidate's
    honest loss is ``proxy + slack``. To be a *lower* bound we take ``mean(delta) - k * std(delta)``
    where ``delta_i = honest_i - proxy_i``: most of the calibration mass on the high side keeps it
    pessimistic (smaller honest predictions => larger remaining lower bounds rarely => never wrong
    stops). With <2 evaluated points the std is undefined; we fall back to ``mean(delta)`` only,
    which still preserves the proxy ordering but with zero safety margin - the calling stop check
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


def _plan_ucb_batch_sizes(cur, n_total, ucb_min_eval_size_eff, outer_workers, seeds_per_cand, batch_sizes):
    """Plan the UCB evaluation batch sizes covering all candidates."""
    while cur < n_total:
        if cur == 0:
            step = min(ucb_min_eval_size_eff, n_total - cur)
        else:
            step = min(max(1, outer_workers // max(1, seeds_per_cand)), n_total - cur)
            step = max(step, 1)
        batch_sizes.append(step)
        cur += step


def _cap_best_unit_members(best_idx, cap, unit_to_members, model_template, X_search, y_search, X_holdout, y_holdout, classification, metric, cache, disk_cache, ranked, lambda_stab, per_candidate, candidates):
    """Cap the members of the best unit when a cap is set."""
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
                # single fit so its std is not refreshed - the capped-template std remains as a
                # cross-seed-stability proxy. Update stable_score to reflect the new mean.
                d["stable_score"] = float(winner_full_loss) + lambda_stab * d["honest_std"]
                d["honest_loss_capped"] = float(np.asarray(per_candidate[next(i for i, (_, ix) in enumerate(candidates) if tuple(ix) == best_idx)]).mean())
                break


def _collect_candidate_scores(candidates, per_candidate, ranked, member_cols, lambda_stab):
    """Collect the scored candidates that reached per-candidate evaluation."""
    for ci, (proxy_loss_val, idx) in enumerate(candidates):
        if ci not in per_candidate:
            continue
        scores = np.asarray(per_candidate[ci], dtype=np.float64)
        mean, std = float(scores.mean()), float(scores.std())
        # Parsimony cardinality = deployed feature count (expanded members), not unit count.
        ranked.append(dict(features=tuple(idx), n_members=len(member_cols[ci]),
                           proxy_loss=float(proxy_loss_val),
                           honest_loss=mean, honest_std=std, stable_score=mean + lambda_stab * std))


def _within_cluster_ref_runs_after_stage_may(member_groups, current, min_multi_clusters, model_template, X_search, y_search, X_holdout, y_holdout, classification, metric, cache, cap, tid, disk_cache, parsimony_tol, protected, n_jobs, inner_n_jobs_cap, ucb_min_eval_size, ucb_enabled, max_drop_rounds, ucb_slack, ucb_stdev_multiplier):
    """Block of within_cluster_refine starting at ``n_multi_eligible = 0``."""
    current = _within_cluster_ref_multi_eligible(member_groups, current, min_multi_clusters, model_template, X_search, y_search, X_holdout, y_holdout, classification, metric, cache, cap, tid, disk_cache, parsimony_tol, protected, n_jobs, inner_n_jobs_cap, ucb_min_eval_size, ucb_enabled, max_drop_rounds, ucb_slack, ucb_stdev_multiplier)
    return current


def _within_cluster_ref_multi_eligible(member_groups, current, min_multi_clusters, model_template, X_search, y_search, X_holdout, y_holdout, classification, metric, cache, cap, tid, disk_cache, parsimony_tol, protected, n_jobs, inner_n_jobs_cap, ucb_min_eval_size, ucb_enabled, max_drop_rounds, ucb_slack, ucb_stdev_multiplier):
    """Block of _within_cluster_ref_runs_after_stage_may starting at ``n_multi_eligible = 0``."""
    threshold: Any = None
    base: Any = None
    n_multi_eligible = 0
    n_multi_eligible = _within_cluster_ref_runs_after_stage_may_2(member_groups, current, n_multi_eligible)
    stage1_will_fire = member_groups is not None and n_multi_eligible >= min_multi_clusters
    if stage1_will_fire:
        base = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout, current,
                            classification, metric, cache=cache, n_estimators_cap=cap,
                            template_id=tid, disk_cache=disk_cache)
        threshold = base + parsimony_tol * abs(base)
    else:
        # Sentinel: Stage 2a will compute ``rank_base`` and seed ``base`` from it.
        base = None
        threshold = None

    # ---- Stage 1: per-cluster collapse (one parallel probe per multi-cluster).
    # Skip when member_groups is missing OR has too few multi-member groups to pay the stage-1 toll
    # (k probes + 1 cumulative verify); on low-redundancy data the cluster-collapse never fires and
    # we just want to fall through to stage 2's legacy single-drop greedy.
    if stage1_will_fire:
        current_set = set(current)
        # Normalize: filter member_groups to columns actually in `current`, drop empties / singletons.
        multi: list[list[int]] = []
        _within_cluster_ref_normalize_filter_member_groups(member_groups, current_set, multi)
        if multi:
            # One probe per multi-cluster: drop ALL members except the first (canonical representative).
            # Other clusters keep FULL membership; the probe asks "can we safely deduplicate THIS one?".
            probes: list[tuple[list[int], int, list[int]]] = []  # (subset, cluster_idx, dropped_members)
            for ci, g in enumerate(multi):
                # g[0] is the surviving representative (the cluster aggregator's first member); g[1:]
                # are the redundant members the probe asks to drop while other clusters stay intact.
                # Protected members are never proposed for removal, even as a cluster-collapse dedupe.
                drop_set = set(g[1:]) - protected
                probe_cols = sorted(c for c in current if c not in drop_set)
                probes.append((probe_cols, ci, sorted(drop_set)))
            losses = _parallel_honest_losses(
                [(p[0], None) for p in probes], model_template, X_search, y_search, X_holdout, y_holdout,
                classification, metric, n_jobs, cache=cache, n_estimators_cap=cap, template_id=tid,
                inner_n_jobs_cap=inner_n_jobs_cap, disk_cache=disk_cache)
            # Each probe is evaluated against the ORIGINAL base/threshold (cluster collapses are
            # measured independently, not against each other). Accepted probes' drops accumulate.
            accepted_drops: set[int] = set()
            _within_cluster_ref_measured_independently_against_each(probes, losses, threshold, accepted_drops)
            if accepted_drops:
                collapsed = sorted(c for c in current if c not in accepted_drops)
                # Verify the union of all accepted cluster-collapses still respects tol (sum-of-parts
                # need not equal whole: pathological mutual dependence between clusters could fail
                # the cumulative drop even if each was independently fine).
                if len(collapsed) < len(current):
                    cum_loss = _honest_loss(
                        model_template, X_search, y_search, X_holdout, y_holdout, collapsed, classification,
                        metric, cache=cache, n_estimators_cap=cap, template_id=tid, disk_cache=disk_cache)
                    base, current = _within_cluster_ref_cum_loss_threshold(cum_loss, threshold, collapsed, base, parsimony_tol, multi, probes, losses, current)

    # ``importance_by_col`` (iter35): persist stage-2a's permutation importances so stage-2b can sort
    # its per-round drop trials in ascending importance order and dispatch UCB-batched. Defaults to
    # empty -> stage-2b falls back to legacy unsorted single-batch dispatch.
    importance_by_col: dict[int, float] = {}
    # ---- Stage 2a: ONE permutation-importance + batch-drop pass on the (possibly stage-1-collapsed)
    # working set. This is the iter11 perf win: a single ranking pass (1 fit + k cheap predicts)
    # ranks every member by drop-safety, then we accept the largest batched drop that respects
    # parsimony_tol - collapsing what would have been many legacy single-drop greedy rounds into
    # ONE verify retrain (with halving fallbacks on rejection). The pass is run AT MOST ONCE per
    # refine call: after the initial bulk-compaction, the working set is small (typically a handful
    # of columns) and the subsequent single-drop greedy stage-2b can polish it in legacy O(k)
    # retrains - the runtime cost of which is now negligible because k is small. Running multiple
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
        # Sort members ascending by importance (lowest = safest to drop first). Protected members are
        # pinned to +inf for THIS batch-drop selection only (never sorted into the "safe to drop"
        # prefix) - ``importance_by_col`` above keeps the real value for reporting/stage-2b priors.
        drop_importances = importances.copy()
        _within_cluster_ref_prefix_importance_col_above(protected, current, drop_importances)
        order = np.argsort(drop_importances, kind="stable")
        sorted_imps = drop_importances[order]
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
        # members one-by-one with a tightening rolling base - the proven informative-preserving
        # path. Measured: this restores 8/8 informative recovery at width=5000 on the regime
        # synthetic while keeping the refine wall-time under iter10's by ~6x.
        # Threshold importance against ``parsimony_tol * |base| / sqrt(n)`` - a per-member-share
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
        # directly to stage-2b's single-drop greedy - equivalent to legacy behaviour on a
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

        # Build (col, importance_prior) pairs, EXCLUDING protected members - they are never proposed
        # for removal, so no trial drops them. Members not in ``importance_by_col`` (e.g. stage-2a was
        # skipped on a degenerate path) fall back to importance = +inf so they sort last; the legacy
        # path also runs them but the UCB path keeps them as last-resort dispatch.
        droppable_current = [c for c in current if int(c) not in protected] if protected else current
        if not droppable_current:
            break
        col_importance = [(int(c), importance_by_col.get(int(c), float("inf"))) for c in droppable_current]
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
                    drop_col = droppable_current[li]
                    survivors = [c for c in current if c != drop_col]
                    tasks.append((survivors, None))
                losses_batch = _parallel_honest_losses(
                    tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                    classification, metric, n_jobs, cache=cache, n_estimators_cap=cap, template_id=tid,
                    inner_n_jobs_cap=inner_n_jobs_cap, disk_cache=disk_cache)
                best_local_idx, best_loss_round = _within_cluster_ref_li_ls_zip_batch(batch_local, losses_batch, evaluated_losses, best_loss_round, best_local_idx)
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
            drop_col = droppable_current[best_local_idx]
            current = [c for c in current if c != drop_col]
            base = min(base, float(best_loss_round))
            # The dropped column's importance entry is no longer needed; left in place because the
            # dict is keyed by column id (the dropped col simply never re-appears in subsequent
            # ``current`` lookups). Avoids mutating the dict in the inner loop.
        else:
            # Legacy single-batch path: ALL droppable trials in one parallel dispatch per round.
            # Bit-identical to pre-``protected_cols`` behaviour when ``protected`` is empty (the
            # default) since ``droppable_current == current`` in that case.
            trials = [[c for c in current if c != drop] for drop in droppable_current]
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


def _within_cluster_ref_runs_after_stage_may_2(member_groups, current, n_multi_eligible):
    """Block of within_cluster_refine starting at ``if member_groups is not None:``."""
    if member_groups is not None:
        current_set_pre = set(current)
        for g in member_groups:
            if sum(1 for c in g if int(c) in current_set_pre) > 1:
                n_multi_eligible += 1
    return n_multi_eligible


def _within_cluster_ref_li_ls_zip_batch(batch_local, losses_batch, evaluated_losses, best_loss_round, best_local_idx):
    """Block of within_cluster_refine starting at ``for li, ls in zip(batch_local, losses_batch):``."""
    for li, ls in zip(batch_local, losses_batch):
        evaluated_losses[li] = float(ls)
        if ls < best_loss_round:
            best_loss_round = float(ls)
            best_local_idx = li
    return best_local_idx, best_loss_round


def _within_cluster_ref_normalize_filter_member_groups(member_groups, current_set, multi):
    """Block of within_cluster_refine starting at ``for g in member_groups:``."""
    for g in member_groups:
        sub = [int(c) for c in g if int(c) in current_set]
        if len(sub) > 1:
            multi.append(sub)


def _within_cluster_ref_measured_independently_against_each(probes, losses, threshold, accepted_drops):
    """Block of within_cluster_refine starting at ``for (_probe_cols, _ci, drops), ls in zip(probes, losses):``."""
    for (_probe_cols, _ci, drops), ls in zip(probes, losses):
        if ls <= threshold:
            accepted_drops.update(drops)


def _within_cluster_ref_cum_loss_threshold(cum_loss, threshold, collapsed, base, parsimony_tol, multi, probes, losses, current):
    """Block of within_cluster_refine starting at ``if cum_loss <= threshold:``."""
    if cum_loss <= threshold:
        current = collapsed
        base = min(base, cum_loss)
        threshold = base + parsimony_tol * abs(base)
    elif len(multi) == 1:
        # Only one cluster was collapsed; the cumulative IS the single probe - if
        # one passed and the other failed, that's just float noise (cache should make
        # them byte-identical, but defend in depth). Accept the probe result anyway.
        current = collapsed
        base = min(base, cum_loss)
        threshold = base + parsimony_tol * abs(base)
    else:
        # Cumulative drop hurts beyond tol: accept only the single best-loss cluster
        # collapse (the safest individual drop set), defer the rest to stage 2.
        best_ci, best_loss = -1, float("inf")
        for (_probe_cols, ci, _drops), ls in zip(probes, losses):
            if ls <= threshold and ls < best_loss:
                best_ci, best_loss = ci, float(ls)
        if best_ci >= 0:
            single_drops = set(probes[best_ci][2])
            current = sorted(c for c in current if c not in single_drops)
            base = min(base, best_loss)
            threshold = base + parsimony_tol * abs(base)
    return base, current


def _within_cluster_ref_prefix_importance_col_above(protected, current, drop_importances):
    """Block of within_cluster_refine starting at ``if protected:``."""
    if protected:
        for i, c in enumerate(current):
            if int(c) in protected:
                drop_importances[i] = float("inf")
