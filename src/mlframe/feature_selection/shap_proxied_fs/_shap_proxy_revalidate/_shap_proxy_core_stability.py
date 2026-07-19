"""Least-core / nucleolus stability refine (gt_02): a principled alternative to greedy parsimony_tol drop.

``within_cluster_refine``'s legacy stage-2b greedy-backward drops any member whose removal keeps the
honest holdout loss within ``parsimony_tol`` of the best seen -- a scalar-threshold decision that
cannot distinguish "redundant given the rest" from "individually small but jointly meaningful". This
module treats the selected proxy UNITS as players of a cooperative game with characteristic function
v(C) = max(0, L_ref - proxy_loss(C)) (higher v = better coalition, monotone transform of the proxy
loss so the game is amenable to a linear allocation) and computes a SAMPLED least-core allocation via
``scipy.optimize.linprog(method="highs")`` (scipy is a core dependency, no new dep). A unit whose
leave-one-out coalition can "block" (v(N \\ {j}) approx= v(N)) gets x_j approx= 0 and is safely
droppable; a weak-but-real unit keeps x_j > 0 because some coalition genuinely needs it.

v(C) is deliberately computed from the PROXY game, never honest retrains: k + n_coalitions honest fits
is exactly the retraining cost the whole selector exists to avoid (see gt_02 sec 7). The ONE honest
verification of the final core-pruned proposal (in ``core_refine``) is what keeps the honesty contract.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def _sample_coalitions(k: int, n_coalitions: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """Return a list of player-index tuples: all singletons + all leave-one-outs + random subsets.

    Singletons and leave-one-outs are included EXACTLY (2k coalitions, deduplicated against each
    other and the grand coalition) because they dominate the binding constraint set for near-additive
    games (gt_02 sec 7); the remainder up to ``n_coalitions`` is filled with uniform-random nonempty
    proper subsets to approximate the rest of the lattice cheaply.
    """
    all_players = tuple(range(k))
    coalitions: set[tuple[int, ...]] = set()
    for j in range(k):
        coalitions.add((j,))
        loo = tuple(i for i in range(k) if i != j)
        if loo:
            coalitions.add(loo)
    max_possible = 2**k - 2  # exclude empty set and grand coalition
    target = min(n_coalitions, max_possible)
    guard = 0
    while len(coalitions) < target and guard < target * 20 + 100:
        guard += 1
        size = int(rng.integers(1, k)) if k > 1 else 1
        subset = tuple(sorted(rng.choice(k, size=size, replace=False).tolist()))
        if subset and subset != all_players:
            coalitions.add(subset)
    return sorted(coalitions, key=lambda c: (len(c), c))


def least_core_allocation(
    evaluator,
    players: tuple[int, ...],
    *,
    n_coalitions: int = 512,
    rng: np.random.Generator,
    nucleolus_refine: bool = False,
    exhaustive: bool = False,
) -> tuple[np.ndarray, float, dict]:
    """Compute a sampled least-core allocation over the proxy game restricted to ``players``.

    ``players`` are unit indices scored via ``evaluator.loss(idx)`` (a memoised proxy-loss oracle,
    e.g. ``_shap_proxy_heuristics._Evaluator``). The characteristic function is
    ``v(C) = max(0, L_ref - evaluator.loss(list(players[i] for i in C)))`` where ``L_ref`` is the
    loss of the WORST (highest-loss) singleton among ``players`` -- a finite, cheap-to-compute zero
    point (the natural "empty coalition" loss is +inf for most metrics and unusable directly). Core
    membership / the eps*-minimising allocation is invariant to this additive shift; only the absolute
    scale of ``x`` moves with it.

    Returns ``(x, eps_star, info)`` where ``x`` has shape ``(k,)`` aligned with ``players`` order,
    ``eps_star`` is the minimal uniform slack achieved by the sampled-coalition LP, and
    ``info = dict(n_constraints, binding_coalitions, lp_status, v_grand)``.

    LP: variables ``x_1..x_k, eps``; objective ``min eps``; constraints
    ``sum(x) == v(N)``, and for each sampled coalition C (excluding the grand coalition N itself):
    ``sum_{j in C} x_j >= v(C) - eps``; ``x_j >= 0``.

    ``exhaustive=True`` enumerates ALL ``2**k - 2`` proper nonempty subsets instead of sampling --
    used by the textbook-game unit tests to prove exact LP correctness against known analytic cores.

    ``nucleolus_refine=True`` runs a truncated (cap 3 iterations) lexicographic-minimisation second
    stage: having fixed ``eps*``, iteratively re-solve minimising the largest remaining excess among
    NOT-yet-tight constraints, freezing newly-tight ones each round. Full nucleolus computation is
    O(2^k) LPs in the worst case (successive constraint-tightening over the whole lattice); we
    deliberately truncate at 3 rounds as a cheap "sharpen toward the nucleolus" pass, not an exact one.
    """
    k = len(players)
    if k == 0:
        return np.zeros(0), 0.0, dict(n_constraints=0, binding_coalitions=0, lp_status="empty", v_grand=0.0)
    if k == 1:
        # Single player: L_ref == its own loss by construction, so v({1}) == v(N) == 0 and x is trivially 0.
        return np.zeros(1), 0.0, dict(n_constraints=0, binding_coalitions=0, lp_status="trivial_k1", v_grand=0.0)

    singleton_losses = [float(evaluator.loss([players[i]])) for i in range(k)]
    L_ref: float = max(singleton_losses)

    def v_of(idx_subset: tuple[int, ...]) -> float:
        """Characteristic function of the proxy game: nonnegative improvement over the worst singleton."""
        cols = [players[i] for i in idx_subset]
        return max(0.0, L_ref - float(evaluator.loss(cols)))

    v_grand = v_of(tuple(range(k)))

    if exhaustive:
        from itertools import combinations

        coalitions: list[tuple[int, ...]] = []
        for size in range(1, k):
            coalitions.extend(combinations(range(k), size))
    else:
        coalitions = _sample_coalitions(k, n_coalitions, rng)

    v_vals = {c: v_of(c) for c in coalitions}

    # LP variables: [x_0..x_{k-1}, eps]. Objective: minimise eps.
    n_vars = k + 1
    c_obj = np.zeros(n_vars)
    c_obj[-1] = 1.0

    # sum_{j in C} x_j + eps >= v(C)  <=>  -sum_{j in C} x_j - eps <= -v(C)
    A_ub = np.zeros((len(coalitions), n_vars))
    b_ub = np.zeros(len(coalitions))
    for row, c in enumerate(coalitions):
        for j in c:
            A_ub[row, j] = -1.0
        A_ub[row, -1] = -1.0
        b_ub[row] = -v_vals[c]

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :k] = 1.0
    b_eq = np.array([v_grand])

    bounds = [(0.0, None)] * k + [(None, None)]  # x_j >= 0; eps unrestricted (can be negative for a nonempty core)

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        # Infeasible / numerically-degenerate LP: fall back to an equal-split allocation with eps=+inf
        # sentinel so callers can detect the failure via info["lp_status"] and treat it as "no signal".
        x = np.full(k, v_grand / k)
        info = dict(n_constraints=len(coalitions), binding_coalitions=0, lp_status=f"infeasible:{res.status}", v_grand=v_grand)
        return x, float("inf"), info

    x = res.x[:k].copy()
    eps_star = float(res.x[-1])
    slack = A_ub @ res.x - b_ub
    binding = int(np.sum(np.abs(slack) < 1e-7))

    if nucleolus_refine:
        x, eps_star = _truncated_nucleolus(coalitions, v_vals, v_grand, k, x, eps_star, max_iters=3)

    info = dict(n_constraints=len(coalitions), binding_coalitions=binding, lp_status="optimal", v_grand=v_grand)
    return x, eps_star, info


def _truncated_nucleolus(
    coalitions: list[tuple[int, ...]], v_vals: dict, v_grand: float, k: int,
    x0: np.ndarray, eps0: float, *, max_iters: int,
) -> tuple[np.ndarray, float]:
    """Lexicographically sharpen the least-core allocation toward the nucleolus, capped at ``max_iters`` rounds.

    Standard iterative scheme: freeze constraints whose excess is already at the current bound (tight),
    then re-solve minimising the largest excess among the remaining (non-frozen) constraints. Each round
    strictly reduces (or holds) the max remaining excess; capped at ``max_iters`` because the exact
    nucleolus needs up to k-1 such rounds with a full re-derivation of the active constraint lattice,
    which is O(2^k) worst case -- we deliberately truncate as a cheap "sharpen" pass, not an exact solve.
    """
    x, eps = x0, eps0
    frozen = np.zeros(len(coalitions), dtype=bool)
    for _ in range(max_iters):
        excess = np.array([v_vals[c] - sum(x[j] for j in c) for c in coalitions])
        tight = np.abs(excess - excess.max()) < 1e-7
        newly_frozen = tight & ~frozen
        if not newly_frozen.any():
            break
        frozen |= tight
        active = [c for c, f in zip(coalitions, frozen) if not f]
        if not active:
            break
        n_vars = k + 1
        c_obj = np.zeros(n_vars)
        c_obj[-1] = 1.0
        A_ub = np.zeros((len(active), n_vars))
        b_ub = np.zeros(len(active))
        for row, c in enumerate(active):
            for j in c:
                A_ub[row, j] = -1.0
            A_ub[row, -1] = -1.0
            b_ub[row] = -v_vals[c]
        # Frozen constraints stay pinned at their already-achieved excess via equality rows appended below.
        frozen_c = [c for c, f in zip(coalitions, frozen) if f]
        if frozen_c:
            A_eq_extra = np.zeros((len(frozen_c) + 1, n_vars))
            b_eq_extra = np.zeros(len(frozen_c) + 1)
            for row, c in enumerate(frozen_c):
                for j in c:
                    A_eq_extra[row, j] = 1.0
                A_eq_extra[row, -1] = 1.0
                b_eq_extra[row] = v_vals[c] - float(excess.max())
            A_eq_extra[-1, :k] = 1.0
            b_eq_extra[-1] = v_grand
        else:
            A_eq_extra = np.zeros((1, n_vars))
            A_eq_extra[0, :k] = 1.0
            b_eq_extra = np.array([v_grand])
        bounds = [(0.0, None)] * k + [(None, None)]
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq_extra, b_eq=b_eq_extra, bounds=bounds, method="highs")
        if not res.success:
            break
        x, eps = res.x[:k].copy(), float(res.x[-1])
    return x, eps


def core_refine(
    members: list[int],
    unit_players: tuple[int, ...],
    evaluator,
    honest_loss_fn,
    *,
    drop_threshold: float = 0.02,
    n_coalitions: int = 512,
    rng: np.random.Generator,
    nucleolus_refine: bool = False,
    unit_to_members: dict,
    legacy_refine_fn,
    legacy_refine_kwargs: dict,
) -> tuple[list[int], dict]:
    """Least-core allocation over ``unit_players``, drop units below ``drop_threshold`` of total credit.

    Drops proxy units whose share ``x_j / sum(x)`` is below ``drop_threshold`` (a FRACTION of total
    allocation, not raw loss units), expands survivors to their member columns via ``unit_to_members``,
    and verifies the proposal ONCE against the honest holdout via ``honest_loss_fn`` (a
    ``(cols) -> bool`` closure that the caller wires to its own tolerance/comparison logic). Accepted
    iff ``honest_loss_fn`` returns True; on rejection, falls back to the legacy greedy
    ``legacy_refine_fn(**legacy_refine_kwargs)`` so ``refine_mode="core"`` never returns a
    silently-worse-than-greedy outcome. ``info["fallback"]`` records whether the fallback fired.

    Returns ``(refined_member_cols, info)`` with
    ``info = dict(allocation, eps_star, dropped_by_core, fallback, lp_status, binding_coalitions)``.
    """
    if len(unit_players) <= 1:
        # A single unit has nothing to allocate away from; core refine is a no-op, mirroring the
        # legacy greedy path's own single-member early return.
        return members, dict(allocation={}, eps_star=0.0, dropped_by_core=[], fallback=False, lp_status="trivial", binding_coalitions=0)

    x, eps_star, lp_info = least_core_allocation(evaluator, unit_players, n_coalitions=n_coalitions, rng=rng, nucleolus_refine=nucleolus_refine)
    total = float(x.sum())
    shares = x / total if total > 0 else np.zeros_like(x)
    keep_mask = shares >= drop_threshold
    if not keep_mask.any():
        # Degenerate: every unit fell below threshold. Keep the single highest-share unit so the
        # proposal is never empty (an empty subset can't be honestly evaluated / isn't a valid model).
        keep_mask[int(np.argmax(shares))] = True
    kept_units = [unit_players[i] for i in range(len(unit_players)) if keep_mask[i]]
    dropped_units = [unit_players[i] for i in range(len(unit_players)) if not keep_mask[i]]
    proposed_cols = sorted({int(c) for u in kept_units for c in unit_to_members[int(u)]})

    allocation = {int(unit_players[i]): float(x[i]) for i in range(len(unit_players))}

    honest_ok = honest_loss_fn(proposed_cols) if proposed_cols else False
    if honest_ok:
        info = dict(allocation=allocation, eps_star=eps_star, dropped_by_core=dropped_units, fallback=False,
                    lp_status=lp_info["lp_status"], binding_coalitions=lp_info["binding_coalitions"])
        return proposed_cols, info

    legacy_result = legacy_refine_fn(**legacy_refine_kwargs)
    info = dict(allocation=allocation, eps_star=eps_star, dropped_by_core=dropped_units, fallback=True,
                lp_status=lp_info["lp_status"], binding_coalitions=lp_info["binding_coalitions"])
    return legacy_result, info
