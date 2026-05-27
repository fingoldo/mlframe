"""Heuristic subset search for when exact enumeration is infeasible (n_features > ~22).

All strategies minimise the same proxy loss as the brute-force kernel (``subset_loss``), so their
output is directly comparable / mergeable with it. Each returns a deduplicated, loss-sorted list of
``(loss, feature_idx_tuple)`` candidates (the top-N), which the facade then honestly re-validates.

Strategies:
  - ``beam_search``     : deterministic forward beam (the default heuristic; the trusted pick).
  - ``greedy_forward`` / ``greedy_backward`` : classic add / drop hill-climb.
  - ``multistart_local`` : random restarts + add/drop/swap local search.
  - ``genetic``         : GA over binary masks (tournament + uniform crossover + bit-flip + elitism).
  - ``simulated_annealing`` : Metropolis bit-flips with geometric cooling.

The proxy scan is the cheap stage, so these stay in vectorised numpy (the Python objective path,
which also supports AUC). A per-subset memo avoids re-scoring repeats.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss, resolve_metric


class _Evaluator:
    """Memoised proxy-loss evaluator over feature-index subsets (keyed by sorted tuple)."""

    def __init__(self, phi, base, y, metric):
        self.phi = phi
        self.base = base
        self.y = y
        self.metric = metric
        self.cache: dict[tuple[int, ...], float] = {}
        self.n_evals = 0

    def loss(self, idx) -> float:
        key = tuple(sorted(int(i) for i in idx))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if len(key) == 0:
            val = float("inf")  # empty subset is not a valid selection
        else:
            val = proxy_loss(coalition_margin(self.phi, self.base, list(key)), self.y, self.metric)
        self.cache[key] = val
        self.n_evals += 1
        return val

    def top_n(self, n: int) -> list[tuple[float, tuple[int, ...]]]:
        items = [(v, k) for k, v in self.cache.items() if k and np.isfinite(v)]
        items.sort(key=lambda t: t[0])
        return items[:n]


def _prep(phi, base, y, classification, metric):
    phi = np.ascontiguousarray(phi, dtype=np.float64)
    base = np.ascontiguousarray(base, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    return phi, base, y, resolve_metric(classification, metric)


def beam_search(phi, base, y, *, classification, metric=None, beam_width=8, min_card=1, max_card=None, top_n=30):
    phi, base, y, metric = _prep(phi, base, y, classification, metric)
    f = phi.shape[1]
    max_card = f if max_card is None else min(max_card, f)
    ev = _Evaluator(phi, base, y, metric)
    # Seed the beam with the best single features.
    singles = sorted(((ev.loss((j,)), (j,)) for j in range(f)), key=lambda t: t[0])
    beam = [c for _, c in singles[:beam_width]]
    for _ in range(min_card, max_card):
        expanded = {}
        for subset in beam:
            for j in range(f):
                if j in subset:
                    continue
                cand = tuple(sorted(subset + (j,)))
                if cand not in expanded:
                    expanded[cand] = ev.loss(cand)
        if not expanded:
            break
        beam = [c for c, _ in sorted(expanded.items(), key=lambda t: t[1])[:beam_width]]
    return ev.top_n(top_n)


def greedy_forward(phi, base, y, *, classification, metric=None, max_card=None, top_n=30):
    phi, base, y, metric = _prep(phi, base, y, classification, metric)
    f = phi.shape[1]
    max_card = f if max_card is None else min(max_card, f)
    ev = _Evaluator(phi, base, y, metric)
    current: tuple[int, ...] = ()
    best_loss = float("inf")
    remaining = set(range(f))
    while remaining and len(current) < max_card:
        cand_best, cand_loss = None, float("inf")
        for j in remaining:
            l = ev.loss(current + (j,))
            if l < cand_loss:
                cand_loss, cand_best = l, j
        if cand_best is None or cand_loss >= best_loss:
            break
        current = tuple(sorted(current + (cand_best,)))
        best_loss = cand_loss
        remaining.discard(cand_best)
    return ev.top_n(top_n)


def greedy_backward(phi, base, y, *, classification, metric=None, min_card=1, top_n=30):
    phi, base, y, metric = _prep(phi, base, y, classification, metric)
    f = phi.shape[1]
    ev = _Evaluator(phi, base, y, metric)
    current = tuple(range(f))
    best_loss = ev.loss(current)
    while len(current) > min_card:
        cand_best, cand_loss = None, float("inf")
        for j in current:
            trial = tuple(x for x in current if x != j)
            l = ev.loss(trial)
            if l < cand_loss:
                cand_loss, cand_best = l, j
        if cand_best is None or cand_loss >= best_loss:
            break
        current = tuple(x for x in current if x != cand_best)
        best_loss = cand_loss
    return ev.top_n(top_n)


def _local_search(ev, start: tuple[int, ...], f: int, max_card: int) -> tuple[int, ...]:
    """Hill-climb by add / drop / swap until no single move improves."""
    current = tuple(sorted(start))
    best = ev.loss(current)
    improved = True
    while improved:
        improved = False
        moves = []
        if len(current) < max_card:
            moves += [tuple(sorted(current + (j,))) for j in range(f) if j not in current]
        if len(current) > 1:
            moves += [tuple(x for x in current if x != j) for j in current]
        for out in current:  # swaps
            for inn in range(f):
                if inn not in current:
                    moves.append(tuple(sorted([x for x in current if x != out] + [inn])))
        for m in moves:
            l = ev.loss(m)
            if l < best:
                best, current, improved = l, m, True
                break
    return current


def multistart_local(phi, base, y, *, classification, metric=None, n_starts=10, max_card=None, rng=None, top_n=30):
    phi, base, y, metric = _prep(phi, base, y, classification, metric)
    f = phi.shape[1]
    max_card = f if max_card is None else min(max_card, f)
    rng = np.random.default_rng(0) if rng is None else rng
    ev = _Evaluator(phi, base, y, metric)
    for _ in range(n_starts):
        k = int(rng.integers(1, max_card + 1))
        start = tuple(sorted(rng.choice(f, size=k, replace=False).tolist()))
        _local_search(ev, start, f, max_card)
    return ev.top_n(top_n)


def genetic(phi, base, y, *, classification, metric=None, pop_size=40, n_generations=30,
            mutation_rate=0.1, elitism=4, rng=None, max_card=None, top_n=30):
    phi, base, y, metric = _prep(phi, base, y, classification, metric)
    f = phi.shape[1]
    max_card = f if max_card is None else min(max_card, f)
    rng = np.random.default_rng(0) if rng is None else rng
    ev = _Evaluator(phi, base, y, metric)

    def mask_to_idx(mask):
        idx = tuple(np.flatnonzero(mask).tolist())
        if len(idx) == 0:  # never allow empty individual
            idx = (int(rng.integers(0, f)),)
        if len(idx) > max_card:
            idx = tuple(sorted(rng.choice(idx, size=max_card, replace=False).tolist()))
        return idx

    pop = (rng.random((pop_size, f)) < 0.4).astype(np.int8)
    for _ in range(n_generations):
        idxs = [mask_to_idx(m) for m in pop]
        losses = np.array([ev.loss(ix) for ix in idxs])
        order = np.argsort(losses)
        new_pop = [pop[order[e]].copy() for e in range(min(elitism, pop_size))]
        while len(new_pop) < pop_size:
            # tournament selection (size 3), lower loss wins
            def pick():
                cand = rng.integers(0, pop_size, size=3)
                return pop[cand[np.argmin(losses[cand])]]
            p1, p2 = pick(), pick()
            cross = rng.random(f) < 0.5
            child = np.where(cross, p1, p2).astype(np.int8)
            flip = rng.random(f) < mutation_rate
            child[flip] = 1 - child[flip]
            new_pop.append(child)
        pop = np.array(new_pop)
    return ev.top_n(top_n)


def simulated_annealing(phi, base, y, *, classification, metric=None, n_iter=2000, t0=1.0,
                        cooling=0.995, rng=None, max_card=None, top_n=30):
    phi, base, y, metric = _prep(phi, base, y, classification, metric)
    f = phi.shape[1]
    max_card = f if max_card is None else min(max_card, f)
    rng = np.random.default_rng(0) if rng is None else rng
    ev = _Evaluator(phi, base, y, metric)
    mask = np.zeros(f, dtype=bool)
    mask[rng.integers(0, f)] = True
    cur_idx = tuple(np.flatnonzero(mask).tolist())
    cur_loss = ev.loss(cur_idx)
    t = t0
    for _ in range(n_iter):
        j = int(rng.integers(0, f))
        trial = mask.copy()
        trial[j] = not trial[j]
        if trial.sum() == 0 or trial.sum() > max_card:
            t *= cooling
            continue
        trial_idx = tuple(np.flatnonzero(trial).tolist())
        trial_loss = ev.loss(trial_idx)
        if trial_loss < cur_loss or rng.random() < np.exp(-(trial_loss - cur_loss) / max(t, 1e-9)):
            mask, cur_loss = trial, trial_loss
        t *= cooling
    return ev.top_n(top_n)
