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

from mlframe.feature_selection._shap_proxy_objective import (
    METRIC_CODES,
    coalition_margin,
    proxy_loss,
    resolve_metric,
    score_margin,
    score_margin_auto,
)


class _Evaluator:
    """Memoised proxy-loss evaluator over feature-index subsets (keyed by sorted tuple).

    Also caches the per-subset coalition *margin* vector so that neighbourhood moves
    (add / drop / flip) which differ from a known parent by one feature can update the
    margin via a single O(n) vector add instead of recomputing the full ``phi[:, idx].sum``
    reduction (the brute-force kernel uses the same incremental-sum trick).
    """

    def __init__(self, phi, base, y, metric):
        self.phi = phi
        self.base = base
        self.y = y
        self.metric = metric
        self.cache: dict[tuple[int, ...], float] = {}
        # Margin cache is opt-in (memory cost = O(|cached subsets| * n_samples)). Beam /
        # greedy callers populate via ``loss_from_margin`` so add-one-feature children are cheap.
        self.margin_cache: dict[tuple[int, ...], np.ndarray] = {}
        self.n_evals = 0
        # iter102 hot-path optimisation: cache the metric integer code + y (already float64
        # via ``_prep``) so the per-add/drop/swap step skips ``proxy_loss``'s METRIC_CODES dict
        # lookup, two ``np.asarray`` no-ops and the wrapper call. score_margin is the njit kernel
        # ``proxy_loss`` wraps; RMSE wraps the MSE code in ``math.sqrt`` (sqrt is monotone so any
        # ranker is invariant, but the absolute loss value matters for cache equality and the
        # corrector fit, so we preserve it). AUC has no njit kernel (per-subset sort), so we
        # leave that path on ``proxy_loss``. Profiled at width=10000 / n_rows=10000 beam_search:
        # 3021 loss_from_parent calls -> ``_loss_fast`` saves ~8us / call (~24ms total, ~1% wall
        # at this regime) by avoiding the wrapper round-trip per child evaluation.
        self._loss_fast = self._make_fast_loss(y, metric)

    def _make_fast_loss(self, y, metric):
        """Return a closure(margin) -> float that skips proxy_loss's per-call dispatch.

        Falls back to ``proxy_loss`` for AUC (needs a per-subset sort, not a pointwise loss).

        iter106: every margin this evaluator scores has the SAME row count (``y.shape[0]``), so the
        serial-vs-prange ``score_margin`` route is resolved ONCE here (via ``score_margin_auto``'s
        crossover) and baked into the closure -- zero per-call dispatch in the beam/greedy hot loop.
        At tall regimes (n_rows >= ~10000) the prange kernel is ~2x on the dominant brier/log-loss
        reductions (profiled width=500 / n_rows=50000: score_margin 1.5s of a 32s fit)."""
        if metric == "auc":
            metric_local = metric
            return lambda margin: proxy_loss(margin, y, metric_local)
        code = METRIC_CODES[metric]
        if metric == "rmse":
            from math import sqrt
            return lambda margin: sqrt(score_margin_auto(margin, y, 1))
        return lambda margin: float(score_margin_auto(margin, y, code))

    @staticmethod
    def _key(idx) -> tuple[int, ...]:
        # Skip the per-element ``int(i)`` cast on the fast path (Python int / numpy scalar tuples
        # iterate fine into ``sorted``); fall back only for exotic inputs.
        if isinstance(idx, tuple):
            return idx if _is_sorted_tuple(idx) else tuple(sorted(idx))
        return tuple(sorted(int(i) for i in idx))

    def loss(self, idx) -> float:
        key = self._key(idx)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if len(key) == 0:
            val = float("inf")  # empty subset is not a valid selection
        else:
            val = self._loss_fast(coalition_margin(self.phi, self.base, list(key)))
        self.cache[key] = val
        self.n_evals += 1
        return val

    def loss_with_margin(self, key: tuple[int, ...]) -> tuple[float, np.ndarray]:
        """Compute (or return cached) loss + margin for a pre-sorted key.

        Caller MUST pass an already sorted tuple of python ints. Used by beam / greedy
        seeding so subsequent ``loss_from_parent`` calls can update the margin incrementally.
        """
        cached_loss = self.cache.get(key)
        cached_margin = self.margin_cache.get(key)
        if cached_loss is not None and cached_margin is not None:
            return cached_loss, cached_margin
        if len(key) == 0:
            margin = self.base.copy()
            val = float("inf")
        else:
            margin = coalition_margin(self.phi, self.base, list(key))
            val = self._loss_fast(margin)
        self.cache[key] = val
        self.margin_cache[key] = margin
        if cached_loss is None:
            self.n_evals += 1
        return val, margin

    def loss_from_parent(self, parent_key: tuple[int, ...], parent_margin: np.ndarray,
                        new_j: int) -> tuple[float, tuple[int, ...], np.ndarray]:
        """Score ``parent_key + {new_j}`` reusing ``parent_margin``; insertion keeps key sorted.

        Returns (loss, child_key, child_margin). ``child_margin`` is cached so the next
        layer's expansion of this child also reuses it. ``new_j`` must NOT be in ``parent_key``.
        """
        # Insert new_j into the sorted parent_key (binary insert keeps tuple ordered).
        lo, hi = 0, len(parent_key)
        while lo < hi:
            mid = (lo + hi) >> 1
            if parent_key[mid] < new_j:
                lo = mid + 1
            else:
                hi = mid
        child_key = parent_key[:lo] + (new_j,) + parent_key[lo:]
        cached_loss = self.cache.get(child_key)
        cached_margin = self.margin_cache.get(child_key)
        if cached_loss is not None and cached_margin is not None:
            return cached_loss, child_key, cached_margin
        child_margin = parent_margin + self.phi[:, new_j]
        val = self._loss_fast(child_margin)
        self.cache[child_key] = val
        self.margin_cache[child_key] = child_margin
        if cached_loss is None:
            self.n_evals += 1
        return val, child_key, child_margin

    def loss_from_parent_drop(self, parent_key: tuple[int, ...], parent_margin: np.ndarray,
                              drop_j: int) -> tuple[float, tuple[int, ...], np.ndarray]:
        """Score ``parent_key - {drop_j}`` reusing ``parent_margin``; ``drop_j`` MUST be in parent.

        One O(n) vector subtract instead of recomputing the full ``phi[:, idx].sum`` reduction.
        Empty result is returned with loss=+inf (matches ``loss`` semantics for the empty subset).
        """
        # Locate drop_j (binary search; parent_key is sorted) and splice it out.
        lo, hi = 0, len(parent_key)
        while lo < hi:
            mid = (lo + hi) >> 1
            if parent_key[mid] < drop_j:
                lo = mid + 1
            else:
                hi = mid
        # parent_key[lo] == drop_j by precondition
        child_key = parent_key[:lo] + parent_key[lo + 1:]
        cached_loss = self.cache.get(child_key)
        cached_margin = self.margin_cache.get(child_key)
        if cached_loss is not None and cached_margin is not None:
            return cached_loss, child_key, cached_margin
        if len(child_key) == 0:
            child_margin = self.base.copy()
            val = float("inf")
        else:
            child_margin = parent_margin - self.phi[:, drop_j]
            val = self._loss_fast(child_margin)
        self.cache[child_key] = val
        self.margin_cache[child_key] = child_margin
        if cached_loss is None:
            self.n_evals += 1
        return val, child_key, child_margin

    def loss_from_parent_swap(self, parent_key: tuple[int, ...], parent_margin: np.ndarray,
                              out_j: int, in_j: int) -> tuple[float, tuple[int, ...], np.ndarray]:
        """Score ``parent_key - {out_j} + {in_j}`` reusing ``parent_margin`` in a single O(n) pass.

        ``out_j`` MUST be in parent; ``in_j`` MUST NOT. Fused vector op avoids the intermediate
        margin allocation of drop-then-add (two O(n) passes -> one O(n) pass).
        """
        # Splice out_j out, then binary-insert in_j (both on the sorted parent_key).
        lo, hi = 0, len(parent_key)
        while lo < hi:
            mid = (lo + hi) >> 1
            if parent_key[mid] < out_j:
                lo = mid + 1
            else:
                hi = mid
        without_out = parent_key[:lo] + parent_key[lo + 1:]
        lo, hi = 0, len(without_out)
        while lo < hi:
            mid = (lo + hi) >> 1
            if without_out[mid] < in_j:
                lo = mid + 1
            else:
                hi = mid
        child_key = without_out[:lo] + (in_j,) + without_out[lo:]
        cached_loss = self.cache.get(child_key)
        cached_margin = self.margin_cache.get(child_key)
        if cached_loss is not None and cached_margin is not None:
            return cached_loss, child_key, cached_margin
        # Fused: parent_margin + phi[:, in_j] - phi[:, out_j] in one np-internal pass.
        child_margin = parent_margin + (self.phi[:, in_j] - self.phi[:, out_j])
        val = self._loss_fast(child_margin)
        self.cache[child_key] = val
        self.margin_cache[child_key] = child_margin
        if cached_loss is None:
            self.n_evals += 1
        return val, child_key, child_margin

    def top_n(self, n: int) -> list[tuple[float, tuple[int, ...]]]:
        items = [(v, k) for k, v in self.cache.items() if k and np.isfinite(v)]
        items.sort(key=lambda t: t[0])
        return items[:n]


def _is_sorted_tuple(t: tuple) -> bool:
    # Fast verify ints + sorted; falls back if any non-int slips in (e.g. numpy scalars).
    for i in range(1, len(t)):
        a, b = t[i - 1], t[i]
        if not (type(a) is int and type(b) is int):
            return False
        if a > b:
            return False
    return len(t) == 0 or type(t[0]) is int


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
    # Seed the beam with the best single features (margin cached so layer-1 expansion reuses it).
    seeded: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    for j in range(f):
        loss, margin = ev.loss_with_margin((j,))
        seeded.append((loss, (j,), margin))
    seeded.sort(key=lambda t: t[0])
    beam = [(key, margin) for _, key, margin in seeded[:beam_width]]
    for _ in range(min_card, max_card):
        # ``expanded`` maps sorted-key -> loss; we evaluate each (parent, new_j) by single
        # O(n) vector add over the cached parent margin instead of recomputing the full sum.
        expanded: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
        for parent_key, parent_margin in beam:
            parent_set = set(parent_key)
            for j in range(f):
                if j in parent_set:
                    continue
                loss, child_key, child_margin = ev.loss_from_parent(parent_key, parent_margin, j)
                # Sibling parents may reach the same child via different add-orders -- keep the
                # already-computed value (deterministic, since the loss is a pure function of key).
                if child_key not in expanded:
                    expanded[child_key] = (loss, child_margin)
        if not expanded:
            break
        ranked = sorted(expanded.items(), key=lambda t: t[1][0])[:beam_width]
        beam = [(key, val[1]) for key, val in ranked]
    return ev.top_n(top_n)


def greedy_forward(phi, base, y, *, classification, metric=None, max_card=None, top_n=30):
    phi, base, y, metric = _prep(phi, base, y, classification, metric)
    f = phi.shape[1]
    max_card = f if max_card is None else min(max_card, f)
    ev = _Evaluator(phi, base, y, metric)
    current: tuple[int, ...] = ()
    # Empty subset has base as margin; cache it so the first add-feature step reuses it.
    cur_margin = base.copy()
    best_loss = float("inf")
    remaining = set(range(f))
    while remaining and len(current) < max_card:
        cand_best, cand_loss, cand_margin = None, float("inf"), None
        for j in remaining:
            l, _, m = ev.loss_from_parent(current, cur_margin, j)
            if l < cand_loss:
                cand_loss, cand_best, cand_margin = l, j, m
        if cand_best is None or cand_loss >= best_loss:
            break
        # Re-derive the sorted key (cheap; len <= max_card) and reuse the chosen child's margin.
        current = tuple(sorted(current + (cand_best,)))
        cur_margin = cand_margin
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
    """Hill-climb by add / drop / swap until no single move improves.

    Maintains ``(current_key, current_margin)`` so every add/drop/swap trial is a single
    O(n) vector op via ``loss_from_parent[_drop|_swap]`` instead of a full reduce-from-scratch
    over the candidate subset (same incremental trick as beam / greedy).
    """
    current = tuple(sorted(start))
    best, cur_margin = ev.loss_with_margin(current)
    improved = True
    while improved:
        improved = False
        cur_set = set(current)
        # First-improvement order: adds, then drops, then swaps -- matches the prior
        # ``moves`` traversal so the chosen move (and hence final subset) is unchanged.
        if len(current) < max_card:
            for j in range(f):
                if j in cur_set:
                    continue
                l, child_key, child_margin = ev.loss_from_parent(current, cur_margin, j)
                if l < best:
                    best, current, cur_margin, improved = l, child_key, child_margin, True
                    break
            if improved:
                continue
        if len(current) > 1:
            for j in current:
                l, child_key, child_margin = ev.loss_from_parent_drop(current, cur_margin, j)
                if l < best:
                    best, current, cur_margin, improved = l, child_key, child_margin, True
                    break
            if improved:
                continue
        for out in current:
            stop = False
            for inn in range(f):
                if inn in cur_set:
                    continue
                l, child_key, child_margin = ev.loss_from_parent_swap(current, cur_margin, out, inn)
                if l < best:
                    best, current, cur_margin, improved = l, child_key, child_margin, True
                    stop = True
                    break
            if stop:
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
    """Metropolis bit-flips with geometric cooling.

    Maintains ``(cur_key, cur_margin)``; each bit-flip is either an add (+phi[:, j]) or a
    drop (-phi[:, j]) via ``loss_from_parent[_drop]`` -- one O(n) vector op per trial instead
    of a full reduce. Math is bit-identical: the same Metropolis decision is taken on the
    same proxy_loss values.
    """
    phi, base, y, metric = _prep(phi, base, y, classification, metric)
    f = phi.shape[1]
    max_card = f if max_card is None else min(max_card, f)
    rng = np.random.default_rng(0) if rng is None else rng
    ev = _Evaluator(phi, base, y, metric)
    mask = np.zeros(f, dtype=bool)
    mask[rng.integers(0, f)] = True
    cur_idx = tuple(np.flatnonzero(mask).tolist())
    cur_loss, cur_margin = ev.loss_with_margin(cur_idx)
    cur_size = int(mask.sum())
    t = t0
    for _ in range(n_iter):
        j = int(rng.integers(0, f))
        flip_adds = not mask[j]
        new_size = cur_size + 1 if flip_adds else cur_size - 1
        if new_size == 0 or new_size > max_card:
            t *= cooling
            continue
        if flip_adds:
            trial_loss, trial_idx, trial_margin = ev.loss_from_parent(cur_idx, cur_margin, j)
        else:
            trial_loss, trial_idx, trial_margin = ev.loss_from_parent_drop(cur_idx, cur_margin, j)
        if trial_loss < cur_loss or rng.random() < np.exp(-(trial_loss - cur_loss) / max(t, 1e-9)):
            # Accept: flip the bit in the mask and roll the cached state forward.
            mask[j] = not mask[j]
            cur_idx, cur_loss, cur_margin, cur_size = trial_idx, trial_loss, trial_margin, new_size
        t *= cooling
    return ev.top_n(top_n)
