"""CPX16 bench: membership-test complexity in ``MBHOptimizer.suggest_candidate``.

The candidate-selection loops scan ``np.argsort(expected_fitness)`` and test
``next_candidate not in self.known_candidates`` (lines ~426/547/564/577). With
``known_candidates`` an ndarray, ``x not in ndarray`` is an O(K) linear scan, so
the loop is O(S*K) when the loop must walk past many already-known/ranked-high
candidates before reaching an unchecked one. The fix builds one
``set(self.known_candidates.tolist())`` per call (O(K) once) and tests against it
(O(1) each) -> O(S+K) per call.

Crossover note: the set construction costs O(K) up front, so the win materialises
only when the loop performs enough membership tests to amortise it. The DECISIVE
realistic case is the exploitation loop (line ~575) where the fitness ranking is
unrelated to the known set, so the scan walks past many KNOWN top-ranked points
(each an O(K) ndarray scan in OLD) before hitting an unchecked candidate. This
bench reproduces that by directly exercising the selection loop with a controlled
number of known points the scan must skip.

Run (CUDA off, python on PATH):
    CUDA_VISIBLE_DEVICES="" python src/mlframe/models/_benchmarks/bench_cpx16_optimization.py
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import time
import numpy as np

from mlframe.models.optimization import MBHOptimizer


def _build_primed_optimizer(n_space: int, n_known: int, n_skip: int, seed: int = 0) -> MBHOptimizer:
    """Optimizer over ``[0, n_space)`` primed for the greedy selection loop, with
    ``best_candidate`` placed so the nearest ``n_skip`` candidates are all known --
    forcing the scan to walk past ``n_skip`` known points (each an O(K) ndarray
    membership test in OLD) before reaching an unchecked candidate."""
    search_space = np.arange(n_space)
    opt = MBHOptimizer(
        search_space=search_space,
        model_name="ETR",
        model_params={},
        init_num_samples=5,
        greedy_prob=1.0,
        random_state=seed,
    )
    opt.pre_seeded_candidates = []
    center = n_space // 2
    # The ``n_skip`` candidates closest to ``center`` are all known, plus filler
    # known points elsewhere to reach ``n_known`` total (so K = n_known is large).
    near = np.arange(center - n_skip // 2, center + n_skip // 2 + 1)
    rng = np.random.default_rng(seed)
    extra = rng.choice(np.setdiff1d(search_space, near), size=max(0, n_known - len(near)), replace=False)
    known = np.unique(np.concatenate([near, extra])).astype(int)
    opt.known_candidates = known
    opt.best_candidate = int(center)
    opt.best_evaluation = 1.0
    opt.n_noimproving_iters = 1
    opt.suggested_candidates.clear()
    return opt


def _time_suggest(n_space: int, n_known: int, n_skip: int, n_calls: int, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        opt = _build_primed_optimizer(n_space, n_known, n_skip)
        opt.suggest_candidate()
        opt.suggested_candidates.clear()
        t0 = time.perf_counter()
        for _ in range(n_calls):
            opt.suggest_candidate()
            opt.suggested_candidates.clear()
        best = min(best, time.perf_counter() - t0)
    return best / n_calls


if __name__ == "__main__":
    print("CPX16 suggest_candidate membership-scan bench (greedy path, best-of-5)\n")
    print(f"{'n_space':>10} {'n_known':>9} {'n_skip':>8} {'us/call':>12}")
    for n_space, n_known, n_skip in [
        (5_000, 2_500, 2_000),
        (10_000, 5_000, 4_000),
        (50_000, 25_000, 20_000),
    ]:
        per_call = _time_suggest(n_space, n_known, n_skip, n_calls=200) * 1e6
        print(f"{n_space:>10} {n_known:>9} {n_skip:>8} {per_call:>12.2f}")
