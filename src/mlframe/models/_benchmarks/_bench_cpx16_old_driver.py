"""Loads the OLD (HEAD) optimization.py source and runs the CPX16 bench logic
against it, so OLD vs NEW timings are apples-to-apples. Invoked by the A/B step."""
import os

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    import sys
    import time
    import types
    import importlib.util

    import numpy as np

    OLD_SRC = sys.argv[1]

    # Import the OLD source under its real package name so relative imports resolve.
    spec = importlib.util.spec_from_file_location("mlframe.models.optimization", OLD_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mlframe.models.optimization"] = mod
    spec.loader.exec_module(mod)
    MBHOptimizer = mod.MBHOptimizer


    def _build(n_space, n_known, n_skip, seed=0):
        opt = MBHOptimizer(
            search_space=np.arange(n_space),
            model_name="ETR",
            model_params={},
            init_num_samples=5,
            greedy_prob=1.0,
            random_state=seed,
        )
        opt.pre_seeded_candidates = []
        center = n_space // 2
        near = np.arange(center - n_skip // 2, center + n_skip // 2 + 1)
        rng = np.random.default_rng(seed)
        extra = rng.choice(np.setdiff1d(np.arange(n_space), near), size=max(0, n_known - len(near)), replace=False)
        known = np.unique(np.concatenate([near, extra])).astype(int)
        opt.known_candidates = known
        opt.best_candidate = int(center)
        opt.best_evaluation = 1.0
        opt.n_noimproving_iters = 1
        opt.suggested_candidates.clear()
        return opt


    def _time(n_space, n_known, n_skip, n_calls=200, repeats=5):
        best = float("inf")
        for _ in range(repeats):
            opt = _build(n_space, n_known, n_skip)
            opt.suggest_candidate()
            opt.suggested_candidates.clear()
            t0 = time.perf_counter()
            for _ in range(n_calls):
                opt.suggest_candidate()
                opt.suggested_candidates.clear()
            best = min(best, time.perf_counter() - t0)
        return best / n_calls


    print(f"{'n_space':>10} {'n_known':>9} {'n_skip':>8} {'us/call':>12}")
    for n_space, n_known, n_skip in [(5_000, 2_500, 2_000), (10_000, 5_000, 4_000), (50_000, 25_000, 20_000)]:
        print(f"{n_space:>10} {n_known:>9} {n_skip:>8} {_time(n_space, n_known, n_skip) * 1e6:>12.2f}")
