"""Bench: minimax_ranking(score_type="winning_votes") redundant pass removal.

OLD code computed ((ranks < ranks.loc[model]) * weights).sum(axis=1) TWICE per
model (once for `models_scores`, once inside the `does_win` comparison's LHS).
NEW code computes the "less" weighted sum once and reuses it for both.

Run:
  CUDA_VISIBLE_DEVICES="" python src/mlframe/votenrank/_benchmarks/bench_minimax_winning_votes.py
"""

import time
import numpy as np
import pandas as pd


def _old_minimax(ranks, weights, models):
    out = []
    for model in models:
        models_scores = ((ranks < ranks.loc[model]) * weights).sum(axis=1)
        does_win = ((ranks < ranks.loc[model]) * weights).sum(axis=1) > ((ranks > ranks.loc[model]) * weights).sum(axis=1)
        models_scores = models_scores * does_win
        out.append(models_scores.drop(model).max())
    return (-pd.Series(data=out, index=pd.Series(models, name="Name"))).sort_values(ascending=False)


def _new_minimax(ranks, weights, models):
    out = []
    for model in models:
        row = ranks.loc[model]
        less = ((ranks < row) * weights).sum(axis=1)
        greater = ((ranks > row) * weights).sum(axis=1)
        models_scores = less * (less > greater)
        out.append(models_scores.drop(model).max())
    return (-pd.Series(data=out, index=pd.Series(models, name="Name"))).sort_values(ascending=False)


def _make(n_models, n_tasks, seed=0):
    rng = np.random.default_rng(seed)
    tbl = pd.DataFrame(
        rng.normal(size=(n_models, n_tasks)),
        index=[f"m{i}" for i in range(n_models)],
        columns=[f"t{j}" for j in range(n_tasks)],
    )
    ranks = tbl.rank(method="min", ascending=False).astype(int)
    weights = pd.Series(index=tbl.columns, data=1.0)
    return ranks, weights, tbl.index.tolist()


def bench(n_models, n_tasks, reps=20):
    ranks, weights, models = _make(n_models, n_tasks)
    # identity
    a = _old_minimax(ranks, weights, models)
    b = _new_minimax(ranks, weights, models)
    assert a.equals(b), "identity FAILED"  # nosec B101 - internal invariant check in src/mlframe/votenrank/_benchmarks, not reachable with untrusted input

    def t(fn):
        best = float("inf")
        for _ in range(reps):
            s = time.perf_counter()
            fn(ranks, weights, models)
            best = min(best, time.perf_counter() - s)
        return best

    old = t(_old_minimax)
    new = t(_new_minimax)
    print(f"n_models={n_models} n_tasks={n_tasks}: OLD={old*1e3:.3f}ms NEW={new*1e3:.3f}ms " f"speedup={old/new:.3f}x identity=OK")


if __name__ == "__main__":
    bench(50, 20)
    bench(100, 50)
    bench(200, 100)
