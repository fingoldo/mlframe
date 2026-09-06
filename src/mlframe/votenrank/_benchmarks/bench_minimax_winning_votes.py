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
        # Production's empty-opponents guard (leaderboard/_rules.py): a 1-model leaderboard has no
        # opponents, .max() on the empty Series is NaN, and NaN never equals itself -- so
        # minimax_election's `ranking == ranking.max()` returns an EMPTY winner list for the one
        # trivially-correct model. Both frozen copies here lacked it, so `assert a.equals(b)` compared two
        # pre-fix implementations and the reported speedup was measured against code that no longer ships.
        opponents = models_scores.drop(model)
        out.append(opponents.max() if not opponents.empty else 0.0)
    return (-pd.Series(data=out, index=pd.Series(models, name="Name"))).sort_values(ascending=False)


def _new_minimax(ranks, weights, models):
    out = []
    for model in models:
        row = ranks.loc[model]
        less = ((ranks < row) * weights).sum(axis=1)
        greater = ((ranks > row) * weights).sum(axis=1)
        models_scores = less * (less > greater)
        opponents = models_scores.drop(model)
        out.append(opponents.max() if not opponents.empty else 0.0)
    return (-pd.Series(data=out, index=pd.Series(models, name="Name"))).sort_values(ascending=False)


def _tbl_for(n_models, n_tasks, seed=0):
    """The raw score table the ranks below are derived from, so the shipped Leaderboard can be built too."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n_models, n_tasks)),
        index=[f"m{i}" for i in range(n_models)],
        columns=[f"t{j}" for j in range(n_tasks)],
    )


def _make(n_models, n_tasks, seed=0):
    tbl = _tbl_for(n_models, n_tasks, seed)
    ranks = tbl.rank(method="min", ascending=False).astype(int)
    weights = pd.Series(index=tbl.columns, data=1.0)
    return ranks, weights, tbl.index.tolist()


def bench(n_models, n_tasks, reps=20):
    ranks, weights, models = _make(n_models, n_tasks)
    # identity
    a = _old_minimax(ranks, weights, models)
    b = _new_minimax(ranks, weights, models)
    assert a.equals(b), "identity FAILED"  # nosec B101 - internal invariant check in src/mlframe/votenrank/_benchmarks, not reachable with untrusted input

    # ...and against what actually ships, not only against the other local copy. Both frozen forms above
    # can agree with each other while diverging from Leaderboard.minimax_ranking.
    from mlframe.votenrank.leaderboard import Leaderboard

    shipped = Leaderboard(_tbl_for(n_models, n_tasks)).minimax_ranking()
    assert np.allclose(shipped.loc[b.index].to_numpy(), b.to_numpy(), equal_nan=True), (  # nosec B101 - internal invariant check in src/mlframe/votenrank/_benchmarks, not reachable with untrusted input
        f"bench copy diverges from the shipped minimax_ranking: shipped={shipped.to_dict()} bench={b.to_dict()}"
    )

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
    # A 1-model leaderboard first: it has no opponents, which is the shape the empty-opponents guard
    # exists for and the one every other size here avoids by construction.
    bench(1, 8)
    bench(50, 20)
    bench(100, 50)
    bench(200, 100)
