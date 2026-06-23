"""CPX36 bench: batched-stack predict vs per-perturbation full predict.

Three FE transformers replace a per-feature loop of full LightGBM predict calls with ONE
predict over a vertically-stacked perturbation matrix, reshaped back. Identity: bit-identical
emitted features (tree models predict per-row independently, so a predict on the stack equals
per-perturbation predict). This bench captures OLD (committed baseline snapshot under
_cpx36_baseline/) vs NEW (current transformer module) timing + identity at a realistic modest
shape (small LightGBM, n_query ~ a few hundred, d ~ 12-20).

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_engineering._benchmarks.bench_cpx36_batched_predict
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_engineering.transformer.counterfactual_substitution import (
    compute_counterfactual_substitution_features as cfact_new,
)
from mlframe.feature_engineering.transformer.adversarial_flip import (
    compute_adversarial_flip_features as advflip_new,
)
from mlframe.feature_engineering.transformer.fisher_weighted_residual import (
    compute_fisher_weighted_residual_features as fisher_new,
)
from mlframe.feature_engineering._benchmarks._cpx36_baseline.counterfactual_substitution_old import (
    compute_counterfactual_substitution_features as cfact_old,
)
from mlframe.feature_engineering._benchmarks._cpx36_baseline.adversarial_flip_old import (
    compute_adversarial_flip_features as advflip_old,
)
from mlframe.feature_engineering._benchmarks._cpx36_baseline.fisher_weighted_residual_old import (
    compute_fisher_weighted_residual_features as fisher_old,
)


def _data(n_train: int, n_query: int, d: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    Xt = rng.standard_normal((n_train, d)).astype(np.float32)
    Xq = rng.standard_normal((n_query, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    yt_cont = Xt @ w + 0.1 * rng.standard_normal(n_train).astype(np.float32)
    return Xt, Xq, yt_cont, w


def _best_of(fn, n: int = 3) -> float:
    best = float("inf")
    out = None
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def bench_one(name: str, old_fn, new_fn, task: str, n_train=2000, n_query=400, d=16):
    Xt, Xq, yt_cont, w = _data(n_train, n_query, d)
    if task == "binary":
        y = (yt_cont > np.median(yt_cont)).astype(np.float32)
    else:
        y = yt_cont
    # warm
    old_fn(Xt, y, Xq, seed=1, task=task)
    new_fn(Xt, y, Xq, seed=1, task=task)

    t_old, df_old = _best_of(lambda: old_fn(Xt, y, Xq, seed=1, task=task))
    t_new, df_new = _best_of(lambda: new_fn(Xt, y, Xq, seed=1, task=task))

    a = df_old.to_numpy()
    b = df_new.to_numpy()
    exact = bool(np.array_equal(a, b))
    max_abs = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")
    print(f"\n=== {name} [{task}] shape=(nt={n_train},nq={n_query},d={d}) ===")
    print(f"  OLD best-of-3: {t_old*1000:8.2f} ms")
    print(f"  NEW best-of-3: {t_new*1000:8.2f} ms   speedup={t_old/t_new:.2f}x")
    print(f"  identity: exact_equal={exact}  max_abs_diff={max_abs:.3e}")
    return exact, max_abs, t_old / t_new


if __name__ == "__main__":
    results = []
    for task in ("regression", "binary"):
        results.append(bench_one("counterfactual_substitution", cfact_old, cfact_new, task))
        results.append(bench_one("adversarial_flip", advflip_old, advflip_new, task))
        results.append(bench_one("fisher_weighted_residual", fisher_old, fisher_new, task))
    print("\nSUMMARY exact_equal:", all(r[0] for r in results))
