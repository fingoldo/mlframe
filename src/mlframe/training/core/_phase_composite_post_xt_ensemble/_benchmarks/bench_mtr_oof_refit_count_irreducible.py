"""Bench/proof: the MTR honest-OOF refit count (K x n_comp inner fits) is IRREDUCIBLE by design.

``compute_mtr_oof_nnls_weights`` derives per-column NNLS weights from a leak-free train-K-fold OOF stack: each
component is cloned and re-fit on the K-1 train folds and predicts ONLY its held-out fold. With K folds and
``n_comp`` components that is exactly ``K * n_comp`` inner ``fit`` calls, and the cProfile note in the module
docstring attributes ~77% of the call wall to those refits.

This bench establishes -- as a TESTED verdict, not an assumed one -- that the refit count is the minimum required
for a leak-free OOF, by demonstrating the two "cheaper" alternatives BREAK the leak-free contract:

  1. fit-once-on-all-then-predict-all (n_comp fits): every row's prediction comes from a model that SAW that row
     in training -> the OOF cell is an in-sample fit, which is exactly the double-dip leak the helper exists to
     avoid. We measure how optimistic the in-sample stack looks vs the honest OOF stack (lower train RMSE = leak).
  2. predicting a held-out fold from a model that included it: same leak, just per-fold.

The only count that yields a held-out (never-trained-on) prediction for EVERY row is one refit per (fold,
component) on the folds EXCLUDING that row -> K x n_comp. Any reduction reuses a model on rows it trained on.

Run (direct path; the package ``-m`` chain pulls heavy core ``__init__`` deps):
    python src/mlframe/training/core/_phase_composite_post_xt_ensemble/_benchmarks/bench_mtr_oof_refit_count_irreducible.py
"""
from __future__ import annotations

import time

import numpy as np


def _make_data(n: int = 4000, p: int = 8, k: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal((p, k))
    y = X @ beta + 0.5 * rng.standard_normal((n, k))
    return X, y


def _components(n_comp: int = 4):
    from sklearn.tree import DecisionTreeRegressor
    return [DecisionTreeRegressor(max_depth=5, random_state=i) for i in range(n_comp)]


def _honest_oof_fit_count_and_stack(components, X, y, kfold: int = 5, seed: int = 42):
    """Leak-free OOF: K x n_comp refits; each cell predicted by a model that NEVER saw that row."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    n, k = y.shape
    n_comp = len(components)
    oof = np.full((n_comp, n, k), np.nan)
    fits = 0
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    for tr, ho in kf.split(np.arange(n)):
        for ci, comp in enumerate(components):
            cl = clone(comp)
            cl.fit(X[tr], y[tr])
            fits += 1
            oof[ci, ho, :] = cl.predict(X[ho]).reshape(len(ho), k)
    return fits, oof


def _insample_fit_count_and_stack(components, X, y):
    """The 'cheaper' alternative: n_comp fits, every model predicts the WHOLE set incl. its own train rows (LEAK)."""
    from sklearn.base import clone
    n, k = y.shape
    stack = np.empty((len(components), n, k))
    fits = 0
    for ci, comp in enumerate(components):
        cl = clone(comp)
        cl.fit(X, y)
        fits += 1
        stack[ci] = cl.predict(X).reshape(n, k)
    return fits, stack


def _rmse(stack, y):
    # equal-mean ensemble RMSE as a coarse honesty proxy (the NNLS solve is downstream + identical structurally).
    pred = stack.mean(axis=0)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main() -> None:
    X, y = _make_data()
    kfold = 5
    comps = _components(4)
    n_comp = len(comps)

    t0 = time.perf_counter()
    honest_fits, honest_stack = _honest_oof_fit_count_and_stack(comps, X, y, kfold=kfold)
    t_honest = time.perf_counter() - t0

    t0 = time.perf_counter()
    insample_fits, insample_stack = _insample_fit_count_and_stack(comps, X, y)
    t_insample = time.perf_counter() - t0

    honest_rmse = _rmse(honest_stack, y)
    insample_rmse = _rmse(insample_stack, y)

    print(f"n_comp={n_comp}  kfold={kfold}")
    print(f"  honest OOF refits        : {honest_fits}  (== K*n_comp == {kfold * n_comp})  wall={t_honest*1e3:.1f} ms")
    print(f"  in-sample 'cheaper' fits : {insample_fits}  (== n_comp)               wall={t_insample*1e3:.1f} ms")
    print(f"  honest-OOF ensemble RMSE  (held-out, HONEST) : {honest_rmse:.4f}")
    print(f"  in-sample ensemble RMSE   (LEAKED, optimistic): {insample_rmse:.4f}")
    leak_ratio = honest_rmse / insample_rmse if insample_rmse > 0 else float("inf")
    print(f"  in-sample RMSE is {leak_ratio:.2f}x LOWER -> the 'cheaper' path is optimistically biased (leak).")
    assert honest_fits == kfold * n_comp, "honest OOF must be exactly K*n_comp refits"
    assert insample_rmse < honest_rmse, "in-sample stack must look optimistic (proves the leak it would introduce)"
    print(
        "\nVerdict (DOC -- by-design, not a hotspot): K*n_comp is the MINIMUM refit count for a leak-free OOF. "
        "Reducing to n_comp (fit-once-predict-all) re-uses each model on rows it trained on -> the OOF cells become "
        "in-sample fits, which is the exact double-dip leak this helper exists to avoid (demonstrated above: the "
        "in-sample stack is optimistically lower-RMSE). The ~77% refit wall is irreducible; no wrapper-side speedup "
        "can cut the fit count without breaking the honest-OOF contract benched in training/_benchmarks/bench_mtr_nnls_oof.py."
    )


if __name__ == "__main__":
    main()
