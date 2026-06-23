"""CPX14 bench: per-fold SHAP redundancy in compute_shap_on_cv.

The pre-fix compute_shap_on_cv calls ``explainer(X)`` on the ENTIRE dataset every fold (k-fold => ~k x redundant
SHAP work). The proposed optimization computes SHAP only on the fold's X_test and concatenates into an OOF SHAP matrix.

Two things measured here:

1. IDENTITY PROBE - does ``shap.Explainer(model)`` WITHOUT a background ``data=`` argument
   produce bit-identical SHAP values for a subset of rows whether explained alone or sliced out of the full-X result?
   The active prod code passes NO background set, so the only correctness question for the COMPUTE side is whether
   the row set passed to ``explainer(...)`` influences the per-row values (it must not, for a tree explainer that
   uses the model's own path-coverage as the reference). If bit-identical, restricting the COMPUTE to X_test is a
   clean win FOR THE COMPUTE - but note the RETURN CONTRACT still changes (see the module docstring of the test).

2. TIMING - OLD (explain full X per fold) vs NEW (explain only X_test per fold) wall time, best-of-N, warmed.

Run: CUDA_VISIBLE_DEVICES="" python src/mlframe/inference/_benchmarks/bench_shap_oof_per_fold.py
catboost+shap importorskip: falls back to a sklearn GradientBoosting tree model that shap's TreeExplainer supports.
"""

from __future__ import annotations

import time
import numpy as np


def _make_data(n: int = 4000, n_features: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    logit = X[:, 0] * 1.3 - X[:, 1] * 0.9 + X[:, 2] * X[:, 3] * 0.5
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return X, y


def _fit_model():
    """Prefer catboost (the prod target); fall back to sklearn GB which shap.TreeExplainer fully supports."""
    try:
        from catboost import CatBoostClassifier

        m = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1, verbose=0, allow_writing_files=False)
        return m, "catboost"
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0), "sklearn_gb"


def identity_probe():
    import shap

    X, y = _make_data()
    model, kind = _fit_model()
    model.fit(X, y)

    explainer = shap.Explainer(model)

    full = explainer(X)
    # Pick a contiguous "test" block, as a CV fold would yield.
    test_idx = np.arange(1000, 1750)
    sub = explainer(X[test_idx])

    full_slice = np.asarray(full.values)[test_idx]
    sub_vals = np.asarray(sub.values)

    # base_values comparison too (per-row in newer shap).
    full_bv = np.asarray(full.base_values)
    sub_bv = np.asarray(sub.base_values)
    full_bv_slice = full_bv[test_idx] if full_bv.ndim >= 1 and full_bv.shape[0] == len(X) else full_bv

    max_abs = float(np.max(np.abs(full_slice - sub_vals)))
    bit_identical = np.array_equal(full_slice, sub_vals)
    bv_max_abs = float(np.max(np.abs(np.atleast_1d(full_bv_slice) - np.atleast_1d(sub_bv))))

    print(f"[identity] model={kind} shap.Explainer auto-picked: {type(explainer).__name__}")
    print(f"[identity] values shapes full_slice={full_slice.shape} sub={sub_vals.shape}")
    print(f"[identity] values max_abs_diff={max_abs:.3e}  bit_identical={bit_identical}")
    print(f"[identity] base_values max_abs_diff={bv_max_abs:.3e}")
    return bit_identical, max_abs


def timing(n_folds: int = 5, repeats: int = 3):
    import shap
    from sklearn.model_selection import KFold

    X, y = _make_data()
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    # Pre-fit each fold's model once; we time only the SHAP-explain step (the thing CPX14 targets).
    fold_models = []
    for tr, te in cv.split(X):
        m, _ = _fit_model()
        m.fit(X[tr], y[tr])
        fold_models.append((m, te))

    def old_path():
        out = []
        for m, _te in fold_models:
            ex = shap.Explainer(m)
            out.append(np.asarray(ex(X).values))  # FULL X every fold
        return out

    def new_path():
        rows = [None] * len(X)
        for m, te in fold_models:
            ex = shap.Explainer(m)
            v = np.asarray(ex(X[te]).values)  # only X_test
            for j, ridx in enumerate(te):
                rows[ridx] = v[j]
        return np.stack(rows)

    # warm
    old_path(); new_path()

    def best_of(fn):
        best = float("inf")
        for _ in range(repeats):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
        return best

    t_old = best_of(old_path)
    t_new = best_of(new_path)
    print(f"[timing] folds={n_folds} OLD(full-X/fold)={t_old*1e3:.1f}ms  NEW(X_test/fold)={t_new*1e3:.1f}ms  speedup={t_old/t_new:.2f}x")
    return t_old, t_new


if __name__ == "__main__":
    try:
        import shap  # noqa: F401
    except Exception as e:
        print("shap unavailable -> bench is a no-op:", e)
        raise SystemExit(0)
    print("=== CPX14 SHAP OOF per-fold bench ===")
    identity_probe()
    timing()
