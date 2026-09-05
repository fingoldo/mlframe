"""One out-of-fold baseline fit, shared by the residual-band transformer cluster.

Ten modules in this package each carried their own ``_fit_baseline_predict`` with the same name and, for
most of them, the same signature. Four were fixed to return inner-KFold(3) out-of-fold predictions and the
rest were not, because the fix was propagated by hand: an in-sample prediction is close to ``y_t`` almost by
construction (the model was just fit on these exact rows), which understates the true baseline residual and
distorts which rows look easy or hard. Every column those modules emit is derived from that judgement.

The copies that shared a signature now all call this one function, so the next correction lands everywhere
at once. All eight copies sharing this signature turned out to be the same function: three of them factored the body
through a nested ``_fit_predict`` helper and were described as doing something else -- ``class_balanced_hard_row``
was believed to refit class-balanced -- but each was verified bit-identical to this implementation on both
task branches before being replaced. ``baseline_surprise`` keeps its own body because it also predicts on a
held-out Xq and returns a pair; ``y_quintile_baseline_knn`` calls this one with its own deeper baseline.
"""

from __future__ import annotations

import numpy as np


def fit_baseline_predict_oof(
    Xt: np.ndarray,
    y_t: np.ndarray,
    task: str,
    seed: int,
    n_estimators: int = 50,
    max_depth: int = 3,
    caller: str = "the residual-band transformer cluster",
) -> np.ndarray:
    """Fit a shallow LightGBM baseline via an inner KFold(3) and return its OUT-OF-FOLD predictions on Xt.

    Returns the predicted probability of the positive class for ``task="binary"`` and the raw value
    otherwise. Falls back to a single in-sample fit when there are too few rows for a 3-fold inner split --
    with fewer than three rows there is no honest split to be had, and the caller's own row-count guards
    keep that path off the sizes these transformers actually run at.

    ``caller`` names the caller in the ImportError. Modules that need no other difference import this
    function under their own name rather than wrapping it, since a wrapper differing only in this string is
    still a near-duplicate body -- the exact drift this consolidation exists to end.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(f"{caller} requires lightgbm") from exc

    n = Xt.shape[0]
    if n < 3:
        if task == "binary":
            model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
            model.fit(Xt, y_t.astype(np.int32))
            return np.asarray(model.predict_proba(Xt))[:, 1].astype(np.float32)
        model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1)
        model.fit(Xt, y_t)
        return np.asarray(model.predict(Xt)).astype(np.float32)

    from sklearn.model_selection import KFold

    preds = np.zeros(n, dtype=np.float32)
    inner_splitter = KFold(n_splits=3, shuffle=True, random_state=int(seed) + 11)
    for inner_idx, (in_tr, in_val) in enumerate(inner_splitter.split(Xt)):
        if task == "binary":
            m = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed) + 7 + inner_idx, verbose=-1, n_jobs=-1)
            m.fit(Xt[in_tr], y_t[in_tr].astype(np.int32))
            preds[in_val] = np.asarray(m.predict_proba(Xt[in_val]))[:, 1].astype(np.float32)
        else:
            m = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed) + 7 + inner_idx, verbose=-1, n_jobs=-1)
            m.fit(Xt[in_tr], y_t[in_tr])
            preds[in_val] = np.asarray(m.predict(Xt[in_val])).astype(np.float32)
    return preds
