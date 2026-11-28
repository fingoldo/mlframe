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

import logging

import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


def fold_train_groups(splitter: Any, train_idx: Any) -> "np.ndarray | None":
    """Group labels of one outer fold's training rows, or ``None`` when the splitter carries no groups.

    A ``BoundSplitter`` (or any splitter exposing ``groups``) hands its labels to the baseline through this helper, so the
    inner baseline folds are group-aware on the same rows the outer fold trains on.
    """
    groups = getattr(splitter, "groups", None)
    return None if groups is None else np.asarray(groups)[np.asarray(train_idx)]


def kfold_splits(X: Any, n_splits: int, random_state: int, groups: Any = None) -> "list[tuple[np.ndarray, np.ndarray]]":
    """Fold index pairs of an out-of-fold pass over ``X``: a shuffled KFold, or a GroupKFold when ``groups`` is given.

    ``groups`` holds one label per row of ``X``. With it, the rows of one group are never split between a fold's training and
    held-out rows, so a duplicate of a row cannot leak that row's own target into its out-of-fold prediction. Without it the
    result is exactly the shuffled KFold the callers used before. With fewer than two distinct groups no group can be held
    out, so the shuffled KFold is used and a warning says so.
    """
    from sklearn.model_selection import GroupKFold, KFold

    if groups is not None:
        g = np.asarray(groups)
        n_groups = int(np.unique(g).size)
        if n_groups >= 2:
            return list(GroupKFold(n_splits=min(n_splits, n_groups)).split(X, groups=g))
        logger.warning("kfold_splits: %d distinct group(s) cannot be held out; using a shuffled KFold, which cannot keep a group on one side.", n_groups)
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X))  # group-blind-ok: with fewer than two groups none can be held out (warned above), and without groups it is the plain KFold


def fit_baseline_predict_oof(
    Xt: np.ndarray,
    y_t: np.ndarray,
    task: str,
    seed: int,
    n_estimators: int = 50,
    max_depth: int = 3,
    caller: str = "the residual-band transformer cluster",
    groups: "np.ndarray | None" = None,
) -> np.ndarray:
    """Fit a shallow LightGBM baseline via an inner KFold(3) and return its OUT-OF-FOLD predictions on Xt.

    Returns the predicted probability of the positive class for ``task="binary"`` and the raw value
    otherwise. Falls back to a single in-sample fit when there are too few rows for a 3-fold inner split --
    with fewer than three rows there is no honest split to be had, and the caller's own row-count guards
    keep that path off the sizes these transformers actually run at.

    ``caller`` names the caller in the ImportError. Modules that need no other difference import this
    function under their own name rather than wrapping it, since a wrapper differing only in this string is
    still a near-duplicate body -- the exact drift this consolidation exists to end.

    ``groups`` (the training rows' group labels) makes the inner split group-aware: the rows of one group are never split
    between the baseline's training and held-out folds, so duplicates of a row cannot leak their own target into its
    out-of-fold prediction. Without it the inner split is a shuffled KFold, which is the leaky default for grouped data.
    With fewer than two distinct groups no group can be held out, so the split falls back to the shuffled KFold with a warning
    (see :func:`kfold_splits`).
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

    preds = np.zeros(n, dtype=np.float32)
    for inner_idx, (in_tr, in_val) in enumerate(kfold_splits(Xt, 3, int(seed) + 11, groups)):
        if task == "binary":
            m = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed) + 7 + inner_idx, verbose=-1, n_jobs=-1)
            m.fit(Xt[in_tr], y_t[in_tr].astype(np.int32))
            preds[in_val] = np.asarray(m.predict_proba(Xt[in_val]))[:, 1].astype(np.float32)
        else:
            m = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, random_state=int(seed) + 7 + inner_idx, verbose=-1, n_jobs=-1)
            m.fit(Xt[in_tr], y_t[in_tr])
            preds[in_val] = np.asarray(m.predict(Xt[in_val])).astype(np.float32)
    return preds


def three_member_predictions(Xa: np.ndarray, ya: np.ndarray, Xb: np.ndarray, task: str, seed: int, owner: str = __name__) -> np.ndarray:
    """Fit 3 diverse baselines (shallow LGB, deeper LGB, linear/logistic) on ``(Xa, ya)`` and return their predictions on ``Xb`` as ``(n_b, 3)``.

    The one definition of the three-member ensemble that ``disagreement_band``, ``multi_baseline_hard_row``, ``ib_baseline_codes``, ``pairwise_kl_divergence`` and
    ``nn_oof_target_mean`` each carried a copy of (audit 09-F31): a fix to a seed offset or to the failure fallback now lands in all of them. A single-class binary
    target, or a linear member that fails to fit, falls back to the class prior. ``owner`` is the caller's ``__name__``: the fallback is logged on that module's logger (INFO), as each copy did before the consolidation.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(f"{owner} requires lightgbm") from exc
    from sklearn.linear_model import LogisticRegression, Ridge

    out = np.zeros((Xb.shape[0], 3), dtype=np.float32)
    if task == "binary":
        ya_i = ya.astype(np.int32)
        prior = float(ya.mean())
        if np.unique(ya_i).size < 2:
            out[:] = prior
            return out
        m1 = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1).fit(Xa, ya_i)
        out[:, 0] = np.asarray(m1.predict_proba(Xb))[:, 1]
        m2 = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1).fit(Xa, ya_i)
        out[:, 1] = np.asarray(m2.predict_proba(Xb))[:, 1]
        try:
            m3 = LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed) + 2).fit(Xa, ya_i)
            out[:, 2] = m3.predict_proba(Xb)[:, 1]
        except Exception as e:
            logging.getLogger(owner).info("%s: LogisticRegression fit failed (%s); falling back to constant class prior.", owner.rsplit(".", 1)[-1], e)
            out[:, 2] = prior
    else:
        m1 = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=int(seed), verbose=-1, n_jobs=-1).fit(Xa, ya)
        out[:, 0] = m1.predict(Xb)
        m2 = lgb.LGBMRegressor(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=int(seed) + 1, verbose=-1, n_jobs=-1).fit(Xa, ya)
        out[:, 1] = m2.predict(Xb)
        m3 = Ridge(alpha=1.0, random_state=int(seed) + 2).fit(Xa, ya)
        out[:, 2] = m3.predict(Xb)
    return out


def three_member_oof(Xt: np.ndarray, y_t: np.ndarray, task: str, seed: int, groups: Any = None, owner: str = __name__) -> np.ndarray:
    """Out-of-fold ``(n, 3)`` predictions of the three-member ensemble over an inner KFold(3); in-sample when there are fewer than 3 rows.

    An in-sample prediction sits close to ``y_t`` by construction, so using it understates every member's error; the OOF form is what the callers judge rows by.
    """
    n = Xt.shape[0]
    if n < 3:
        return three_member_predictions(Xt, y_t, Xt, task, seed, owner)
    preds = np.zeros((n, 3), dtype=np.float32)
    for k, (tr, va) in enumerate(kfold_splits(Xt, 3, int(seed) + 11, groups)):
        preds[va] = three_member_predictions(Xt[tr], y_t[tr], Xt[va], task, int(seed) + 7 + k, owner)
    return preds
