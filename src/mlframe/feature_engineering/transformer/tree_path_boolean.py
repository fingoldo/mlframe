"""Tree-path boolean features: extract LGB root→leaf decision paths as boolean conjunctions.

Iter 92 mechanism. Agent A #1 ranked. Mammography target.

Train baseline LGB depth=4-5 per OOF fold; for top-K leaves (highest leaf-value abs), extract the
path (sequence of split conditions root→leaf); evaluate each path as a boolean column on query rows.

Per query emit n_paths boolean columns + n_paths_matched + pred_prob_baseline.
Default K=8 paths → 10 features total.
"""
from __future__ import annotations
import logging
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _extract_top_paths(booster, X_sample: np.ndarray, top_k: int = 8) -> list[list[tuple[int, float, bool]]]:
    """Extract top-K leaf paths as list of (feature_idx, threshold, go_left) conditions.

    Ranks leaves by abs(leaf_value) × leaf_count. Returns list of conditions per path.
    """
    model_dump = booster.dump_model()
    paths: list[list[tuple[int, float, bool]]] = []
    leaf_scores: list[float] = []

    def _walk(node: dict, conditions: list[tuple[int, float, bool]]):
        """Recurse down LightGBM's dumped tree structure, accumulating split conditions until a leaf is reached, at which point the accumulated path and its (leaf_value * leaf_count) score are recorded."""
        if "leaf_value" in node:
            leaf_val = abs(float(node["leaf_value"]))
            leaf_cnt = float(node.get("leaf_count", 1))
            paths.append(list(conditions))
            leaf_scores.append(leaf_val * leaf_cnt)
            return
        feat = int(node["split_feature"])
        thresh = float(node["threshold"])
        # Left child: feat <= thresh (if decision_type is "<=")
        _walk(node["left_child"], [*conditions, (feat, thresh, True)])
        _walk(node["right_child"], [*conditions, (feat, thresh, False)])

    for tree_info in model_dump["tree_info"]:
        _walk(tree_info["tree_structure"], [])

    # Rank by score, take top_k.
    # Wave 62 (2026-05-20): lexsort with path-index tiebreak so tied leaf scores
    # give deterministic top-K paths across runs.
    _scores_arr = np.asarray(leaf_scores)
    order = np.lexsort((np.arange(len(_scores_arr)), -_scores_arr))[:top_k]
    return [paths[i] for i in order]


def _evaluate_paths(paths: list, X: np.ndarray) -> np.ndarray:
    """Return (n_rows, n_paths) boolean matrix."""
    n_rows = X.shape[0]
    n_paths = len(paths)
    out = np.zeros((n_rows, n_paths), dtype=np.float32)
    for p, conditions in enumerate(paths):
        if not conditions:
            out[:, p] = 1.0
            continue
        mask = np.ones(n_rows, dtype=bool)
        for feat, thresh, go_left in conditions:
            if go_left:
                mask &= X[:, feat] <= thresh
            else:
                mask &= X[:, feat] > thresh
        out[:, p] = mask.astype(np.float32)
    return out


def compute_tree_path_boolean_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    n_paths=8, baseline_max_depth=4, standardize=True, column_prefix="tpath", dtype=np.float32,
):
    """Train a shallow LightGBM baseline, extract its top-K highest-impact root->leaf decision paths, and emit them as boolean-conjunction features (plus match count and the baseline's own prediction) for the query rows.

    With ``X_query`` given, fits once on all of ``X_train`` and scores ``X_query`` (Mode B, inference). Without it, requires a ``splitter`` and produces out-of-fold features over ``X_train`` itself (Mode A, training-time FE), refitting per fold so no fold leaks its own rows into its baseline.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("tree_path_boolean requires lightgbm") from exc
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = n_paths + 2

    def _process(Xt, Xq, y_t, fold_seed):
        """Fit the shallow baseline booster on ``(Xt, y_t)`` and emit query-row features for ``Xq``: the top-K path boolean matches, their count, and the booster's own prediction."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s, Xq_s = Xt, Xq
        if task == "binary":
            m = lgb.LGBMClassifier(n_estimators=20, max_depth=baseline_max_depth, learning_rate=0.1,
                                   random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t.astype(np.int32))
            pred_q = np.asarray(m.predict_proba(Xq_s))[:, 1].astype(np.float32)
        else:
            m = lgb.LGBMRegressor(n_estimators=20, max_depth=baseline_max_depth, learning_rate=0.1,
                                  random_state=int(fold_seed), verbose=-1, n_jobs=-1).fit(Xt_s, y_t)
            pred_q = np.asarray(m.predict(Xq_s)).astype(np.float32)
        try:
            paths = _extract_top_paths(m.booster_, Xt_s, top_k=n_paths)
        except Exception:
            paths = []
        # Pad to n_paths if fewer extracted
        while len(paths) < n_paths:
            paths.append([])
        bool_matrix = _evaluate_paths(paths[:n_paths], Xq_s)
        n_matched = bool_matrix.sum(axis=1).astype(np.float32)
        return np.column_stack([bool_matrix, n_matched, pred_q])

    def _make_df(feats):
        """Slice the raw ``(n_rows, n_paths+2)`` feature matrix into named, dtype-cast columns for the output frame."""
        cols = {}
        for k in range(n_paths):
            cols[f"{column_prefix}_path{k}"] = feats[:, k].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_matched"] = feats[:, n_paths].astype(dtype, copy=False)
        cols[f"{column_prefix}_baseline_pred"] = feats[:, n_paths + 1].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        return pl.DataFrame(_make_df(_process(X_train_f, Xq, y_train_f, seed)))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100).astype(dtype, copy=False)
        logger.info("tree_path_boolean: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
