"""Formal Concept Analysis closed itemsets as equivalence-class features.

Iter 100 mechanism. Agent A #5 ranked. Diabetes small balanced binary target.

Discretize X to boolean attributes (above/below median per feature), compute top-K closed concepts
via `concepts` library, emit membership in each concept's extent as a categorical feature.
"""
from __future__ import annotations
import logging
import numpy as np
import polars as pl
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_fca_closed_concepts_features(
    X_train, y_train, X_query=None, splitter=None, *, seed, task="regression",
    top_k=8, standardize=True, column_prefix="fca", dtype=np.float32,
):
    """Formal Concept Analysis membership features: emits per-query indicators for the top-K closed concepts
    (by extent size) of a boolean context built from median-thresholded train attributes, plus n_matched
    and n_concepts summary columns. top_k + 2 features total.
    """
    try:
        import concepts as _concepts  # noqa: F401 -- probe import to fail fast with a clear error if concepts is missing
    except ImportError as exc:
        raise ImportError("fca_closed_concepts requires concepts library") from exc

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = top_k + 2

    def _process(Xt, Xq, y_t):
        """Build the FCA context from a 200-row subsample of Xt, extract the top-K closed concepts, and score the query rows against each concept's intent."""
        d = Xt.shape[1]
        medians = np.median(Xt, axis=0)
        # Boolean attributes: x_j > median(x_j)
        bool_train = (Xt > medians).astype(bool)
        bool_query = (Xq > medians).astype(bool)
        # For practicality on N=4000: sample down to 200 train rows for FCA (concept lattice is exponential)
        n_sample = min(200, Xt.shape[0])
        rng = np.random.default_rng(int(seed))
        sample_idx = rng.choice(Xt.shape[0], n_sample, replace=False)
        # Build context for `concepts` lib: objects=row indices, properties=feature_above_median
        objects = [f"r{i}" for i in range(n_sample)]
        properties = [f"f{j}" for j in range(d)]
        bools = bool_train[sample_idx]
        try:
            # concepts.Context expects rows = strings, cols = strings, with boolean matrix
            from concepts import Context
            ctx_data = [tuple(row) for row in bools.astype(bool).tolist()]
            ctx = Context(objects, properties, ctx_data)
            lattice = ctx.lattice
            # Concepts: take top K by extent size (most common)
            all_concepts = [(c.extent, c.intent) for c in lattice if 0 < len(c.extent) < n_sample]
            # Secondary content key (the intent tuple) so equal-extent-size concepts break ties by content,
            # not by the `concepts` lib's lattice iteration order -- otherwise the top_k selection could
            # depend on a library-internal ordering rather than the data.
            all_concepts.sort(key=lambda x: (-len(x[0]), tuple(sorted(x[1]))))
            top_concepts = all_concepts[:top_k]
        except Exception as exc:
            logger.info("fca_closed_concepts: lattice construction failed (%s); falling back to no concepts.", exc)
            top_concepts = []
        # Evaluate concept-membership on query rows (does query row satisfy concept's intent?)
        top_indicators_q = np.zeros((Xq.shape[0], top_k), dtype=np.float32)
        for k, (_extent, intent) in enumerate(top_concepts):
            if not intent:
                top_indicators_q[:, k] = 1.0
                continue
            intent_features = [properties.index(p) for p in intent]
            mask = np.all(bool_query[:, intent_features], axis=1).astype(np.float32)
            top_indicators_q[:, k] = mask
        n_matched = top_indicators_q.sum(axis=1).astype(np.float32)
        return np.column_stack([top_indicators_q, n_matched, np.full(Xq.shape[0], float(len(top_concepts)), dtype=np.float32)])

    def _make_df(feats):
        """Assign column names to the per-concept indicator, n_matched, and n_concepts feature slots."""
        cols = {}
        for k in range(top_k):
            cols[f"{column_prefix}_concept{k}"] = feats[:, k].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_matched"] = feats[:, top_k].astype(dtype, copy=False)
        cols[f"{column_prefix}_n_concepts"] = feats[:, top_k + 1].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        return pl.DataFrame(_make_df(_process(X_train_f, Xq, y_train_f)))
    if splitter is None:
        raise ValueError("Mode A requires splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features_out), dtype=dtype)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_train_f)):
        out[val_idx] = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx]).astype(dtype, copy=False)
        logger.info("fca_closed_concepts: fold %d done", fold_idx + 1)
    return pl.DataFrame(_make_df(out))
