"""Categorical / embedding-text fit-time preprocessing carved out of ``_base_fit``.

``_FitPrepMixin`` holds the three cohesive pre-network feature-prep methods
(``_encode_emb_text_fit`` / ``_factorize_cats_fit`` / ``_apply_cat_codes``) that
convert raw embedding-list / free-text / categorical columns into the pure-numeric,
cat-codes-leading layout the tabular MLP expects. ``_FitMixin`` inherits from it so
the estimator mixes in the full fit surface unchanged. Operates purely on ``self``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class _FitPrepMixin:
    """Fit-time categorical / embedding-text feature prep for the neural estimator."""

    def _encode_emb_text_fit(self, X, eval_set, fit_params):
        """Expand embedding-``List`` columns + HF-embed free-text columns to numeric for the tabular MLP, which has no
        native embedding/text input layers. The feature-name lists arrive via ``fit_params`` (``embedding_features`` /
        ``text_features``, popped so they never reach Lightning); the fitted encoder is stashed on
        ``self._emb_text_encoder_`` so ``predict`` applies the identical transform. No-op when no such columns are named
        OR when they were already encoded upstream by the strategy pre-pipeline (the encoder skips absent columns), so
        this can't double-encode. Applies to train X and the eval_set val X, for every target type.
        """
        self._emb_text_encoder_ = None
        emb = fit_params.pop("embedding_features", None)
        txt = fit_params.pop("text_features", None)
        named = list(emb or []) + list(txt or [])
        if not named or not hasattr(X, "columns") or not any(c in X.columns for c in named):
            return X, eval_set
        from ..feature_prep import NeuralEmbeddingTextEncoder
        enc = NeuralEmbeddingTextEncoder(embedding_features=list(emb or []), text_features=list(txt or []))
        X = enc.fit_transform(X)
        self._emb_text_encoder_ = enc
        if eval_set is not None:
            _is_list = isinstance(eval_set, list) and len(eval_set) > 0
            _ev = eval_set[0] if _is_list else eval_set
            if isinstance(_ev, tuple) and len(_ev) >= 1 and _ev[0] is not None and hasattr(_ev[0], "columns"):
                _ev_new = (enc.transform(_ev[0]),) + tuple(_ev[1:])
                eval_set = ([_ev_new] + list(eval_set[1:])) if _is_list else _ev_new
        return X, eval_set

    def _factorize_cats_fit(self, X, eval_set, fit_params):
        """Factorize raw categorical columns to integer codes and reorder them leading so the network's ``CategoricalEmbedding`` can index them.

        ``cat_features`` arrives via ``fit_params`` (popped so it never reaches Lightning). For each named column present in X, build a
        ``value -> code`` map (``pandas.factorize`` order, codes ``0..card-1``; the reserved code ``card`` is the unknown/overflow slot used at
        predict for values unseen at fit). The cat-code columns are moved to the FRONT of X (numeric columns trail) as float32 -- the layout
        contract ``CategoricalEmbedding.forward`` expects. Stores ``self._cat_code_maps_`` (dict col -> {value: code}), ``self._cat_cols_`` (the
        fixed leading column order), ``self._cat_cardinalities_`` (per-cat card, same order) and ``self._n_cat_features_`` for predict-time
        replay + the network builder. No-op (cardinalities stays None) when no cat column is named OR present, OR when the learnable-embedding
        knob is off (the strategy's CatBoostEncoder path then handles cats upstream). Runs BEFORE ``_validate_no_nan_inf`` so the validator
        (which rejects object dtype) sees a pure-numeric frame. Mirrors the eval_set handling for the val frame.
        """
        self._cat_code_maps_ = None
        self._cat_cols_ = None
        self._cat_cardinalities_ = None
        self._n_cat_features_ = 0

        cats = fit_params.pop("cat_features", None)
        if not getattr(self, "use_learnable_cat_embeddings", True):
            return X, eval_set
        if not hasattr(X, "columns"):
            return X, eval_set
        present = [c for c in (cats or []) if c in X.columns]
        # With requires_encoding off (learnable embeddings on) the strategy skips the CatBoostEncoder, so raw object/category/string columns
        # reach the estimator. If the suite did not thread an explicit cat_features list, the MLP must still factorize+embed every non-numeric
        # column itself -- otherwise _validate_no_nan_inf rejects the object dtype ("requires numeric dtype"). Union the explicit list with any
        # remaining non-numeric column so no raw categorical slips through to the network (auto-detected cats are embedded just like named ones).
        _auto = [c for c in X.columns if c not in present and getattr(X[c], "ndim", 1) == 1 and not pd.api.types.is_numeric_dtype(X[c])]
        present = present + _auto
        if not present:
            return X, eval_set

        code_maps: dict = {}
        cardinalities: list[int] = []
        # Build code columns WITHOUT mutating the caller's frame (MEMORY.md: never mutate caller X in place -- a 100 GB frame would double
        # RAM). ``X.assign`` returns a new frame sharing untouched columns; only the small cat columns are replaced. We then reorder via a
        # column selection (also a view-backed new frame, no deep copy).
        encoded_cols: dict = {}
        for col in present:
            codes, uniques = pd.factorize(X[col], sort=False)
            # ``factorize`` returns -1 for NaN; remap to the reserved unknown row (``card``) so the embedding sees a valid index, not -1.
            card = int(len(uniques))
            value_to_code = {val: i for i, val in enumerate(uniques)}
            code_maps[col] = value_to_code
            cardinalities.append(card)
            codes = codes.astype(np.float32)
            np.putmask(codes, codes < 0, float(card))
            encoded_cols[col] = codes

        other_cols = [c for c in X.columns if c not in present]
        ordered = present + other_cols
        # Reorder so cat-code columns lead, numeric trail. ``assign`` overlays the encoded cat columns onto a shallow copy, then the selection
        # picks the final order (numeric columns keep their dtype; cats are float32 codes the embedding casts back to long internally).
        X = X.assign(**encoded_cols)[ordered]

        self._cat_code_maps_ = code_maps
        self._cat_cols_ = list(present)
        self._cat_cardinalities_ = cardinalities
        self._n_cat_features_ = len(present)

        if eval_set is not None:
            _is_list = isinstance(eval_set, list) and len(eval_set) > 0
            _ev = eval_set[0] if _is_list else eval_set
            if isinstance(_ev, tuple) and len(_ev) >= 1 and _ev[0] is not None and hasattr(_ev[0], "columns"):
                _ev_X = self._apply_cat_codes(_ev[0])
                _ev_new = (_ev_X,) + tuple(_ev[1:])
                eval_set = ([_ev_new] + list(eval_set[1:])) if _is_list else _ev_new
        return X, eval_set

    def _apply_cat_codes(self, X):
        """Replay the fit-time categorical factorization on a fresh frame (val at fit, or X at predict): map each value through the stored
        ``value -> code`` map (unseen -> the reserved unknown code = the column's cardinality), then reorder cat columns leading as float32.
        No-op when no cat factorization was fitted. Returns a new column-ordered frame (selection view, not a deep copy).
        """
        code_maps = getattr(self, "_cat_code_maps_", None)
        if not code_maps:
            return X
        if not hasattr(X, "columns"):
            return X
        if self._cat_cols_ is None or self._cat_cardinalities_ is None:
            return X
        encoded_cols: dict = {}
        for col, card in zip(self._cat_cols_, self._cat_cardinalities_):
            if col not in X.columns:
                continue
            mapping = code_maps[col]
            # ``.astype(object)`` BEFORE ``.map``: Series.map on a pandas CATEGORICAL column returns a Categorical (it maps the categories and keeps
            # the dtype), so the ``.fillna(card)`` unknown-code fill below would try to add ``card`` as a NEW category and raise "Cannot setitem on a
            # Categorical with a new category". Mapping the plain object values yields a numeric/object Series whose NaN fill (values unseen at fit)
            # lands as the reserved unknown code, never a new category.
            mapped = X[col].astype(object).map(mapping)
            encoded_cols[col] = mapped.fillna(float(card)).astype(np.float32)
        other_cols = [c for c in X.columns if c not in self._cat_cols_]
        ordered = [c for c in self._cat_cols_ if c in X.columns] + other_cols
        return X.assign(**encoded_cols)[ordered]
