"""Learnable categorical-embedding boundary for the recurrent wrappers (tabular block only).

Carved out of ``recurrent.py`` to keep that facade under the 1k-line monolith threshold. ``_RecurrentCatEmbeddingMixin`` holds the fit-boundary
categorical factorization (``_factorize_cats_fit``), the predict-time replay (``_apply_cat_codes`` / ``_prepare_predict_features``), the
numeric-only scaler fit (``_scaler_fit_numeric_only``), and the content-hash predict cache key (``_compute_cache_key``). It operates purely on
``self`` so ``_RecurrentWrapperBase`` mixes it in unchanged; mirrors the flat-MLP ``_FitMixin`` cat-embedding contract but scopes everything to the
TABULAR ``features`` block -- the SEQUENCES are never factorized (they carry no named cat columns) and SEQUENCE_ONLY mode no-ops cleanly.
"""
from __future__ import annotations

import hashlib as _hashlib

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import _ensure_numpy

try:
    import xxhash as _xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False


class _RecurrentCatEmbeddingMixin:
    """Tabular-block categorical-embedding boundary shared by the recurrent sklearn wrappers."""

    def _factorize_cats_fit(self, features, cat_features):
        """Factorize raw categorical columns of the TABULAR ``features`` frame to integer codes and reorder them leading so the network's
        ``CategoricalEmbedding`` (on the aux block) can index them. SEQUENCES are never touched -- they have no named cat columns. ``cat_features``
        is the list of categorical column names; for each present column build a ``value -> code`` map (``pandas.factorize`` order, codes
        ``0..card-1``; the reserved code ``card`` is the unknown/overflow slot used at predict). The cat-code columns move to the FRONT of the
        returned frame (numerics trail) as float32 -- the layout contract ``CategoricalEmbedding.forward`` expects. Stores ``_cat_code_maps_`` /
        ``_cat_cols_`` / ``_cat_cardinalities_`` / ``_n_cat_features_`` for predict-time replay + network construction. No-op (cardinalities stays
        None) when the knob is off, no cat column is named/present, or ``features`` is not a DataFrame. Returns the (possibly reordered) frame.
        """
        self._cat_code_maps_ = None
        self._cat_cols_ = None
        self._cat_cardinalities_ = None
        self._n_cat_features_ = 0

        if not getattr(self, "use_learnable_cat_embeddings", True):
            return features
        if not cat_features or features is None or not hasattr(features, "columns"):
            return features
        present = [c for c in cat_features if c in features.columns]
        if not present:
            return features

        import pandas as _pd

        code_maps: dict = {}
        cardinalities: list[int] = []
        encoded_cols: dict = {}
        for col in present:
            codes, uniques = _pd.factorize(features[col], sort=False)
            card = len(uniques)
            code_maps[col] = {val: i for i, val in enumerate(uniques)}
            cardinalities.append(card)
            codes = codes.astype(np.float32)
            # ``factorize`` returns -1 for NaN; route to the reserved unknown row (``card``) so the embedding sees a valid index, not -1.
            np.putmask(codes, codes < 0, float(card))
            encoded_cols[col] = codes

        other_cols = [c for c in features.columns if c not in present]
        ordered = present + other_cols
        # ``assign`` overlays the encoded cat columns onto a shallow copy (no deep copy of the whole frame); the selection picks the final order.
        features = features.assign(**encoded_cols)[ordered]

        self._cat_code_maps_ = code_maps
        self._cat_cols_ = list(present)
        self._cat_cardinalities_ = cardinalities
        self._n_cat_features_ = len(present)
        return features

    def _apply_cat_codes(self, features):
        """Replay the fit-time categorical factorization on a fresh tabular frame (val at fit, or features at predict): map each value through
        the stored ``value -> code`` map (unseen -> the reserved unknown code = the column's cardinality), then reorder cat columns leading as
        float32. No-op when no cat factorization was fitted or ``features`` is not a DataFrame. Returns the reordered frame.
        """
        code_maps = getattr(self, "_cat_code_maps_", None)
        if not code_maps or features is None or not hasattr(features, "columns"):
            return features
        if self._cat_cols_ is None or self._cat_cardinalities_ is None:
            return features
        encoded_cols: dict = {}
        for col, card in zip(self._cat_cols_, self._cat_cardinalities_):
            if col not in features.columns:
                continue
            mapped = features[col].map(code_maps[col])
            encoded_cols[col] = mapped.fillna(float(card)).astype(np.float32)
        other_cols = [c for c in features.columns if c not in self._cat_cols_]
        ordered = [c for c in self._cat_cols_ if c in features.columns] + other_cols
        return features.assign(**encoded_cols)[ordered]

    def _prepare_predict_features(self, features):
        """Predict-time tabular prep: replay the cat factorization (reorder cat-codes leading; unseen -> reserved code), then coerce the result
        to a contiguous float32 ndarray. Returning ndarray here means ``_compute_cache_key`` + ``_create_dataset`` see a pure-numeric array (the
        cache key reads ``.dtype`` which a DataFrame lacks). No-op shape-wise when no cat factorization was fitted; ``None`` passes through.
        """
        if features is None:
            return None
        features = self._apply_cat_codes(features)
        arr = _ensure_numpy(features)
        if arr is None:
            return None
        if isinstance(arr, np.ndarray) and arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
            return arr
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _scaler_fit_numeric_only(self, features) -> None:
        """Fit ``self._feature_scaler`` on the NUMERIC block only when learnable cat embeddings are active, leaving the leading cat-code columns
        unscaled (scaling integer codes would corrupt them as embedding indices). The scaler is fit on the trailing ``aux_input - n_cat`` numeric
        columns; ``_create_dataset`` then scales only those at transform time. When no cats were factorized this is a plain full fit.
        """
        scaled = _ensure_numpy(features)
        if scaled is None:
            self._feature_scaler = None
            return
        self._feature_scaler = StandardScaler()
        n_cat = self._n_cat_features_ if self._cat_cardinalities_ else 0
        if n_cat > 0 and scaled.shape[1] > n_cat:
            self._feature_scaler.fit(scaled[:, n_cat:])
        elif n_cat > 0:
            # All columns are cat codes -- nothing numeric to scale; disable the scaler so transform is a clean passthrough.
            self._feature_scaler = None
        else:
            self._feature_scaler.fit(scaled)

    @staticmethod
    def _compute_cache_key(
        features: np.ndarray | None,
        sequences: list[np.ndarray] | None,
    ) -> bytes:
        """Compute content-hash cache key from input arrays.

        Sampling 3 scalars + shape and hashing the tuple-of-str was
        collision-prone: any two predict-batches that agreed on (shape,
        dtype, first/middle/last value) returned the cached prediction of
        the first, which silently mis-predicted on near-duplicate inputs.

        Hash full ``tobytes()`` payload (xxhash if available, blake2b
        otherwise) keyed on shape+dtype to make sub-cell changes always
        invalidate the cache.
        """
        # Module-top imports (hashlib + optional xxhash) keep the predict-hot
        # path import-free; the previous try/except ImportError ran on EVERY
        # predict call. xxhash is ~5x faster than blake2b on tobytes payloads.
        if _HAS_XXHASH:
            _hasher = _xxhash.xxh3_128()
        else:
            _hasher = _hashlib.blake2b(digest_size=16)
        _update = _hasher.update
        _digest = _hasher.digest

        if features is not None:
            _update(b"FEAT")
            _update(str(features.shape).encode())
            _update(features.dtype.str.encode())
            _arr = np.ascontiguousarray(features)
            _update(_arr.tobytes())
        if sequences is not None:
            _update(b"SEQ")
            _update(str(len(sequences)).encode())
            for seq in sequences:
                _arr = np.ascontiguousarray(seq)
                _update(str(seq.shape).encode())
                _update(seq.dtype.str.encode())
                _update(_arr.tobytes())
        return bytes(_digest())
