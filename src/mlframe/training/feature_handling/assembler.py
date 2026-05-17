"""
Multi-handler concat assembler.

Round-3 plan §5 + A8: insertion order locks the column-name order in
the final feature matrix; same column + same method on the same axis
raises at config validation; same column + DIFFERENT methods is
allowed and produces disambiguated names like ``text_a__tfidf__token_word``
+ ``text_a__hash__bucket_42``.

Phase E v1: takes a list of fitted handler outputs (sparse / dense /
embedding) + a target model_kind, returns one of:

  * For sparse-aware models -- a tuple ``(sparse_block, dense_block)``
    so the consumer (``train_and_evaluate_model``) can pass the
    sparse block directly into XGB DMatrix / LGB Dataset / sklearn
    linear without densification.
  * For dense-only models -- a single dense ``np.ndarray`` after
    auto-SVD on any sparse blocks > 512 cols.

Auto-SVD parameters are per-handler (``TextHandlerSpec.svd_dim``)
with a global default in ``FeatureHandlingConfig.svd_globals.default_dim``
(phase A; default 256).

Output column-name disambiguation is done at construct time for
sparse blocks; dense blocks get the prefix prepended in-name. The
final feature-name vector is exposed via :func:`assembled_column_names`
for SHAP / feature-importance attribution downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from sklearn.decomposition import TruncatedSVD

from mlframe.training.feature_handling.routing import (
    DEFAULT_SVD_TRIGGER_NCOLS,
    accepts_sparse,
    emit_svd_warning,
    is_dense_only,
    should_apply_svd,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import scipy.sparse as sp  # noqa: F401


# =====================================================================
# Block descriptor
# =====================================================================


@dataclass
class HandlerOutput:
    """One fitted handler's contribution to the final matrix.

    ``data`` is one of: ``np.ndarray``, ``scipy.sparse.csr_matrix``.
    ``column`` + ``method`` together produce the disambiguating prefix
    ``"{column}__{method_short}__"`` so SHAP / FI attribution stays
    sane when multiple handlers fire on the same column.
    """
    column: str  # source column name
    method: str  # handler method (tfidf / hashing / frozen_text_embedding / ...)
    data: Any
    n_features: int
    output_kind: Literal["sparse", "dense", "embedding"]
    feature_names: Optional[List[str]] = None  # None => auto-generate

    def name_prefix(self) -> str:
        """Disambiguating prefix for column names. Round-3 A8 fix."""
        method_short = {
            "tfidf": "tfidf",
            "hashing": "hash",
            "frozen_text_embedding": "frozen_emb",
            "learnable_text_embedding": "learn_emb",
            "ordinal": "ord",
            "onehot": "onehot",
            "embedding": "emb",
            "target_mean": "tgt_mean",
            "target_m_estimate": "tgt_m",
            "target_james_stein": "tgt_js",
            "target_loo": "tgt_loo",
            "woe": "woe",
        }.get(self.method, self.method)
        return f"{self.column}__{method_short}"

    def expanded_feature_names(self) -> List[str]:
        """Expand to a list of length ``n_features`` with the
        disambiguating prefix applied.
        """
        prefix = self.name_prefix()
        if self.feature_names is not None:
            return [f"{prefix}__{n}" for n in self.feature_names]
        # Auto-generate by index.
        return [f"{prefix}__{i}" for i in range(self.n_features)]


# =====================================================================
# Assembled output
# =====================================================================


@dataclass
class AssembledMatrix:
    """Final assembly result.

    For sparse-aware models: ``sparse_block`` carries the (potentially
    very wide) TF-IDF / hashing matrix; ``dense_block`` carries
    transformer embeddings + numeric. Consumer hands them to the
    model-native sparse+dense API.

    For dense-only models: ``sparse_block`` is None (everything got
    auto-SVD-reduced and concatenated into ``dense_block``).

    ``feature_names`` always carries the FULL list of names in the
    final concatenated order (sparse_block names first, then dense_block).
    """
    sparse_block: Optional[sp.csr_matrix]
    dense_block: Optional[np.ndarray]
    feature_names: List[str]
    routing_applied: List[str]  # human-readable trace for debugging

    @property
    def is_two_track(self) -> bool:
        return self.sparse_block is not None and self.dense_block is not None

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# =====================================================================
# SVD with WARN
# =====================================================================


def _apply_svd(
    sparse_mat: sp.csr_matrix,
    n_components: int,
    column: str,
    model_kind: str,
    random_state: int = 0,
) -> Tuple[np.ndarray, Any]:
    """Apply ``TruncatedSVD`` to a sparse matrix. Returns
    ``(dense_array, fitted_svd)`` so callers can cache the fitted
    object via :class:`FeatureCache` keyed on ``(column,
    train_idx_sig, sparse_dim, svd_dim, random_state)``.

    Round-3 R3-08 fix: pinned ``n_iter=5`` + ``power_iteration_normalizer="auto"``
    so cross-version sklearn doesn't drift the SVD basis on cache hit.
    """
    n_components_actual = min(n_components, max(1, sparse_mat.shape[1] - 1))

    emit_svd_warning(
        column=column, model_kind=model_kind,
        n_sparse_cols=sparse_mat.shape[1], svd_dim=n_components_actual,
    )
    svd = TruncatedSVD(
        n_components=n_components_actual,
        random_state=random_state,
        n_iter=5,
        power_iteration_normalizer="auto",
    )
    reduced = svd.fit_transform(sparse_mat).astype(np.float32)
    return reduced, svd


# =====================================================================
# Assembler
# =====================================================================


def assemble_for_model(
    blocks: Sequence[HandlerOutput],
    *,
    model_kind: str,
    numeric_block: Optional[np.ndarray] = None,
    numeric_feature_names: Optional[List[str]] = None,
    svd_default_dim: int = 256,
    svd_trigger_ncols: int = DEFAULT_SVD_TRIGGER_NCOLS,
    per_handler_svd_dim: Optional[Dict[str, int]] = None,
) -> AssembledMatrix:
    """Concatenate handler outputs for a specific model.

    Routing logic:

    1. Validate same-column + same-method overlap raises.
    2. Group blocks by output_kind: sparse / dense / embedding.
    3. If model is sparse-aware: hstack all sparse blocks into ONE
       sparse matrix; concat all dense + embedding blocks; output
       two-track matrix.
    4. If model is dense-only: for each sparse block, decide if SVD
       fires (n_features > svd_trigger_ncols) -> reduce to dense;
       otherwise ``.toarray()`` densify in place. Then concat
       everything into a single dense matrix.

    ``numeric_block`` is the upstream-derived numeric feature matrix
    (post-imputer / scaler from PreprocessingBackendConfig). Its
    column names go into ``feature_names`` UNCHANGED (no prefix
    disambiguation -- they're already user-facing).

    ``per_handler_svd_dim`` maps ``"{column}__{method}"`` to a custom
    SVD dim, overriding ``svd_default_dim``.
    """
    import scipy.sparse as sp

    # ---- 1. Validate overlap (same col + same method = error) ---
    seen = set()
    for b in blocks:
        sig = (b.column, b.method)
        if sig in seen:
            raise ValueError(
                f"duplicate handler on column={b.column!r} method={b.method!r}; "
                f"same-column + same-method handlers are not allowed."
            )
        seen.add(sig)

    sparse_blocks = [b for b in blocks if b.output_kind == "sparse"]
    dense_blocks = [b for b in blocks if b.output_kind == "dense"]
    embedding_blocks = [b for b in blocks if b.output_kind == "embedding"]
    routing_trace: List[str] = []

    if model_kind in ("cb", "xgb", "lgb") or accepts_sparse(model_kind):
        # ---- Two-track concat -------------------------------------
        sparse_mat: Optional[sp.csr_matrix] = None
        sparse_names: List[str] = []
        if sparse_blocks:
            stacked = sp.hstack(
                [b.data.tocsr() for b in sparse_blocks], format="csr",
            )
            sparse_mat = stacked
            for b in sparse_blocks:
                sparse_names.extend(b.expanded_feature_names())
            routing_trace.append(
                f"sparse_block: {len(sparse_blocks)} handlers -> "
                f"{stacked.shape[1]} cols"
            )
        dense_parts: List[np.ndarray] = []
        dense_names: List[str] = []
        for b in dense_blocks + embedding_blocks:
            arr = b.data
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            dense_parts.append(arr)
            dense_names.extend(b.expanded_feature_names())
            routing_trace.append(f"dense_block: {b.column!r} {b.method!r} {arr.shape}")
        if numeric_block is not None:
            dense_parts.append(numeric_block)
            if numeric_feature_names is not None:
                dense_names.extend(numeric_feature_names)
            else:
                dense_names.extend(
                    [f"num__{i}" for i in range(numeric_block.shape[1])]
                )
        if dense_parts:
            dense_mat = np.concatenate(dense_parts, axis=1)
        else:
            dense_mat = None

        all_names = sparse_names + dense_names
        return AssembledMatrix(
            sparse_block=sparse_mat,
            dense_block=dense_mat,
            feature_names=all_names,
            routing_applied=routing_trace,
        )

    # ---- Single-track DENSE concat (with auto-SVD on big sparse) -
    # is_dense_only(model_kind) OR unknown model -> safest path.
    dense_parts: List[np.ndarray] = []
    all_names: List[str] = []

    for b in sparse_blocks:
        per_key = per_handler_svd_dim.get(f"{b.column}__{b.method}") if per_handler_svd_dim else None
        svd_dim = per_key if per_key is not None else svd_default_dim
        if should_apply_svd(model_kind, b.n_features, svd_trigger_ncols):
            reduced, _svd = _apply_svd(
                b.data.tocsr(), svd_dim,
                column=b.column, model_kind=model_kind,
            )
            dense_parts.append(reduced)
            # Renamed: <prefix>__svd<dim>__<i>
            prefix = b.name_prefix()
            all_names.extend([f"{prefix}__svd{reduced.shape[1]}__{i}" for i in range(reduced.shape[1])])
            routing_trace.append(
                f"svd: {b.column!r} {b.method!r} {b.n_features} -> {reduced.shape[1]}"
            )
        else:
            arr = b.data.toarray().astype(np.float32, copy=False)
            dense_parts.append(arr)
            all_names.extend(b.expanded_feature_names())
            routing_trace.append(
                f"densify: {b.column!r} {b.method!r} {b.n_features} cols (under SVD trigger)"
            )

    for b in dense_blocks + embedding_blocks:
        arr = b.data
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        dense_parts.append(arr)
        all_names.extend(b.expanded_feature_names())
        routing_trace.append(f"dense_pass: {b.column!r} {b.method!r} {arr.shape}")

    if numeric_block is not None:
        dense_parts.append(numeric_block)
        if numeric_feature_names is not None:
            all_names.extend(numeric_feature_names)
        else:
            all_names.extend([f"num__{i}" for i in range(numeric_block.shape[1])])

    if not dense_parts:
        # Empty assembly -- caller decides what to do.
        return AssembledMatrix(
            sparse_block=None, dense_block=None,
            feature_names=[], routing_applied=routing_trace,
        )

    dense_mat = np.concatenate(dense_parts, axis=1)
    return AssembledMatrix(
        sparse_block=None,
        dense_block=dense_mat,
        feature_names=all_names,
        routing_applied=routing_trace,
    )


def assembled_column_names(asm: AssembledMatrix) -> List[str]:
    """Public alias for :attr:`AssembledMatrix.feature_names`."""
    return list(asm.feature_names)


__all__ = [
    "HandlerOutput",
    "AssembledMatrix",
    "assemble_for_model",
    "assembled_column_names",
]
