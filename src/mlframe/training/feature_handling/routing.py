"""
Sparse / dense routing decision: per-model output kind.

Round-3 plan §4 sparse/dense matrix + CC5: ``scipy.sparse.hstack`` of
TF-IDF (sparse) + transformer (dense) + numeric (dense) at 7M rows
allocates **143 GB** if everything densifies. Two-track concat is
mandatory for sparse-aware models; dim-reduction (TruncatedSVD) is
auto-applied with a WARN for dense-only models when a handler emits
>512 sparse cols.

Public surface:

  * :data:`SPARSE_AWARE_MODELS` -- frozen set of models that accept
    sparse inputs natively (scipy.sparse CSR / CSC -- via DMatrix /
    Dataset / Pool).
  * :data:`DENSE_ONLY_MODELS` -- models that require ``.toarray()``
    (HGB, RF, NGB, MLP, recurrent, TabNet).
  * :func:`accepts_sparse(model_kind)` -- ``model_kind in SPARSE_AWARE``.
  * :func:`should_apply_svd(model_kind, n_sparse_cols, threshold)`
    -- True iff the auto-SVD trigger fires (auto-apply with warning,
    not error).
  * :func:`hgb_max_features_cap(model_kind, requested_max_features)`
    -- enforces HGB's 500-col cap with WARN+cap semantics.
"""

from __future__ import annotations

import logging
from typing import FrozenSet

logger = logging.getLogger(__name__)


# Models with native sparse-matrix support. CB, XGB, LGB take sparse
# via Pool / DMatrix / Dataset; sklearn linear / SGD also accept it
# via the standard sparse handling.
SPARSE_AWARE_MODELS: FrozenSet[str] = frozenset(
    {"cb", "xgb", "lgb", "linear", "ridge", "sgd"}
)

# Dense-only consumers. HGB cannot ingest sparse; RF deprecated sparse
# in 1.0+; NGB / MLP / recurrent / TabNet always operate on dense.
DENSE_ONLY_MODELS: FrozenSet[str] = frozenset(
    {"hgb", "rf", "ngb", "mlp", "recurrent", "tabnet"}
)

# Default trigger threshold for auto-SVD on dense-only models.
DEFAULT_SVD_TRIGGER_NCOLS: int = 512

# HGB-specific cap on TF-IDF max_features. Above this the densified
# matrix at 7M rows blows out RAM regardless of SVD.
HGB_TFIDF_MAX_FEATURES_CAP: int = 500


def accepts_sparse(model_kind: str) -> bool:
    """Returns True iff the model accepts ``scipy.sparse`` matrices
    natively. Round-3 plan §4 sparse/dense compatibility table."""
    return model_kind in SPARSE_AWARE_MODELS


def is_dense_only(model_kind: str) -> bool:
    """Returns True iff the model requires ``.toarray()`` densification
    of any sparse handler output."""
    return model_kind in DENSE_ONLY_MODELS


def should_apply_svd(
    model_kind: str,
    n_sparse_cols: int,
    threshold: int = DEFAULT_SVD_TRIGGER_NCOLS,
) -> bool:
    """Decide whether to auto-apply TruncatedSVD before feeding the
    sparse handler output into a dense-only model.

    Returns True iff:
      * ``model_kind`` is dense-only (otherwise sparse passes through);
      * ``n_sparse_cols`` exceeds ``threshold`` (default 512;
        configurable via per-handler ``svd_dim`` or
        ``svd_globals.default_dim``).

    Round-3 user-confirmed: this is auto-apply WITH WARNING, not
    hard-error. The warning surfaces the rationale + override path.
    """
    return is_dense_only(model_kind) and n_sparse_cols > threshold


def hgb_max_features_cap(
    model_kind: str,
    requested_max_features: int,
    cap: int = HGB_TFIDF_MAX_FEATURES_CAP,
    allow_high: bool = False,
) -> int:
    """HGB-specific TF-IDF cap. Round-3 A18 fix: the previous silent
    cap was a footgun -- user sets 5000, gets 500, performance drops,
    no log line. This now WARNs + caps explicitly. ``allow_high=True``
    skips the cap (escape hatch).
    """
    if model_kind != "hgb":
        return requested_max_features
    if requested_max_features <= cap:
        return requested_max_features
    if allow_high:
        logger.info(
            "[fhc] HGB tfidf cap bypassed via allow_high=True; max_features=%d "
            "(densified TF-IDF on a multi-million-row frame may exhaust RAM)",
            requested_max_features,
        )
        return requested_max_features
    logger.warning(
        "[fhc] HGB is dense-only; TF-IDF max_features=%d would densify into a "
        "%dx%d float32 matrix. Capping to %d for safety. Override: pass "
        "allow_high=True or use a sparse-aware model (cb/xgb/lgb/linear).",
        requested_max_features, requested_max_features, requested_max_features, cap,
    )
    return cap


def emit_svd_warning(
    column: str,
    model_kind: str,
    n_sparse_cols: int,
    svd_dim: int,
    threshold: int = DEFAULT_SVD_TRIGGER_NCOLS,
) -> None:
    """Standardised WARN for the auto-SVD path. Round-3 user-confirmed:
    surface the trigger, the action, and the override knob in one
    line so users see why the SVD applied and how to disable it.
    """
    estimated_dense_gb = (n_sparse_cols * 4 * 1_000_000) / 1e9  # rough at 1M rows fp32
    logger.warning(
        "[fhc] Handler on column %r emits %d sparse cols, but model %r is "
        "dense-only. Auto-applying TruncatedSVD(n_components=%d) to avoid "
        "~%.1f GB densification per million rows. To disable: set "
        "TextHandlerSpec.svd_dim=None and switch to a sparse-aware model. "
        "To customise: TextHandlerSpec.svd_dim=<int>.",
        column, n_sparse_cols, model_kind, svd_dim, estimated_dense_gb,
    )


__all__ = [
    "SPARSE_AWARE_MODELS",
    "DENSE_ONLY_MODELS",
    "DEFAULT_SVD_TRIGGER_NCOLS",
    "HGB_TFIDF_MAX_FEATURES_CAP",
    "accepts_sparse",
    "is_dense_only",
    "should_apply_svd",
    "hgb_max_features_cap",
    "emit_svd_warning",
]
