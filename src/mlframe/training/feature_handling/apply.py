"""
End-to-end integration helper: ``feature_handling_apply``.

The single callable that bridges a :class:`FeatureHandlingConfig`,
a dataset, and a target model into the assembled feature matrix the
model.fit() / .predict() path consumes. It's the migration seam --
existing consumer code in :mod:`mlframe.training.core` continues to
work unchanged; consumers that opt into FHC route through this
helper.

The helper:

  1. Auto-detects text columns (multi-criteria + anti-UUID).
  2. Resolves the per-model effective handler chain (defaults +
     overrides + append).
  3. Validates the chain against the model's compat matrix entry
     - raises a single combined :class:`ValueError` listing every
     mismatch.
  4. Fits each handler on train rows; transforms train/val/test;
     hands the outputs to :func:`assemble_for_model`.
  5. Wires through :class:`FeatureCache` so the same handler outputs
     are reused across multiple consumer fits within one
     ``train_mlframe_models_suite`` call; in-memory cache uses cheap
     session-keyed identity, no content hashing in hot path.

Phase Q is the final foundation piece. Concrete model-specific
wiring (CB ``embedding_features=`` plumbing in phase F, MLP-as-
``TabularInputEncoder`` in phase G, etc) builds on top of this
helper rather than replacing it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from mlframe.training.feature_handling.assembler import (
    AssembledMatrix,
    HandlerOutput,
    assemble_for_model,
)
from mlframe.training.feature_handling.cache import FeatureCache
from mlframe.training.feature_handling.config import FeatureHandlingConfig
from mlframe.training.feature_handling.fingerprint import (
    InMemoryKey,
    canonical_params_hash,
    current_session,
)
from mlframe.training.feature_handling.handlers import (
    CatHandlerSpec,
    HashingParams,
    TextHandlerSpec,
    TfidfParams,
)
from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder
from mlframe.training.feature_handling.text_column_encoder_alias import (  # see below
    build_text_encoder,
)
from mlframe.training.feature_handling.text_detection import (
    TextDetectionDecision,
    detect_text_columns,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401
    import polars as pl  # noqa: F401


# =====================================================================
# Result dataclass
# =====================================================================


@dataclass
class FeatureHandlingResult:
    """Output of :func:`feature_handling_apply`. Carries the assembled
    matrices for train / val / test (val/test optional) plus the
    decisions trace for ``fhc.describe()``-style introspection.
    """
    train: AssembledMatrix
    val: Optional[AssembledMatrix] = None
    test: Optional[AssembledMatrix] = None
    text_columns_detected: List[str] = field(default_factory=list)
    detection_decisions: List[TextDetectionDecision] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)


# =====================================================================
# Public API
# =====================================================================


def feature_handling_apply(
    *,
    train_df: Any,
    val_df: Optional[Any] = None,
    test_df: Optional[Any] = None,
    train_target: Optional[Any] = None,
    fhc: FeatureHandlingConfig,
    model_kind: str,
    cache: Optional[FeatureCache] = None,
    candidate_text_columns: Optional[List[str]] = None,
    candidate_cat_columns: Optional[List[str]] = None,
    numeric_block_train: Optional[np.ndarray] = None,
    numeric_block_val: Optional[np.ndarray] = None,
    numeric_block_test: Optional[np.ndarray] = None,
    numeric_feature_names: Optional[List[str]] = None,
) -> FeatureHandlingResult:
    """Apply a ``FeatureHandlingConfig`` to a dataset for a specific
    model.

    Parameters
    ----------
    train_df : polars.DataFrame | pandas.DataFrame
        The training data. Required.
    val_df, test_df : optional held-out frames; transformed via the
        train-fitted handlers (no leak).
    train_target : optional 1-D target vector. Required when any
        cat handler in the resolved chain is a ``target_*`` method
        (target_mean / m_estimate / james_stein / loo / woe).
    fhc : :class:`FeatureHandlingConfig`
        The user's config. The helper reads ``per_model[model_kind]``,
        the defaults, and the sub-configs.
    model_kind : str
        Routing target. Determines:
          * which handler-method validations run;
          * sparse-aware vs dense-only assembly path;
          * auto-SVD trigger for >512 sparse cols.
    cache : optional :class:`FeatureCache` instance
        When supplied, fitted-handler outputs are looked up / stored
        keyed on (session_id, df_id, train_idx, column, params, provider).
        Cache survives across multiple model fits within one
        ``train_mlframe_models_suite`` call.
    candidate_text_columns : optional explicit list to skip the text
        auto-detector. When ``None``, the helper auto-detects via
        :func:`detect_text_columns`.
    candidate_cat_columns : explicit list of cat columns. When ``None``,
        NO cat detection happens and the cat handler chain runs over zero
        columns (silently dropping any ``target_mean`` / WoE / etc handlers
        the FHC configured). Pre-2026-05-20 the docstring claimed symmetric
        auto-detect with text but the implementation was "phase Q v1 --
        caller passes explicit cat cols". Documenting the asymmetry so the
        caller doesn't get a silent feature drop when they relied on the
        symmetric promise. Pass an explicit list (e.g. from a polars
        ``cs.string() / cs.categorical()`` selector or a pandas
        ``select_dtypes(['category','object']).columns`` snapshot) to enable
        cat handlers; a symmetric ``detect_cat_columns`` analog is a
        follow-up item.
    numeric_block_* : optional pre-built numeric matrices (post-imputer
        / scaler from the legacy pipeline). Concatenated to the
        assembled matrix as the "numeric" block.

    Returns
    -------
    :class:`FeatureHandlingResult` with train/val/test assembled
    matrices + the auto-detection decisions trace + final
    ``feature_names``.

    Raises
    ------
    ValueError
        If the resolved handler chain is incompatible with
        ``model_kind`` (see :func:`validate_against_models`).
    """
    fhc.validate_against_models([model_kind])

    # ---- 1. Detect or accept text columns -----------------------
    if candidate_text_columns is None:
        text_cols, decisions = detect_text_columns(
            train_df, config=fhc.text_detection,
        )
    else:
        text_cols = list(candidate_text_columns)
        decisions = []

    # ---- 2. Resolve effective handler chain --------------------
    text_specs = fhc._effective_text_specs(model_kind)
    cat_specs = fhc._effective_cat_specs(model_kind)

    # Filter chain by which handlers actually have target columns.
    # text_specs apply to text_cols (or the spec's apply_to_columns
    # restriction); cat_specs to cat_cols.
    if candidate_cat_columns is None:
        cat_cols = []  # phase Q v1 -- caller passes explicit cat cols
    else:
        cat_cols = list(candidate_cat_columns)

    # ---- 3. Run handlers on each column, collect blocks --------
    train_blocks: List[HandlerOutput] = []
    val_blocks: List[HandlerOutput] = []
    test_blocks: List[HandlerOutput] = []

    sess = current_session()
    train_id = id(train_df)

    # Text axis
    for spec in text_specs:
        target_cols = spec.apply_to_columns or text_cols
        for col in target_cols:
            if spec.method == "drop":
                continue
            if spec.method == "native":
                # Defer to model-native consumer (e.g. CB text_features).
                # Phase F wires this into the CB-specific consumer path;
                # phase Q hands back the column as-is via a passthrough
                # block so the caller can route it natively.
                continue

            # TF-IDF / Hashing path (phase C TextColumnEncoder)
            if spec.method in ("tfidf", "hashing"):
                params = spec.params if not isinstance(spec.params, type(None)) else None
                tr_block, val_block, te_block = _apply_text_encoder(
                    train_df=train_df, val_df=val_df, test_df=test_df,
                    column=col, params=params,
                    cache=cache, session_id=sess.session_id,
                    train_id=train_id,
                )
                if tr_block is not None:
                    train_blocks.append(tr_block)
                if val_block is not None:
                    val_blocks.append(val_block)
                if te_block is not None:
                    test_blocks.append(te_block)
                continue

            if spec.method == "frozen_text_embedding":
                # Phase B HuggingFaceProvider path. Defer to phase F/G
                # for the actual end-to-end inference; phase Q docs the
                # path but doesn't run inference here (heavy + needs
                # provider lifecycle setup).
                logger.info(
                    "[fhc] frozen_text_embedding on %r deferred to phase F/G consumer wiring",
                    col,
                )
                continue

            if spec.method == "learnable_text_embedding":
                # Same: this lives in TabularInputEncoder (phase G)
                # because it's an nn.Module not an offline transform.
                logger.info(
                    "[fhc] learnable_text_embedding on %r deferred to phase G",
                    col,
                )
                continue

            if spec.method == "custom":
                # Phase P CustomHandler path -- wires through here.
                from mlframe.training.feature_handling.custom_handler import (
                    CustomHandler,
                )
                handler = CustomHandler(column=col, params=spec.params)
                handler.fit(train_df, train_target)
                tr_out = handler.transform(train_df)
                tr_block = HandlerOutput(
                    column=col, method="custom", data=tr_out,
                    n_features=tr_out.shape[1] if hasattr(tr_out, "shape") else 0,
                    output_kind=spec.params.output_kind,
                )
                train_blocks.append(tr_block)
                if val_df is not None:
                    v_out = handler.transform(val_df)
                    val_blocks.append(HandlerOutput(
                        column=col, method="custom", data=v_out,
                        n_features=v_out.shape[1] if hasattr(v_out, "shape") else 0,
                        output_kind=spec.params.output_kind,
                    ))
                if test_df is not None:
                    t_out = handler.transform(test_df)
                    test_blocks.append(HandlerOutput(
                        column=col, method="custom", data=t_out,
                        n_features=t_out.shape[1] if hasattr(t_out, "shape") else 0,
                        output_kind=spec.params.output_kind,
                    ))
                continue

    # Cat axis (target encoders only -- ordinal/onehot/native handled
    # elsewhere in the existing pipeline path)
    for spec in cat_specs:
        target_cols = spec.apply_to_columns or cat_cols
        for col in target_cols:
            if spec.method.startswith("target_") or spec.method == "woe":
                if train_target is None:
                    raise ValueError(
                        f"target encoder method={spec.method!r} on column "
                        f"{col!r} requires train_target argument; got None."
                    )
                # Target encoders are single-output by construction. Multi-output (multilabel /
                # multi-regression) inputs slip through downstream sklearn's ``np.asarray(list(y))`` with
                # shape (n, k); the LeakageSafeEncoder then crashes with an opaque length-mismatch trace.
                # Raise here with the actual shape so the wiring bug is visible at the call site.
                _tt = train_target
                _shape = None
                if hasattr(_tt, "shape"):
                    _shape = _tt.shape
                elif hasattr(_tt, "ndim"):
                    _shape = (len(_tt),) if _tt.ndim == 1 else None
                if _shape is not None and len(_shape) > 1 and _shape[1] > 1:
                    raise ValueError(
                        f"target encoder method={spec.method!r} on column {col!r}: train_target is "
                        f"multi-output (shape={_shape}); collapse via mean / pick-target before passing in."
                    )
                tr_block, val_block, te_block = _apply_target_encoder(
                    train_df=train_df, val_df=val_df, test_df=test_df,
                    column=col, params=spec.params,
                    train_target=train_target,
                    cache=cache, session_id=sess.session_id,
                    train_id=train_id,
                )
                train_blocks.append(tr_block)
                if val_block is not None:
                    val_blocks.append(val_block)
                if te_block is not None:
                    test_blocks.append(te_block)

    # ---- 4. Assemble per model_kind ----------------------------
    train_asm = assemble_for_model(
        train_blocks, model_kind=model_kind,
        numeric_block=numeric_block_train,
        numeric_feature_names=numeric_feature_names,
        svd_default_dim=256,
    )
    val_asm = (
        assemble_for_model(
            val_blocks, model_kind=model_kind,
            numeric_block=numeric_block_val,
            numeric_feature_names=numeric_feature_names,
            svd_default_dim=256,
        ) if val_df is not None else None
    )
    test_asm = (
        assemble_for_model(
            test_blocks, model_kind=model_kind,
            numeric_block=numeric_block_test,
            numeric_feature_names=numeric_feature_names,
            svd_default_dim=256,
        ) if test_df is not None else None
    )

    return FeatureHandlingResult(
        train=train_asm,
        val=val_asm,
        test=test_asm,
        text_columns_detected=text_cols,
        detection_decisions=decisions,
        feature_names=list(train_asm.feature_names),
    )


# =====================================================================
# Internal: text encoder
# =====================================================================


def _apply_text_encoder(
    *,
    train_df: Any,
    val_df: Optional[Any],
    test_df: Optional[Any],
    column: str,
    params,
    cache: Optional[FeatureCache],
    session_id: str,
    train_id: int,
) -> Tuple[Optional[HandlerOutput], Optional[HandlerOutput], Optional[HandlerOutput]]:
    """Fit a TextColumnEncoder on train, transform train/val/test.

    Cache key: (session_id, train_id, column, params_hash) so multiple
    models in the same suite call reuse the same fitted encoder.
    """
    method = "tfidf" if isinstance(params, TfidfParams) else "hashing"

    if cache is not None:
        key = InMemoryKey(
            session_id=session_id,
            df_token=train_id,
            # Use a column-content-derived token so cache keys discriminate different OD masks / row
            # subsets even when ``train_id`` (id(df)) coincides. Literal ``0`` collided across OD-masked
            # vs full-train fits (different rows, same df identity).
            train_idx_token=_text_column_content_token(train_df, column),
            column=column,
            params_canonical_hash=canonical_params_hash(params),
            provider_signature=f"text_encoder:{method}",
        )

        def _fit():
            enc = build_text_encoder(column=column, params=params)
            enc.fit(train_df)
            return enc

        encoder = cache.get_or_compute(key, _fit)
    else:
        encoder = build_text_encoder(column=column, params=params)
        encoder.fit(train_df)

    train_mat = encoder.transform(train_df)
    train_block = HandlerOutput(
        column=column, method=method, data=train_mat,
        n_features=train_mat.shape[1], output_kind="sparse",
    )
    val_block = None
    if val_df is not None:
        val_mat = encoder.transform(val_df)
        val_block = HandlerOutput(
            column=column, method=method, data=val_mat,
            n_features=val_mat.shape[1], output_kind="sparse",
        )
    test_block = None
    if test_df is not None:
        test_mat = encoder.transform(test_df)
        test_block = HandlerOutput(
            column=column, method=method, data=test_mat,
            n_features=test_mat.shape[1], output_kind="sparse",
        )
    return train_block, val_block, test_block


# =====================================================================
# Internal: target encoder
# =====================================================================


def _apply_target_encoder(
    *,
    train_df: Any,
    val_df: Optional[Any],
    test_df: Optional[Any],
    column: str,
    params,
    train_target: Any,
    cache: Optional[FeatureCache],
    session_id: str,
    train_id: int,
) -> Tuple[HandlerOutput, Optional[HandlerOutput], Optional[HandlerOutput]]:
    """Fit a LeakageSafeEncoder OOF on train, transform held-out via
    full-train statistic. Cache the fitted encoder keyed on (column,
    method, params_hash) so multiple models share it.
    """
    method = params.kind  # target_mean / target_m_estimate / ...

    def _fit():
        enc = LeakageSafeEncoder(
            method=method,
            smoothing=params.smoothing,
            cv=params.cv,
            prior=params.prior,
            random_state=params.random_state,
        )
        # fit_transform on train so OOF encodings are returned
        train_col = _extract_column_values(train_df, column)
        oof_train = enc.fit_transform(train_col, list(train_target))
        return (enc, oof_train)

    if cache is not None:
        key = InMemoryKey(
            session_id=session_id,
            df_token=train_id,
            # Hash y content so multi-target suites don't collide on the same df_token/column slot.
            train_idx_token=_target_content_token(train_target),
            column=column,
            params_canonical_hash=canonical_params_hash(params),
            provider_signature=f"target_encoder:{method}",
        )
        encoder, oof_train = cache.get_or_compute(key, _fit)
    else:
        encoder, oof_train = _fit()

    train_block = HandlerOutput(
        column=column, method=method, data=oof_train.reshape(-1, 1).astype(np.float32),
        n_features=1, output_kind="dense",
    )
    val_block = None
    if val_df is not None:
        val_enc = encoder.transform(_extract_column_values(val_df, column))
        val_block = HandlerOutput(
            column=column, method=method, data=val_enc.reshape(-1, 1).astype(np.float32),
            n_features=1, output_kind="dense",
        )
    test_block = None
    if test_df is not None:
        test_enc = encoder.transform(_extract_column_values(test_df, column))
        test_block = HandlerOutput(
            column=column, method=method, data=test_enc.reshape(-1, 1).astype(np.float32),
            n_features=1, output_kind="dense",
        )
    return train_block, val_block, test_block


def _extract_column_values(df: Any, column: str) -> List:
    """Polars / pandas -> python list."""
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df[column].to_list()
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return df[column].tolist()
    except ImportError:  # pragma: no cover
        pass
    return list(df)


def _text_column_content_token(train_df: Any, column: str) -> int:
    """63-bit content fingerprint for a single text column used to disambiguate text-encoder cache
    entries when ``id(train_df)`` recycles across OD-masked / pre-clone variants. Mirrors the
    ``_target_content_token`` strategy below but operates on a single column to avoid full-frame
    hashing cost. Returns 0 on any backend error; the caller still keys on (df_token, column, params)
    so collision risk degrades to the literal-zero baseline (no worse than pre-fix)."""
    try:
        import polars as pl
        if isinstance(train_df, pl.DataFrame):
            ser = train_df[column]
            n = ser.len()
            sample_idx = [0, n // 4, n // 2, 3 * n // 4, max(0, n - 1)] if n else []
            sampled = [str(ser[i]) for i in sample_idx]
            buf = ("|".join(sampled) + f"|{n}|{column}").encode("utf-8", errors="replace")
        else:
            import pandas as pd
            if isinstance(train_df, pd.DataFrame):
                ser = train_df[column]
                n = len(ser)
                sample_idx = [0, n // 4, n // 2, 3 * n // 4, max(0, n - 1)] if n else []
                sampled = [str(ser.iloc[i]) if i < n else "" for i in sample_idx]
                buf = ("|".join(sampled) + f"|{n}|{column}").encode("utf-8", errors="replace")
            else:
                return 0
        digest = hashlib.blake2b(buf, digest_size=8).digest()
        return int.from_bytes(digest, "big") & ((1 << 63) - 1)
    except Exception:
        return 0


def _target_content_token(train_target: Any) -> int:
    """Return a 63-bit integer fingerprint of ``train_target`` for the InMemoryKey.

    The literal ``0`` placeholder collides across targets: a multi-target suite using a target-mean /
    WoE encoder for two different y's would hit the same cache slot and reuse target-1's OOF
    encodings for target-2. We hash the actual y content so distinct targets always produce
    distinct keys, even within one session.

    Returns ``0`` on conversion failure (the caller will then still collide -- documented fallback
    rather than a silent wrong result, because the encoder will at least be re-fit at the next
    call when this helper recovers).
    """
    try:
        if hasattr(train_target, "to_numpy"):
            arr = train_target.to_numpy()
        elif hasattr(train_target, "values"):
            arr = train_target.values
        else:
            arr = np.asarray(list(train_target))
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        buf = np.ascontiguousarray(arr).tobytes()
        digest = hashlib.blake2b(buf, digest_size=8).digest()
        digest += str(arr.shape).encode() + str(arr.dtype).encode()
        # Re-hash to 8 bytes (folded with shape/dtype) and clip to 63 bits to stay in signed int range.
        final = hashlib.blake2b(digest, digest_size=8).digest()
        return int.from_bytes(final, "big") & ((1 << 63) - 1)
    except Exception:
        return 0


__all__ = [
    "FeatureHandlingResult",
    "feature_handling_apply",
]
