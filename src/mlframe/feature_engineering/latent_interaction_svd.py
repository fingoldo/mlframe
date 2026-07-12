"""SVD latent-factor embeddings of an (entity x item) interaction matrix, optionally TF-IDF/time weighted.

Distinct from the pre-existing ``cat_cooccurrence_svd`` (a per-ROW categorical-pair encoder producing a
single-column embedding for a `src_col` category from its contingency with `other_col`): this builds a full
sparse ENTITY x ITEM interaction matrix from an events log (one row per interaction, e.g. customer x item
purchase), optionally TF-IDF-reweighted and/or time-decayed, and returns BOTH row (entity) and column (item)
latent embeddings via truncated SVD -- the pattern used for customer-item, customer-brand, customer-coupon
relational features in a 5th-place AmExpert-2019 solution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

if TYPE_CHECKING:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer


def _weighted_matrix(
    events_df: pd.DataFrame,
    row_entity: str,
    col_entity: str,
    weight_col: Optional[str],
    time_col: Optional[str],
    time_decay_half_life: Optional[float],
    reference_time: Optional[float],
    row_uniq: np.ndarray,
    col_uniq: Optional[np.ndarray],
) -> Tuple["sparse.csr_matrix", np.ndarray, np.ndarray, np.ndarray]:
    """Build the (row x col) weighted count matrix, reusing ``col_uniq`` as-is when given (frozen-basis path).

    Returns ``(matrix, row_uniq, col_uniq, row_oov_mass_fraction)`` -- the last is, per row entity, the
    fraction of its RAW (pre-TF-IDF) interaction weight mass that landed on column entities absent from
    ``col_uniq`` (always 0 when ``col_uniq`` is None, i.e. the normal fit path with no frozen vocabulary).
    """
    weights = events_df[weight_col].to_numpy(dtype=np.float64) if weight_col is not None else np.ones(len(events_df), dtype=np.float64)

    if time_col is not None and time_decay_half_life is not None:
        t = events_df[time_col].to_numpy(dtype=np.float64)
        ref = float(reference_time) if reference_time is not None else float(t.max())
        decay = 0.5 ** ((ref - t) / time_decay_half_life)
        weights = weights * decay

    row_vals = events_df[row_entity].to_numpy()
    col_vals = events_df[col_entity].to_numpy()

    if col_uniq is None:
        col_uniq_out, col_inv = np.unique(col_vals, return_inverse=True)
        in_vocab = np.ones(len(events_df), dtype=bool)
    else:
        col_uniq_out = col_uniq
        col_index = {v: i for i, v in enumerate(col_uniq_out)}
        col_inv = np.array([col_index.get(v, -1) for v in col_vals], dtype=np.int64)
        in_vocab = col_inv >= 0

    row_index = {v: i for i, v in enumerate(row_uniq)}
    row_inv = np.array([row_index[v] for v in row_vals], dtype=np.int64)

    n_rows, n_cols = len(row_uniq), len(col_uniq_out)
    total_mass = np.zeros(n_rows, dtype=np.float64)
    np.add.at(total_mass, row_inv, weights)
    oov_mass = np.zeros(n_rows, dtype=np.float64)
    if not in_vocab.all():
        np.add.at(oov_mass, row_inv[~in_vocab], weights[~in_vocab])
    with np.errstate(divide="ignore", invalid="ignore"):
        row_oov_fraction = np.divide(oov_mass, total_mass, out=np.zeros(n_rows, dtype=np.float64), where=total_mass > 0)
    row_oov_fraction[total_mass == 0] = 1.0

    M = sparse.coo_matrix((weights[in_vocab], (row_inv[in_vocab], col_inv[in_vocab])), shape=(n_rows, n_cols)).tocsr()
    return M, row_uniq, col_uniq_out, row_oov_fraction


@dataclass
class FittedLatentInteractionSvd:
    """Fitted TF-IDF (optional) + SVD basis over a frozen item vocabulary, reusable for NEW row entities.

    Refitting per inference batch would let the item vocabulary/SVD axes drift from the training corpus;
    ``transform_new_entities`` instead applies the already-fitted ``tfidf``/``svd`` transformers on the SAME
    ``col_uniq`` item vocabulary, and reports each new entity's out-of-vocabulary interaction-weight mass
    (interactions with items never seen during fit, which are silently dropped from the frozen vocabulary) as
    a reliability diagnostic -- mirrors ``tfidf_svd_entity_embedding.FittedTfidfSvdEntityEmbedding``.
    """

    col_uniq: np.ndarray
    tfidf: Optional["TfidfTransformer"]
    svd: "TruncatedSVD"
    row_entity: str
    col_entity: str

    def transform_new_entities(
        self,
        events_df: pd.DataFrame,
        weight_col: Optional[str] = None,
        time_col: Optional[str] = None,
        time_decay_half_life: Optional[float] = None,
        reference_time: Optional[float] = None,
    ) -> pd.DataFrame:
        """Embed row entities in ``events_df`` on the already-fitted item vocabulary/SVD basis (no refitting).

        Returns one row per unique row entity, columns ``svd_0..svd_{n-1}`` plus ``oov_weight_fraction`` --
        the fraction of that entity's raw interaction weight mass spent on items absent from the fitted item
        vocabulary (0 = every interaction was with a known item, 1 = every interaction is with a novel item
        and the row is embedded as an all-zero vector, i.e. a cold-start fallback at the SVD origin).
        """
        if self.row_entity not in events_df.columns or self.col_entity not in events_df.columns:
            raise ValueError(
                f"transform_new_entities: {self.row_entity!r}/{self.col_entity!r} must both be columns of events_df"
            )
        row_uniq = np.unique(events_df[self.row_entity].to_numpy())
        M, row_uniq, _, row_oov_fraction = _weighted_matrix(
            events_df,
            self.row_entity,
            self.col_entity,
            weight_col,
            time_col,
            time_decay_half_life,
            reference_time,
            row_uniq,
            self.col_uniq,
        )
        if self.tfidf is not None:
            M = self.tfidf.transform(M)
        row_emb = self.svd.transform(M)
        col_names = [f"svd_{i}" for i in range(row_emb.shape[1])]
        row_df = pd.DataFrame(row_emb, index=row_uniq, columns=col_names)
        row_df.index.name = self.row_entity
        row_df["oov_weight_fraction"] = row_oov_fraction
        return row_df


def latent_interaction_features(
    events_df: pd.DataFrame,
    row_entity: str,
    col_entity: str,
    weight_col: Optional[str] = None,
    time_col: Optional[str] = None,
    time_decay_half_life: Optional[float] = None,
    reference_time: Optional[float] = None,
    use_tfidf: bool = True,
    n_components: int = 10,
    random_state: int = 0,
    return_fitted: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, FittedLatentInteractionSvd]]:
    """Build a sparse (row_entity x col_entity) interaction matrix and return SVD row/column embeddings.

    Parameters
    ----------
    events_df
        One row per interaction event (e.g. a purchase, a click).
    row_entity, col_entity
        Columns defining the two axes of the interaction matrix (e.g. ``"customer_id"``, ``"item_id"``).
    weight_col
        Optional numeric column to sum per ``(row_entity, col_entity)`` cell instead of a plain interaction
        count (e.g. purchase amount).
    time_col, time_decay_half_life, reference_time
        When all three are given, each event's weight (count or ``weight_col``) is multiplied by
        ``0.5 ** ((reference_time - time_col) / time_decay_half_life)`` before aggregation -- more recent
        events contribute more. ``reference_time`` defaults to ``events_df[time_col].max()``.
    use_tfidf
        Apply TF-IDF reweighting (via ``sklearn.feature_extraction.text.TfidfTransformer``, treating each
        row-entity's interaction-count vector as a "document" over item "terms") to the (possibly time-
        decayed) count matrix before SVD -- downweights ubiquitous items, upweights entity-distinctive ones.
    n_components
        Number of SVD components (capped at the matrix's rank).
    random_state
        Passed to ``TruncatedSVD``.
    return_fitted
        When ``True``, also return a :class:`FittedLatentInteractionSvd` capturing the fitted item vocabulary,
        TF-IDF transformer (if used) and SVD basis, so NEW row entities (e.g. cold-start customers seen only
        at inference) can be embedded later via ``transform_new_entities`` on the SAME frozen basis, with an
        OOV interaction-weight-fraction reliability diagnostic, instead of leaking/refitting on the new batch.
        Default ``False`` preserves the prior return type and is bit-identical to the pre-extension behavior.

    Returns
    -------
    tuple
        ``(row_embeddings, col_embeddings)`` -- DataFrames indexed by the entity id, columns
        ``svd_0..svd_{n-1}``. If ``return_fitted=True``, a
        ``(row_embeddings, col_embeddings, FittedLatentInteractionSvd)`` triple instead.
    """
    if row_entity not in events_df.columns or col_entity not in events_df.columns:
        raise ValueError(f"latent_interaction_features: {row_entity!r}/{col_entity!r} must both be columns of events_df")

    weights = events_df[weight_col].to_numpy(dtype=np.float64) if weight_col is not None else np.ones(len(events_df), dtype=np.float64)

    if time_col is not None and time_decay_half_life is not None:
        t = events_df[time_col].to_numpy(dtype=np.float64)
        ref = float(reference_time) if reference_time is not None else float(t.max())
        decay = 0.5 ** ((ref - t) / time_decay_half_life)
        weights = weights * decay

    row_uniq, row_inv = np.unique(events_df[row_entity].to_numpy(), return_inverse=True)
    col_uniq, col_inv = np.unique(events_df[col_entity].to_numpy(), return_inverse=True)
    n_rows, n_cols = len(row_uniq), len(col_uniq)

    M = sparse.coo_matrix((weights, (row_inv, col_inv)), shape=(n_rows, n_cols)).tocsr()

    tfidf = None
    if use_tfidf:
        from sklearn.feature_extraction.text import TfidfTransformer

        tfidf = TfidfTransformer()
        M = tfidf.fit_transform(M)

    from sklearn.decomposition import TruncatedSVD

    n_eff = min(n_components, max(1, min(n_rows, n_cols) - 1))
    svd = TruncatedSVD(n_components=n_eff, random_state=random_state)
    row_emb = svd.fit_transform(M)
    col_emb = svd.components_.T  # (n_cols, n_eff), the item-side embedding sharing the same latent axes

    col_names = [f"svd_{i}" for i in range(n_eff)]
    row_df = pd.DataFrame(row_emb, index=row_uniq, columns=col_names)
    col_df = pd.DataFrame(col_emb, index=col_uniq, columns=col_names)
    row_df.index.name = row_entity
    col_df.index.name = col_entity

    if not return_fitted:
        return row_df, col_df
    fitted = FittedLatentInteractionSvd(col_uniq=col_uniq, tfidf=tfidf, svd=svd, row_entity=row_entity, col_entity=col_entity)
    return row_df, col_df, fitted


__all__ = ["latent_interaction_features", "FittedLatentInteractionSvd"]
