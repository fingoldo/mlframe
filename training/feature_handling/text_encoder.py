"""
:class:`TextColumnEncoder` -- TF-IDF / Hashing per-column encoder
with polars / pandas / sparse symmetry.

Round-3 plan §1.4 + user's "polars-ds dispatch" requirement: the
encoder probes :class:`PolarsNativeDispatcher` first and uses
polars-ds' native ``Blueprint.tfidf`` / ``Blueprint.hashing_encode``
when available; otherwise falls back to sklearn ``TfidfVectorizer``
/ ``HashingVectorizer``.

Output format follows input by default (``set_output_kind="auto"``):
  * polars input -> polars output (or sparse matrix when handler is
    sparse-block; the cache layer in phase D handles concat).
  * pandas input -> sparse ``scipy.sparse.csr_matrix`` (default for
    TF-IDF / hashing; tree models accept sparse natively).

Round-3 R2-9 + plan §3 sparse/dense routing: when downstream model
is dense-only (HGB / RF / NGB / MLP / TabNet / Recurrent) AND the
sparse output exceeds 512 cols, the cache layer auto-applies
TruncatedSVD with a WARN (phase E wires that). Phase C is just the
encoder; the routing decision lives upstream.

Phase C v1: sklearn-only fallback (polars-ds 0.11.x doesn't ship
TF-IDF / hashing as Blueprint ops -- audit confirmed). When polars-ds
adds them upstream (the user is contributing!) the dispatcher picks
them up automatically without a code change here.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from mlframe.training.feature_handling.handlers import (
    HashingParams,
    TfidfParams,
)
from mlframe.training.feature_handling.polars_capability import (
    PolarsNativeDispatcher,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401
    import polars as pl  # noqa: F401
    import scipy.sparse as sp  # noqa: F401


def _column_to_string_list(
    df: Any,
    column: str,
) -> List[str]:
    """Polars / pandas -agnostic extraction of a column as List[str].

    Coerces None / NaN to "" so the underlying vectoriser doesn't
    crash (round-3 T8). Non-string types are stringified -- caller
    should not pass numeric columns here (auto-detector excludes them).
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            ser = df[column]
            return [
                "" if v is None else (v if isinstance(v, str) else str(v))
                for v in ser.to_list()
            ]
    except ImportError:  # pragma: no cover
        pass

    try:
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            ser = df[column]
            return [
                "" if (v is None or (isinstance(v, float) and np.isnan(v)))
                else (v if isinstance(v, str) else str(v))
                for v in ser.tolist()
            ]
    except ImportError:  # pragma: no cover
        pass

    raise TypeError(
        f"unsupported DataFrame type {type(df).__name__}; expected polars or pandas"
    )


class TextColumnEncoder:
    """Per-column text vectoriser. Constructed by phase E for each
    text column the FHC marks for ``method="tfidf"`` or ``"hashing"``.

    Invariants (locked by tests):
      * Output column count matches ``params.max_features`` (TF-IDF)
        or ``params.n_features`` (hashing).
      * Vocab fitted on TRAIN rows only (caller passes ``train_df``);
        ``transform()`` applies the fitted vocab to any frame.
      * ``transform()`` on a fit-row frame returns the same matrix as
        ``fit_transform()`` (idempotency).
      * Sparse output by default (csr_matrix). Caller densifies if
        needed.
      * Polars input + polars-ds capability available -> polars-native
        path. Phase C v1 doesn't yet hit this branch (polars-ds 0.11.x
        lacks TF-IDF as a Blueprint op); reserved for upstream
        coverage.
    """

    def __init__(
        self,
        column: str,
        params: Union[TfidfParams, HashingParams],
        prefer_polarsds: bool = True,
    ):
        self.column = column
        self.params = params
        self._dispatcher = PolarsNativeDispatcher(prefer_polarsds=prefer_polarsds)
        self._vectorizer: Optional[Any] = None
        self._fitted: bool = False
        self._n_features_out: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_features_out(self) -> Optional[int]:
        return self._n_features_out

    @property
    def output_kind(self) -> Literal["sparse", "dense"]:
        """Both TF-IDF and Hashing emit sparse CSR by default. The
        consumer (phase E concat layer) decides whether to densify."""
        return "sparse"

    def signature(self) -> str:
        """Stable string for cache keys. Phase D upgrades to the
        full :class:`ProviderIdentity`-style canonical hash; phase C
        uses a simple representation."""
        return f"{type(self.params).__name__}:{self.params.kind}:{self.params.model_dump_json()}"

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, train_df: Any) -> "TextColumnEncoder":
        """Fit the vectoriser on the column extracted from
        ``train_df``. Returns self."""
        texts = _column_to_string_list(train_df, self.column)
        self._fit_vectorizer(texts)
        self._fitted = True
        return self

    def transform(self, df: Any) -> "sp.csr_matrix":
        """Apply the fitted vectoriser to ``df[self.column]``. Returns
        a sparse CSR matrix.
        """
        if not self._fitted:
            raise RuntimeError(
                f"TextColumnEncoder({self.column!r}) not fitted. "
                f"Call .fit(train_df) first."
            )
        texts = _column_to_string_list(df, self.column)
        # sklearn's TfidfVectorizer.transform / HashingVectorizer.transform
        # both return scipy.sparse.csr_matrix.
        return self._vectorizer.transform(texts)

    def fit_transform(self, train_df: Any) -> "sp.csr_matrix":
        texts = _column_to_string_list(train_df, self.column)
        out = self._fit_vectorizer(texts, also_transform=True)
        self._fitted = True
        return out

    # ------------------------------------------------------------------
    # Vectoriser construction
    # ------------------------------------------------------------------

    def _fit_vectorizer(
        self,
        texts: Sequence[str],
        also_transform: bool = False,
    ):
        # TODO(phase upstream): when polars-ds ships native TF-IDF /
        # hashing in Blueprint, branch here via self._dispatcher.has()
        # and feed the polars-native path. For now, sklearn is the
        # only impl (round-3 audit confirmed polars-ds 0.11.x absence).
        from sklearn.feature_extraction.text import (
            HashingVectorizer,
            TfidfVectorizer,
        )

        if isinstance(self.params, TfidfParams):
            self._vectorizer = TfidfVectorizer(
                max_features=self.params.max_features,
                ngram_range=self.params.ngram_range,
                min_df=self.params.min_df,
                max_df=self.params.max_df,
                sublinear_tf=self.params.sublinear_tf,
                norm=self.params.norm,
            )
        elif isinstance(self.params, HashingParams):
            self._vectorizer = HashingVectorizer(
                n_features=self.params.n_features,
                ngram_range=self.params.ngram_range,
                norm=self.params.norm,
                alternate_sign=False,  # all-positive for log-friendly downstream
            )
        else:
            raise TypeError(
                f"unsupported params type {type(self.params).__name__}; "
                f"expected TfidfParams or HashingParams"
            )

        if also_transform:
            out = self._vectorizer.fit_transform(texts)
        else:
            self._vectorizer.fit(texts)
            out = None

        # Determine n_features_out: TF-IDF can be smaller than max_features
        # if the corpus has fewer unique tokens; hashing always exact.
        if isinstance(self.params, TfidfParams):
            self._n_features_out = len(self._vectorizer.vocabulary_)
        else:
            self._n_features_out = self.params.n_features

        return out

    def __repr__(self) -> str:
        kind = type(self.params).__name__
        return (
            f"TextColumnEncoder(column={self.column!r}, params_kind={kind}, "
            f"fitted={self._fitted}, n_features_out={self._n_features_out})"
        )


__all__ = ["TextColumnEncoder"]
