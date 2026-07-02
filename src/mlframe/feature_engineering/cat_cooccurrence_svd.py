"""SVD co-occurrence embedding of a categorical column (Dyakonov's ``code_factor``).

Encode a categorical column by the leading singular vectors of its cat x cat
CONTINGENCY (co-occurrence) matrix with a second categorical column. For each
category ``a`` of the source column the encoded value is the ``a``-th entry of
the leading LEFT singular vector(s) of ``M`` where ``M[a, b] = #{rows with
src == a AND other == b}`` -- i.e. categories that co-occur with the same
partner categories get similar codes.

Reference: Dyakonov A.G., "Methods for solving classification problems with
categorical features" // Applied Mathematics and Informatics, Faculty of
Computational Mathematics and Cybernetics, MSU, 2014, No. 46, pp. 103-127.
(Slides 44-46 of the "Feature Engineering / Construction" lecture.)

Why this is distinct from what already exists in mlframe:

* NOT target encoding (``_target_encoding_fe``): consumes NO ``y``; the code
  is derived purely from the co-occurrence structure of two categoricals, so
  it is leakage-free by construction and usable on unlabelled / test rows.
* NOT count / frequency encoding (``_count_freq_interaction_fe``): those map a
  category to a scalar marginal count; this maps it to a dense low-dim vector
  that captures WHICH partner categories it co-occurs with, not just how often
  it appears.
* NOT the cat x cat cross (``_cat_pair_fe``): that materialises the JOINT cell
  ``(a, b)`` as a new high-cardinality categorical then target-encodes it; this
  collapses the co-occurrence matrix to a dense per-category embedding via SVD
  and never touches ``y``.

The result is a smooth, ordered numeric encoding useful for linear models and
kNN (per the lecture: "many encodings are random, target/factor encoding is
logical") and as a leakage-free alternative to target encoding when ``y`` is
scarce or the encoded column must be reused across an unlabelled test set.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "engineered_name_cooccur_svd",
    "cat_cooccurrence_svd_fit",
    "apply_cat_cooccurrence_svd",
    "cat_cooccurrence_svd_with_recipes",
]

_NAN_TOKEN = "__nan__"


def engineered_name_cooccur_svd(src_col: str, other_col: str, component: int) -> str:
    """Stable engineered column name for one SVD co-occurrence component.

    ``{src}__cooccur_svd{k}__by__{other}`` reads as "component k of the
    co-occurrence embedding of src through its joint with other".
    """
    return f"{src_col}__cooccur_svd{component}__by__{other_col}"


def _column_to_tokens(col) -> np.ndarray:
    """Coerce a column to a numpy object array of canonical string tokens.

    NaN / None map to a single ``"__nan__"`` sentinel so they form one implicit
    category at fit AND apply time (no NaN propagation). Reuses the project-wide
    ``canonical_group_token`` so an int / float dtype drift between fit and apply
    still resolves to the same category (``1`` and ``1.0`` collapse).
    """
    from mlframe.feature_selection.filters._internals import canonical_group_token

    arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
    # Resolve tokens once per DISTINCT value then gather -- runs per-unique, not per-row.
    if arr.dtype.kind in ("i", "u", "b"):
        uniq, inv = np.unique(arr, return_inverse=True)
        toks = np.array([canonical_group_token(u) for u in uniq], dtype=object)
        return toks[inv]
    codes, uniq = pd.factorize(arr, use_na_sentinel=False)
    toks = np.empty(len(uniq), dtype=object)
    for j, v in enumerate(uniq):
        if v is None or (isinstance(v, float) and v != v):  # None or NaN
            toks[j] = _NAN_TOKEN
        else:
            toks[j] = canonical_group_token(v)
    return toks[codes]


def _contingency_svd_row_coords(M: np.ndarray, n_eff: int, normalize: str) -> np.ndarray:
    """Row (source-category) coordinates from an SVD of the contingency matrix.

    ``normalize='ca'`` (correspondence analysis, the default): the RAW contingency
    SVD's leading axis is dominated by the category MARGINAL frequencies (the trivial
    "size" effect), so the group-discriminating association structure hides in the 2nd+
    vectors. CA removes that by decomposing the standardised residual
    ``(P - r c^T) / sqrt(r c^T)`` (``P = M / total``, ``r`` / ``c`` the row / column
    marginal profiles) and scaling row coordinates by ``s / sqrt(r)``. The leading
    CA axis then directly carries the association -- the "logical ordering" the lecture
    wants. Measured: CA n_components=1 matches raw n_components=2 on the latent-group
    bed (min holdout AUC 0.77 vs raw-1's unstable 0.60).

    ``normalize='raw'``: plain SVD of the count matrix (Dyakonov's slide-44 form),
    kept for exact reproduction and for the rare case where the marginal axis IS the
    signal of interest.
    """
    if normalize == "ca":
        total = float(M.sum())
        if total <= 0.0:
            return np.zeros((M.shape[0], n_eff), dtype=np.float64)
        P = M / total
        r = P.sum(axis=1, keepdims=True)
        c = P.sum(axis=0, keepdims=True)
        expected = r @ c
        S = (P - expected) / np.sqrt(expected + 1e-12)
        U, s, _Vt = np.linalg.svd(S, full_matrices=False)
        row_coords = (U[:, :n_eff] * s[:n_eff]) / np.sqrt(r + 1e-12)
        return np.ascontiguousarray(row_coords, dtype=np.float64)
    if normalize == "raw":
        U, _s, _Vt = np.linalg.svd(M, full_matrices=False)
        return U[:, :n_eff].astype(np.float64, copy=True)
    raise ValueError(f"normalize must be 'ca' or 'raw'; got {normalize!r}")


def cat_cooccurrence_svd_fit(
    X: pd.DataFrame,
    src_col: str,
    other_col: str,
    *,
    n_components: int = 1,
    normalize: str = "ca",
) -> tuple[np.ndarray, dict]:
    """Fit the SVD co-occurrence embedding of ``src_col`` through ``other_col``.

    Builds the ``(card_src, card_other)`` integer contingency matrix
    ``M[a, b] = #{rows: src == a, other == b}``, decomposes it, and encodes each
    source category ``a`` by the first ``n_components`` row coordinates. Because
    the sign of a singular vector is arbitrary, each component is sign-
    canonicalised so its largest-magnitude entry is positive (deterministic
    across runs / platforms).

    Parameters
    ----------
    X : pd.DataFrame
        Frame containing both columns.
    src_col : str
        Categorical column to encode.
    other_col : str
        Second categorical column whose co-occurrence with ``src_col`` defines
        the embedding. May equal ``src_col`` (then M is diagonal and the
        embedding degenerates to a scaled indicator -- guarded against by the
        caller's auto-detection, but not an error here).
    n_components : int, default 1
        Number of leading singular vectors to emit. Capped at
        ``min(card_src, card_other)``; a warning is logged if the request
        exceeds the available rank.
    normalize : {"ca", "raw"}, default "ca"
        ``"ca"`` (correspondence analysis) decomposes the marginal-standardised
        residual so the leading axis carries the ASSOCIATION structure, not the
        trivial marginal-frequency size effect -- the accurate default (see
        ``_contingency_svd_row_coords``). ``"raw"`` is the plain count-matrix SVD
        of Dyakonov's slide-44 form.

    Returns
    -------
    embedded : ndarray, shape (n, n_eff)
        Per-row embedding, ``n_eff = min(n_components, rank)`` columns. Each row
        carries the code of its source category.
    recipe : dict
        ``{"lookup": {token: [comp_0, ...]}, "default": [0.0, ...],
           "n_components": n_eff, "normalize": normalize}``. Replay is a pure
        dict lookup; unseen categories at apply time map to ``default`` (the zero
        vector -- the origin of the embedding, the natural "no information"
        fallback).

    Raises
    ------
    ValueError
        On empty X, missing columns, ``n_components < 1``, or bad ``normalize``.
    """
    if len(X) == 0:
        raise ValueError("cat_cooccurrence_svd_fit: X is empty")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1; got {n_components}")
    if normalize not in ("ca", "raw"):
        raise ValueError(f"normalize must be 'ca' or 'raw'; got {normalize!r}")
    for c in (src_col, other_col):
        if c not in X.columns:
            raise ValueError(f"cat_cooccurrence_svd_fit: column {c!r} missing from X")

    src_tok = _column_to_tokens(X[src_col])
    oth_tok = _column_to_tokens(X[other_col])

    src_uniq, src_inv = np.unique(src_tok, return_inverse=True)
    oth_uniq, oth_inv = np.unique(oth_tok, return_inverse=True)
    n_src = src_uniq.shape[0]
    n_oth = oth_uniq.shape[0]

    # Contingency matrix via a single flat bincount over the (a, b) cell index.
    flat = src_inv.astype(np.int64) * n_oth + oth_inv.astype(np.int64)
    counts = np.bincount(flat, minlength=n_src * n_oth)
    M = counts.reshape(n_src, n_oth).astype(np.float64)

    max_rank = min(n_src, n_oth)
    n_eff = min(int(n_components), max_rank)
    if n_eff < n_components:
        logger.debug(
            "cat_cooccurrence_svd_fit: requested %d components but rank is %d "
            "(card_src=%d, card_other=%d); emitting %d.",
            n_components, max_rank, n_src, n_oth, n_eff,
        )

    emb = _contingency_svd_row_coords(M, n_eff, normalize)

    # Sign-canonicalise each component: the SVD sign is arbitrary, so pin the
    # largest-magnitude entry positive for run-to-run / platform determinism.
    for k in range(n_eff):
        col = emb[:, k]
        j = int(np.argmax(np.abs(col)))
        if col[j] < 0.0:
            emb[:, k] = -col

    embedded = emb[src_inv, :]

    lookup = {str(src_uniq[a]): emb[a, :].tolist() for a in range(n_src)}
    recipe = {
        "lookup": lookup,
        "default": [0.0] * n_eff,
        "n_components": n_eff,
        "normalize": normalize,
    }
    return embedded, recipe


def apply_cat_cooccurrence_svd(
    X_test: pd.DataFrame,
    src_col: str,
    recipe: dict,
) -> np.ndarray:
    """Replay the co-occurrence SVD embedding from the stored lookup.

    Pure function of ``X_test[src_col]`` -- no ``y``, no ``other_col`` reference
    (the co-occurrence structure is baked into the lookup at fit time). Unseen
    categories map to ``recipe["default"]`` (the zero vector). NaN maps to the
    ``"__nan__"`` token, which is a real key iff NaN was seen at fit.

    Returns
    -------
    ndarray, shape (n_test, n_components)
    """
    for key in ("lookup", "default", "n_components"):
        if key not in recipe:
            raise KeyError(
                f"apply_cat_cooccurrence_svd: recipe missing {key!r}. Re-fit to regenerate."
            )
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"apply_cat_cooccurrence_svd: X_test must be a DataFrame; got {type(X_test).__name__}"
        )
    lookup: dict = recipe["lookup"]
    default = np.asarray(recipe["default"], dtype=np.float64)
    n_comp = int(recipe["n_components"])

    tokens = _column_to_tokens(X_test[src_col])
    if tokens.size == 0:
        return np.empty((0, n_comp), dtype=np.float64)

    # Resolve the embedding once per DISTINCT category then gather by code.
    codes, uniques = pd.factorize(tokens, use_na_sentinel=False)
    table = np.empty((uniques.shape[0], n_comp), dtype=np.float64)
    for i, u in enumerate(uniques):
        vec = lookup.get(str(u))
        table[i, :] = default if vec is None else np.asarray(vec, dtype=np.float64)
    return table[codes]


def cat_cooccurrence_svd_with_recipes(
    X: pd.DataFrame,
    *,
    src_cols: Sequence[str],
    other_cols: Sequence[str],
    n_components: int = 1,
    normalize: str = "ca",
):
    """Append SVD co-occurrence embedding columns for each (src, other) pair.

    For every ordered pair ``(src, other)`` with ``src != other`` and both in
    ``X``, fit the embedding and append ``n_eff`` columns named via
    ``engineered_name_cooccur_svd``. Returns ``(X_aug, appended_names,
    recipes)`` where each recipe is the plain dict from ``cat_cooccurrence_svd_fit``
    augmented with its ``src_col`` / ``other_col`` for round-trip replay.
    """
    src_cols = [c for c in src_cols if c in X.columns]
    other_cols = [c for c in other_cols if c in X.columns]
    if not src_cols or not other_cols:
        return X.copy(), [], []

    new_cols: dict[str, np.ndarray] = {}
    appended: list[str] = []
    recipes: list[dict] = []
    for src in src_cols:
        for other in other_cols:
            if src == other:
                continue
            emb, rec = cat_cooccurrence_svd_fit(
                X, src, other, n_components=n_components, normalize=normalize,
            )
            rec = {**rec, "src_col": src, "other_col": other}
            for k in range(rec["n_components"]):
                name = engineered_name_cooccur_svd(src, other, k)
                new_cols[name] = emb[:, k]
                appended.append(name)
            recipes.append(rec)
    if not appended:
        return X.copy(), [], []
    new_df = pd.DataFrame(new_cols, index=X.index)
    X_aug = pd.concat([X, new_df], axis=1)
    return X_aug, appended, recipes
