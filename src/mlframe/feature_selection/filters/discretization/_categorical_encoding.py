"""Pure-numpy categorical / quantile encoders used by the discretisation facade (moved verbatim from ``__init__``)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _native_ordinal_encode_2d(vals: np.ndarray) -> np.ndarray:
    """Drop-in pure-numpy replacement for sklearn OrdinalEncoder().fit_transform on a (n, 1) array.

    Returns float64 ordinals so downstream digitize / dtype-promotion logic stays bit-for-bit
    identical to the sklearn path. ``pd.factorize`` is asymptotically the same numpy unique
    + inverse-index lookup but skips estimator-validation overhead (~6x faster at n=10k).
    """
    flat = vals.reshape(-1)
    codes, _ = pd.factorize(flat, use_na_sentinel=True)
    return np.asarray(codes, dtype=np.float64).reshape(vals.shape)


def cap_categorical_cardinality(codes_2d: np.ndarray, max_cardinality: int) -> np.ndarray:
    """Collapse the rare-category tail of each column so no column exceeds ``max_cardinality`` distinct codes.

    Per column: the ``max_cardinality - 1`` MOST FREQUENT codes keep distinct ids ``0..max_cardinality-2`` (ordered by
    frequency desc), all rarer codes fold into one "other" bucket ``max_cardinality-1``; the ``-1`` NaN sentinel is
    preserved. Columns already at/below the cap are returned UNCHANGED (bit-identical). This is the standard high-card
    reduction: a categorical whose cardinality approaches N has sparse contingency cells, so its plug-in MI/CMI is
    unreliable anyway (the analytic null already guards on >=5 expected/cell) - folding the rare tail DENSIFIES the cells
    (better MI) while capping the code range so the whole codes matrix fits a narrow int (int8 for cap<=127). Operates on
    a float64 code matrix (the ``_multi_col_factorize_native`` output) and returns float64.
    """
    if max_cardinality is None or max_cardinality < 2 or codes_2d.size == 0:
        return codes_2d
    out = codes_2d
    _copied = False
    _n, p = codes_2d.shape
    for j in range(p):
        col = codes_2d[:, j]
        finite = col[col >= 0]
        if finite.size == 0:
            continue
        # cardinality = number of distinct non-negative codes; cheap via max (codes are dense 0..k-1 from factorize).
        k = int(finite.max()) + 1
        if k <= max_cardinality:
            continue
        counts = np.bincount(finite.astype(np.int64), minlength=k)
        # keep the (cap-1) most frequent as 0..cap-2 (freq desc); everything else -> the "other" bucket cap-1.
        # kind="stable" so two categories tied exactly at
        # the cutoff boundary have a deterministic, documented tie-break (first-seen/lowest-index order)
        # rather than relying on the non-guaranteed default sort's tie behaviour across numpy
        # versions/architectures. Sort by NEGATED counts (descending) directly, rather than
        # ascending-then-reverse - reversing a stable ascending sort would itself un-stabilize the tie
        # order (it reverses ties' relative order too), defeating the point of "stable".
        keep = np.argsort(-counts, kind="stable")[: max_cardinality - 1]
        remap = np.full(k, max_cardinality - 1, dtype=np.float64)
        remap[keep] = np.arange(max_cardinality - 1, dtype=np.float64)
        if not _copied:
            out = codes_2d.copy()
            _copied = True
        mask = col >= 0
        out[mask, j] = remap[col[mask].astype(np.int64)]
    return out


def _multi_col_factorize_native(categorical_df: "pd.DataFrame") -> np.ndarray:
    """Multi-column ordinal encoding without sklearn OrdinalEncoder.

    Strategy (in order of preference):

    1. Pre-Categorical columns -> read ``.cat.codes`` directly (single C-level
       attribute access, no recomputation, no GIL contention). NaN is already
       encoded as -1 by pandas convention - matches downstream contract.
    2. Non-Categorical object / string / bool columns -> joblib-threaded
       ``pd.factorize`` (releases GIL on the hash-table fill, threading wins).
    3. Single-column fallback -> sequential loop (zero overhead).

    Ordering contract: distinct categories get distinct integer codes; NaN -> -1.
    Code values themselves are NOT bit-for-bit identical to sklearn's
    OrdinalEncoder (.cat.codes uses category-dictionary order; OrdinalEncoder
    uses first-occurrence). For downstream MI estimation the value mapping is
    semantically equivalent (MI is invariant under bijective relabeling).

    Bench on 100-col 200k-row pre-Categorical DF (representative MRMR workload):
    ~7x faster than the sequential pd.factorize loop AND no GIL contention
    so callers can multi-thread on top.
    """
    n_rows = len(categorical_df)
    cols = list(categorical_df.columns)
    if not cols:
        return np.empty((n_rows, 0), dtype=np.float64)

    out = np.empty((n_rows, len(cols)), dtype=np.float64)
    needs_factorize: list = []  # (j, col) for non-pre-categorical columns
    for _j, _c in enumerate(cols):
        _ser = categorical_df[_c]
        if isinstance(_ser.dtype, pd.CategoricalDtype):
            # Fast path: ``.cat.codes`` is a vectorised C-level attribute
            # access; NaN already encoded as -1.
            out[:, _j] = _ser.cat.codes.to_numpy(dtype=np.float64, copy=False)
        else:
            needs_factorize.append((_j, _c))

    if needs_factorize:
        # Threshold raised 1 -> 8 (7-site joblib.Parallel audit): isolated/warmed/best-of-3+
        # measurement at the realistic column-count range for this branch (2-8 non-Categorical columns)
        # found the joblib threading pool never clearly wins there - 2 cols -> 1.29x (but that case was
        # already serial pre-fix, since the old threshold was ``<= 1``), 8 cols -> 0.52x (loses), 40 cols
        # -> 0.93x (roughly even). No clean win was found across 2-8 columns, so that whole range now stays
        # serial; the pool is still used above 8 columns where the per-thread pd.factorize work is enough
        # to amortise joblib's dispatch overhead.
        if len(needs_factorize) <= 8:
            for _j, _c in needs_factorize:
                _codes, _ = pd.factorize(categorical_df[_c], use_na_sentinel=True)
                out[:, _j] = _codes.astype(np.float64)
        else:
            # joblib threading. pd.factorize releases the GIL on the hash build,
            # so threads parallelise. prefer='threads' avoids the pickling cost
            # of process workers on a categorical DF view.
            from joblib import Parallel, delayed as _delayed

            _results = Parallel(n_jobs=min(8, len(needs_factorize)), prefer="threads")(
                _delayed(lambda c: pd.factorize(categorical_df[c], use_na_sentinel=True)[0].astype(np.float64))(_c) for _j, _c in needs_factorize
            )
            for (_j, _), _codes in zip(needs_factorize, _results):
                out[:, _j] = _codes
    return out


def _native_kbins_quantile(vals: np.ndarray, n_bins: int) -> np.ndarray:
    """Drop-in pure-numpy replacement for sklearn KBinsDiscretizer(strategy='quantile', encode='ordinal').

    Uses np.nanpercentile for edge calc + np.searchsorted for bin lookup. ~12x faster than
    KBinsDiscretizer at n=10k single-column because we skip BaseEstimator validation +
    sklearn's CSR-friendly hot-path scaffolding. Output shape matches sklearn's: (n, 1) float64.
    """
    flat = np.asarray(vals, dtype=np.float64).reshape(-1)
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.nanpercentile(flat, quantiles)
    # Inner edges only (drop both extremes, like sklearn's KBinsDiscretizer does internally).
    inner = bin_edges[1:-1]
    codes = np.searchsorted(inner, flat, side="right").astype(np.float64)
    return codes.reshape(vals.shape if vals.ndim == 2 else (-1, 1))
