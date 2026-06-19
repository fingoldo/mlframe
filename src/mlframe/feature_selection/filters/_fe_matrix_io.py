"""P0 of the matrix-native FE replatform: a single-copy float32 matrix adapter.

GATED OFF by default (``MLFRAME_FE_MATRIX_P0``) and UN-WIRED -- nothing in the FE path imports this
yet, so landing it is safe during any run. It is the framework boundary the later phases (P-seam ->
P2 streaming -> P3 synergy screen -> P5 polars-native exit, and ultimately GPU-resident candidates)
build on: ONE place that turns pandas / polars / pyarrow / ndarray into a contiguous numeric block
and back, so the SAME numba (and, later, cupy) kernels serve every framework.

Design (corrections folded in from the architecture/correctness review):

* ONE copy in. The numeric plane is a pre-allocated ``(n, p_num)`` array filled column-by-column
  (one write per column), NOT a per-column ``astype`` + ``column_stack`` (which is two copies and can
  spike memory at 1M rows). The ``dtype`` arg is honoured on every path.

* float32 by default -- an INTENTIONAL behaviour change (footprint + one reused contiguous block).
  Parity gates must reference a float32-cast baseline, not the raw float64 frame.

* Categorical codes live ONLY in a separate int plane, never in the float plane: a float32 mantissa
  is 24 bits, so high-cardinality codes (> 2**24) would alias and silently collapse distinct
  categories. Numeric and categorical columns are tracked by position so the frame round-trips.

* Nulls: polars via ``Series.is_null()`` (NOT a -1 sentinel code -- a real -1 category would clash);
  pandas via ``isna``; numpy via ``~isfinite``. A per-column boolean null mask is preserved so
  ``from_feature_matrix`` can restore missingness exactly.

* Output routes back to the SOURCE framework (pandas in -> pandas out, polars in -> polars out,
  ndarray in -> ndarray out), not always pandas.

Non-pure FE families (the contract for the streaming phase -- these CANNOT be width-blocked or
per-block kerneled without their full-column anchor, so later phases must freeze the anchor over the
FULL column first): ``smart_log`` / ``logn`` (nanmin/min shift), ``_safe_div`` (exact ``y==0``
branch), ``prewarp`` / ``gate_med`` (fit-time coefficients), ``grad1`` / ``grad2`` (``np.gradient``
-- CROSS-ROW, cannot be width-blocked at all). Enumerated here so a later kernel author sees them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Families that are NOT pure-elementwise -- a per-block kernel corrupts them unless their full-column
# anchor (min/nanmin/median/fit-coeff) is frozen first, or (gradient) they are computed full-column.
NON_PURE_FE_FAMILIES: tuple[str, ...] = (
    "smart_log", "logn", "safe_div", "prewarp", "gate_med", "grad1", "grad2",
)


def fe_matrix_p0_enabled() -> bool:
    """Whether the P0 matrix adapter is active. OFF unless ``MLFRAME_FE_MATRIX_P0`` is truthy."""
    return os.environ.get("MLFRAME_FE_MATRIX_P0", "").strip().lower() in ("1", "true", "on", "yes")


@dataclass
class FeatureMatrix:
    """Framework-agnostic numeric view of an input frame.

    ``numeric`` is a contiguous ``(n, p_num)`` array (``dtype``, default float32). ``categorical`` is a
    parallel ``(n, p_cat)`` int plane of dense category codes (-1 for null), kept SEPARATE from the
    float plane. ``columns`` is the original column order; ``col_kind[j]`` is ``"numeric"`` or
    ``"categorical"`` and ``col_index[j]`` indexes into the matching plane. ``null_mask`` (optional)
    is a ``(n, p)`` bool of original missingness, in ``columns`` order, for exact round-trip.
    ``categories`` maps a categorical column name -> its ordered category labels (code i -> labels[i]).
    """

    numeric: np.ndarray
    categorical: np.ndarray
    columns: list[str]
    col_kind: list[str]
    col_index: list[int]
    n_rows: int
    framework: str
    categories: dict[str, list] = field(default_factory=dict)
    null_mask: Optional[np.ndarray] = None
    dtype: Any = np.float32

    def numeric_column(self, name: str) -> np.ndarray:
        """The numeric plane column for ``name`` (KeyError-equivalent ValueError if not numeric)."""
        j = self.columns.index(name)
        if self.col_kind[j] != "numeric":
            raise ValueError(f"column {name!r} is not numeric (kind={self.col_kind[j]})")
        return self.numeric[:, self.col_index[j]]


def _detect_framework(X) -> str:
    mod = type(X).__module__.split(".")[0]
    if mod == "pandas":
        return "pandas"
    if mod == "polars":
        return "polars"
    if mod in ("pyarrow",):
        return "pyarrow"
    return "numpy"


def _polars_column_to_arrays(s, dtype):
    """(numeric_values_or_None, codes_or_None, categories_or_None, null_mask) for a polars Series.

    Categorical / Enum -> (None, int codes, category labels, null mask). Numeric/bool -> (float values,
    None, None, null mask). Nulls are read from ``is_null`` (never a -1 code that could clash with a
    real category)."""
    import polars as pl

    null_mask = s.is_null().to_numpy()
    if s.dtype in (pl.Categorical, pl.Enum):
        # physical() gives the UInt32 codes; fill nulls with 0 (a VALID unsigned value -- a -1 sentinel
        # would overflow UInt32 and blow polars up) and then stamp -1 into the null positions via the
        # mask below, so the public contract (null -> -1 code) still holds.
        # via to_list (materialised) not to_numpy: the zero-copy Arrow buffer to_numpy returns can trip
        # array-introspecting pytest plugins (typeguard / jaxtyping) into a spurious MemoryError; to_list
        # is robust. Nulls become None in the list -> np.array makes them 0 after the null_mask stamp.
        codes = np.array([(c if c is not None else 0) for c in s.to_physical().to_list()], dtype=np.int64)
        # Enum: ordered categories live on the dtype. Categorical: get them via cat.get_categories() --
        # NOT dtype.categories, which for a Categorical is a global-string-cache view that can blow up
        # (MemoryError) when materialised. Distinguish by dtype, don't getattr-probe.
        if s.dtype == pl.Enum:
            categories = list(s.dtype.categories)
        else:  # Categorical
            try:
                categories = s.cat.get_categories().to_list()
            except Exception:
                categories = []
        codes = np.where(null_mask, -1, codes)
        return None, codes, categories, null_mask
    vals = s.to_numpy().astype(dtype, copy=False)
    return vals, None, None, null_mask


def _pandas_column_to_arrays(s, dtype):
    import pandas as pd

    null_mask = s.isna().to_numpy()
    if isinstance(s.dtype, pd.CategoricalDtype):
        codes = s.cat.codes.to_numpy().astype(np.int64, copy=False)  # already -1 for NaN
        return None, codes, list(s.cat.categories), null_mask
    vals = s.to_numpy(dtype=dtype, copy=False) if s.dtype != object else s.to_numpy().astype(dtype)
    return vals, None, None, null_mask


def to_feature_matrix(X, *, dtype: Any = np.float32) -> FeatureMatrix:
    """Convert ``X`` (pandas / polars / pyarrow / ndarray) to a :class:`FeatureMatrix` with ONE copy
    of the numeric plane (pre-allocated + filled column-by-column). Categorical columns go to a
    separate int code plane; numeric data is cast to ``dtype`` (default float32)."""
    framework = _detect_framework(X)
    if framework == "pyarrow":
        import pandas as pd  # pyarrow Table -> pandas is the cheapest correct bridge for P0
        X = X.to_pandas()
        framework = "pyarrow"  # remember the SOURCE so we route back to pyarrow on the way out

    if framework == "numpy":
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n, p = arr.shape
        numeric = np.ascontiguousarray(arr, dtype=dtype)
        cols = [str(i) for i in range(p)]
        return FeatureMatrix(
            numeric=numeric, categorical=np.empty((n, 0), dtype=np.int64),
            columns=cols, col_kind=["numeric"] * p, col_index=list(range(p)),
            n_rows=n, framework="numpy", null_mask=~np.isfinite(numeric),
            dtype=dtype,
        )

    if framework == "polars":
        columns = list(X.columns)
        getcol = lambda nm: _polars_column_to_arrays(X[nm], dtype)  # noqa: E731
    else:  # pandas
        columns = [str(c) for c in X.columns]
        getcol = lambda nm: _pandas_column_to_arrays(X[nm], dtype)  # noqa: E731

    n = int(X.shape[0])
    p = len(columns)
    # First pass: classify columns (cheap) so we can pre-size both planes for a single fill pass.
    parsed = [getcol(nm if framework == "polars" else columns[i]) for i, nm in enumerate(columns)]
    num_positions = [i for i, pr in enumerate(parsed) if pr[0] is not None]
    cat_positions = [i for i, pr in enumerate(parsed) if pr[1] is not None]

    numeric = np.empty((n, len(num_positions)), dtype=dtype)
    categorical = np.empty((n, len(cat_positions)), dtype=np.int64)
    null_mask = np.zeros((n, p), dtype=bool)
    col_kind: list[str] = [""] * p
    col_index: list[int] = [0] * p
    categories: dict[str, list] = {}

    nj = cj = 0
    for i, (vals, codes, cats, nmask) in enumerate(parsed):
        null_mask[:, i] = nmask
        if vals is not None:
            numeric[:, nj] = vals  # ONE write per numeric column into the pre-allocated plane
            col_kind[i] = "numeric"; col_index[i] = nj; nj += 1
        else:
            categorical[:, cj] = codes
            col_kind[i] = "categorical"; col_index[i] = cj; cj += 1
            categories[columns[i]] = list(cats) if cats is not None else []

    return FeatureMatrix(
        numeric=numeric, categorical=categorical, columns=columns,
        col_kind=col_kind, col_index=col_index, n_rows=n, framework=framework,
        categories=categories, null_mask=null_mask, dtype=dtype,
    )


def from_feature_matrix(fm: FeatureMatrix):
    """Reconstruct the source-framework object from a :class:`FeatureMatrix`, restoring categorical
    labels and original missingness. Routes back to the framework that produced ``fm``."""
    n = fm.n_rows
    data: dict[str, np.ndarray] = {}
    col_objs: dict[str, Any] = {}
    for j, name in enumerate(fm.columns):
        if fm.col_kind[j] == "numeric":
            col = fm.numeric[:, fm.col_index[j]].copy()
            if fm.null_mask is not None:
                col = col.astype(np.float64)
                col[fm.null_mask[:, j]] = np.nan
            data[name] = col
            col_objs[name] = col
        else:
            codes = fm.categorical[:, fm.col_index[j]]
            labels = fm.categories.get(name, [])
            col_objs[name] = (codes, labels)

    if fm.framework == "numpy":
        # numeric-only round-trip (numpy input had no categoricals)
        return fm.numeric.copy()

    if fm.framework == "polars":
        import polars as pl
        out = {}
        for name in fm.columns:
            if fm.col_kind[fm.columns.index(name)] == "numeric":
                out[name] = col_objs[name]
            else:
                codes, labels = col_objs[name]
                vals = [labels[c] if 0 <= c < len(labels) else None for c in codes.tolist()]
                out[name] = vals
        return pl.DataFrame(out)

    # pandas (and pyarrow source -> return pandas, the cheapest faithful bridge for P0)
    import pandas as pd
    out_pd = {}
    for name in fm.columns:
        if fm.col_kind[fm.columns.index(name)] == "numeric":
            out_pd[name] = col_objs[name]
        else:
            codes, labels = col_objs[name]
            out_pd[name] = pd.Categorical.from_codes(
                np.clip(codes, -1, len(labels) - 1), categories=labels,
            )
    df = pd.DataFrame(out_pd)
    if fm.framework == "pyarrow":
        import pyarrow as pa
        return pa.Table.from_pandas(df, preserve_index=False)
    return df
