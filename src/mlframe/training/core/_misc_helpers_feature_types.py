"""Feature-type auto-detection carved out of ``_misc_helpers``.

Holds the three functions that decide which columns are text / embedding / categorical and validate that no
column claims two of those roles at once. Split off because the parent crossed the 1000-LOC budget; the parent
re-exports every name so historical ``from ._misc_helpers import _auto_detect_feature_types`` imports resolve.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)


def _filter_polars_cat_features_by_dtype(
    df: pl.DataFrame,
    cat_features: list[str],
) -> list[str]:
    """Defensive filter for CB Polars fastpath ``cat_features``: keep only Categorical/Enum dtypes.

    CB 1.2.x's Cython fused cpdef dispatcher only matches pl.Categorical (and on some builds pl.Enum); other dtypes
    raise opaque ``TypeError: No matching signature found``. Drops mismatched columns with WARNING; missing columns silently.
    """
    valid: list = []
    dropped: list = []
    for c in cat_features or []:
        if c not in df.columns:
            continue
        dt = df.schema[c]
        is_cat = (dt == pl.Categorical) or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum))
        if is_cat:
            valid.append(c)
        else:
            dropped.append((c, str(dt)))
    if dropped:
        logger.warning(
            "Dropping %d column(s) from CB cat_features because their "
            "Polars dtype is not Categorical/Enum: %s. CatBoost's fastpath "
            "dispatcher has no overload for those types and would raise "
            "'No matching signature found'. Most likely cause: the column "
            "was promoted from cat_features to text_features and cast to "
            "pl.String, but the caller is still passing the pre-promotion "
            "list. Fix the caller to use the post-promotion cat_features.",
            len(dropped), dropped,
        )
    return valid


def _auto_detect_feature_types(
    df,
    feature_types_config,
    cat_features: list,
    verbose: bool = False,
    pandas_meta: dict | None = None,
) -> tuple:
    """Auto-detect text/embedding features and promote high-cardinality string/categorical columns to text_features.

    Promotion criteria: not user-assigned, dtype is pl.String/pl.Utf8/pl.Categorical (pl.Enum stays nominal),
    n_unique > threshold. Does NOT mutate ``cat_features`` (caller filters via set-difference).

    ``pandas_meta`` is the mutation-immune snapshot built by ``_phase_fit_pipeline`` (``train_df_pandas_pre_meta``);
    when supplied AND the caller is on the pandas path, every read goes through the dict instead of ``df``, so any
    later in-place mutation on the source frame cannot corrupt the detection result. Polars path is unchanged (the
    polars-pre frame is already a public-API alias and is conceptually immutable).

    Returns: ``(text_features, embedding_features, auto_detected_high_card_to_drop)``.
    """
    import polars as pl

    _ftc = feature_types_config
    text_features = list(_ftc.text_features or []) if _ftc is not None else []
    embedding_features = list(_ftc.embedding_features or []) if _ftc is not None else []
    # ``use_text_features=True``: auto-detected cols -> text_features. ``False``: -> auto_detected_high_card_to_drop
    # so caller drops them entirely (prevents XGB QuantileDMatrix OOM and CB artefact bloat on 2M-level cats).
    # User-supplied explicit text_features/embedding_features are honored regardless.
    auto_detected_high_card_to_drop: list = []

    if feature_types_config is None or not feature_types_config.auto_detect_feature_types:
        return text_features, embedding_features, auto_detected_high_card_to_drop

    if cat_features is None:
        cat_features = []

    # Metadata-dict path is only meaningful for pandas inputs; polars inputs use the (immutable-by-API) polars-pre frame.
    use_meta = pandas_meta is not None and not isinstance(df, pl.DataFrame)

    abs_threshold = feature_types_config.cat_text_cardinality_threshold
    # Minimum non-null FRACTION to promote; below it CB's TF-IDF estimator yields an empty dictionary and raises
    # "Dictionary size is 0" (text_feature_estimators.cpp). Fraction (not count) scales with dataset size.
    min_non_null_frac = getattr(feature_types_config, "min_non_null_fraction_for_text_promotion", 0.01)
    if use_meta:
        assert pandas_meta is not None  # guaranteed by the ``use_meta`` construction above
        total_rows = int(pandas_meta["shape"][0])
    else:
        total_rows = df.height if hasattr(df, "height") else len(df)
    min_non_null_abs = max(1, round(total_rows * min_non_null_frac))
    # Size-aware effective promotion threshold: a flat 300-uniq floor is wrong at both ends of the data-size axis
    # (on 100-row data every string column stays "cat"; on 10M-row data 300 is still a sliver). pct=0 keeps legacy
    # behaviour (effective == absolute). The 50-uniq floor prevents pathologically tiny effective thresholds.
    pct_threshold = getattr(feature_types_config, "cat_text_cardinality_threshold_pct", 0.0) or 0.0
    if pct_threshold > 0.0:
        threshold = min(abs_threshold, max(50, int(total_rows * pct_threshold)))
    else:
        threshold = abs_threshold
    user_assigned = set(text_features) | set(embedding_features)
    promoted: list = []
    cardinalities: dict = {}
    skipped_low_non_null: list = []
    promote_text = feature_types_config.use_text_features
    # honor_user_dtype: pre-cast categorical dtypes (pl.Categorical / pl.Enum / pandas category) are treated
    # as user-declared and skip auto-promotion; raw pl.String / pl.Utf8 / object/string stay candidates.
    honor_user_dtype = getattr(feature_types_config, "honor_user_dtype", False)
    honored_user_dtype_cols: list = []

    if isinstance(df, pl.DataFrame):
        # Accept all embedding-shaped dtypes:
        # - pl.List(pl.Float32/Float64): the legacy variable-length float embedding.
        # - pl.Array(<inner>, N):        polars>=0.20 fixed-size embeddings; backends
        #                                that auto-densify treat the row as a length-N
        #                                vector regardless of inner dtype.
        # - pl.List(pl.Int*):            quantized 8/16/32-bit embeddings (e.g.
        #                                Sentence-Transformers int8 export); the row
        #                                is still a vector, just stored compact.
        _pl_array_cls = getattr(pl, "Array", None)
        _int_inner_dtypes = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)

        def _is_embedding_dtype(dt) -> bool:
            """True when ``dt`` is a polars dtype that represents an embedding row: variable-length float/int list, or a fixed-size ``pl.Array``."""
            # Variable-length float embedding (legacy path).
            if dt == pl.List(pl.Float32) or dt == pl.List(pl.Float64):
                return True
            # Quantized int embedding stored as variable-length list.
            inner = getattr(dt, "inner", None)
            if isinstance(dt, pl.List) and inner is not None and inner in _int_inner_dtypes:
                return True
            # Fixed-size pl.Array(...) - any numeric inner is an embedding.
            if _pl_array_cls is not None and isinstance(dt, _pl_array_cls):
                if inner is not None and (inner in (pl.Float32, pl.Float64) or inner in _int_inner_dtypes):
                    return True
            return False

        # First pass is dtype-only (cheap, no kernel launches): route embeddings + honored + user_assigned cols, and
        # collect the residual text-like list. Then ONE lazy aggregation computes n_unique + count for every text-like
        # col in a single collect (was: 2 eager Series calls per col = 2N kernel launches; on 60 cols this was 50-200 ms).
        text_like_cols: list = []
        for name, dtype in df.schema.items():
            if name in user_assigned:
                continue
            if _is_embedding_dtype(dtype):
                if name not in cat_features:
                    embedding_features.append(name)
                continue
            # pl.Enum is a CLOSED, already-encoded nominal categorical (its category set is fixed at schema
            # time) -- never free text, so it must stay nominal unconditionally, not only when
            # honor_user_dtype is set (this function's own docstring already documents "pl.Enum stays
            # nominal" as the intended behavior; the code used to contradict it by including Enum in
            # is_text_like whenever honor_user_dtype was left at its False default). Promoting an Enum
            # column to text_features leaks its physical integer code (not the decoded string label) into
            # CatBoost's text-feature Pool construction, which then rejects it: "text_features must have
            # string type" -- caught live via a fuzz combo with a high-cardinality polars Enum column.
            # pl.Enum is an instance-level dtype (not a class), so isinstance() is required alongside the class-level check.
            is_enum = isinstance(dtype, pl.Enum)
            is_text_like = dtype in (pl.String, pl.Utf8, pl.Categorical) and not is_enum
            is_user_categorical_dtype = dtype == pl.Categorical or is_enum
            if is_enum or (honor_user_dtype and is_user_categorical_dtype):
                honored_user_dtype_cols.append(name)
                continue
            if is_text_like:
                text_like_cols.append(name)

        if text_like_cols:
            # Index-based aliases (__autodetect_nu_{i}__ / __autodetect_cnt_{i}__) are collision-proof: even a user
            # column literally named "__autodetect_nu_0__" cannot collide because we only read the aggregation
            # output, not the input frame's columns.
            _aggs = [pl.col(c).n_unique().alias(f"__autodetect_nu_{i}__") for i, c in enumerate(text_like_cols)] + [
                pl.col(c).count().alias(f"__autodetect_cnt_{i}__") for i, c in enumerate(text_like_cols)
            ]
            _agg_row = df.lazy().select(_aggs).collect()
            for i, name in enumerate(text_like_cols):
                n_unique = int(_agg_row[f"__autodetect_nu_{i}__"][0])
                if n_unique > threshold:
                    non_null = int(_agg_row[f"__autodetect_cnt_{i}__"][0])
                    if non_null < min_non_null_abs:
                        skipped_low_non_null.append((name, n_unique, non_null))
                        continue
                    cardinalities[name] = n_unique
                    if promote_text:
                        text_features.append(name)
                        if name in cat_features:
                            promoted.append(name)
                    else:
                        auto_detected_high_card_to_drop.append(name)
    else:
        # pandas path: prefer the mutation-immune ``pandas_meta`` dict snapshot when supplied (built by
        # ``_phase_fit_pipeline`` before the pipeline mutates dtypes / column set). Both branches share
        # the same promotion logic; only the source of column-list / dtype-string / n_unique / non-null /
        # embedding-shape-sniff differs.
        if use_meta:
            assert pandas_meta is not None  # guaranteed by the ``use_meta`` construction above
            _columns = pandas_meta["columns"]
            _dtypes = pandas_meta["dtypes"]
            _meta_n_unique = pandas_meta.get("n_unique", {})
            _meta_non_null = pandas_meta.get("non_null", {})
            _meta_embed_obj = set(pandas_meta.get("embedding_object_cols", []))
        else:
            _columns = list(df.columns)
            # same dupe-column hazard as _phase_helpers.py:1114;
            # silently-collapsing dtype dict would feed a wrong schema-hash downstream.
            if len(set(_columns)) != len(_columns):
                from collections import Counter as _Counter
                _dupes = [_c for _c, _n in _Counter(_columns).items() if _n > 1]
                raise ValueError(f"df has {len(_dupes)} duplicate column name(s) " f"({_dupes[:5]}); deduplicate before predict() to keep schema-hash honest.")
            _dtypes = {c: str(df[c].dtype) for c in _columns}
            _meta_n_unique = None
            _meta_non_null = None
            _meta_embed_obj = None

        nunique_cols: list = []
        # pandas 2.3+ / 3.0 surface object string columns under several
        # ``str(dtype)`` spellings that the legacy ("object","string",...)
        # prefix list missed, silently dropping every high-cardinality
        # text column to the numeric-only path (skills_text -> text=[]):
        #   * ``pd.StringDtype(na_value=nan)`` -> ``'<StringDtype(na_value=nan)>'``
        #     (observed big machine)
        #   * ``future.infer_string`` / pandas 3.0 default -> ``'str'``
        #     (observed big machine). ``'str'.startswith('string')``
        #     is False, so a bare ``str`` dtype slipped through.
        # The ``"str"`` token is a prefix of every string spelling
        # (str / string / string[python] / StringDtype...), so it
        # subsumes the old "string"/"stringdtype" tokens; "object" and
        # "category" stay explicit (they don't start with "str").
        _string_like_dtype_tokens = ("object", "str", "category")
        if use_meta:
            assert pandas_meta is not None  # guaranteed by the ``use_meta`` construction above
            _meta_non_string_cat = set(pandas_meta.get("non_string_category_cols", []))
        else:
            _meta_non_string_cat = None
        for col in _columns:
            if col in user_assigned:
                continue
            dtype_name = _dtypes[col]
            if honor_user_dtype and dtype_name == "category":
                honored_user_dtype_cols.append(col)
                continue
            _dtype_lc = dtype_name.lower().lstrip("<")
            _is_string_like = any(_dtype_lc.startswith(tok) for tok in _string_like_dtype_tokens) or "stringdtype" in _dtype_lc
            # A pandas 'category' dtype's categories can be ANY value type (bool/int/float), not just strings --
            # unlike polars Categorical/Enum, which are always string-backed. A non-string-categories column
            # promoted to text_features leaks its raw category value (e.g. a literal ``True``/``1``) into
            # CatBoost's text-feature Pool construction, which rejects it: "text_features must have string type"
            # (caught live via a fuzz combo with a non-string-categories 'category' column). Treat it like an
            # honored user dtype: never text-auto-promoted, regardless of cardinality.
            if _is_string_like and dtype_name.startswith("category"):
                if _meta_non_string_cat is not None:
                    _is_non_string_cat = col in _meta_non_string_cat
                else:
                    _cats_dtype = getattr(df[col].dtype, "categories", None)
                    _is_non_string_cat = _cats_dtype is not None and _cats_dtype.dtype.kind not in "OU"
                if _is_non_string_cat:
                    honored_user_dtype_cols.append(col)
                    continue
            if _is_string_like:
                # Skip object columns whose cells are ndarray / list (embedding vectors). nunique() hashes
                # the cells via PyObjectHashTable which raises ``TypeError: unhashable type: 'numpy.ndarray'``.
                # Treat them as embeddings: route to embedding_features and skip the cardinality check
                # (iter#44 fuzz finding). With the metadata dict the sniff was done at snapshot time so we
                # only consult the precomputed list; the legacy fallback path still probes the live series.
                if dtype_name.startswith("object"):
                    if use_meta:
                        _is_embedding = _meta_embed_obj is not None and col in _meta_embed_obj
                    else:
                        _series = df[col]
                        try:
                            _first = next((v for v in _series.head(8) if v is not None), None)
                        except Exception:
                            logger.debug("failed probing object column %r for embedding detection; treating as non-embedding", col, exc_info=True)
                            _first = None
                        _is_embedding = _first is not None and (
                            hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
                        )
                    if _is_embedding:
                        embedding_features.append(col)
                        if col in cat_features:
                            promoted.append(col)
                        continue
                nunique_cols.append(col)

        if nunique_cols:
            if use_meta:
                # n_unique / non_null are precomputed in the metadata snapshot for every text-candidate
                # column (string / object / category / bool). No frame is touched here -- the dict is the
                # sole source of truth, immune to any in-place mutation on the source train_df.
                _stats = [(col, int(_meta_n_unique[col]), int(_meta_non_null[col])) for col in nunique_cols]
            else:
                # Legacy fallback: ``df[cols].agg(["nunique","count"])`` returns a 2 x len(cols) frame
                # where row 0 is nunique and row 1 is count. pandas dispatches both reductions via its
                # block manager which is materially cheaper than the legacy N x (nunique + notna().sum())
                # per-column Python -> C round-trip.
                # PANDAS-AT-IN-AUDIT: one .loc(...).to_dict() per row beats N ``_agg.at`` lookups; .at is
                # a single-cell scalar accessor and pays a row-level reindex on each call.
                _agg = df[nunique_cols].agg(["nunique", "count"])
                _nunique_map = _agg.loc["nunique"].to_dict()
                _count_map = _agg.loc["count"].to_dict()
                _stats = [(col, int(_nunique_map[col]), int(_count_map[col])) for col in nunique_cols]
            for col, n_unique, non_null in _stats:
                if n_unique > threshold:
                    if non_null < min_non_null_abs:
                        skipped_low_non_null.append((col, n_unique, non_null))
                        continue
                    cardinalities[col] = n_unique
                    if promote_text:
                        text_features.append(col)
                        if col in cat_features:
                            promoted.append(col)
                    else:
                        auto_detected_high_card_to_drop.append(col)

    def _fmt_with_cardinality(names):
        """Format each name as ``name:n_unique`` (thousands-separated) when a cardinality was recorded, else the bare name, for log messages."""
        parts = []
        for n in names:
            nu = cardinalities.get(n)
            parts.append(f"{n}:{nu:_}" if nu is not None else n)
        return "[" + ", ".join(parts) + "]"

    if verbose and (text_features or embedding_features or promoted):
        if promoted:
            logger.info(
                "  Promoted %d high-cardinality column(s) from cat_features to text_features "
                "(threshold>%s): %s",
                len(promoted), threshold, _fmt_with_cardinality(promoted),
            )
        logger.info(
            "  Auto-detected feature types -- text: %s, embedding: %s",
            _fmt_with_cardinality(text_features) if text_features else "(none)",
            embedding_features or "(none)",
        )

    # Load-bearing: log drop-list regardless of verbose so operators see auto-dropped columns and why (silent drop bites).
    if auto_detected_high_card_to_drop:
        logger.warning(
            "  use_text_features=False: auto-dropping %d high-cardinality "
            "text-like column(s) (n_unique > %d) to prevent "
            "XGB QuantileDMatrix OOM / CB model-artefact bloat: %s. "
            "To keep these columns, set use_text_features=True (routes "
            "them to text_features -- CB uses them, XGB/LGB drop them) "
            "or add them explicitly to feature_types_config.text_features.",
            len(auto_detected_high_card_to_drop),
            threshold,
            _fmt_with_cardinality(auto_detected_high_card_to_drop),
        )

    # Load-bearing diagnostic: columns silently kept as cat_features instead of being promoted (avoids "Dictionary size is 0").
    if skipped_low_non_null:
        formatted = ", ".join(f"{name}:{n_unique:_} (non_null={nn:_}/{total_rows:_})" for name, n_unique, nn in skipped_low_non_null)
        logger.warning(
            "  Auto-detection: %d column(s) had n_unique>%d (would be "
            "promoted to text_features) but non_null<%d (%.1f%% of %d rows, "
            "below the %.2f%% floor) -- kept as cat_features to avoid "
            "CatBoost's 'Dictionary size is 0' error on sparse text "
            "columns: %s",
            len(skipped_low_non_null), threshold, min_non_null_abs,
            min_non_null_frac * 100, total_rows, min_non_null_frac * 100,
            formatted,
        )
    if honored_user_dtype_cols and verbose:
        logger.info(
            "  %d column(s) with explicit categorical dtype (pl.Categorical / pl.Enum / pandas category) "
            "kept out of text-auto-promotion regardless of cardinality: pl.Enum always (a closed, "
            "already-encoded nominal categorical); others only when honor_user_dtype=True: %s",
            len(honored_user_dtype_cols), sorted(honored_user_dtype_cols),
        )

    return text_features, embedding_features, auto_detected_high_card_to_drop


def _validate_feature_type_exclusivity(
    text_features: list,
    embedding_features: list,
    cat_features: list,
) -> None:
    """Raise ValueError if any column appears in multiple feature type lists. Each argument may be ``None`` (treated as empty)."""
    text_features = text_features or []
    embedding_features = embedding_features or []
    cat_features = cat_features or []
    overlap_tc = set(text_features) & set(cat_features)
    if overlap_tc:
        raise ValueError(f"Columns cannot be both text_features and cat_features: {overlap_tc}")
    overlap_ec = set(embedding_features) & set(cat_features)
    if overlap_ec:
        raise ValueError(f"Columns cannot be both embedding_features and cat_features: {overlap_ec}")
    overlap_te = set(text_features) & set(embedding_features)
    if overlap_te:
        raise ValueError(f"Columns cannot be both text_features and embedding_features: {overlap_te}")
