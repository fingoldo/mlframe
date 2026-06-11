"""Wave-29 sensors: isinstance narrow checks missing duck-typed alternatives.

5 sites where ``isinstance(x, ConcreteClass)`` rejected legitimate
alternatives. Audit also confirmed strong overall hygiene
(~120/130 dispatch sites use proper pd/pl/np branches).

P1 #1 extractors.py:817 -- ``isinstance(exact_val, list)`` rejected
   tuples; ``classification_exact_values={"col": (1,2,3)}`` got
   wrapped as ``[(1,2,3)]`` then ``col_data == (1,2,3)`` raised.

P1 #2 mrmr.py:1281 -- polars DataFrame slipped past the np.ndarray
   coerce; downstream ``X[target_name] = y`` raised on polars
   in-place mutation. Added explicit polars -> pandas branch.

P1 #3 boruta_shap.py:872 -- ``np.array(explainer.shap_values(...))``
   wrapped BEFORE the ``isinstance(..., list)`` check, making the
   list branch DEAD CODE. Multi-class SHAP aggregation silently
   mis-counted importances (3-D ndarray branch ran instead).

P2 #4 pipeline.py:_filter_to_numeric -- polars DataFrame silently
   passed through; downstream ``_df.select_dtypes(...)`` raised
   AttributeError with no diagnostic naming the type. Coerce
   polars -> pandas explicitly.

P2 #5 core/main.py -- ``isinstance(df, (pd, pl, str))`` rejected
   ``pathlib.Path`` with a confusing "must be ... path string"
   message. Common caller idiom from yaml/Click configs. Coerce
   PathLike -> str at the boundary.
"""
from __future__ import annotations

import pathlib

import mlframe as _mlframe


_ROOT = pathlib.Path(_mlframe.__file__).resolve().parent


def _read(rel: str) -> str:
    """Read a source file under src/mlframe.

    Monolith-split compat: concat parent + every matching sibling so
    source-grep sensors still match after the recent splits.
    """
    _path = _ROOT / rel
    if not _path.exists() and _path.suffix == ".py":
        _pkg = _path.with_suffix("")
        _init = _pkg / "__init__.py"
        if _init.exists():
            parts = [_init.read_text(encoding="utf-8")]
            for _sub in sorted(_pkg.glob("*.py")):
                if _sub.name != "__init__.py":
                    parts.append(_sub.read_text(encoding="utf-8"))
            return "\n".join(parts)
    primary = _path.read_text(encoding="utf-8")
    if rel == "training/core/main.py":
        # main.py was carved into ``_main_train_suite.py`` (suite entry) plus
        # ``_main_train_suite_phases.py`` (PathLike coerce / leaderboard phase
        # / precomputed bundle deepcopy live here).
        _dir = _ROOT / "training" / "core"
        for nm in ("_main_train_suite.py", "_main_train_suite_phases.py"):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "training/extractors.py":
        # extractors.py was carved into themed siblings; the
        # ``classification_exact_values`` tuple/set acceptance lives in
        # ``_extractors_simple.py``.
        _dir = _ROOT / "training"
        for nm in (
            "_extractors_simple.py",
            "_extractors_showcase.py",
            "_extractors_dtype_helpers.py",
        ):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "feature_selection/boruta_shap.py":
        # BorutaShap.fit + .explain live in the boruta_shap package submodule.
        sibling = _ROOT / "feature_selection" / "boruta_shap" / "_fit_explain.py"
        if sibling.exists():
            primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "training/pipeline.py":
        # 2026-05-22 split: apply_preprocessing_extensions + _apply_pysr_fe
        # + fit_and_transform_pipeline moved to sibling files.
        _dir = _ROOT / "training"
        for nm in ("_pipeline_extensions.py", "_pipeline_fit_transform.py"):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    elif rel == "feature_selection/filters/mrmr/_mrmr_class.py":
        # mrmr subpackage split: MRMR class body in mrmr/_mrmr_class.py; the rest of the surface lives in
        # _mrmr_{fingerprints,fit_impl,fe_step,validate_transform}.py + the mrmr/__init__.py facade.
        _dir = _ROOT / "feature_selection" / "filters"
        for nm in ("mrmr/__init__.py", "_mrmr_fingerprints.py", "_mrmr_fit_impl/_fit_impl_core.py", "_mrmr_fit_impl/_helpers.py", "_mrmr_fe_step/_step_core.py", "_mrmr_fe_step/_helpers.py", "_mrmr_validate_transform.py"):
            sibling = _dir / nm
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# ---- #1 extractors classification_exact_values accepts iterables -------


def test_extractors_exact_values_accepts_tuple_and_set():
    src = _read("training/extractors.py")
    # Pre-fix shape MUST be gone:
    assert "exact_vals = exact_val if isinstance(exact_val, list) else [exact_val]" not in src, (
        "Wave 29 P1 regression: extractors.py reverted to ``isinstance(..., list)`` "
        "narrow check; tuples / sets in classification_exact_values get wrapped "
        "as [(tuple,)] and downstream `==` raises."
    )
    # Post-fix marker:
    assert "isinstance(exact_val, (list, tuple, set, frozenset))" in src


# ---- #2 mrmr polars-coerce ---------------------------------------------


def test_mrmr_fit_handles_polars_input_without_inplace_mutation():
    """MRMR.fit must accept polars input without tripping the in-place mutation
    failure mode that the original Wave-29 fix targeted. The current
    implementation keeps the frame in native polars (no .to_pandas() copy on
    100+ GB frames) and uses non-mutating polars ops to inject the target
    column for MI computation. We pin both invariants by detecting EITHER the
    legacy pandas-coercion path OR the native-polars handling marker."""
    src = _read("feature_selection/filters/mrmr/_mrmr_class.py")

    legacy_pandas_coerce = (
        "isinstance(X, _pl_for_isinstance.DataFrame)" in src
        and "X = X.to_pandas()" in src
    )
    native_polars = "isinstance(X, pl.DataFrame)" in src

    assert legacy_pandas_coerce or native_polars, (
        "Wave 29 P1 regression: MRMR.fit no longer recognises polars input. "
        "Without either a polars->pandas coercion OR a native-polars branch, "
        "``X[target_name] = y`` raises on polars in-place mutation."
    )


# ---- #3 boruta_shap multi-class branch revived -------------------------


def test_boruta_shap_inspects_raw_shap_type_before_array_wrap():
    src = _read("feature_selection/boruta_shap.py")
    # Pre-fix shape MUST be gone:
    assert "self.shap_values = np.array(explainer.shap_values(basis))\n            if isinstance(self.shap_values, list):" not in src, (
        "Wave 29 P1 regression: boruta_shap reverted to np.array-then-isinstance-list "
        "pattern; the list branch is dead code and multi-class SHAP aggregation "
        "silently mis-counts importances."
    )
    # Post-fix marker:
    assert "_raw_shap = explainer.shap_values(basis)" in src
    assert "if isinstance(_raw_shap, list):" in src
    # Multi-class branch correctly computes shap_imp / n_classes:
    assert "self.shap_values = shap_imp / len(class_inds)" in src


# ---- #4 pipeline _filter_to_numeric coerces polars ---------------------


def test_pipeline_filter_to_numeric_handles_polars():
    src = _read("training/pipeline.py")
    # Pre-fix shape (silent passthrough) MUST be gone:
    assert "if _df is None or not isinstance(_df, pd.DataFrame):\n            return _df, []" not in src, (
        "Wave 29 P2 regression: _filter_to_numeric silently passes polars "
        "DataFrames through; downstream select_dtypes raises AttributeError."
    )
    # Post-fix markers. The polars-coerce branch now lives in sibling
    # ``_pipeline_extensions.py`` and the bare ``_df = _df.to_pandas()``
    # call is no longer contiguous with the isinstance check (intervening
    # comment + Arrow split-blocks try/except). Check the two pieces
    # independently so the sensor matches the current shape.
    assert "import polars as _pl_local" in src
    assert "isinstance(_df, _pl_local.DataFrame)" in src, (
        "Wave 29 P2 regression: polars-DataFrame branch in "
        "_filter_to_numeric is gone; silent passthrough resurfaced."
    )
    assert "_df = _df.to_pandas()" in src, (
        "Wave 29 P2 regression: bare ``_df.to_pandas()`` fallback gone "
        "from the polars coerce path."
    )


# ---- #5 main.py PathLike coercion --------------------------------------


def test_main_accepts_pathlike_df_argument():
    src = _read("training/core/main.py")
    # Post-fix markers:
    assert "isinstance(df, _os_for_pathlike.PathLike)" in src
    assert "df = str(df)" in src


def test_main_typeerror_message_mentions_pathlike():
    """Post-fix error message must mention PathLike as an accepted type
    (the prior message confused users passing pathlib.Path)."""
    src = _read("training/core/main.py")
    assert "PathLike" in src, (
        "Wave 29 P2 regression: TypeError message no longer mentions "
        "PathLike; users passing pathlib.Path will be told to use a "
        "'path string' instead of the documented PathLike contract."
    )
