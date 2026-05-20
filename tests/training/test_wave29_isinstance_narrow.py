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
    return (_ROOT / rel).read_text(encoding="utf-8")


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


def test_mrmr_fit_coerces_polars_to_pandas():
    src = _read("feature_selection/filters/mrmr.py")
    assert "isinstance(X, _pl_for_isinstance.DataFrame)" in src, (
        "Wave 29 P1 regression: MRMR.fit no longer coerces polars DataFrame "
        "to pandas; downstream ``X[target_name] = y`` raises on polars "
        "in-place mutation."
    )
    assert "X = X.to_pandas()" in src


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
    # Post-fix marker:
    assert "import polars as _pl_local" in src
    assert "if isinstance(_df, _pl_local.DataFrame):\n                    _df = _df.to_pandas()" in src


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
