"""Wave 54 (2026-05-20): dict comprehension / dict(zip) silently drops dup keys.

Audit class: `{x.key: x.value for x in items}` / `dict(zip(keys, values))` /
`dict(pairs)` where keys can collide silently keeps only the LAST entry --
earlier entries vanish without any warning, count discrepancy, or signal.

1 P1 + 4 P2 fixes applied:

  P1:
    1. feature_selection/boruta_shap.py:646 (BorutaShap mapping)
       dict(zip(self.X.columns, np.arange(...))) silently collapsed duplicate
       column names to the LAST index; any earlier-duplicated column would
       never be shuffled/tested by the shadow-feature loop. Now raises.

  P2:
    2. training/core/_phase_helpers.py:1114 (train_df dtype snapshot)
       {c: str(train_df[c].dtype) for c in train_df.columns} silently
       collapsed dupe columns to one entry, feeding a wrong schema-hash
       downstream. Now raises explicitly.

    3. training/core/_misc_helpers.py:615 (predict-time df dtype snapshot)
       Same shape as #2 at the predict-time validate path.

    4. feature_selection/general.py:274 (MI features per target)
       pd.DataFrame({target_columns[col]: mi[col, :] for col in range(...)})
       silently dropped MI rows when target_columns had dupes. Now raises.

    5. feature_engineering/bruteforce.py:168 (column rename collisions)
       "col-x" + "col=x" both become "col_x" -> dupe columns. Now suffixes
       collisions with _2, _3, ... so each column retains unique identity.

Verified clean (do not refactor): all other dict-comp / dict-zip sites have
key sources guaranteed unique by upstream contract (sklearn classes_,
enumerate, range, families list, hash output, or all-equal-value init).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Duplicate-name guards, exercised through the code that owns them
# ---------------------------------------------------------------------------


def test_boruta_shap_rejects_dup_columns() -> None:
    """The shadow-feature column mapping refuses duplicate names and tells the caller how to fix the frame."""
    from mlframe.feature_selection import boruta_shap as bs_mod

    inst = bs_mod.BorutaShap.__new__(bs_mod.BorutaShap)
    inst.X = pd.DataFrame(np.zeros((3, 3)), columns=["a", "b", "a"])
    with pytest.raises(ValueError, match=r"duplicate column name.*deduplicate before fit\(\) to avoid silently dropping shadow indices"):
        inst.create_mapping_between_cols_and_indices()
    inst.X = pd.DataFrame(np.zeros((3, 3)), columns=["a", "b", "c"])
    assert inst.create_mapping_between_cols_and_indices() == {"a": 0, "b": 1, "c": 2}


def test_phase_helpers_rejects_dup_columns_in_train_df() -> None:
    """The fit-time dtype snapshot refuses a train frame with duplicate column names instead of collapsing them."""
    from mlframe.training.core._phase_helpers_fit_pipeline import _phase_fit_pipeline_string_object_category_columns

    frame = pd.DataFrame({"x": ["u", "v", "u"], "y": [1.0, 2.0, 3.0]})
    frame.columns = ["x", "x"]
    with pytest.raises(ValueError, match=r"train_df has 1 duplicate column name\(s\).*deduplicate before fit\(\) to keep schema-hash honest"):
        _phase_fit_pipeline_string_object_category_columns(True, False, frame, None)
    unique = frame.set_axis(["x", "z"], axis=1)
    _phase_fit_pipeline_string_object_category_columns(True, False, unique, None)


def test_misc_helpers_rejects_dup_columns_in_predict_df() -> None:
    """The predict-time feature-type detection refuses a frame with duplicate column names."""
    from mlframe.training.configs import FeatureTypesConfig
    from mlframe.training.core._misc_helpers_feature_types import _auto_detect_feature_types

    frame = pd.DataFrame({"x": ["u", "v", "u"], "y": [1.0, 2.0, 3.0]})
    frame.columns = ["x", "x"]
    cfg = FeatureTypesConfig(auto_detect_feature_types=True)
    with pytest.raises(ValueError, match=r"df has 1 duplicate column name\(s\).*deduplicate before predict\(\) to keep schema-hash honest"):
        _auto_detect_feature_types(frame, cfg, [])


def test_general_mi_rejects_dup_target_columns(monkeypatch) -> None:
    """Exhaustive feature search refuses a target list naming the same column twice rather than dropping an MI row."""
    import polars as pl

    from mlframe.feature_selection import general

    n_feat = 2
    bins = pl.DataFrame({"f0": [0, 1, 0], "f1": [1, 0, 1]}).to_pandas()
    monkeypatch.setattr(general, "clean_ram", lambda *a, **k: None)
    monkeypatch.setattr(general, "bin_numerical_columns", lambda **kw: (bins, kw["binned_targets"], None, [], None))
    monkeypatch.setattr(
        general,
        "estimate_features_relevancy",
        lambda **kw: ([], np.zeros((len(kw["target_columns"]), n_feat)), {}, []),
    )
    df = pl.DataFrame({"f0": [0.0, 1.0, 2.0], "f1": [1.0, 0.0, 1.0], "t": [0, 1, 0]})
    common = dict(
        df=df,
        exclude_columns=[],
        permuted_mutual_informations={},
        binned_targets=df.select("t"),
        mi_algorithms_ranking=[],
        binning_params={},
        efs_params={},
    )
    with pytest.raises(ValueError, match=r"target_columns has 1 duplicate\(s\).*deduplicate to avoid silently dropping MI rows"):
        general.run_efs(target_columns=["t", "t"], **common)
    *_head, features_mis = general.run_efs(target_columns=["t"], **common)
    assert list(features_mis.columns) == ["t", "feature"]


def test_bruteforce_renames_handle_collisions() -> None:
    """PySR column sanitising maps "-" and "=" to "_" and suffixes every collision, so no two columns share a name."""
    from mlframe.feature_engineering.bruteforce import sanitize_pysr_column_names

    cols = ["a-x", "a=x", "b", "c-y", "c=y", "c-y"]
    assert sanitize_pysr_column_names(cols) == ["a_x", "a_x_2", "b", "c_y", "c_y_2", "c_y_3"]
    assert sanitize_pysr_column_names(["p", "q"]) == ["p", "q"]


# ---------------------------------------------------------------------------
# Behavioural sensors
# ---------------------------------------------------------------------------


def test_boruta_shap_raises_on_dup_input_columns() -> None:
    """BorutaShap.create_mapping_between_cols_and_indices must raise on dupes."""
    import pandas as pd
    from mlframe.feature_selection import boruta_shap as bs_mod

    if "src" + "\\" + "mlframe" not in bs_mod.__file__ and "src/mlframe" not in bs_mod.__file__:
        pytest.skip(f"boruta_shap loaded from stale build path {bs_mod.__file__}")

    # Build a minimal instance with X having duplicate column names.
    inst = bs_mod.BorutaShap.__new__(bs_mod.BorutaShap)
    # pd.DataFrame allows non-unique columns via list-based ctor.
    inst.X = pd.DataFrame(np.zeros((3, 3)), columns=["a", "b", "a"])
    with pytest.raises(ValueError, match="duplicate column name"):
        inst.create_mapping_between_cols_and_indices()
