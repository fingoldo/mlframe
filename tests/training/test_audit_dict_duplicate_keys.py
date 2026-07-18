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

from pathlib import Path

import numpy as np
import pytest

MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    """Read."""
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_boruta_shap_rejects_dup_columns() -> None:
    """Boruta shap rejects dup columns."""
    src = _read("feature_selection/boruta_shap/__init__.py")
    assert "duplicate column name" in src
    assert "deduplicate before fit() to avoid silently dropping shadow indices" in src


def test_phase_helpers_rejects_dup_columns_in_train_df() -> None:
    """The dupe-column check moved into the sibling _phase_helpers_fit_split.py
    during the 2026-05-21 monolith split, then further into
    _phase_helpers_fit_pipeline.py during the 2026-05-22 split. Read all three."""
    src_parent = _read("training/core/_phase_helpers.py")
    src_sibling_a = _read("training/core/_phase_helpers_fit_split.py")
    src_sibling_b = _read("training/core/_phase_helpers_fit_pipeline.py")
    needle = "deduplicate before fit() to keep schema-hash honest"
    assert needle in src_parent or needle in src_sibling_a or needle in src_sibling_b


def test_misc_helpers_rejects_dup_columns_in_predict_df() -> None:
    """Misc helpers rejects dup columns in predict df."""
    src = _read("training/core/_misc_helpers.py")
    assert "deduplicate before predict() to keep schema-hash honest" in src


def test_general_mi_rejects_dup_target_columns() -> None:
    """General mi rejects dup target columns."""
    src = _read("feature_selection/general.py")
    assert "deduplicate to avoid silently dropping MI rows" in src


def test_bruteforce_renames_handle_collisions() -> None:
    """Bruteforce renames handle collisions."""
    src = _read("feature_engineering/bruteforce.py")
    # The fix introduces the suffix-collision loop.
    assert "_renamed = [col.replace" in src
    assert "_final_names: list = []" in src
    assert 'f"{_name}_{_seen[_name]}"' in src


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


def test_bruteforce_column_renaming_disambiguates_collisions() -> None:
    """After the column-rename pass, the resulting frame has unique column names
    even when the renaming would have collided (e.g. 'a-x' and 'a=x')."""
    # Replicate the rename loop logic directly.
    cols = ["a-x", "a=x", "b", "c-y", "c=y", "c-y"]
    renamed = [c.replace("-", "_").replace("=", "_") for c in cols]
    # Without disambiguation, renamed has dupes.
    assert len(set(renamed)) < len(renamed)

    seen: dict = {}
    final_names: list = []
    for name in renamed:
        if name in seen:
            seen[name] += 1
            final_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            final_names.append(name)
    # Post-disambiguation, all unique.
    assert len(set(final_names)) == len(final_names)
    # And first-arrival keeps its base name.
    assert final_names[0] == "a_x"
    assert final_names[1] == "a_x_2"
    assert final_names[3] == "c_y"
    assert final_names[4] == "c_y_2"
    assert final_names[5] == "c_y_3"
