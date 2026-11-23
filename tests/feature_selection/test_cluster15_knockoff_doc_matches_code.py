"""Cluster 15 regression: the knockoff helper's docstring describes the real construction (audit 2026-11-20, finding 15-F19).

The knockoffs are built from the raw values and only the real slot is mean-imputed afterwards. Source-level pin on the docstring and on
the code order it describes.
"""

from pathlib import Path

from mlframe.feature_selection.wrappers import _knockoffs as ko

_SRC = Path(ko.__file__).resolve()


def test_docstring_says_knockoffs_use_raw_values():
    """The docstring no longer claims the knockoff is built from mean-imputed values (15-F19)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "The knockoffs are built from the raw values" in text
    assert "the knockoff is built from mean-imputed values" not in text


def test_knockoff_built_before_real_slot_imputation():
    """The knockoff is constructed from X_arr before the real slot is imputed (15-F19)."""
    text = _SRC.read_text(encoding="utf-8")
    build = text.index("X_tilde = make_gaussian_knockoffs(X_arr")
    impute = text.index("X_arr = np.where(np.isnan(X_f), np.nanmean(X_f, axis=0), X_arr)")
    assert build < impute
