"""Regression test: _pair_eng_col_name must not mislabel mixed-basis pairs.

Pre-fix: the sole call site passed ``basis_i if basis_i == basis_j else basis_i`` -- a useless
if-else (RUF034) that always evaluated to ``basis_i`` regardless of ``basis_j``. When
``basis="auto"`` routes the two columns of a pair to DIFFERENT basis families (independent
per-column moment routing), the emitted column name silently claimed both legs used ``basis_i``'s
code even though leg b was actually evaluated with ``basis_j``. The computed VALUES were always
correct (``h_a``/``h_b`` used their own bases); only the metadata name lied.
"""
from __future__ import annotations


def test_pair_eng_col_name_same_basis_unchanged():
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import _pair_eng_col_name

    assert _pair_eng_col_name("a", "b", "hermite", "hermite", 2, 3) == "a*b__He2_He3"


def test_pair_eng_col_name_mixed_basis_reflects_both_legs():
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import _pair_eng_col_name

    name = _pair_eng_col_name("a", "b", "hermite", "chebyshev", 2, 3)
    assert name == "a*b__He2_T3", f"pre-fix this collapsed to 'a*b__He2_He3' (leg b's basis silently dropped); got {name}"
