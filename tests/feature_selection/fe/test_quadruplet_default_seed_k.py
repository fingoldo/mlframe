"""The quadruplet stage seeds enough raw columns to cover several independent four-way structures."""
import inspect

from mlframe.feature_selection.filters.mrmr import MRMR


def test_class_defaults_cover_several_quadruplets():
    """seed_k 12 and top_count 6 are the constructor defaults (4 seeds meant a single candidate quadruplet and 2 of 6 planted products found)."""
    params = inspect.signature(MRMR.__init__).parameters
    assert params["fe_hybrid_orth_quadruplet_seed_k"].default == 12
    assert params["fe_hybrid_orth_quadruplet_top_count"].default == 6
