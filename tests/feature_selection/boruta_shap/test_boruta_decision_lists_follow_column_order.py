"""BorutaShap's accepted/rejected/tentative lists follow the column order, not set iteration (tracker 2026-11-02 wave 8, RNG8-1)."""

import types

import numpy as np

from mlframe.feature_selection.boruta_shap import BorutaShap


def test_lists_are_in_all_columns_order():
    """Set iteration of strings varies with PYTHONHASHSEED; the lists must keep the data's column order."""
    cols = [f"feat_{i:02d}" for i in range(40)]
    fake = types.SimpleNamespace(
        all_columns_=np.array(cols),
        accepted_columns_=[[c] for c in cols[::3]],
        rejected_columns_=[[c] for c in cols[1::3]],
        flatten_list=lambda xs: [x for sub in xs for x in sub],
        accepted_=None,
        y_=None,
    )
    BorutaShap.calculate_rejected_accepted_tentative(fake, verbose=False)
    assert fake.accepted_ == cols[::3]
    assert fake.rejected_ == cols[1::3]
    assert fake.tentative_ == cols[2::3]
