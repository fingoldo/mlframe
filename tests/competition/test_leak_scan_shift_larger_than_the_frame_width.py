"""sort_by_density_leak_scan clamps max_col_shift to the frame width instead of failing on an empty column overlap."""
import numpy as np
import pandas as pd

from mlframe.competition.leak_scan import sort_by_density_leak_scan


def test_three_columns_with_max_col_shift_five_runs():
    """At shift 4 the slice mask_int[i, :3 - 4] kept two columns while block[:, 4:] was empty, and the matmul raised."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((20, 3)), columns=list("abc"))
    df.iloc[::3, 0] = np.nan
    out = sort_by_density_leak_scan(df, max_col_shift=5)
    assert isinstance(out, dict)
