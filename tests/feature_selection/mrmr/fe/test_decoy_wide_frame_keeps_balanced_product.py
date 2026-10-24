"""A balanced product survives MRMR on a wide frame full of near-copies of the target.

The target is the sign of ``pa * pb`` plus noise, so neither factor has any marginal relevance. 260 weak copies of the target and 138 noise columns
surround them (p=400, n=5000). Two stages used to lose the pair: the sufficient-summary early stop, whose marginal-only test declared the selection
complete, and the end-of-fit subsumption pass, which conditioned on so many anchor compounds that it had no power and called the product redundant.
The fit takes about 25 minutes on a GTX 1050 Ti host, so the test is skipped in ``--fast`` mode.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.mark.slow
@pytest.mark.slow_only
def test_balanced_product_survives_260_weak_copies_of_the_target():
    """The product of the two balanced factors is selected as an engineered feature."""
    rng = np.random.default_rng(500)
    n = 5000
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    y = (a * b + 0.3 * rng.standard_normal(n) > 0).astype(int)
    cols = {f"d{k}": y + (3.0 + 0.002 * k) * rng.standard_normal(n) for k in range(260)}
    cols.update({f"n{k}": rng.standard_normal(n) for k in range(138)})
    cols["pa"], cols["pb"] = a, b
    X = pd.DataFrame(cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fs = MRMR(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, verbose=0, n_jobs=1, random_seed=0).fit(X, y)
    names = [str(s) for s in fs.get_feature_names_out()]
    assert any("pa" in name and "pb" in name for name in names), [name for name in names if "(" in name]
