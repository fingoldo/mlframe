"""MRMR accepts ``redundancy_aggregator`` 'jmim' and 'auto' and rejects an unknown one (the Fleuret synergy mitigation docs/MRMR_RESEARCH.md calls shipped).

It fits a real MRMR, so it is an ordinary test: it used to sit in tests/test_meta, which is for properties of the repository, and cost that suite minutes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_fleuret_module_confirms_jmim_synergy_mitigation_shipped():
    """The mitigation the doc calls shipped is real: MRMR fits under ``redundancy_aggregator`` 'jmim' and 'auto' and rejects an unknown aggregator."""
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = (X["a"] + X["b"] > 0).astype(int)
    for aggregator in ("jmim", "auto"):
        selector = MRMR(redundancy_aggregator=aggregator, verbose=0, n_workers=1, full_npermutations=0, baseline_npermutations=5)
        assert selector.fit(X, y) is selector
        assert selector.get_params()["redundancy_aggregator"] == aggregator
    with pytest.raises(ValueError, match="redundancy_aggregator must be one of"):
        MRMR(redundancy_aggregator="bogus", verbose=0, n_workers=1, full_npermutations=0, baseline_npermutations=5).fit(X, y)
