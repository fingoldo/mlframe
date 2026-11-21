"""The target-mix second pass records which bases it mixed, their slopes and the candidates each produced."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _target_mix_fe as target_mix_fe
from mlframe.feature_selection.filters.mrmr import MRMR


def test_provenance_names_the_base_and_slope_of_a_strong_column():
    """A design column that tracks y is chosen as the base, its OLS slope is recorded, and the provenance has one entry for it."""
    rng = np.random.default_rng(0)
    n = 2000
    u = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    y = 2.0 * u + 0.3 * x2 + 0.05 * rng.standard_normal(n)
    X = pd.DataFrame({"u": u, "x2": x2})
    found, provenance = target_mix_fe.second_pass_recipes(MRMR(fe_max_steps=1), X, y, {"u": u}, max_bases=1, scan_rows=n, min_relative_gain=0.001)
    assert len(provenance) == 1
    assert provenance[0]["base"] == "u"
    assert abs(provenance[0]["beta"] - 2.0) < 0.1
    assert set(provenance[0]["features"]) <= set(found)


def test_no_base_gives_empty_provenance():
    """Pure noise against y has no base that clears the held-out bar, so no recipes and no provenance."""
    rng = np.random.default_rng(1)
    n = 2000
    u = rng.standard_normal(n)
    y = rng.standard_normal(n)
    found, provenance = target_mix_fe.second_pass_recipes(MRMR(fe_max_steps=1), pd.DataFrame({"u": u}), y, {"u": u}, max_bases=1, scan_rows=n, min_relative_gain=0.001)
    assert found == {}
    assert provenance == []


def test_provenance_entries_record_the_child_fit_wall_and_rows():
    """Every base's provenance entry carries the wall and the row count of its child fit, so the cost of the second pass is visible per base."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_selection.filters import _target_mix_fe as tm

    class _Child:
        """Stand-in child selector: fits instantly, finds nothing."""

        _engineered_features_: list = []
        _engineered_recipes_: list = []

        def fit(self, X, y):
            """Fit nothing."""
            return self

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = rng.normal(size=n)
    u = rng.normal(size=n)
    orig = (tm.choose_bases, tm._child)
    try:
        tm.choose_bases = lambda design, y_, max_bases, min_relative_gain=0.0: [("u", u, 0.5)]
        tm._child = lambda estimator: _Child()
        import mlframe.feature_selection.filters._target_mix_gate as gate

        orig_gate = gate.target_has_structure
        gate.target_has_structure = lambda *a, **k: True
        try:
            _, prov = tm.second_pass_recipes(object(), X, y, {"u": u}, max_bases=1, scan_rows=1000, min_relative_gain=0.0)
        finally:
            gate.target_has_structure = orig_gate
    finally:
        tm.choose_bases, tm._child = orig
    assert len(prov) == 1
    assert prov[0]["child_rows"] == 200 and prov[0]["child_fit_s"] >= 0.0
