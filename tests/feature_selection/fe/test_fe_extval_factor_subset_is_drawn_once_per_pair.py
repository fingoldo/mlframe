"""External-validation factors are subsampled once per pair, so tied leaders are compared on the same factor set."""
import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit_extval as extval


def test_every_config_of_a_pair_sees_the_same_capped_factor_set(monkeypatch):
    """The cap used to be applied inside the per-config helper (a fresh draw per config); the helper must now receive an already-capped factor list."""
    seen = []
    real = extval._emit_pair_features_step2_read_here_once_part2

    def spy(*args, **kwargs):
        """Record the factor list and cap the helper was handed."""
        seen.append((len(args[12]), args[13]))
        return real(*args, **kwargs)

    monkeypatch.setattr(extval, "_emit_pair_features_step2_read_here_once_part2", spy)
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(6)})
    y = ((X["f0"] * X["f1"] + X["f2"] / (1 + X["f3"].abs()) + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    MRMR(verbose=0, random_state=0, n_jobs=1, fit_cache_max=0, fe_max_external_validation_factors=2).fit(X, y)
    assert seen, "no pair reached the external-validation loop with tied leaders; the fixture needs adjusting"
    assert all(n_ext <= 2 and cap in (0, None, 2) for n_ext, cap in seen), seen
    assert all(cap in (0, None) for n_ext, cap in seen if n_ext == 2), seen
