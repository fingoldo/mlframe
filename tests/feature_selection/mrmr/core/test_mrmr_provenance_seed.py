"""Wave 9.1 loop-iter-7 regression: provenance trail MUST record the
actual seed regardless of which ctor API the user used.

Pre-fix: ``MRMR.fit`` recorded ``self.provenance_['seed']`` by reading
``self.random_state``, but the ctor at ``mrmr.py:641`` promotes
``random_state -> random_seed`` (one-way, not symmetric). When the user
passed the documented ``random_seed=42`` API directly, ``self.random_state``
stayed at its default ``None`` and the provenance silently recorded
``seed=None`` even though the actual kernel seed was 42.

Effect: downstream consumers (mlflow.log_param, training_provenance,
reproducibility audit trails) could not reconstruct the seed used.

Fix: read ``self.random_seed`` (the public-API canonical name); both
APIs end up with it populated because of the ctor promotion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _frame(n=80, seed=0):
    """Build a small random classification frame used purely to drive an MRMR fit for provenance inspection."""
    rng = np.random.default_rng(int(seed))
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.integers(0, 2, n), name="y")
    return X, y


def test_provenance_seed_records_when_user_passes_random_seed():
    """The documented public API ``random_seed=`` must populate the
    provenance trail. Pre-fix this returned ``None`` silently.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame()
    sel = MRMR(random_seed=42, verbose=0).fit(X, y)
    assert (
        sel.provenance_["seed"] == 42
    ), f"random_seed=42 must produce provenance seed=42; got {sel.provenance_['seed']!r} - reproducibility audit trail silently lost."


def test_provenance_seed_records_when_user_passes_random_state():
    """sklearn-compat alias ``random_state=`` must also flow through."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame()
    sel = MRMR(random_state=42, verbose=0).fit(X, y)
    assert sel.provenance_["seed"] == 42


def test_provenance_seed_none_when_unset():
    """No seed passed -> provenance records None (not 0, not int(None))."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame()
    sel = MRMR(verbose=0).fit(X, y)
    assert sel.provenance_["seed"] is None
