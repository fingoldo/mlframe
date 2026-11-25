"""Cluster 32 regression: the init-score combiner is fit on the training rows only (audit 2026-11-20, finding 32-F11).

The quick model holds out 20% of the rows. A combiner fit on every row let the holdout labels shape the holdout's init_score, so the delta was
optimistic. The combiner now fits on the same training rows the quick model uses; flipping holdout labels must not move any base value.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.baselines._baseline_diagnostics_init_score import _combined_features_base
from mlframe.training.baselines._baseline_diagnostics_quick_model import _quick_train_mask
from mlframe.training.configs import BaselineDiagnosticsConfig


def _frame(n: int = 120, seed: int = 0):
    """Two numeric features and a regression target driven by both."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 2.0 * X["a"].to_numpy() - X["b"].to_numpy() + 0.1 * rng.normal(size=n)
    return X, y


def test_train_mask_matches_the_eighty_twenty_split():
    """The mask keeps 80% of the rows as training, and is None below 50 rows (32-F11)."""
    cfg = BaselineDiagnosticsConfig()
    _X, y = _frame(120)
    mask = _quick_train_mask(cfg, y, "regression")
    assert mask.sum() == 96
    assert _quick_train_mask(cfg, y[:40], "regression") is None


def test_holdout_labels_do_not_move_any_base_value():
    """Flipping only holdout labels leaves every base value unchanged, so the combiner never saw them (32-F11)."""
    cfg = BaselineDiagnosticsConfig()
    X, y = _frame(120)
    chosen = [SimpleNamespace(feature="a"), SimpleNamespace(feature="b")]
    mask = _quick_train_mask(cfg, y, "regression")
    base, _ = _combined_features_base(X, y, chosen, "regression", fit_rows=mask)
    y_flipped = y.copy()
    y_flipped[~mask] = -y_flipped[~mask] * 10.0  # holdout labels change, training labels do not
    base2, _ = _combined_features_base(X, y_flipped, chosen, "regression", fit_rows=mask)
    assert np.allclose(base, base2)


def test_training_labels_do_move_the_base_values():
    """Control: changing a TRAINING label does move the combiner, so the mask is not a no-op (32-F11)."""
    cfg = BaselineDiagnosticsConfig()
    X, y = _frame(120)
    chosen = [SimpleNamespace(feature="a"), SimpleNamespace(feature="b")]
    mask = _quick_train_mask(cfg, y, "regression")
    base, _ = _combined_features_base(X, y, chosen, "regression", fit_rows=mask)
    y_changed = y.copy()
    y_changed[mask] = y_changed[mask] + 5.0
    base2, _ = _combined_features_base(X, y_changed, chosen, "regression", fit_rows=mask)
    assert not np.allclose(base, base2)
