"""Bootstrap / calibration diagnostics seeds derive from the suite master seed (A7-14).

Previously each per-target block used a fixed 0 seed, so distinct targets shared identical bootstrap draws and the run
was not reproducible from the single suite seed. The per-target seed must now be a deterministic function of
(master_seed, key): stable across runs with the same master seed, distinct across targets, and varying with the seed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_derive_seed_is_deterministic_and_key_dependent():
    """Derive seed is deterministic and key dependent."""
    from mlframe.training.honest_diagnostics import _derive_seed

    assert _derive_seed(42, "binary/t1/cb") == _derive_seed(42, "binary/t1/cb")  # stable
    assert _derive_seed(42, "binary/t1/cb") != _derive_seed(42, "binary/t2/cb")  # distinct targets
    assert _derive_seed(42, "binary/t1/cb") != _derive_seed(7, "binary/t1/cb")  # varies with master seed
    s = _derive_seed(42, "binary/t1/cb")
    assert 0 <= s < 2**31 - 1  # int32-safe for sklearn / numpy splitters


def test_bootstrap_seed_flows_from_master_seed():
    """run_honest_diagnostics must derive per-target bootstrap seeds from ctx.split_config.random_seed."""
    from mlframe.training.honest_diagnostics import run_honest_diagnostics

    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) > 0.5).astype(int)
    p = np.clip(y * 0.6 + rng.normal(scale=0.3, size=n) + 0.2, 0.01, 0.99)
    probs = np.column_stack([1 - p, p])

    entry = SimpleNamespace(model_name="cb", model=None, test_target=y, test_probs=probs, test_preds=(p > 0.5).astype(int))
    models = {"binary_classification": {"tgt": [entry]}}

    ctx = SimpleNamespace(train_df=None, val_df=None, test_df=None, data_dir="", models_dir="", split_config=SimpleNamespace(random_seed=1234))
    out = run_honest_diagnostics(ctx, models, metadata={})
    # The bootstrap block ran with the derived seed; assert it produced CIs (reproducible artefact present).
    key = "binary_classification/tgt/cb"
    block = out["bootstrap_ci"][key]
    assert "roc_auc" in block and "ci_lo" in block["roc_auc"], block
    # Reproducibility: a second run with the same master seed yields an identical AUC point + CI.
    out2 = run_honest_diagnostics(ctx, models, metadata={})
    b2 = out2["bootstrap_ci"][key]
    assert block["roc_auc"]["ci_lo"] == b2["roc_auc"]["ci_lo"]
    assert block["roc_auc"]["ci_hi"] == b2["roc_auc"]["ci_hi"]
