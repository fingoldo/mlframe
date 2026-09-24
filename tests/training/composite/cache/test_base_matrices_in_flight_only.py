"""A base's x_remaining copies exist only while its transforms are being scored.

Every base context used to hold its own (screen rows x features) float and prebinned copies from the build until the
candidate loop ended, so the loop's peak grew with the base count. The contexts now keep the dropped columns and dedup mask;
the copies are gathered by the base's first transform and dropped after its last.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def test_a_serial_screen_holds_one_bases_matrices_at_a_time(monkeypatch):
    import mlframe.training.composite.discovery._fit as fit_mod
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.normal(size=2000) for i in range(12)})
    X["y"] = 2 * X["f0"] + np.sin(X["f1"]) + rng.normal(0, 0.3, 2000)
    live: list = []
    real = fit_mod.eval_one_transform

    def spy(self, base, tn, t, *, base_contexts, **k):
        others = [ctx for b, ctx in base_contexts.items() if b != base and "_unary_result_memo" not in ctx]
        live.append(sum(ctx.get("x_remaining_matrix") is not None for ctx in others))
        return real(self, base, tn, t, base_contexts=base_contexts, **k)

    monkeypatch.setattr(fit_mod, "eval_one_transform", spy)
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, discovery_n_jobs=1, base_candidates=[f"f{i}" for i in range(6)],
                                         transforms=["diff", "linear_residual"], interaction_base_discovery_enabled=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CompositeTargetDiscovery(cfg).fit(X, "y", [f"f{i}" for i in range(12)], np.arange(len(X)))
    assert live, "no transform was scored"
    assert max(live) == 0, f"other bases held their matrices while one was scored: {live}"
