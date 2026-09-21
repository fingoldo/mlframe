"""The wrap-pass watchdogs must share one inner predict per (entry, split), not run two identical ones.

With the y-scale metric block on (``skip_wrap_pass_predict=False``), the universal watchdog and the additive watchdog
each called ``estimator_.predict`` on the same split frame and got the same array. The second now comes from the phase's
prediction memo; the wrapper's own predict and pre-clip predict are untouched, since their runtime statistics feed the
report and the model card.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

pytest.importorskip("sklearn")


def _run_suite(tmp_path, monkeypatch, *, memo: bool) -> collections.Counter:
    """Train a small composite suite with the metric block on, counting inner Ridge predicts per frame shape."""
    from sklearn.linear_model import Ridge

    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    from mlframe.training.core import train_mlframe_models_suite
    from tests.training.composite.test_composite_integration import (
        _LEAN_OUTPUT_CONFIG_KWARGS,
        _LEAN_REPORTING_CONFIG_KWARGS,
        _build_minimal_fte,
        _tvt_dataset,
    )

    calls: collections.Counter = collections.Counter()
    real_predict = Ridge.predict

    def counting_predict(self, X):
        """Count each inner predict by the frame it scores."""
        calls[getattr(X, "shape", None)] += 1
        return real_predict(self, X)

    monkeypatch.setattr(Ridge, "predict", counting_predict)
    if not memo:
        import mlframe.training.core._phase_composite_wrapping as wrapping

        monkeypatch.setattr(wrapping, "memo_predict", lambda model, frame: np.asarray(model.predict(frame), dtype=np.float64).reshape(-1))
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["TVT_prev"], transforms=["diff", "linear_residual"],
        mi_sample_n=200, top_k_after_mi=2, eps_mi_gain=-1.0, skip_wrap_pass_predict=False,
    )
    train_mlframe_models_suite(
        df=_tvt_dataset(n=400), target_name="target", model_name="m",
        features_and_targets_extractor=_build_minimal_fte(), mlframe_models=["linear"],
        output_config={"data_dir": str(tmp_path / "data"), "models_dir": "models", **_LEAN_OUTPUT_CONFIG_KWARGS},
        reporting_config=_LEAN_REPORTING_CONFIG_KWARGS, verbose=0, composite_target_discovery_config=cfg,
    )
    return calls


def test_the_watchdog_pair_costs_one_inner_predict(tmp_path, monkeypatch):
    """Against a run whose watchdogs predict independently, every split frame is scored exactly once less."""
    with_memo = _run_suite(tmp_path / "a", monkeypatch, memo=True)
    monkeypatch.undo()
    without = _run_suite(tmp_path / "b", monkeypatch, memo=False)
    saved = {shape: without[shape] - with_memo.get(shape, 0) for shape in without}
    assert sum(saved.values()) > 0, f"the memo saved no inner predicts: {dict(with_memo)} vs {dict(without)}"
    assert all(v >= 0 for v in saved.values()), f"the memo must never add predicts: {saved}"
