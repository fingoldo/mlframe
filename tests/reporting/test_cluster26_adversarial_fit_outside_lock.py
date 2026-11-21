"""Audit 2026-11-20 finding 26-F22: the adversarial fit runs while the cache lock is free.

The fit is seconds long. Holding the lock across it queued every other target's panel behind it. The test patches the fit and
records whether the lock is held at the moment the fit runs; it must be free.
"""

import numpy as np
import pandas as pd

from mlframe.reporting import _diagnostics_adversarial as adv


def test_fit_runs_with_the_cache_lock_released(monkeypatch, tmp_path):
    """The validation fit is called without _ADVERSARIAL_LOCK held."""
    seen = []

    def fit_probe(*args, **kwargs):
        """Record whether the cache lock is held while the fit runs."""
        seen.append(adv._ADVERSARIAL_LOCK.locked())
        raise RuntimeError("stop after the probe")

    from mlframe.reporting.charts import drift

    monkeypatch.setattr(drift, "adversarial_validation", fit_probe)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    adv._render_adversarial_panel(
        train_frame=frame, test_frame=frame.iloc[:20], val_frame=None, non_calendar=["a", "b"], calendar=[],
        plot_outputs="none", base_path=str(tmp_path / "run"), charts=None, seed=0,
    )
    assert seen == [False]
