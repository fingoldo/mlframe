"""When most resamples fail, the CI spans the surviving range and is flagged degraded (audit 2026-11-20, finding 23-F1)."""
import numpy as np

from mlframe.evaluation.bootstrap import bootstrap_metrics


def test_mostly_failing_metric_gets_a_full_range_and_a_flag():
    """A metric that fails on most resamples: the interval spans every survivor and the result is marked degraded."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(float)
    p = rng.uniform(size=400)
    calls = {"n": 0}

    def flaky(yt, yp):
        """Fail on 80% of calls, return the mean error otherwise."""
        calls["n"] += 1
        if calls["n"] > 1 and calls["n"] % 5 != 0:  # the first call is the point estimate and must succeed
            raise ValueError("boom")
        return float(np.mean(np.abs(yt - yp)))

    out = bootstrap_metrics(y, p, {"flaky": flaky}, n_bootstrap=200, method="percentile", random_state=0, n_jobs=1)["flaky"]
    assert out["degraded"] is True
    assert out["lo"] == float(np.min(out["samples"]))
    assert out["hi"] == float(np.max(out["samples"]))
