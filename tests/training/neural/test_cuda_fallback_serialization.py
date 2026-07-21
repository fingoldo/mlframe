"""Regression: run_with_cuda_cpu_fallback must fully serialize concurrent Trainer-construction/predict calls.

Lightning's ``_replace_dunder_methods`` monkeypatches the ``DataLoader`` class itself (not per-instance) for the
duration of each Trainer call. Two threads entering concurrently -- exactly what sklearn's
``permutation_importance(..., n_jobs=-1)`` under joblib's threading backend does when calling
``estimator.predict()`` on ONE shared model from many worker threads -- leaves the class-level monkeypatch
wrap depth growing without bound, eventually blowing Python's recursion limit in an unrelated LATER Trainer
call (caught live: a fuzz combo's permutation-importance loop corrupted enough state that the SAME combo's
later PDP/ICE predict loop crashed with "RecursionError: maximum recursion depth exceeded" inside Lightning's
own wrapper). The identical concurrent-access root cause also underlies the CUDA device-churn race tested
elsewhere (test_fit_cuda_cpu_fallback.py / test_predict_cuda_cpu_fallback.py). This test doesn't need real
CUDA/Lightning -- it verifies the lock itself: run_with_cuda_cpu_fallback's run_fn never executes concurrently
across threads.
"""

from __future__ import annotations

import threading
import time

from mlframe.training.neural.base._cuda_fallback import run_with_cuda_cpu_fallback


def test_run_with_cuda_cpu_fallback_serializes_concurrent_calls():
    """Many threads calling run_with_cuda_cpu_fallback concurrently must never overlap inside run_fn."""
    n_threads = 8
    in_flight = {"count": 0, "max_seen": 0}
    lock = threading.Lock()

    def _run_fn(trainer):
        """Fake trainer call: record entry, sleep briefly to widen the race window, record exit."""
        with lock:
            in_flight["count"] += 1
            in_flight["max_seen"] = max(in_flight["max_seen"], in_flight["count"])
        time.sleep(0.02)
        with lock:
            in_flight["count"] -= 1
        return "ok"

    def _worker():
        """Worker."""
        result, trainer_used = run_with_cuda_cpu_fallback(
            action="predict",
            primary_trainer="primary",
            model=object(),
            accelerator="cpu",  # non-cuda accelerator: is_cuda_runtime_error always False, so no retry path taken
            run_fn=_run_fn,
            build_cpu_trainer=lambda: "cpu",
        )
        assert result == "ok"
        assert trainer_used == "primary"

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert in_flight["max_seen"] == 1, f"expected run_fn to never execute concurrently (max_seen=1), got max_seen={in_flight['max_seen']}"


def test_run_with_cuda_cpu_fallback_lock_does_not_deadlock_on_reentry():
    """The lock is an RLock -- the SAME thread must be able to call run_with_cuda_cpu_fallback reentrantly
    (e.g. a nested predict call from within another locked predict call) without deadlocking."""

    def _inner_run_fn(trainer):
        """Inner run fn."""
        return "inner"

    def _outer_run_fn(trainer):
        """Outer run fn triggers a nested call to run_with_cuda_cpu_fallback on the SAME thread."""
        result, _ = run_with_cuda_cpu_fallback(
            action="predict", primary_trainer="p2", model=object(), accelerator="cpu",
            run_fn=_inner_run_fn, build_cpu_trainer=lambda: "cpu2",
        )
        return f"outer+{result}"

    result, _ = run_with_cuda_cpu_fallback(
        action="predict", primary_trainer="p1", model=object(), accelerator="cpu",
        run_fn=_outer_run_fn, build_cpu_trainer=lambda: "cpu1",
    )
    assert result == "outer+inner"


def test_real_model_survives_concurrent_predict_stress():
    """End-to-end regression for the RecursionError caught live via permutation-importance's threading backend:
    many threads calling predict() concurrently on ONE shared, already-fitted PytorchLightningRegressor must
    all succeed with no RecursionError / no crash. This is the exact access pattern
    sklearn.inspection.permutation_importance(..., n_jobs=-1) under joblib's threading backend uses. Runs on
    CPU (no CUDA hardware assumed) -- the recursion bug is CUDA-independent (Lightning's DataLoader-class
    monkeypatch race triggers regardless of accelerator), so a CPU-only repro is a faithful regression guard."""
    import pytest

    torch = pytest.importorskip("torch")
    import numpy as np

    from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor, TorchDataModule

    est = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 1, "first_layer_num_neurons": 8},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={"max_epochs": 1, "logger": False, "accelerator": "cpu", "devices": 1, "enable_progress_bar": False},
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 4)).astype(np.float32)
    y = rng.normal(size=64).astype(np.float32)
    est.fit(X, y)

    errors: list = []

    def _worker():
        """Worker."""
        try:
            for _ in range(30):
                Xi = rng.normal(size=(20, 4)).astype(np.float32)
                est.predict(Xi)
        except Exception as e:  # noqa: BLE001 - capturing for the assertion below, not swallowing silently
            errors.append((type(e).__name__, str(e)))

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"expected zero errors across concurrent predict calls, got: {errors[:5]}"
