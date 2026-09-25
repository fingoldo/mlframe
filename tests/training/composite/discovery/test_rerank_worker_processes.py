"""The parallel tiny-model rerank runs its specs in worker processes, and a dying worker does not take the caller down.

A production Jupyter kernel died three times out of three with a heap corruption while the rerank trained LightGBM on 16
threads of one process; the serial rerank passed. Worker processes give each spec's fits their own heap, and a fault
kills only the worker: the rerank then rescores its specs serially in the calling process.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.composite.discovery._tiny_rerank_process import (
    make_spec_task,
    rerank_backend,
    score_spec,
    score_specs_in_processes,
)
from mlframe.training.composite.transforms import get_transform

pytest.importorskip("lightgbm")


def _common(n=600, seed=0):
    """Inputs shared by every spec, sized so a 3-fold tiny LightGBM CV runs in well under a second."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    base = rng.normal(size=n)
    y = 2.0 * base + x[:, 0] + rng.normal(size=n)
    common = dict(
        y_screen=y, families=["lgb"], per_bin_enabled=False, per_bin_n_bins=5, use_wilcoxon=False, n_estimators=10,
        num_leaves=7, learning_rate=0.1, cv_folds=3, deterministic=True, n_seed_repeats=1, random_state=0, time_aware=False,
        groups=None, cv_selector_mode="mean", cv_selector_alpha=0.05, cv_selector_confidence=0.9,
        cv_selector_quantile_level=0.5, early_stop_threshold=float("inf"), fold_n_jobs=1, inner_n_jobs=1,
    )
    return common, base, x


def _spec(name: str, transform_name: str = "diff", y=None, base=None):
    """A spec whose fitted parameters come from the transform's own fit, as discovery's do."""
    params = get_transform(transform_name).fit(y, base) if y is not None else {}
    return SimpleNamespace(name=name, fitted_params=params, base_column="base")


def test_auto_and_unknown_backends_mean_processes():
    """Processes are the default because they are the mode that cannot take the kernel down; threads stay selectable."""
    assert rerank_backend("auto") == "processes"
    assert rerank_backend(None) == "processes"
    assert rerank_backend("threads") == "threads"
    assert rerank_backend("fibres") == "processes"


def test_a_worker_gathers_its_own_base_matrix_and_matches_the_in_process_score():
    """The process task carries the full matrix plus the dropped columns; the worker's gather gives the same score."""
    common, base, x = _common()
    x_full = np.column_stack([base.astype(np.float32), x])
    in_process = score_spec(make_spec_task(_spec("s"), get_transform("diff"), common, False, base, x_matrix=x))
    via_gather = score_spec(make_spec_task(_spec("s"), get_transform("diff"), common, False, base, x_full=x_full, drop_idx=[0]))
    assert in_process[1]["lgb"] == pytest.approx(via_gather[1]["lgb"], abs=1e-12)


def test_processes_return_the_serial_scores_in_task_order():
    common, base, x = _common()
    y = common["y_screen"]
    tasks = [
        make_spec_task(_spec(f"s{i}", t, y, base), get_transform(t), common, False, base, x_matrix=x)
        for i, t in enumerate(["diff", "ratio", "linear_residual"])
    ]
    serial = [score_spec(t) for t in tasks]
    parallel = score_specs_in_processes(tasks, n_jobs=2)
    assert [r[0] for r in parallel] == ["s0", "s1", "s2"]
    for s, p in zip(serial, parallel):
        assert s[1]["lgb"] == pytest.approx(p[1]["lgb"], abs=1e-12)


def test_a_dying_worker_leaves_the_caller_alive_and_rescored_serially(caplog):
    """The kernel survived nothing before: a native fault in any thread killed it. Now only the worker dies."""
    common, base, x = _common()
    parent = os.getpid()

    def _die_in_a_worker(parent_pid):
        """Unpickled in a worker: exit abruptly, as a native fault would. Unpickled in the caller: a normal transform."""
        import os as _os

        if _os.getpid() != parent_pid:
            _os._exit(3)
        from mlframe.training.composite.transforms import get_transform as _gt

        return _gt("diff")

    class _Fatal:
        def __reduce__(self):
            return (_die_in_a_worker, (parent,))

    tasks = [
        make_spec_task(_spec("fatal"), _Fatal(), common, False, base, x_matrix=x),
        make_spec_task(_spec("ok", "ratio", common["y_screen"], base), get_transform("ratio"), common, False, base, x_matrix=x),
    ]
    with caplog.at_level(logging.WARNING):
        results = score_specs_in_processes(tasks, n_jobs=2)
    assert [r[0] for r in results] == ["fatal", "ok"]
    assert all(np.isfinite(r[1]["lgb"]) for r in results)
    assert any("worker process died" in r.getMessage() for r in caplog.records)


def test_a_skipped_spec_does_no_work():
    """Honest-OOF measured it already: the CV fits would be discarded."""
    common, base, x = _common()
    assert score_spec(make_spec_task(_spec("s"), get_transform("diff"), common, True, base, x_matrix=x)) == ("s", {}, {}, None)
