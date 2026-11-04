"""MBHOptimizer counts warm-start evaluations toward its best and does not let a failed (infinite) evaluation become the best or worst."""
import numpy as np

from mlframe.models._optimization_search import MBHOptimizer, OptimizationDirection


def _optimizer(direction, known_c=None, known_e=None):
    """A tiny optimizer over the integers 0..19."""
    return MBHOptimizer(search_space=np.arange(20), direction=direction, known_candidates=known_c, known_evaluations=known_e, init_num_samples=2, verbose=0)


def test_warm_start_best_is_the_returned_best():
    """known_candidates=[10], known_evaluations=[0.99] with Maximize: the 0.99 point can never be re-evaluated, so it must already be the best."""
    opt = _optimizer(OptimizationDirection.Maximize, [10], [0.99])
    assert opt.best_candidate == 10 and abs(opt.best_evaluation - 0.99) < 1e-12
    opt.submit_evaluations(candidates=[3], evaluations=[0.5], durations=[0.0])
    assert opt.best_candidate == 10


def test_infinite_evaluation_does_not_become_best_or_worst():
    """Minimize with +inf from a crashed candidate: +inf was recorded as the worst, and -inf as the best, scaling the exploration distances by an infinite range."""
    opt = _optimizer(OptimizationDirection.Minimize)
    opt.submit_evaluations(candidates=[3, 4], evaluations=[1.0, np.inf], durations=[0.0, 0.0])
    assert opt.best_evaluation == 1.0 and np.isfinite(opt.worst_evaluation)
    opt.submit_evaluations(candidates=[5], evaluations=[-np.inf], durations=[0.0])
    assert opt.best_evaluation == 1.0
