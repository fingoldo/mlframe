"""Constants, enums, and small helpers shared by ``optimization.py`` and ``_optimization_search.py``.

Split out (2026-07-11) to break the import cycle the Wave 100 monolith split introduced: ``optimization.py``
defined these names and imported ``MBHOptimizer``/``_ETRWithStd`` back from ``_optimization_search.py`` at the
bottom (for BC re-export), while ``_optimization_search.py`` imported these shared names from ``optimization``
at the top -- two modules each partially importing the other. Both siblings now import from this leaf module
instead, so neither has to wait on the other's ``__init__``.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any, Optional, Sequence, Union

import numpy as np


class _LazyModule:
    """Transparent lazy proxy: imports the wrapped module on first attribute
    access. Keeps matplotlib (~0.15s) off the eager import path -- this module
    is reachable from feature-selection imports, yet plt is only used by the
    optimizer's plotting callbacks.
    """

    def __init__(self, name: str):
        self._lm_name = name
        self._lm_mod: Optional[Any] = None

    def __getattr__(self, attr: str) -> Any:
        if self._lm_mod is None:
            import importlib

            self._lm_mod = importlib.import_module(self._lm_name)
        return getattr(self._lm_mod, attr)


plt = _LazyModule("matplotlib.pyplot")

SMALL_VALUE = 1e-4
BIG_VALUE = 1e6


# Sentinel distinguishing "surrogate not yet trainable" (transient: no evaluations submitted yet, or all
# known targets identical so the model cannot be fit) from a genuine None == "search space exhausted".
# Callers that drive the optimizer in a loop must NOT terminate on _NOT_READY -- it means "try again after
# submitting more evaluations", whereas None means "every candidate has been checked / suggested".
class _SearchNotReady:
    """Sentinel type for :data:`NOT_READY`; see the module comment above for the NOT_READY-vs-None contract."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "NOT_READY"


NOT_READY = _SearchNotReady()


class OptimizationDirection(Enum):
    """Whether the optimizer is hunting for the highest or lowest evaluation."""

    Minimize = auto()
    Maximize = auto()


class CandidateSamplingMethod(Enum):
    """Strategy for picking the initial (pre-surrogate) exploration candidates."""

    Random = auto()
    Equidistant = auto()
    Fibonacci = auto()
    ReversedFibonacci = auto()


class OptimizationProgressPlotting(Enum):
    """How often ``MBHOptimizer`` renders its diagnostic plot during the search loop."""

    No = auto()
    Final = auto()  # Plotting is done once, after the search finishes
    OnScoreImprovement = auto()  # Plotting is done on every improving candidate
    Regular = auto()  # Plotting is done on every candidate


def compute_candidates_exploration_scores(search_space: Sequence, known_candidates: Sequence) -> np.ndarray:
    """Compute distances from all candidates to known points.
    Assuming search_space is sorted.
    """

    distances = np.zeros(len(search_space))  # distances to closest checked points
    if len(known_candidates) == 0:
        # No checked points yet -> every search-space point is maximally far; the loop below would leave r/lo unbound.
        return distances
    indices = {el: i for i, el in enumerate(search_space)}

    lo = None
    for i in sorted(known_candidates):
        r = indices[i]
        if lo is None:
            distances[:r] = np.abs(search_space[0:r] - search_space[r])
        else:
            m = (lo + r) // 2
            distances[lo:m] = np.abs(search_space[lo:m] - search_space[lo])
            distances[m:r] = np.abs(search_space[m:r] - search_space[r])
        lo = r
    distances[r:] = np.abs(search_space[r:] - search_space[r])

    return distances


def generate_fibonacci(n: int) -> np.ndarray:
    """Creates Fibonacci sequence for a given n."""

    if n <= 0:
        return np.array([], dtype=np.int64)

    fibonacci_sequence = [0, 1]
    while len(fibonacci_sequence) < n:
        next_number = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_number)

    return np.array(fibonacci_sequence, dtype=np.int64)


def plot_search_state(
    search_space: Union[Sequence, np.ndarray],
    next_cand: int,
    new_y: float,
    best_candidate: Optional[int],
    best_evaluation: float,
    nsteps: int,
    expected_fitness: Optional[np.ndarray],
    y_pred: Optional[np.ndarray],
    y_std: Optional[np.ndarray],
    ground_truth: Optional[np.ndarray],
    known_candidates: np.ndarray,
    known_evaluations: np.ndarray,
    skip_candidates: Sequence,
    acquisition_method: str,
    mode: str,
    additional_info: str,
    figsize: tuple = (8, 4),
    font_size: int = 10,
    x_label: str = "nfeatures",
    y_label: str = "score",
    expected_fitness_color: str = "green",
    legend_location: str = "lower right",
) -> None:
    """Render the current optimizer state: known evaluations, the surrogate's predicted fitness (+ std band when available), and the just-suggested candidate, on a dual-axis matplotlib figure.

    Purely diagnostic -- called from :meth:`MBHOptimizer.submit_evaluations` when ``plotting`` requests it (``OnScoreImprovement`` / ``Regular``); never affects the search itself.
    """

    # ---------------------------------------------------------------------------------------------------------------
    # Plot expected fitness of the points
    # ---------------------------------------------------------------------------------------------------------------

    plt.rcParams.update({"font.size": font_size})
    fig, axMain = plt.subplots(sharex=True, figsize=figsize, layout="tight")
    axExpectedFitness = axMain.twinx()

    if expected_fitness is not None:
        axExpectedFitness.plot(search_space, expected_fitness, color=expected_fitness_color, linestyle="dashed", label=acquisition_method, alpha=0.3)
        # axExpectedFitness.plot(search_space, y_std, color=expected_fitness_color,linestyle='dashed', label='y_std')
        # axExpectedFitness.plot(search_space, distances, color=expected_fitness_color,linestyle='dotted', label='distances')

    # ---------------------------------------------------------------------------------------------------------------
    # Plot the black box function, surrogate function, known points
    # ---------------------------------------------------------------------------------------------------------------

    if ground_truth is not None:
        axMain.plot(search_space, ground_truth, color="black", label="Ground truth")
    if y_pred is not None:
        axMain.plot(search_space, y_pred, color="red", linestyle="dashed", label="Surrogate Function")
        axMain.fill_between(search_space, y_pred - y_std, y_pred + y_std, color="blue", alpha=0.2)

    axMain.scatter(known_candidates, known_evaluations, color="blue", label="Known Points")

    if skip_candidates:
        idx = ~np.isin(known_candidates, skip_candidates)
        if idx.sum() > 0:
            axMain.set_ylim([known_evaluations[idx].min(), None])

    axExpectedFitness.set_yticklabels([])
    axExpectedFitness.set_yticks([])
    axExpectedFitness.set_ylabel(acquisition_method, color=expected_fitness_color)
    # axExpectedFitness.legend()
    axMain.set_xlabel(x_label)
    axMain.set_ylabel(y_label)

    # ---------------------------------------------------------------------------------------------------------------
    # Plot next candidate
    # ---------------------------------------------------------------------------------------------------------------

    axMain.scatter(next_cand, new_y, color="red", marker="D", label="Next candidate")

    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.title(f"Iteration #{nsteps}, mode={mode} {additional_info}")
    axMain.set_title(f"Iteration #{nsteps}, mode={mode} {additional_info}, best={best_evaluation:.6f}@{best_candidate:_}")
    axMain.legend(loc=legend_location)
    # Non-blocking show: ``plt.show()`` (default block=True) made the Qt
    # window MODAL, freezing the optimisation loop until the user closed
    # every figure manually. RFECV with optimizer_plotting='OnScoreImprovement'
    # spawns a window per score improvement -> dozens of stuck modals on
    # the desktop. block=False renders the window non-modally; the
    # plt.pause() flush gives the GUI event loop a tick to draw.
    try:
        plt.show(block=False)
        plt.pause(0.001)
    except Exception:  # nosec B110 - non-trivial body
        # Headless / Agg backend: show is a no-op, pause may not work
        # without a backend. Failure here must NEVER block training.
        pass
    plt.close(fig)
