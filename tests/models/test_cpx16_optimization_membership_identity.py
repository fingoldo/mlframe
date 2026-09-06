"""CPX16 regression: ``MBHOptimizer.suggest_candidate`` membership-test optimization.

The candidate-selection loops replaced ``x not in self.known_candidates`` (an O(K)
ndarray scan) with ``x not in known_candidates_set`` (a Python set built once per
call from ``self.known_candidates.tolist()``). This must NOT change which candidate
is suggested for a given RNG state / inputs: ``np.int64(5) in {5}`` resolves via
``__hash__``/``__eq__`` exactly like the ndarray scan, regardless of np-scalar vs
python-scalar dtype, as long as the values match.

The identity test drives a full optimization run (suggest -> submit -> repeat) with
a fixed seed and asserts the suggested-candidate sequence is bit-identical to the
sequence produced by the OLD (HEAD) implementation loaded via ``git show``.
"""

import numpy as np
import pytest

from mlframe.models.optimization import MBHOptimizer, NOT_READY


def _run_sequence(MBHOpt, n_space: int, n_steps: int, seed: int) -> list:
    """Drive a full optimization loop and return the suggested-candidate sequence."""
    rng = np.random.default_rng(seed)
    search_space = np.arange(n_space)
    # A deterministic synthetic ground truth so submit_evaluations has real signal.
    ground_truth = np.sin(search_space / 7.0) + 0.1 * search_space / n_space
    opt = MBHOpt(
        search_space=search_space,
        ground_truth=ground_truth,
        model_name="ETR",
        model_params={"n_estimators": 8, "random_state": 0},
        init_num_samples=8,
        random_state=seed,
    )
    suggested = []
    for _ in range(n_steps):
        c = opt.suggest_candidate()
        if c is None or c is NOT_READY:
            # Submit a seed point to make the surrogate trainable, then retry once.
            seed_pt = int(rng.integers(0, n_space))
            opt.submit_evaluations([seed_pt], [float(ground_truth[seed_pt])], [0.0])
            continue
        suggested.append(int(c))
        opt.submit_evaluations([int(c)], [float(ground_truth[int(c)])], [0.0])
    return suggested


# The commit that introduced the CPX16 membership set. Its PARENT is the last revision whose
# ``suggest_candidate`` still tested membership with ``next_candidate not in self.known_candidates``, the
# O(K) ndarray scan this identity test exists to compare against.
#
# It has to be a pinned SHA. The previous form read ``HEAD:src/mlframe/models/optimization.py``, which on a
# clean worktree is byte-identical to the live file -- and worse, ``MBHOptimizer`` has since moved out of
# that module entirely, leaving only a re-export, so exec'ing it handed back the live class object. Both
# sides of the comparison were the same code: reverting the optimization, or reintroducing the int-cast that
# corrupts fractional search spaces, would have been applied to both and the test could not fail.


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_the_membership_set_agrees_with_the_ndarray_on_every_candidate_a_real_run_considers(seed):
    """The set membership CPX16 introduced must return the ndarray's verdict on real candidate values.

    This replaces an old-versus-new sequence comparison that could not fail. It exec'd
    ``git show HEAD:src/mlframe/models/optimization.py`` and called the ``MBHOptimizer`` it found there --
    but on a clean worktree that file is byte-identical to the live one, and ``MBHOptimizer`` has since
    moved out of it entirely, leaving a re-export, so both sides of the comparison were the same class
    object. Reverting the optimization would have been applied to both.

    Pinning the pre-CPX16 revision instead does not work either: that code stores its constructor
    arguments through ``pyutilz.store_params_in_object`` without ``postfix=""``, and the helper's default
    postfix has since changed to ``_param_``, so the old class no longer even builds here (it raises
    ``AttributeError: 'MBHOptimizer' object has no attribute 'direction'``). A historical revision is not a
    usable reference once its dependencies have moved under it.

    What CPX16 actually had to preserve is testable directly, and on the values a real run produces rather
    than a fabricated comparison: for every candidate the loop considers, ``x in set(known.tolist())`` must
    give the same answer as ``x in known``. The numpy scalars coming out of ``search_space[idx]`` are where
    that equivalence is not obvious.
    """
    rng = np.random.default_rng(seed)
    n_space, n_steps = 400, 120
    search_space = np.arange(n_space)
    ground_truth = np.sin(search_space / 7.0) + 0.1 * search_space / n_space
    opt = MBHOptimizer(
        search_space=search_space,
        ground_truth=ground_truth,
        model_name="ETR",
        model_params={"n_estimators": 8, "random_state": 0},
        init_num_samples=8,
        random_state=seed,
    )

    checked = 0
    suggested = []
    for _ in range(n_steps):
        known = np.asarray(opt.known_candidates)
        if known.size:
            known_set = set(known.tolist())
            for idx in range(n_space):
                x = search_space[idx]
                assert (x in known_set) == bool(x in known), f"membership verdicts disagree for {x!r} (seed={seed})"
                checked += 1
        c = opt.suggest_candidate()
        if c is None or c is NOT_READY:
            seed_pt = int(rng.integers(0, n_space))
            opt.submit_evaluations([seed_pt], [float(ground_truth[seed_pt])], [0.0])
            continue
        suggested.append(int(c))
        opt.submit_evaluations([int(c)], [float(ground_truth[int(c)])], [0.0])

    assert len(suggested) > 20, f"run too short to be a meaningful check (seed={seed}): {len(suggested)} suggestions"
    assert checked > 1000, f"the membership equivalence was barely exercised (seed={seed}): {checked} comparisons"


def test_membership_set_matches_ndarray_for_np_scalar_keys():
    """The set built from ``known_candidates.tolist()`` must yield the same membership
    verdict as ``x in ndarray`` for the np-scalar keys produced by ``search_space[idx]``."""
    known = np.array([3, 17, 999, 50000], dtype=np.int64)
    known_set = set(known.tolist())
    search_space = np.arange(100001)
    for idx in (3, 4, 17, 18, 999, 1000, 50000, 50001):
        x = search_space[idx]  # numpy scalar, as in suggest_candidate
        assert (x in known_set) == (x in known), f"membership mismatch at {idx}"


def test_known_candidates_preserve_float_dtype_on_continuous_search_space():
    """MODELS-1 (2026-08-05 audit): submit_evaluations force-cast known_candidates to int (twice --
    once for the live tracker, once for the plotting-only temp array), corrupting/mis-tracking a
    continuous (non-integer) search_space. Once truncated to int, `next_candidate not in
    known_candidates_set` (built from the same truncated ints) can never recognise an already-evaluated
    FRACTIONAL candidate as known, so the optimizer keeps re-suggesting points it already evaluated.
    Drives a real optimization loop over a fractional search_space and asserts every suggested candidate
    is genuinely new, and known_candidates keeps its float dtype throughout."""
    search_space = np.linspace(0.0, 10.0, 401)  # step 0.025 -- every value fractional
    ground_truth = np.sin(search_space)
    opt = MBHOptimizer(
        search_space=search_space,
        ground_truth=ground_truth,
        model_name="ETR",
        model_params={"n_estimators": 8, "random_state": 0},
        init_num_samples=8,
        random_state=0,
    )
    suggested_so_far = set()
    rng = np.random.default_rng(1)
    n_resuggested = 0
    for _ in range(80):
        c = opt.suggest_candidate()
        if c is None or c is NOT_READY:
            seed_pt = float(rng.choice(search_space))
            opt.submit_evaluations([seed_pt], [float(np.sin(seed_pt))], [0.0])
            continue
        if c in suggested_so_far:
            n_resuggested += 1
        suggested_so_far.add(c)
        opt.submit_evaluations([c], [float(np.sin(c))], [0.0])

    assert n_resuggested == 0, f"{n_resuggested} already-evaluated fractional candidates were re-suggested"
    assert opt.known_candidates.dtype.kind == "f", f"expected known_candidates to retain float dtype, got {opt.known_candidates.dtype}"
