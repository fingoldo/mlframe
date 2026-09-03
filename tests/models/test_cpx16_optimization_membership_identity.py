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

import importlib.util
import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys
from pathlib import Path

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


def _new_sequence(n_space, n_steps, seed):
    """Helper: New sequence."""
    return _run_sequence(MBHOptimizer, n_space, n_steps, seed)


def _old_sequence(n_space, n_steps, seed):
    """Load the HEAD version of optimization.py and run the same sequence on it."""
    repo_root = Path(__file__).resolve().parents[2]
    rel = "src/mlframe/models/optimization.py"
    out = subprocess.run(  # nosec B603 B607 -- fixed local argv (sys.executable/git + literal args), not a partial/searched path from untrusted input, no shell
        ["git", "show", f"HEAD:{rel}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        pytest.skip(f"git show unavailable: {out.stderr.strip()}")
    old_src = out.stdout
    tmp = repo_root / "tests" / "models" / "_cpx16_optimization_OLD_tmp.py"
    tmp.write_text(old_src, encoding="utf-8")
    try:
        spec = importlib.util.spec_from_file_location("mlframe.models.optimization", tmp)
        mod = importlib.util.module_from_spec(spec)
        saved = sys.modules.get("mlframe.models.optimization")
        sys.modules["mlframe.models.optimization"] = mod
        try:
            spec.loader.exec_module(mod)
            return _run_sequence(mod.MBHOptimizer, n_space, n_steps, seed)
        finally:
            if saved is not None:
                sys.modules["mlframe.models.optimization"] = saved
            else:
                del sys.modules["mlframe.models.optimization"]
    finally:
        tmp.unlink(missing_ok=True)


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_suggest_sequence_identical_old_vs_new(seed):
    """Suggest sequence identical old vs new."""
    n_space, n_steps = 400, 120
    new_seq = _new_sequence(n_space, n_steps, seed)
    old_seq = _old_sequence(n_space, n_steps, seed)
    assert len(new_seq) > 20, "run too short to be a meaningful identity check"
    assert new_seq == old_seq, (
        f"suggested-candidate sequence diverged (seed={seed}): "
        f"first mismatch at {next((i for i, (a, b) in enumerate(zip(new_seq, old_seq)) if a != b), 'len-only')}"
    )


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
