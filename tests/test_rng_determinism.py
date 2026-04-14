"""RNG determinism tests for mlframe.tuning and mlframe.optimization.

Verifies that:
- MBHOptimizer(random_state=42) produces identical suggest_candidate sequence
- ParamsOptimizer(random_state=42) produces identical generate_valid_candidates output
- Module bodies of tuning.py / optimization.py do not call np.random.* or bare random.random
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

import mlframe.tuning as tuning_mod
import mlframe.optimization as opt_mod


# --------------------------------------------------------------------
# AST check: no module-level np.random.* calls or bare random.random
# --------------------------------------------------------------------


def _collect_forbidden_calls(source: str) -> list:
    """Walk an AST for forbidden RNG patterns: `np.random.<x>()` and `random()` (bare)."""
    tree = ast.parse(source)
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # np.random.<x> pattern: Attribute(value=Attribute(value=Name('np'), attr='random'), attr='<x>')
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
                inner = func.value
                if isinstance(inner.value, ast.Name) and inner.value.id == "np" and inner.attr == "random":
                    # np.random.default_rng is the sanctioned seed-threadable factory; allow it.
                    if func.attr == "default_rng":
                        continue
                    forbidden.append(f"np.random.{func.attr} at line {node.lineno}")
            # bare `random()` — Call with func=Name('random')
            if isinstance(func, ast.Name) and func.id == "random":
                forbidden.append(f"bare random() call at line {node.lineno}")
        # module-level np.random.seed(...) ban also covered above
    return forbidden


@pytest.mark.parametrize("mod", [tuning_mod, opt_mod])
def test_no_np_random_or_bare_random(mod):
    path = Path(inspect.getsourcefile(mod))
    source = path.read_text(encoding="utf-8")
    forbidden = _collect_forbidden_calls(source)
    assert not forbidden, f"{path.name} contains forbidden RNG calls: {forbidden}"


# --------------------------------------------------------------------
# MBHOptimizer determinism
# --------------------------------------------------------------------


def _make_mbh(seed):
    pytest.importorskip("catboost")
    search_space = np.arange(1, 101, dtype=np.int32)
    return opt_mod.MBHOptimizer(
        search_space=search_space,
        init_num_samples=5,
        init_sampling_method=opt_mod.CandidateSamplingMethod.Random,
        random_state=seed,
    )


def test_mbh_suggest_candidate_deterministic():
    pytest.importorskip("catboost")
    opt1 = _make_mbh(42)
    opt2 = _make_mbh(42)

    seq1 = [opt1.suggest_candidate() for _ in range(5)]
    seq2 = [opt2.suggest_candidate() for _ in range(5)]
    assert seq1 == seq2, f"MBHOptimizer not deterministic: {seq1} vs {seq2}"


def test_mbh_init_sampled_inputs_deterministic():
    pytest.importorskip("catboost")
    opt1 = _make_mbh(123)
    opt2 = _make_mbh(123)
    assert list(opt1.pre_seeded_candidates) == list(opt2.pre_seeded_candidates)


# --------------------------------------------------------------------
# ParamsOptimizer / generate_valid_candidates determinism
# --------------------------------------------------------------------


def test_generate_valid_candidates_deterministic():
    from scipy.stats import randint, loguniform

    params = {
        "depth": randint(1, 10),
        "lr": loguniform(1e-3, 0.3),
        "kind": ["a", "b", "c"],
    }
    c1 = tuning_mod.generate_valid_candidates(params=params, n=5, random_state=42)
    c2 = tuning_mod.generate_valid_candidates(params=params, n=5, random_state=42)
    assert c1 == c2, f"generate_valid_candidates not deterministic: {c1} vs {c2}"


def test_params_optimizer_rng_is_seeded():
    o1 = tuning_mod.ParamsOptimizer(random_state=7)
    o2 = tuning_mod.ParamsOptimizer(random_state=7)
    # Draw a sequence from each rng; must match.
    a = [int(o1._rng.integers(0, 10**9)) for _ in range(10)]
    b = [int(o2._rng.integers(0, 10**9)) for _ in range(10)]
    assert a == b
    # stdlib rngs too
    a2 = [o1._stdlib_rng.random() for _ in range(10)]
    b2 = [o2._stdlib_rng.random() for _ in range(10)]
    assert a2 == b2
