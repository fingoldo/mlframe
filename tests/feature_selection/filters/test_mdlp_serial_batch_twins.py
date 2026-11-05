"""The serial per-node permutation test and the batched one draw the same permutations: their verdicts agree draw for draw."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._mdlp_validated_split import _mdlp_permutation_batch_njit
from mlframe.feature_selection.filters._numba_utils_cross_module import _accept_split_serial
from mlframe.feature_selection.filters.supervised_binning import _mdlp_best_split_njit


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("effect", [0.0, 0.35])
def test_serial_and_batched_verdicts_agree(seed, effect):
    """Borderline nodes (weak effect, small n) are where two different permutation streams would disagree; the two twins must not."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(60, 400))
    x = np.sort(rng.standard_normal(n))
    y = (rng.random(n) < 0.5 + effect * (x > 0)).astype(np.int64)
    _, gain, _, _, _ = _mdlp_best_split_njit(x, y, 2, 5)
    n_perm, alpha, base_seed = 99, 0.05, 1234 + seed
    serial = _accept_split_serial(x, y, 2, float(gain), 5, n_perm, base_seed, alpha)
    batched = _mdlp_permutation_batch_njit(
        x[None, :].copy(), y[None, :].copy(), np.array([n], dtype=np.int64), np.array([gain]), np.array([2], dtype=np.int64), 5, n_perm, np.array([base_seed], dtype=np.int64), alpha
    )
    assert bool(serial) == bool(batched[0])


@pytest.mark.parametrize("seed", [0, 1, 99, 123456789])
def test_shared_shuffle_is_the_honest_null_permutation(seed):
    """The labels every MDLP split test permutes are read through the permutation the honest null draws for the same seed (``seeded_permutation``)."""
    from mlframe.feature_selection.filters._numba_utils_shared import seeded_permutation
    from mlframe.feature_selection.filters._numba_utils_shuffle import shuffle_labels_into

    y = np.random.default_rng(seed).integers(0, 5, 300).astype(np.int64)
    out = np.empty_like(y)
    shuffle_labels_into(y, out, seed)
    assert np.array_equal(out, y[seeded_permutation(len(y), seed)])
    assert np.array_equal(np.sort(out), np.sort(y))


def test_twin_modules_have_no_private_permutation_draw():
    """Every MDLP split test draws through ``shuffle_labels_into``: none of the twin modules calls numba's ``np.random`` on its own (a private copy is how the streams drifted apart)."""
    import ast
    import inspect
    import textwrap

    from mlframe.feature_selection.filters import _mdlp_validated_split, _numba_utils_cross_module

    for mod, funcs in ((_mdlp_validated_split, ("_permutation_null_gain_njit", "_mdlp_permutation_batch_njit")), (_numba_utils_cross_module, ("_accept_split_serial",))):
        for name in funcs:
            tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(mod, name).py_func)))
            called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            random_attrs = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "random"]
            assert "shuffle_labels_into" in called, f"{mod.__name__}.{name} must draw through shuffle_labels_into"
            assert not random_attrs, f"{mod.__name__}.{name} must not draw from numba's np.random"
