"""The screening path (the greedy screen, its confirmations and the Fleuret recheck) builds no joblib pool: the multi-worker recheck is a device launch or sequential njit segments."""

import ast
from pathlib import Path

import pytest

FILTERS = Path(__file__).resolve().parents[3] / "src" / "mlframe" / "feature_selection" / "filters"
SCREENING = [
    "fleuret.py",
    "_confirm_predictor_helpers.py",
    "_confirm_predictor.py",
    "_fleuret_confirm_gpu.py",
    "_fleuret_confidence_prange.py",
]


def _imports_joblib(path: Path) -> list:
    """The import statements of ``path`` that name joblib."""
    hits = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name.split(".")[0] == "joblib"]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "joblib":
            hits.append(node.module)
    return hits


@pytest.mark.parametrize("name", SCREENING)
def test_module_imports_no_joblib(name):
    """None of the screening modules imports joblib."""
    path = FILTERS / name
    if not path.exists():
        pytest.skip(f"{name} does not exist")
    assert _imports_joblib(path) == []


def test_the_worker_segment_route_gives_the_same_counts_for_any_worker_count_it_is_asked():
    """``fleuret_segment_counts`` with one segment equals the serial njit kernel (the multi-worker route differs only by how the budget is cut)."""
    import numpy as np

    from mlframe.feature_selection.filters._confirm_predictor_helpers import _compact_confidence_inputs
    from mlframe.feature_selection.filters._numba_utils import get_fleuret_criteria_confidence
    from mlframe.feature_selection.filters.fleuret import fleuret_segment_counts

    rng = np.random.default_rng(0)
    n = 6000
    cols = [rng.integers(0, 5, n) for _ in range(6)]
    y = (cols[0] + cols[1] + rng.integers(0, 2, n)) % 3
    data = np.column_stack([*cols, y]).astype(np.int8)
    nbins = np.asarray([5] * 6 + [3], dtype=np.int64)
    cd, cnb, cx, cy, csel, cent = _compact_confidence_inputs(data.copy(), nbins, (3,), (6,), [0, 1, 2], None)
    want = get_fleuret_criteria_confidence(
        data_copy=cd.copy(), factors_nbins=cnb, x=cx, y=cy, selected_vars=csel, bootstrapped_gain=0.0005, npermutations=9, max_failed=3, nexisting=1,
        cached_cond_MIs=cent[1], entropy_cache=cent[0], extra_x_shuffling=True, base_seed=np.uint64(11),
    )
    cd2, cnb2, cx2, cy2, csel2, cent2 = _compact_confidence_inputs(data.copy(), nbins, (3,), (6,), [0, 1, 2], None)
    nfailed, nchecked, _ = fleuret_segment_counts(
        data_copy=cd2, factors_nbins=cnb2, x=cx2, y=cy2, selected_vars=csel2, bootstrapped_gain=0.0005, npermutations=9, max_failed=3, nexisting=1, cached_cond_MIs=dict(cent2[1]),
        entropy_cache=dict(cent2[0]), extra_x_shuffling=True, n_workers=1, base_seed=11, toggles=(False, False, False, False),
    )
    assert (nfailed, nchecked) == tuple(want)
