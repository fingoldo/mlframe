"""Audit 2026-11-20 finding 05-F1: the recursion benchmark labels its second arm as the interpreted body, not numpy.

The arm runs ``_knn_shapley_recursion.py_func``, the CPython code the njit kernel is compiled from. The printed label must
say so, and must not claim a numpy backend that the package does not have.
"""

from mlframe.data_valuation._benchmarks.profile_data_valuation import bench_njit_vs_interpreted_recursion


def test_second_arm_is_labelled_interpreted_not_numpy(capsys):
    """At a tiny size the printed row names interpreted_py and never numpy_loop."""
    bench_njit_vs_interpreted_recursion(sizes=(64,), repeats=2)
    out = capsys.readouterr().out
    assert "interpreted_py=" in out
    assert "numpy_loop" not in out
