"""A limit compared against similarly named operands in several modules is one decision written many times.

``np.unique(y).size <= 20`` appeared in a dozen cardinality gates; re-measuring the limit in one left the others behind. The py-ci-shared `threshold` rule finds a value compared
against similarly named operands in at least three modules. Each accepted group in the baseline says why its copies are legitimately independent (a numerical-zero guard on its
own scale, a minimum row count) or the copies were replaced by one named constant (`FEW_CLASSES_MAX`). Kept apart from the other adopted gates so it runs wherever the installed
py-ci-shared has this rule.
"""

from __future__ import annotations

from pathlib import Path

from py_ci_shared.drifted_duplicate_literals import RULE_THRESHOLD, assert_no_drifted_duplicate_literals

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src" / "mlframe"


def test_no_threshold_literal_is_restated_across_modules():
    """Every group of modules comparing one limit against similarly named operands is accepted with a reason in the baseline."""
    assert_no_drifted_duplicate_literals(
        root=SRC,
        rules=(RULE_THRESHOLD,),
        skip_dir_names=("_benchmarks",),
        baseline_path=HERE / "_drifted_threshold_literals_baseline.json",
        min_files=1000,
    )
