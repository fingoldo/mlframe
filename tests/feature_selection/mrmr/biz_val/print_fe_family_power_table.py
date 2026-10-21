"""Print the detection-power and false-positive table of every FE family (not collected by pytest; run it with the repository's Python).

Usage: ``python tests/feature_selection/mrmr/biz_val/print_fe_family_power_table.py [family ...]``. With no argument every family of both survival modules is fitted, which takes
hours; the tests themselves fit only the families without an open finding. The families with an open finding are listed in ``OPEN_FINDINGS`` of the common module.
"""

from __future__ import annotations

import sys

from tests.feature_selection.mrmr.biz_val import test_biz_value_mrmr_fe_family_survival as numeric
from tests.feature_selection.mrmr.biz_val import test_biz_value_mrmr_fe_family_survival_encodings as encodings
from tests.feature_selection.mrmr.biz_val._fe_family_survival_common import print_power_table


def main(names: list) -> None:
    """Fit and print the table for the named families, or for all of them when no name is given."""
    families = [f for f in numeric.FAMILIES + encodings.FAMILIES if not names or f.name in names]
    print_power_table(families)


if __name__ == "__main__":
    main(sys.argv[1:])
