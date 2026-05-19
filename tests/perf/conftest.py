"""Pytest collection hooks for tests/perf/.

`profile_*.py` files in this directory are standalone runtime-profiling
scripts (cProfile harnesses, training-suite hot-path probes) - not unit
tests. They were getting picked up by default pytest collection because
their filenames don't start with `test_`, but pytest still tried to
import them when run as `pytest tests/perf/`. Block collection here so
the default suite doesn't pay their import cost (each one pulls heavy
deps: lightgbm/catboost/torch).

To run a profile script intentionally, invoke `python tests/perf/profile_<name>.py`
directly, or pass `-o python_files=profile_*.py` to pytest.
"""

from __future__ import annotations

collect_ignore_glob = ["profile_*.py"]
