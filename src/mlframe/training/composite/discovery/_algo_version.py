"""The version of the discovery selection logic, part of the discovery disk-cache key.

``mlframe.__version__`` changes per release, so a fix to discovery inside a release left warm caches replaying specs the
old code selected. Bump this integer with every change under ``composite/discovery`` or ``composite/transforms`` that can
change which specs are selected; ``tests/test_meta/test_discovery_algo_version_bumped.py`` fails when those sources change
without a bump (a bump re-pins the source hash automatically).
"""

DISCOVERY_ALGO_VERSION: int = 5
