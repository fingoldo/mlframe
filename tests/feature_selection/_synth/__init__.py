"""Verbatim homes for the shared synthetic-data helpers that ``tests/`` imports.

``tests/feature_selection/_biz_val_synth.py``, ``_synthetic_distributions.py`` and
``_mrmr_realistic_data.py`` are now pure re-export shims over the modules in this package.
The split freezes the ~53-file import surface: when these generators are promoted into
production (``mlframe.data.datasets``), the shims are repointed in one place instead of
53 importers being rewritten.
"""
