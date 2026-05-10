"""Golden regression test fixtures for the MRMR refactor.

Two snapshot tiers per the plan:

* ``pre_refactor/`` -- captured at etap 0a before any code change. Contains
  ``support_set`` + scalar counters only (no keyed cached_MIs values, since
  the pre-B12 ``arr2str`` collapses multiset variants to identical strings
  -- a confirmed silent correctness bug).
* ``intermediate/`` -- captured at etap 9 on post-cleanup code (B1-B12, B14
  applied). Includes keyed cached_MIs values in the new collision-safe key
  format. Used as the reference baseline for the screen_predictors
  decomposition (etaps 10a/10b/11).

See ``tests/feature_selection/test_screen_golden.py`` for the comparison
machinery (tied-score-aware support comparator, ``np.isclose`` rtol policy).
"""
