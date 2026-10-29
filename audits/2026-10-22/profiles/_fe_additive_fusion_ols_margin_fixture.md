# `test_regression_fe_additive_fusion_gpu_ols_margin_uses_full_n_rows`, 2026-10-29

Failed on HEAD before any 2026-10-29 change: the GPU fusion now also requires the unweighted sum to keep the pair's linear gain (`_keeps_both_halves`, commit 7400f0715), and the old
fixture (`y = h_c + 0.2 h_d`) has a sum that is worse than the stronger half. The code is the intended behaviour (`test_additive_fusion_keeps_both_halves.py` pins it); the fixture no longer
isolated the OLS margin.

Why equal weights cannot work: a half clears its relevance floor on the n_sc scoring rows only when its correlation with y is at least about 5.7/sqrt(n_sc); the OLS gain of two equal
halves is then about 2.1/sqrt(n_sc), just above the cap-sized margin 2/sqrt(n_sc), so the window (2/sqrt(n), 2/sqrt(n_sc)) is empty. With the second half at 0.6 of the first the gain
shrinks faster than the floor binds. New fixture (n=4000, cap=1000): `y = h_c + 0.6 h_d + 5 noise`; on the scoring sample OLS gain 0.053, unweighted-sum gain 0.049, window (0.032, 0.063).

The test also gained a control: with the margin widened to the pre-fix 4/sqrt(4000) the same fusion is rejected, so the test fails if the margin ever reverts to the subsample size.
Scans: `D:/Temp/ols_dbg.py`, `D:/Temp/ols_dbg2.py`, `D:/Temp/ols_margin_scan2.py`.
