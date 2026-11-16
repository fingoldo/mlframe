# Audit 01: calibration / metrics / evaluation / reporting / estimators / inference / inspection (MRMR excluded)

| ID | file:line | finding | disposition | reason/test |
|---|---|---|---|---|
| A1 | estimators/pipelines.py:186 | `joblib.dump` of cv_results straight to the final path: a crash mid-write truncates the previous dump, and the .sha256 sidecar then no longer matches or is absent. | RESOLVED | Now uses `atomic_write_bytes`. Test: tests/estimators/test_pipelines_cv_dump_atomic.py (the old code passes the path to the patched dump, so the OSError never occurs and the test fails). mypy clean. |
| A2 | evaluation/reports.py:381 | Confusion-matrix `savefig` has no try/finally, so a savefig failure skips `plt.close` and leaks the figure. | RESOLVED | try/finally around savefig/plt.close in reports.py; test `test_confusion_matrix_savefig_failure_still_closes_figure` (tests/evaluation/test_savefig_closes_figure_on_failure.py). |
| A3 | evaluation/reports.py:863, 918 | `savefig` failure leaves the returned figure open, but the figure is the caller's return value. | REJECTED | The caller owns the figure; closing it would break the API. |
| A4 | calibration/_post_train_calibrators.py:73, 334 and calibration/_policy_pick_helpers.py:116 | Broad `except Exception`. | REJECTED | Each logs a warning through log_throttle and states what falls back (leakage check skipped, sidecar optional). Not silent. |
| A5 | reporting/report_html.py:261 | HTML write is non-atomic. | REJECTED | Already writes a .partial file then `os.replace`, with cleanup on failure. |
| A6 | calibration/_post_train_calibrators.py:27 | Calibrator dump write. | REJECTED | Already uses `atomic_write_bytes`. |

Coverage note: this was a targeted sweep (except-handlers, writers, global state, figure lifecycle) over about 31k LOC. It was not a line-by-line read of every metric kernel. Numerical-stability and leakage paths I opened (probabilities.py logit clipping, threshold stability held-out scoring, calibrator-vs-test overlap guards) were already handled.
