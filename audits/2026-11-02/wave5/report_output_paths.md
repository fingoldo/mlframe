# Wave 5 audit 2026-11-02: output paths, symlinks and non-atomic writes

Scope: write and delete sites, read-only, Opus 5.5 low effort (targeted). Condensed from the agent's returned text.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| FSW5-1 | Medium | `integrations/mlflow.py:107-111` `embed_website_to_mlflow`: caller name opened with `open(..., "w")` | RESOLVED |
| FSW5-2 | Low | `training/_training_loop_refit.py:70-80`: `os.remove(snapshot_file)` with no containment | RESOLVED |
| FSW5-3 | Low | `feature_selection/wrappers/rfecv/_cb_border_cache.py:122-132, 185-191`: `mkstemp`, close, reopen by name in the shared temp root | RESOLVED |
| FSW5-4 | Low | `utils/numba_cache_guard.py:72-86`: stamp written in place while other processes read it | RESOLVED |
| FSW5-5 | Info | `votenrank/stability_exp.py:159`: `savefig(ensure_parent_dir(filename))` | DOC |

FSW5-1: a `..` component raises `ValueError`; the page is written to a temporary name and renamed, so a planted symlink is replaced, not followed. Test: `tests/integrations/test_mlflow.py`.

FSW5-2: a relative snapshot name must resolve inside `train_dir` and a symlink is left alone. Test: `tests/training/test_catboost_snapshot_discard_containment.py`.

FSW5-3 and FSW5-4: border files go to the per-user private directory (`utils/private_temp.py`); the stamp is renamed into place. Test: `tests/utils/test_stamp_filter_scratch_hygiene.py`.

FSW5-5: `filename` is an explicit parameter chosen by the caller, as in any `savefig`; the open lead from the report is closed.

Checked and safe: atomic temp-and-replace writers (param oracle store, disk cache, family budget, RFECV checkpoint, bundle writer, locking, composite cache store, kaleido scratch, kernel-tuning ensure, metadata); cleanup confined to directories the code created; the SHAP beeswarm file name is sanitised; shared-memory segments are unlinked by their creator.
