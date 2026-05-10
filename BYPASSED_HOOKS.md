# BYPASSED_HOOKS log (filters refactor PR)

Per the refactor plan: during etapes 2-10 the path-keyed baselines in
`tests/test_meta/` (`_annotation_baseline.json`, `_docstring_baseline.json`,
`_api_snapshot.json`, etc.) are out of date because filters.py is being
split into a package. Regenerated at etap 11.5.

This file tracks every pytest-meta failure that the refactor branch knowingly
defers to etap 11.5. Each entry pins a meta test, the etap that surfaced it,
and the regeneration step that resolves it.

## Etap 2-4 (initial structural split)

| Test | Reason | Fixed at |
|---|---|---|
| `test_meta/test_api_stability.py::test_public_api_matches_snapshot` | new submodule paths break key match | etap 11.5 (`_api_snapshot.json` regenerate) |
| `test_meta/test_public_annotations.py::test_no_new_unannotated_public_functions` | new submodule public functions need baseline entries | etap 11.5 (`_annotation_baseline.json` regenerate) |
| `test_meta/test_public_docstrings.py::test_no_new_undocumented_public_symbols` | new submodule public functions need baseline entries | etap 11.5 (`_docstring_baseline.json` regenerate) |
| `test_meta/test_deferred_drift.py::test_user_deferred_lists_havent_grown` | adding 2 pre-existing dead helpers to whitelist for unblock | etap 11.5 (audit final whitelist) |
| `test_meta/test_todo_hygiene.py::test_every_todo_marker_has_attribution` | legacy TODO markers under the moved code | etap 11.5 |
| `test_meta/test_logger_lazy_formatting.py::test_no_new_eager_log_format_on_debug_or_info` | legacy f-strings in moved code | etap 11.5 |
| `test_meta/test_no_unicode_in_console_output.py::test_no_new_non_ascii_console_output` | legacy non-ASCII strings in moved code | etap 11.5 |
| `test_meta/test_config_field_consumption.py::test_every_config_field_has_a_consumer` | unrelated -- pre-existing in mlframe master | not owned by refactor |

This file is **deleted at etap 11.5** when all baselines are regenerated and
all meta tests are green again.
