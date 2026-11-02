# Wave 6 audit 2026-11-02: log forging through data-derived names in log calls

Scope: `logger.*` and `log_throttle` calls outside benchmarks and tests, read-only Grep pass, Opus 5.5 low effort. Site counts are pattern counts, not a line-by-line read. Condensed from the agent's returned text.

| ID | Severity | Sites | Disposition |
|---|---|---|---|
| LOGF6-1 | Low | about 55 calls put a column, feature or target name into the message with a bare `%s` | RESOLVED |
| LOGF6-2 | Low | about 760 calls pass an exception's text through `%s`, and third-party errors echo column names and category values | RESOLVED |
| LOGF6-3 | Info | about 15 calls log names read from file paths or files on disk | RESOLVED |
| LOGF6-4 | Info | one f-string log in `training/phases.py:328` | RESOLVED |
| LOGF6-5 | Low | no control-character escaping anywhere in mlframe's logging setup (root cause of the four above) | RESOLVED |
| LOGF6-6 | Low | a newline followed by `::error::` or `::add-mask::` in a name is read as a GitHub Actions workflow command in CI logs | RESOLVED |
| LOGF6-7 | Info | calls that already use `%r` (`evaluation/bootstrap.py:184, 197, 335, 342`, `reporting/colors.py:209`) | REJECTED |

LOGF6-1 to LOGF6-6 are fixed together, centrally, as the report recommended: editing about 815 sites would drift and `!r` on exception text makes messages harder to read. `mlframe.utils.log_sanitize` installs a logging record factory (at `import mlframe`) that rewrites the ARGUMENTS of records created by loggers named `mlframe*`:
- control characters except tab and newline (carriage return, ESC, the C0 and C1 ranges) are written as `\xNN`;
- a newline is written as backslash-n in arguments of at most 3 newlines and 512 characters, so names and short messages stay on one line while long, deliberately formatted blocks (a classification report) keep their line breaks;
- exception arguments are replaced by their escaped message only when it needed escaping;
- the format string, which is code, and records of other loggers are not touched.

Tests: `tests/utils/test_log_sanitize.py` (newline in a column name, ESC and CR, exception text, clean text unchanged, multi-line report preserved, mapping arguments, foreign loggers untouched, idempotent install).
A deliberately long hostile value (more than 3 newlines or 512 characters) still passes with its line breaks; the fix raises the cost of forging and removes the common cases, it is not a guarantee.
LOGF6-7: already safe.
