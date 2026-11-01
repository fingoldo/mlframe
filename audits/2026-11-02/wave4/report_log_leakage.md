# Wave 4 audit 2026-11-02: log forging and data leakage through logs, reports and errors

Scope: `src/mlframe`, read-only, an Opus 5.5 low-effort agent (targeted grep sweep plus some full reads). The agent could not write files; this is its returned text, condensed.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| LOG5-1 | Low | `utils/_param_oracle.py:117-120` (host key falls back to `platform.node()`, stored with every row) | RESOLVED |
| LOG5-2 | Low | label lists in `evaluation/bootstrap.py:625,754`, `evaluation/reports.py:810`, `training/composite/venn_abers.py:231`, `conformal_classification.py:133` | DOC |
| LOG5-3 | Low | `reporting/_async_render.py:233` (worker env snapshot filtered to `MLFRAME_PLOT_*`) | RESOLVED |
| LOG5-4 | Info | `__init__.py:119` (CUDA path probe) | RESOLVED |
| LOG5-5 | Info | benchmark `bench_sf3_jmim_exponent_selection.py:73` copies the environment | REJECTED |

LOG5-1: the fallback host key `node_<hostname>` is persisted in the oracle store. Fixed: the node name is hashed (`sha256`, 16 hex digits), `tests/utils/test_param_oracle_host_key_hashed.py`.
LOG5-2: the messages list the distinct label values (unbounded, `list` repr so newlines are escaped). Harmless for numeric labels; with string labels they could copy class names into centralised logs. Documented, not capped.
Checked and safe: no object whose `repr` holds a credential; environment dumps only at the sites above; column and entity names in exceptions use `!r`; about 25 data-dependent exception messages print shapes and counts only.
Not covered: f-string log calls that put `{col}` without `!r` in an INFO/WARNING message (a follow-up grep settles it).
