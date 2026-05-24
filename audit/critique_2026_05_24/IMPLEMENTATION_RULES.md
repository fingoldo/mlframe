# Implementation Rules для всех wave-агентов (mlframe critique 2026-05-24)

Этот файл — обязательная памятка для КАЖДОГО implementation-агента. Прочитать ДО любых правок.

## Среда
- Репо: `d:/Upd/Programming/PythonCodeRepository/mlframe/`
- Python: `D:/ProgramData/anaconda3/python.exe` (НИКОГДА `python3`)
- Windows shell, OS Windows 10
- Encoding: ASCII only в Python print, никаких emojis/arrows (cp1251 crashes per memory)
- Branch: master (НЕ создавать новые branches — пушим в master per user OK)

## Git дисциплина
- `git commit -o -- <file1> <file2> -m "..."` для КАЖДОГО коммита (parallel-agent safe, не цепляет sibling работу)
- Уже modified в working tree: `src/mlframe/feature_engineering/wavelet_dwt.py` — user's WIP, **НЕ ТРОГАТЬ, НЕ КОММИТИТЬ**
- НИКАКИХ `git stash`, `git reset --hard`, `git checkout --`, `git restore` для inspection — используй `git show HEAD:<path>` чтобы посмотреть предыдущий state
- Commit message format: `fix(scope): one-line WHY` или `test(scope): regression sensor for <bug>`. Никаких Co-Authored-By, никаких "as flagged in audit", никаких quotes из чата, никаких "user asked"
- Никаких `--no-verify` / `--no-gpg-sign`
- НЕ пушить — это сделает orchestrator (Claude в parent context) после tests+verification

## Sensor-тесты (обязательно для каждого bug fix)
1. **Сначала** напиши regression-test, который воспроизводит баг
2. Запусти тест → должен **FAIL** на текущем коде (RED)
3. Implement fix
4. Запусти тест → должен **PASS** (GREEN)
5. Если для проверки RED state нужен предыдущий код — `git show HEAD:<path>` (read-only), не stash
6. Test file naming: `tests/<package>/test_regression_<finding_id>_<short_slug>.py` (например `test_regression_S05_oof_silent_fallback.py`)

## Benchmark-дисциплина (для perf-optimizations)
1. Используй `cProfile` + `time.perf_counter` или `timeit` для measure-before
2. Implement candidate
3. Measure-after с тем же fixture
4. Если новый вариант НЕ быстрее (или код не короче/чище) — **не применять**, оставить inline-comment `# bench-attempt-rejected (2026-05-24): X ms -> Y ms, reason ...` в месте, чтобы следующий агент не повторил
5. Если есть несколько кандидатов (numba vs vectorized vs polars) — бенчмаркай все, выбирай самый быстрый, документируй в комментарии 
6. Не предлагать tradeoffs, теряющие safety (per memory `feedback_no_tradeoff_optimizations`)
7. Для CUDA/numba — интегрируй с `pyutilz.system.kernel_tuning_cache` (НИКАКИХ hardcoded thresholds)
8. Сохрани все вариации (per memory `feedback_keep_all_kernel_versions`) если кernel — назови _v2/_shared/_warp, dispatcher выбирает

## Pytest вызов (обязательно)
```
$env:PYTHONUNBUFFERED='1'; D:/ProgramData/anaconda3/python.exe -m pytest <path> --no-cov --timeout=60 -x -s
```
- `--no-cov` обязателен на Windows (per `feedback_pytest_no_cov`)
- `--timeout=60` обязателен (per `feedback_parallel_agents_heartbeat`)
- `-x` останавливаемся на первой ошибке (быстрая итерация)
- `-s` no capture (видеть прогресс)
- НЕ `| tail` / `| head` на background pytest — слепит на 5-30 мин (per `feedback_no_tail_pipe_on_long_runs`)
- Если test fails из-за OOM/paging — retry 1 раз через 1 мин (per `feedback_retry_on_oom`)

## Heartbeat / progress (обязательно для агентов работающих >5 мин)
- Создай `audit/critique_2026_05_24/heartbeats/HEARTBEAT_<your_slug>.txt`
- ПЕРЕЗАПИШИ его (touch + 1 строка статуса) ПЕРЕД каждым Edit/Bash/pytest. mtime = liveness signal.
- Hard budget: 45 мин wall. Если упёрся — финализируй частичный прогресс в DONE_MANIFEST.json и выйди.

## DONE_MANIFEST (каждый агент)
В конце работы запиши `audit/critique_2026_05_24/manifests/DONE_<your_slug>.json`:
```json
{
  "agent_slug": "w1a-leakage",
  "started_utc": "...",
  "finished_utc": "...",
  "findings_assigned": ["S01", "S03", "S04", "S05"],
  "findings_status": {
    "S01": {"status": "DONE", "commit": "abc123", "test_file": "tests/training/test_regression_S01_cache_fp.py", "notes": "..."},
    "S03": {"status": "DEFERRED", "reason": "blocked by S02 fix", "notes": "..."},
    ...
  },
  "blocked": [],
  "test_results": {"green_paths": [...], "red_paths": [...]}
}
```

## Code-style правила (per memory)
- Comments до 160 chars (НЕ wrap на 72-80)
- НЕ писать `# Phase N`, `# audit ID Sxx`, `# fix for ...`, date stamps, refactor-history — это git territory
- НЕ писать `# natural Python idiom`, `# elegant`, `# obvious choice` — AI-fingerprint
- Только WHY-комментарии (если non-obvious); по default — никаких
- `orjson` over `json` для сериализации; compile regexes at module level (per `feedback_orjson_compile_regex`)
- JSON hashing: `sort_keys=True` (per `feedback_json_hash_sort_keys`)

## ML/data-integrity правила
- НЕ делать `.copy()` / `.clone()` / `pd.DataFrame(df)` на горячем пути — frames 100+GB
- Predпочитать views, lazy eval (polars), mutate-and-restore
- `pl.Categorical` → НЕ использовать, только `pl.Enum(train+val union)` (per `reference_polars_global_string_cache`)
- НЕ silent coerce (`pd.to_datetime(errors='coerce')`, bare except return None) (per `feedback_silent_correctness_bug_classes`)
- НЕ забывай `eq_missing` вместо `==` для polars null-safe (per `feedback_eq_missing_null_handling`)
- Architectural changes (новый модуль, новая абстракция, redesign, новая политика) — **НЕ ИМПЛЕМЕНТИРОВАТЬ**, оставить TODO + написать предложение в `audit/critique_2026_05_24/architectural_proposals/<finding_id>.md` — user будет approve в финальной волне

## Search-before-write
- Перед написанием helper/util — grep по mlframe + pyutilz, может уже есть (per `feedback_search_for_reuse_first`)
- Перед reimplementation — pip install библиотеку если она missing (per `feedback_install_libs_first`)

## Что НЕ делать
- НЕ trunc DB / БД миграции / shared infra — это destructive (per `feedback_no_auto_truncate`)
- НЕ убивать USER's процессы (per `feedback_never_kill_processes`)
- НЕ запускать `git push` (это делает orchestrator)
- НЕ trigger pre-commit/ci changes без user OK
- НЕ delete benchmark results files под `profiling/` или `_benchmarks/_results/` — это user data
- НЕ touch `src/mlframe/feature_engineering/wavelet_dwt.py` (user's WIP)
- НЕ исправлять issue которое не в твоём списке findings (no scope creep)
- НЕ скипай test fail "as pre-existing" — диагностируй (per `feedback_just_fix_dont_dig`, `feedback_fix_all_failures`)

## Architectural-proposal формат (если попалось)
Если finding требует new module / new abstraction / new policy — НЕ имплементируй. Вместо этого:
1. Создай `audit/critique_2026_05_24/architectural_proposals/<finding_id>.md`
2. В нём: проблема, 2-3 варианта решения с tradeoffs (LOC, deps, blast radius, perf impact), рекомендация, риски
3. В DONE_MANIFEST.json mark `status: "ARCH-DEFER"`, `proposal: "<path>"`
4. User approve в финальной волне

## Финальный self-check (перед exit агента)
- [ ] Все findings из моего списка — status в manifest (DONE / ARCH-DEFER / BLOCKED / REJECTED-with-reason)
- [ ] Каждый bug fix имеет sensor-тест который GREEN
- [ ] Каждый perf fix имеет bench numbers в commit message или inline comment
- [ ] Все мои изменения commited (через `-o` flag)
- [ ] manifest файл записан
- [ ] heartbeat файл финально обновлён со статусом DONE
- [ ] НЕ запушено
