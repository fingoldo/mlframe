# Upstream PR #2: Support polars-lts-cpu / polars-u64-idx via extras

## Проблема
На машинах без AVX2 (Sandy/Ivy Bridge, некоторые Atom/VIA) обычный `polars` падает с `SIGILL`. Решение Polars-экосистемы — отдельный distribution `polars-lts-cpu` (тот же Python-модуль `polars`, но Rust-код собран без AVX2). Однако `polars_ds` в `pyproject.toml` жёстко зависит от `polars`:

```toml
dependencies = [
    "polars >= 1.4.0, != 1.25, ...",
    'typing-extensions; python_version <= "3.11"',
]
```

При `pip install polars_ds` резолвер всегда тянет `polars` (не `polars-lts-cpu`), игнорируя уже установленный `polars-lts-cpu`, потому что для pip это **разные пакеты**. Устанавливается несовместимый binary → SIGILL.

## Патч

```diff
 [project]
 name = "polars_ds"
 requires-python = ">=3.9"
 version = "0.11.2"

 dependencies = [
-    "polars >= 1.4.0, != 1.25, != 1.26, !=1.32.1, !=1.32.2",
     'typing-extensions; python_version <= "3.11"',
 ]

 [project.optional-dependencies]
+default = ["polars >= 1.4.0, != 1.25, != 1.26, !=1.32.1, !=1.32.2"]
+lts-cpu = ["polars-lts-cpu >= 1.4.0, != 1.25, != 1.26, !=1.32.1, !=1.32.2"]
+u64-idx = ["polars-u64-idx >= 1.4.0, != 1.25, != 1.26, !=1.32.1, !=1.32.2"]
 plot = ["great-tables>=0.9", "graphviz>=0.20", "altair >= 5.4.0", "vegafusion[embed]"]
 models = ["numpy>=1.16"]
 compat = ["numpy>=1.16"]
-all = ["great-tables>=0.9", "graphviz>=0.20", "numpy>=1.16", "altair >= 5.4.0", "vegafusion[embed]"]
+all = ["polars >= 1.4.0, != 1.25, != 1.26, !=1.32.1, !=1.32.2", "great-tables>=0.9", "graphviz>=0.20", "numpy>=1.16", "altair >= 5.4.0", "vegafusion[embed]"]
```

## Установка после патча

| Ситуация | Команда |
|---|---|
| Стандартная установка | `pip install "polars_ds[default]"` |
| Старый CPU (без AVX2) | `pip install "polars_ds[lts-cpu]"` |
| >4.3B строк | `pip install "polars_ds[u64-idx]"` |
| Через uv/poetry | те же extras работают |
| Ничего не выбрано | `pip install polars_ds` → **ImportError** при первом импорте (т.к. polars не установлен) — это приемлемая регрессия, лечится подсказкой в README |

## Обратная совместимость
**Ломающее изменение.** Существующие `pip install polars_ds` без extras перестанут тянуть `polars` автоматически. Митигация:
- Задокументировать в CHANGELOG как breaking, version bump до 0.12.0 (minor per semver, polars_ds в 0.x).
- В `python/polars_ds/__init__.py` — добавить user-friendly ImportError:
  ```python
  try:
      import polars as pl
  except ImportError as e:
      raise ImportError(
          "polars_ds requires a polars backend. Install one of:\n"
          "  pip install 'polars_ds[default]'   # standard\n"
          "  pip install 'polars_ds[lts-cpu]'   # old CPU (no AVX2)\n"
          "  pip install 'polars_ds[u64-idx]'   # large rowcount\n"
      ) from e
  ```

## Альтернативное (менее ломающее) решение
Оставить `polars` в dependencies, но дополнительно добавить extras `lts-cpu` / `u64-idx`. Тогда пользователи старых CPU делают:
```bash
pip install --no-deps polars_ds
pip install polars-lts-cpu typing-extensions
```
Это работает уже сейчас (workaround), но принуждает к `--no-deps`. С extras же pip-резолвер сам распутает.

Для "цивилизованного" решения всё равно нужно убрать `polars` из mandatory deps.

## Прецеденты в экосистеме
- [polars-business](https://github.com/MarcoGorelli/polars-business/blob/main/pyproject.toml) — extras `[polars]`, `[polars-lts-cpu]`.
- [patito](https://github.com/JakobGM/patito) — `polars-extra-dependencies` optional group.
- [great-tables](https://github.com/posit-dev/great-tables) — `polars` как optional (можно использовать с pandas backend).

## Порядок подачи PR в `abstractqqq/polars_ds_extension`
1. Branch в форке `D:\Temp\polars_ds_fork`: `git checkout -b feature/lts-cpu-extras`.
2. Правка `pyproject.toml` (diff выше).
3. Правка `python/polars_ds/__init__.py` — friendly ImportError.
4. Правка `README.md` — install section с 3 вариантами.
5. CHANGELOG: "0.12.0 (breaking): polars no longer installed by default — choose `[default]`, `[lts-cpu]`, or `[u64-idx]` extras."
6. Открыть PR с referenc'ами на polars-business как прецедент.

## Для немедленного использования на машине с LTS CPU
Без ожидания upstream PR:
```bash
pip install polars-lts-cpu "typing-extensions; python_version<='3.11'"
pip install --no-deps polars_ds
# опционально:
pip install numpy great-tables graphviz altair  # для plot/models/all features
```
Проверка:
```python
import polars as pl; print(pl.__version__)  # должен быть lts-cpu сборка
import polars_ds as pds; print(pds.__version__)
```
