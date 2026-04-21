# PATTERN_SCAN — поиск прецедентов в polars_ds для OOF-дизайна

**Дата:** 2026-04-16  **Фокус:** есть ли в самом polars_ds паттерн, разделяющий train- и test-семантику в pipeline.

## Краткий вывод

**Прямого прецедента OOF/cross-fitting в polars_ds нет.** Но есть **косвенный, мощный прецедент**, переворачивающий выбор (a) vs (b).

## Что искали и что нашли

### 1. Прямые ключевые слова
Grep по `D:\Temp\polars_ds_fork\python\` и `src\` для: `fit_transform`, `KFold`, `cv=`, `oof`, `out_of_fold`, `cross_fit`, `fold_idx` — **0 совпадений**.

- `linear_models.py` — только `.fit()` / `.predict()`, никакого CV.
- `sample_and_split.py` — `split_by_ratio()` + stratified split, но это data-partitioning утилиты, не fold-iterator.
- `eda.py` — нет stateful train-only логики.

### 2. Ключевая находка: `Pipeline.materialize(return_df=True)`

Файл: `python/polars_ds/pipeline/pipeline.py:1026-1069`.

```python
def materialize(
    self,
    df: pl.DataFrame | pl.LazyFrame,
    return_df: bool = False,
) -> Pipeline | tuple[Pipeline, pl.LazyFrame]:
    ...
    if return_df:
        return pipeline, transformed_train_lazyframe
    return pipeline
```

**Что это значит:**
- API уже **знает**, что у train-прохода через pipeline есть особый статус.
- Возврат `(Pipeline, train_lazy)` — уже устоявшийся паттерн, **не breaking**.
- Сейчас `transformed_train_lazyframe` = «pipeline применённый к train», то есть для encoder'ов это full-mapping (с утечкой!).

### 3. Текущие encoder'ы — uniformly applied
`target_encode` / `woe_encode` (`pipeline/transforms.py:503-612`) учат mapping и возвращают `pl.replace_strict(...)` — одно и то же выражение применяется к любому df.

### 4. Тесты не проверяют train/test split вообще
`tests/test_transforms.py:175-295` проверяют только согласованность с sklearn на одном датасете. Leakage-сценарий не покрыт.

## Следствие для дизайна (a) vs (b)

`return_df=True` **снимает главный минус варианта (a)**: «breaking change сигнатуры materialize». Сигнатура `materialize(df, return_df=True) -> (Pipeline, LazyFrame)` уже существует. Нам остаётся:

1. При `cv > 0` для encoder-шагов: **подменить колонки в `transformed_train_lazyframe`** на OOF-версии (вместо full-mapping применения).
2. Сам `Pipeline` хранит финальный `mapping(full_train)` — для `pipeline.transform(test)` поведение не меняется.

То есть **(a) перестаёт быть breaking**. Пользователи, которые звали `materialize(df)` без `return_df=True`, ничего не замечают (но получают молчаливую утечку — это нужно митигировать через warning и/или дефолт `cv=5` в новой версии).

Пользователи, которые звали `materialize(df, return_df=True)`, получают LazyFrame с OOF-значениями вместо full-mapping — поведение «улучшается» для них без слома сигнатуры.

## Вариант (b) в контексте находки

`pipeline.fit_transform(train)` всё ещё валиден, но дублирует функциональность `materialize(df, return_df=True)` — пользователю придётся выбирать между двумя API, делающими почти одно и то же. Это **избыточно** и противоречит сложившимся конвенциям polars_ds.

## Обновлённый вердикт

**Рекомендация: вариант (a)**, но переформулированный:
- Использовать существующий `materialize(df, return_df=True)` контракт.
- При `cv > 0` для encoder-шагов класть в возвращаемый train LazyFrame OOF-значения, а не full-mapping.
- В Pipeline сохраняется финальный mapping для `transform(test)`.
- Документировать: «для train используйте `bp.materialize(train, return_df=True)`; полученный LazyFrame уже содержит OOF».
- Дополнительно: добавить warning при `pipeline.transform(train_df)` для шагов с `cv>0` — невозможно надёжно (Pipeline не знает, что ему передали именно train), но можно требовать от пользователя явный флаг `pipeline.transform(df, is_train=False)` либо просто документировать risk.

Это решение:
- Не breaking (использует существующий API).
- Не вводит дублирующий метод (`fit_transform`).
- Семантически согласовано с тем, как polars_ds уже думает о train-прохождении (`return_df=True` как «получи преобразованный train»).
- Минус (×K цена `materialize` всегда) частично митигируется: `cv` опциональный параметр, по умолчанию `None` → текущая цена.

## Что ещё стоит обсудить с мейнтейнером

1. Должен ли `materialize` менять поведение по умолчанию (`cv=None` vs `cv=5`)? — Скорее нет, не breaking.
2. Нужен ли `pipeline.transform(df, is_train=False)` флаг для защиты от misuse? — Дискуссионно.
3. Lazy-режим: OOF требует материализации фолдов, можно ли это сделать на LazyFrame через `collect()` внутри? — Технический вопрос.
