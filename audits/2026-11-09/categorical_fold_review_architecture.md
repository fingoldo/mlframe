# Ревью categorical_fold_strategy: реализация и архитектура

Область: `src/mlframe/feature_selection/filters`. Только чтение кода; исходники не менялись.

## 1. Где cap применяется сейчас (факты)

- `discretization/_categorical_encoding.py::cap_categorical_cardinality(codes_2d, max_cardinality)`: чисто numpy, работает на double-матрице кодов (n, p), блоками по 16 колонок через транспонированные тайлы. Оставляет cap-1 самых частых (stable argsort по -counts), остальное в бакет `cap-1`, `-1` (NaN) сохраняется. Колонки с card <= cap возвращаются без изменений. **Маппинг нигде не сохраняется**: функция возвращает только новую матрицу.
- `discretization/_discretization_dataset.py::_factorize_categorical_columns` (стр. 576-615): polars-ветка (physical codes -> Int64, fill_null(-1), затем astype float64, потом cap) и pandas-ветка (`_multi_col_factorize_native`: `.cat.codes` либо `pd.factorize`, затем cap). Затем `_encode_categorical_block` сдвигает коды на +1, если есть NaN (-1 -> 0), и проверяет переполнение dtype.
- Вызывается из `categorize_dataset(...)`, а тот из `_mrmr_fit_impl/_fit_impl_stages/_binning.py`: `_discretize_inputs_part1` (стр. 470), `_honest_null_codes_uncached` (стр. 285, перекатегоризация под пермутированный y), warm-up-поток (стр. 167). Ещё `PermutedTargetBinner` в `discretization/shared`.
- `_code_storage_plan.py::plan_code_storage`: cap входит как верхняя граница кода категориальных колонок (`bound = max(bound, cap)`); `_planned_code_itemsize` в `_mrmr_validate_transform.py:144` дублирует вызов. Параметры в `_mrmr_class.py:348` (`max_categorical_cardinality=127`), `quantization_dtype=int8`.
- Заметка про коды: после +1 сдвига при NaN максимальный код = cap, т.е. при cap=127 это 127, влезает в int8 впритык. Любая новая стратегия обязана сохранять инвариант "max код <= cap (с учётом NaN-сдвига)", иначе `_encode_categorical_block` молча расширит dtype до int16 и ломает compact-storage бюджет (`code_storage_plan`).

## 2. Главные находки

### F1 (критично). В пайплайне нет состояния уровней и replay-пути для кодов
Свёртка делается только внутри `categorize_dataset` над кодами, и они живут только в fit. `transform()` возвращает выбранные исходные колонки и FE-рецепты; перекодирования категориальных входа в int8 при transform нет. Значит:
- "Маппинг сохраняется в recipe и применяется в transform идентично" (proposal, шаг 2) относится к несуществующему механизму. Для selection-only MRMR маппинг нужен лишь для диагностики (`fold_report_`), а не для transform. Нужно явно решить: либо маппинг только отчётный (дёшево, рекомендую), либо он нужен рецептам FE на сложенных кодах (тогда у рецептов появляется зависимость от маппинга; см. F4).
- Unseen-уровни при transform не возникают в коде свёртки; вопрос "ближайшая статистика или other" актуален только если свёрнутые коды потребляют рецепты. Для них политика проста: unseen -> bucket "other" (тот же код, что у хвоста), NaN -> NaN-код. Группа "с ближайшей статистикой" требует хранить статистики и даёт непредсказуемое поведение; отклонить.
- Нужна и текущая слабость: `pd.factorize` / `.cat.codes` дают коды в порядке появления/словаря; маппинг "код -> исходный уровень" теряется. Для отчёта нужно хранить `uniques` (в pandas ветке `pd.factorize` их выбрасывает: `_multi_col_factorize_native` возвращает только коды; polars: physical-коды вообще не несут строк, нужен `cat.get_categories()` / `to_physical` + словарь).

### F2 (критично). Таргет-ориентированная свёртка ломает пермутационный нуль
`_target_dependent_columns` (`_binning.py:41`) явно предполагает, что категориальные колонки НЕ зависят от таргета ("Every other column (categorical, ...) is binned without the target, so its observed codes are also its codes under any permutation"). Эти коды копируются в `_fixed_codes` и не пересчитываются под пермутированным y. Для `target_stat` это неверно: in-sample смещение свёртки (то, что proposal сам называет переобучением) нуль не увидит, и именно здесь значимость завышается. Нужно:
- включить таргет-зависимые категориальные колонки в `_dependent` (и в `PermutedTargetBinner.supported`, который сейчас работает только для optimal_joint/cv; `_make_null_binner` вернёт None для других стратегий, и тогда каждый пермутационный проход идёт через полный `categorize_dataset`, то есть стоимость свёртки умножается на число пермутаций);
- либо принять OOF-схему на этапе свёртки (proposal раздел 4) и считать, что нуль корректен, но это надо доказать тестом (колонка с шумовыми уровнями при n=2000 не проходит порог). Рекомендация: и то, и другое, а OOF только как защита, не как замена пермутации.
- Стоимость: fold-DP на 20 колонках x 500 уровней мала (O(L^2) или жадно O(L log L), L=500), но делается один раз на fit + K фолдов + на каждую пермутацию, если колонка помечена зависимой. Пермутационный нуль надо считать через кэшируемые `(level, class)` таблицы сочетаемости (см. 4.3), а не повторным проходом по n.

### F3. y доступен, но дискретизированным только по месту
- `categorize_dataset` получает `y_for_strategy` ТОЛЬКО для supervised-стратегий (`_binning.py:448-469`), и это сырой столбец target из `_x_for_cat[target_names[0]]` (непрерывный для регрессии, метки для классификации, первый таргет). Для `nbins_strategy=None` (дефолт) y сюда не попадает. Параметр `y_for_strategy` придётся пробрасывать и в unsupervised-случае, если `categorical_fold_strategy != "frequency"`. Не допускать, чтобы `frequency` менял это (регрессионный тест на отсутствие чтения y).
- Дискретизированный `classes_y` на момент `categorize_dataset` ещё не существует: таргет кодируется вместе с фреймом (он колонка `_x_for_cat`, см. комментарий "target injection happens upstream"), т.е. y-коды появляются в том же `data`. Практичный вариант: свёртка категориальных колонок выполняется ПОСЛЕ дискретизации числового блока, на уже готовом столбце кода таргета в `data` (он дискретизирован тем же методом, что и MRMR использует для MI). Это даёт мультикласс-форму для всех типов таргета без отдельных статистик на раздел 5 (регрессия, счёт, multitarget сводятся к `classes_y`). Непрерывный `_fe_prewarp_y_continuous_` (используется в 17 файлах FE-стадий, ставится позже в fit) на этом шаге НЕ доступен; не опираться на него. Разделы 5 про survival, ANOVA, Poisson, log-rank предлагаю убрать: YAGNI, один merge-cost по MI на `classes_y`.
- Мультитаргет: `target_names[0]` уже единственный сигнал для стратегий; сохраняем то же поведение (документировать).

### F4. Взаимодействие с grouped_quantile / composite_group_agg
`_grouped_quantile_fe.py`, `_grouped_agg_fe.py`, `_composite_group_agg_fe.py` берут группировочные колонки напрямую из сырого `X` (`auto_detect_group_cols`, `pd.factorize` по `g_keys`), а не из int8-кодов MRMR. Поэтому свёртка категорий, применённая в `categorize_dataset`, их не затрагивает ни в положительную, ни в отрицательную сторону: у групповых FE собственные пороги кардинальности. Решение: не менять; в docstring параметра явно написать, что свёртка действует только на MI-коды, а FE-группировка использует полный словарь. Если позже захочется согласовать, это отдельный параметр, не часть этой фичи. Риск: рецепт группового FE на сырой колонке и MI-код сложенной колонки расходятся, это ожидаемо и уже так сейчас.

### F5. int8 / GPU
- Выход свёртки обязан остаться в диапазоне 0..cap-1 (+NaN-код). Коды матрицы кладутся в `_code_alloc_dtype`; свёртка работает до приведения (double). Для 1M x p double-матрица кодов уже сейчас промежуточно в float64 (8 Б/ячейку; polars-ветка делает `astype(float64)` всей категориальной подматрицы). Это уже bottleneck при 1M x 100 категорий (800 МБ на 100 колонок); рекомендация для новых модулей: работать на int32 по колонкам (per-column функция `(codes_1d int32, y_codes) -> remap int16`) и записывать результат прямо в целевой dtype блока, не создавая double-матрицу. Существующий `cap_categorical_cardinality` оставить как есть (политика "никогда не удалять старые варианты", `frequency` == он).
- Device-born пути (GPU-FE): они потребляют уже готовую матрицу кодов `data` (int8 резидентная копия); свёртка полностью на хосте до этого, поэтому GPU-контракты не затрагиваются, при условии что `nbins` (число уровней колонки) считается ПОСЛЕ свёртки (`data.max(axis=0)+1` уже так). Нужен тест, что `nbins` свёрнутой колонки = число групп (+1 при NaN), иначе device-аллокации гистограмм разойдутся с кодами.
- Не добавлять GPU-ядро свёртки: L=500 уровней на колонку, вычисление ничтожно; GPU-путь здесь не нужен, но нужен только расчёт таблицы `counts[level, class]` одним `np.add.at`/`bincount` по `level*K+class`, что на 1M строк ~ несколько мс на колонку.

### F6. pickle
Все состояние свёртки (`categorical_fold_report_`: по колонке `levels_before, groups_after, other_share, strategy_used, mapping uniques->group` как списки/np.ndarray/dict) обычные атрибуты, без замыканий и кэшей. Нельзя хранить таблицы `(level, class)`, DP-состояние, ссылки на `categorize_dataset`, замыкание honest-null (оно уже вырезается в `__getstate__`). Новые constructor-параметры должны попасть в `_mrmr_setstate_defaults.py`, чтобы старые pickle без них загружались с `"frequency"`; плюс get_params/clone-тест. Новые `*_` атрибуты fit-состояния добавить туда же. По правилам проекта: если у атрибута есть кэш, он должен быть в списке стриппинга `__getstate__`.

### F7. polars vs pandas
Две ветки сегодня уже расходятся: polars `Utf8`-колонки кодируются через `cast(pl.Categorical).to_physical()` (глобальный string cache, коды зависят от кэша), pandas через `.cat.codes` / factorize. Новая свёртка должна работать на ОБЩЕМ представлении "int32 коды + int32 y-коды", чтобы не иметь двух реализаций. Реализация: единая функция принимает numpy-колонку кодов; различие веток остаётся только на этапе получения кодов (как сейчас). Для отчёта со строками уровней нужны отдельные адаптеры `levels_labels_pandas / levels_labels_polars` (только если решено сохранять метки, см. F1; иначе хранить только коды).

### F8. Размер модулей и размещение
- `_discretization_dataset.py` = 682 LOC, `_categorical_encoding.py` = 163, `_binning.py` = 513, `_mrmr_class.py` = 3633 (трогать только: добавление параметров в `__init__` и docstring разрешено, как дефолт/поведение in-place; без разбиения, без carve-out, без регенерации baseline).
- Новый код только в новых модулях (<1k LOC каждый), рядом с `discretization/`. В `_discretization_dataset.py` добавить ТОЛЬКО диспетчеризацию (несколько строк в `_factorize_categorical_columns`).
- `categorical_fold_strategy` + 3 сопутствующих параметра: 4 новых аргумента в и так огромном `__init__`; рекомендую сжать до `categorical_fold_strategy` + одного `categorical_fold_kwargs: dict | None` (по образцу `nbins_strategy` / `nbins_strategy_kwargs`), чтобы не раздувать `__init__` и мета-тест "параметр документирован".
- Принять предложение префикса env-флага: `MLFRAME_MRMR_...` для отката, если нужен (политика: чистые ускорения без флагов, но смена поведения по умолчанию возможна через параметр, не флаг).

## 3. Критика самого предложения (с точки зрения реализации)
1. Раздел 5 раздут: 7 типов таргета, но MRMR всё равно работает с дискретизированным `classes_y`. Достаточно одного merge-cost на `classes_y`. Убрать специальные статистики (hazard, log-rank, KM, Poisson).
2. "Оптимальная 1-D кластеризация по сглаженной статистике" для K>2 классов не определена (порядок на векторах). Для K>2 использовать иерархическое слияние по `delta_MI` (жадно, heap), порядок не нужен; для бинарного достаточно сортировки по доле. Один алгоритм: жадное слияние пары с минимальной потерей MI (`delta_MI` считается из таблицы `counts[level, class]`, O(L) на шаг через пересчёт только затронутой группы, L=500 -> миллисекунды).
3. `min_support`: порог `max(5, n/(cap*20))` для n=10M и cap=127 даёт 3937 строк на уровень, то есть сворачивает уровни, которые при таком n вполне надёжны; для n=2000 даёт 5. Формулу нужно вывести из условия "ожидаемая ячейка >= 5 в таблице level x classes_y" (то же, что аналитический нуль уже использует: guard >=5 expected/cell), а не из cap. Это согласуется с политикой "без фиксированных порогов".
4. `auto` без фиксированных порогов: выбирать по OOF-MI (K-fold по таблицам `counts`, дёшево) между `frequency` и `target_stat`; никаких порогов на n/levels. То же OOF-сравнение защищает от переобучения; пермутационный контроль из раздела 6 заменить пермутационным нулём пайплайна (F2), а не отдельной ad-hoc процедурой.
5. NaN-политика ("отдельный уровень, если доля NaN >= порога") ещё один порог; сегодня NaN уже получает собственный код через +1 сдвиг в `_encode_categorical_block` (missing_strategy). Не вводить второй механизм: свёртка работает только на не-NaN кодах, NaN остаётся как есть.
6. Бинарный/мультикласс shrinkage с `k`: избыточен при выборе по merge-cost MI на OOF; убрать `categorical_fold_smoothing`.

## 4. Конкретный план реализации

### 4.1 Модули (все новые, `filters/discretization/`)
1. `_categorical_fold_tables.py` (~150 LOC): таблицы и утилиты.
   - `level_class_table(codes: np.ndarray[int32], y_codes: np.ndarray, n_levels: int, n_classes: int) -> np.ndarray[int32]` одним `np.bincount(codes*K + y, minlength=L*K)` (без NaN: codes>=0).
   - `min_support_from_classes(n_classes: int, n_rows: int) -> int`, из условия ожидаемой ячейки >= 5.
2. `_categorical_fold_mi.py` (~200 LOC): `merge_cost` и MI из таблицы `counts` тем же оценщиком, что MRMR (plug-in MI из `(L,K)` таблицы; переиспользовать существующую функцию MI из `filters` вместо новой; сначала grep на reuse).
3. `_categorical_fold_strategies.py` (~350 LOC):
   - `fold_frequency(codes, cap) -> remap` (обёртка вокруг логики `cap_categorical_cardinality`, ДОЛЖНА давать те же коды; старую функцию не удалять).
   - `fold_min_support(codes, cap, min_support) -> remap`.
   - `fold_target_stat(counts: np.ndarray, cap: int, min_support: int) -> remap` (жадное слияние по delta_MI, heap).
   - `fold_auto(codes, y_codes, cap, n_folds) -> (remap, strategy_used)` (выбор по OOF-MI).
   - общий тип: `remap: np.ndarray[int16]` длины n_levels, значения 0..cap-1.
4. `_categorical_fold_apply.py` (~250 LOC): `fold_categorical_block(codes_2d, y_codes, cap, strategy, kwargs) -> (codes_2d_folded, report)`. Работает по колонкам, пишет сразу в целевой блок, не создавая double-копию. `report` = `list[dict]` (колонка, levels_before, groups_after, other_share, strategy_used, remap as np.ndarray) для `fold_report_` и `log_throttle`.
5. Диспетчеризация: в `_factorize_categorical_columns(..., fold_strategy=None, y_codes=None, fold_kwargs=None)`: если `fold_strategy in (None, "frequency")` вызывается текущий путь БЕЗ изменений (бит-идентично). Иначе `fold_categorical_block`. Новые аргументы пробросить через `categorize_dataset` (сигнатура +3 аргумента), `_binning.py` (3 call site + `PermutedTargetBinner`), `_mrmr_class.py` (`__init__`, docstring), `_mrmr_setstate_defaults.py`.
6. `_code_storage_plan.py`: без изменения логики; тест, что bound не превышает cap при любой стратегии.

### 4.2 Порядок шагов внутри fit
1. Коды числовых колонок и y: `categorize_dataset` дискретизирует числовой блок (включая столбец таргета) как сейчас.
2. Факторизация категориальных колонок (pandas/polars) в int32 коды (без double) с сохранением -1 NaN.
3. Достать `y_codes` из уже готового `data` (колонка таргета `target_names[0]`), только если стратегия != frequency.
4. `fold_categorical_block`, запись в итоговый dtype, `+1` NaN-сдвиг как сейчас (`_encode_categorical_block` без изменений).
5. Запись `fold_report_` в `self` (в `_discretize_inputs_part1` после вызова `categorize_dataset`: возвращать report третьим элементом нельзя без поломки сигнатуры; вернуть через необязательный `report_out: list | None` аргумент или атрибут-приёмник).
6. Пермутационный нуль: добавить в `_target_dependent_columns` категориальные колонки при стратегии != frequency; в `PermutedTargetBinner`/`_honest_null_codes_uncached` переиспользовать закэшированные `(L,K)` таблицы (пермутировать y_codes, пересчитать bincount, заново свернуть). Не пересчитывать факторизацию.
7. transform: без изменений (selection-only). Рецепты FE на свёрнутых категориях не вводить в этой фиче.

### 4.3 Структуры данных
- `fold_report_: list[dict]` (pickle-безопасно, см. F6).
- Параметры: `categorical_fold_strategy: str = "frequency"`, `categorical_fold_kwargs: dict | None = None` (ключи `min_support`, `cv_folds`).
- Внутренние: `counts[(L, K)] int32` на колонку, живут только внутри вызова; не сохраняются на `self`.

### 4.4 Производительность (1M строк)
- `bincount` на колонку ~ 5-10 мс при 1M строк; 20 колонок ~ 0.2 с.
- Жадное слияние L=500 -> 127 групп с пересчётом delta_MI затронутых пар: ~ мс на колонку (нужно измерить cProfile'ом; не гадать).
- OOF для `auto`: K таблиц из той же `counts` вычитанием фолд-таблиц (одним проходом `bincount` с ключом `fold*L*K+...`), не K проходов по данным.
- Главный риск по скорости: пермутационный нуль (F2): если нет кэша таблиц, цена свёртки умножается на число пермутаций; закладываем `counts` из шага 3 и пересчёт только `y_codes` перестановкой.
- Память: целевые блоки int8/int16; избавиться от float64 промежуточной матрицы для новых стратегий; текущая float64 для `frequency` не трогать.
- Профилирование: 100k x 100 (20 кат x 500 ур.), затем 1M; сначала замерить текущий `cap_categorical_cardinality` как baseline.

### 4.5 Тесты
Все в `tests/` рядом с существующими тестами фильтров; запуск целевой, `-n 0`, по одному pytest-процессу.
- `test_categorical_fold_unit.py`: детерминизм `remap`; `frequency` даёт бит-идентичные коды с `cap_categorical_cardinality` (рег. тест на дефолт); `max код <= cap` и `nbins` согласован с числом групп; NaN сохраняется; polars и pandas дают одинаковые группы (с учётом различия порядка кодов); cap=127 граница int8 с NaN-сдвигом; колонки с card <= cap нетронуты; все уровни редкие; один уровень; константная колонка.
- `test_categorical_fold_biz_value.py`: редкий информативный уровень (30 строк из n=20000) сохраняется при `target_stat`, теряется при `frequency`; шумовые уровни не получают MI выше пермутационного нуля (n=2000, 60 уровней); бинарный, мультикласс, регрессия (через `classes_y`), счётный таргет; регрессия на выбор `auto` (OOF) в обе стороны.
- `test_categorical_fold_null.py`: honest null видит таргет-зависимую свёртку (категориальная колонка в `_dependent`; значимость шумовой категориальной не завышена); кэш таблиц не меняет результат.
- `test_categorical_fold_pickle_params.py`: pickle round trip, `clone`/`get_params`, загрузка старого pickle без новых атрибутов (дефолт frequency), `fold_report_` сериализуем.
- `test_categorical_fold_profile.py` (cProfile, маркер slow, не в локальном прогоне): 100k x 100 / 1M.
- Мета-тест: параметры в docstring `MRMR` (существующие мета-тесты, py-ci-shared; не плодить локальных).

## 5. Рекомендуемый срез поставки
1. Шаг 1 proposal (min_support без таргета) как `fold_min_support`, с выводом порога из таблицы, а не из cap; без изменений честной нулевой машины (таргет не используется).
2. `target_stat` через жадное слияние по delta_MI + OOF `auto`, вместе с F2 (нуль) в ОДНОМ пакете, не раньше: без F2 фича даёт ложные находки.
3. Не делать: разделы 5-6 по специальным статистикам таргета, smoothing, "группа с ближайшей статистикой для unseen", recipe-replay категорий.
