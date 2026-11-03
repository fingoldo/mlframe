
## 2026-11-03: no per-group Python lists for candidates

100000 x 100 (4 group columns, 32 numeric columns, 384 candidates): the stage sample put 12 of 21 samples on the main thread waiting for the column pool, whose
tasks converted every group's sorted values and the global sorted column to Python lists (`.tolist()`, GIL-bound, so the threads could not overlap it). The
candidate payloads now keep the arrays; `build_grouped_quantile_recipe` already converts the values of the `top_k` winners to float lists. Family wall 3.96 -> 2.73 s
(one process, old and new module side by side), same eight winners, equal recipes and columns.
