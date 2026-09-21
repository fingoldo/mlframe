# Exact, reusable train/val/test splits: usage examples

Copy-pasteable recipes for pinning the holdout sets of `train_mlframe_models_suite` so runs stay comparable:
retrain with more history on the identical test (and val) rows, or fix test/val to exact calendar windows.
Everything is configured on `TrainingSplitConfig`; without the fields below the split is the usual
fraction-based one. Implementation and rules: `mlframe/training/_fixed_splits.py`.

## Decision tree: which recipe do I need?

```
What must stay fixed between runs?
├── The exact rows of test (and val) from an earlier run   -> Recipe 1 + Recipe 2 (id file)
│   └── only test; val may be re-drawn from the new data  -> Recipe 3
├── A calendar period ("test = March 2025")                -> Recipe 4 (date windows)
└── Nothing, I just want to be able to reproduce it later  -> Recipe 1 alone
```

## Recipe 1: record the split of a run

Name a stable, unique row key with `id_column` and give the suite a `data_dir`. The run writes
`split_ids.parquet` (columns `[<id_column>, "split"]`, split in `train`/`val`/`test`/`calib`) plus a
`split_ids.json` sidecar next to its other split artifacts.

```python
from mlframe.training.configs import OutputConfig, TrainingSplitConfig
from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

fte = SimpleFeaturesAndTargetsExtractor(ts_field="ts", regression_targets=["y"])

models, metadata = train_mlframe_models_suite(
    df=df_2024,                      # must contain the key column, here "job_id"
    target_name="y",
    model_name="run1",
    features_and_targets_extractor=fte,
    mlframe_models=["lgb"],
    split_config=TrainingSplitConfig(id_column="job_id", test_size=0.1, val_size=0.1),
    output_config=OutputConfig(data_dir="D:/experiments", models_dir="models"),
)

mem = metadata["split_membership"]
print(mem["path"])            # <data_dir>/models/<target_name>/<model_name>/split_ids.parquet
print(mem["counts"])          # {'train': ..., 'val': ..., 'test': ..., 'calib': ...}
print(mem["holdout_starts"])  # first timestamp of val's / test's sequential block
```

- The key is read right after the features/targets extractor and then dropped, so it never becomes a model
  feature. It is dropped from predict frames too (`predict_mlframe_models_suite` / `predict_from_models` read it
  from `metadata["split_membership"]["id_column"]`), so scoring frames may keep the column.
- Listing the key for dropping as well is fine: `SimpleFeaturesAndTargetsExtractor(columns_to_drop={"job_id"})` or
  `PreprocessingConfig(drop_columns=["job_id"])` both drop it after it has been read. Only an extractor that
  removes the column from the frame it returns (a custom `transform`) makes it unreadable; that raises a
  `ValueError` naming the column.
- The key must be unique and non-null. Duplicates or nulls raise `ValueError`: a duplicated key cannot pin one
  row to one split when the membership is replayed. A key literally named `"split"` is rejected (it would
  collide with the file's own column).
- Without `data_dir` nothing is written (a WARNING says so) and `mem["path"]` is `None`.

## Recipe 2: retrain with more history, same test and val

Point `split_ids_path` at the file from Recipe 1. By default (`reuse_splits=("test", "val")`) every id listed
as test goes to test and every id listed as val goes to val, exactly. All other rows, including the new
history, form the pool for train.

```python
models2, metadata2 = train_mlframe_models_suite(
    df=df_2022_2024,                 # more history + the rows of run 1
    target_name="y",
    model_name="run2_more_history",
    features_and_targets_extractor=fte,
    mlframe_models=["lgb"],
    split_config=TrainingSplitConfig(
        id_column="job_id",
        split_ids_path=metadata["split_membership"]["path"],  # the file run 1 wrote
    ),
    output_config=OutputConfig(data_dir="D:/experiments", models_dir="models"),
)

print(metadata2["split_pinning"])
# {'missing_ids': {}, 'cutoff': '2024-11-02T00:00:00', 'n_excluded_after_cutoff': 0, 'sources': {'test': 'file', 'val': 'file'}}
```

Run 2 records its own `split_ids.parquet` as well, so run 3 can pin to either run.

What to expect in the log and metadata:

- **Rows newer than the holdouts stay out of train.** With `train_before_holdout=True` (the default), pool rows
  whose timestamp is at or after the earliest pinned holdout start are excluded from train, with an INFO line
  and their count in `split_pinning["n_excluded_after_cutoff"]`. The start comes from the sidecar: it is the
  first timestamp of each holdout's sequential block, so a shuffled val whose random part was drawn from the
  train period does not move the cutoff. Without the sidecar the pinned test's earliest timestamp is used. Pass
  `train_before_holdout=False` to keep those rows in train.
- **Ids missing from the new frame** are WARNed per split and counted in `split_pinning["missing_ids"]`; the
  pinned set just gets smaller. A reused split that matches zero rows raises (almost always the wrong file or an
  id dtype mismatch, e.g. int in the file and str in the frame).
- **Groups.** With a `group_field`, a group that has rows both in train and in a pinned holdout is WARNed and
  counted in `split_pinning["groups_spanning"]`. Rows are not moved: the pinned membership wins, so expect some
  group leakage in that split's metrics.

## Recipe 3: pin only test, re-draw val from the larger data

```python
split_config = TrainingSplitConfig(
    id_column="job_id",
    split_ids_path="D:/experiments/models/y/run1/split_ids.parquet",
    reuse_splits=("test",),
    val_size=0.1,          # carved by the normal splitter from the non-test pool
    calib_size=0.05,       # optional; also carved from that pool
)
```

Unpinned splits are carved by the regular splitter from the remaining rows, so `val_size`, `val_placement`,
`shuffle_val`, groups and stratification behave as they do without pinning. `test_size` is ignored for a pinned
test. `reuse_splits` accepts any non-empty subset of `train`/`val`/`test`/`calib`. When `train` is reused, train
is exactly the recorded ids and the unpinned rows the splitter would add to it are left out.

## Recipe 4: pin test/val to calendar windows

No id file needed: rows whose timestamp falls in a half-open `[start, end)` window go to that split. Either bound
may be omitted (unbounded). The timestamps are the extractor's `ts_field`.

```python
split_config = TrainingSplitConfig(
    val_start="2025-01-01", val_end="2025-03-01",
    test_start="2025-03-01", test_end="2025-04-01",
)
```

- Rows at or after the earliest window start (here `val_start`) that are not in a window are kept out of train
  (`train_before_holdout`); that includes rows after the test window, which are simply not used.
- The windows must not overlap, and `end` must be after `start`; tz-aware bounds are aligned to tz-aware
  timestamps. A window that selects zero rows raises.
- Windows combine with an id file: pin test from the file and val by a window (`reuse_splits=("test",)` plus
  `val_start`/`val_end`), or the other way round. The same split pinned both ways raises.
- Add `id_column` to record the resulting membership for later runs, as in Recipe 1.

## Comparing the runs

With test pinned, the test metrics of run 1 and run 2 are computed on the same rows, so the difference is the
effect of the extra history (plus training noise). Check that the sets really match before comparing:

```python
import pandas as pd

s1 = pd.read_parquet(metadata["split_membership"]["path"])
s2 = pd.read_parquet(metadata2["split_membership"]["path"])
t1 = set(s1.loc[s1["split"] == "test", "job_id"])
t2 = set(s2.loc[s2["split"] == "test", "job_id"])
assert t1 == t2, (len(t1 - t2), len(t2 - t1))
```

`t1 - t2` is non-empty only when some test ids were missing from run 2's frame (see the WARNING from Recipe 2).
