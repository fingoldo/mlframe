# Kaggle / Industry Recommendations: Date Feature Engineering

Research compiled 2026-05-26 to inform the extended defaults shipped in
`mlframe.feature_engineering.basic.create_date_features` and the new
`add_cyclical_date_features` helper. The aim is to capture what top
Kaggle GMs and the industry-canonical tabular libraries actually emit
from a single datetime column.

## 1. Default integer date parts (the "calendar" features)

Across the canonical recipes the universally-emitted set is:

| Component | Why it earns its slot | Recommended dtype |
|---|---|---|
| `year` | Multi-year drift / regime changes (M5 2017->2019, COVID jumps) | `int32` (NOT int8 -- 2100 overflows int16) |
| `month` | Annual seasonality (1-12) | `int8` |
| `day` (of month) | Pay-cycle / month-end effects | `int8` |
| `weekday` | Strongest seasonality on retail / web / mobility data | `int8` |
| `quarter` | Fiscal-quarter accounting cycles | `int8` |
| `week_of_year` | Holiday alignment (ISO week) | `int8` |
| `day_of_year` | Annual cycle continuous proxy (1-366) | `int16` |
| `is_weekend` | Cheap binary, materially helps on retail / commute targets | `bool` |

This matches the fastai `add_datepart` shape closely (Year / Month / Week
/ Day / Dayofweek / Dayofyear / Is_month_end / Is_month_start /
Is_quarter_end / Is_quarter_start / Is_year_end / Is_year_start /
Elapsed). fastai also emits an `Elapsed` Unix-seconds feature for
across-period continuity -- not in our defaults yet because polars
`.dt.timestamp("s")` is already the natural way to get it.

Sources:

- fastai tabular `add_datepart`: <https://docs.fast.ai/tabular.core.html>
- fastai datepart article (KDnuggets): <https://www.kdnuggets.com/2018/03/feature-engineering-dates-fastai.html>
- M5 winning recipes (Yakovlev / Tunguz threads): <https://www.kaggle.com/code/sergioli212/m5-all-feature-engineering-ready-to-use>
- Generic FE tutorial (Medium / Piyumal): <https://medium.com/@sasinduha/feature-engineering-on-date-feature-a7c00e0b955f>

### dtype overflow note

`year` MUST NOT be int8 (max 127) or int16 (max 32767 -- fine in
practice but defensive int32 is one byte every 1k rows and removes the
question). Tree models do not benefit from the narrower dtype; linear
models do not benefit either. int32 is the right default.

## 2. Cyclical sin/cos encoding (Kaggle-canonical for linear / NN)

The integer encoding implies `Jan == 1` is closer to `Feb == 2` than to
`Dec == 12`, but on the actual calendar Jan and Dec are adjacent. Trees
can learn the wrap-around with enough splits; linear models, kNN, and
neural networks cannot.

The Kaggle-canonical fix is the sin/cos pair, popularised on scikit-
learn examples and used in every NN-tabular write-up since ~2018:

```python
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

Properties:

- `month_sin^2 + month_cos^2 == 1` exactly. Invariant for unit tests.
- Distance between (Dec, Jan) in the (sin, cos) plane equals distance
  between (Jun, Jul). Adjacency preserved.
- Each periodic component takes 2 floats (vs N one-hots for N levels).
- Range is `[-1, 1]`, float32 is enough precision; saves 4x bytes vs
  float64.

### Granularities we emit by default

- `hour` (period 24) -- intraday / circadian
- `day` (period 31 for day-of-month) -- pay-cycle
- `weekday` (period 7) -- weekly seasonality
- `month` (period 12) -- annual seasonality
- `day_of_year` (period 365.25) -- continuous annual cycle, finest

(`week_of_year` and `quarter` are usually NOT cyclical-encoded -- they
benefit more from being kept as a raw integer that trees can split on.)

Sources:

- scikit-learn cyclical FE example: <https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html>
- NVIDIA blog (sin/cos vs RBF vs spline): <https://developer.nvidia.com/blog/three-approaches-to-encoding-time-information-as-features-for-ml-models/>
- skforecast cyclical features: <https://skforecast.org/0.10.1/faq/cyclical-features-time-series>
- TDS cyclical encoding write-up: <https://towardsdatascience.com/cyclical-encoding-an-alternative-to-one-hot-encoding-for-time-series-features-4db46248ebba/>
- feature-engine `CyclicalFeatures` (library API mirror): <https://feature-engine.trainindata.com/en/latest/user_guide/creation/CyclicalFeatures.html>
- Kaggle deep-learning cyclical-encoding kernel (van Wyk): <https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning>

## 3. Categorical vs numeric encoding tradeoffs

Top Kaggle solutions ship BOTH:

- raw integers (for trees -- LightGBM / XGBoost / CatBoost handle the
  wrap-around fine via deep enough splits)
- sin/cos pairs (for linear ridges and NN blending heads)

The integer form costs ~1 byte/col with int8; the sin/cos pair costs
8 bytes/col (2 x float32). On a 100M row frame the cyclical encoding
adds ~600MB across all 5 default periods -- not free, but the standard
stacker / NN-blender will use them. The default we ship is OPT-IN
(`cyclical=False`) so the unconditional cost is zero for users who only
train trees. Callers that include a linear / NN base learner flip
`cyclical=True`.

One-hot encoding of date parts (NVIDIA blog Approach 1) is generally
discouraged for high-cardinality components (day_of_year -> 366 OHE
columns) -- sin/cos is strictly more expressive AND smaller. We do not
provide a one-hot path.

## 4. Multi-timezone handling

The thorniest correctness issue in date FE is silent timezone mismatch.
When two columns in the same frame carry different tz (e.g. UTC + local
+ tz-naive), the extracted `hour` / `weekday` / `day` are MEANINGLESS
across rows -- the same instant maps to different hours in different
rows.

**No library handles this for you.** pandas raises on mixed-tz
arithmetic in some paths but happily extracts inconsistent `.dt.hour`
in others; polars stores tz at the Series level so cross-column tz
mismatch is silent.

Recommendations (synthesised from Yakovlev's defensive coding style and
the M5 competition discussion forum):

1. Normalise upstream: convert every datetime column to UTC at ingest;
   keep one separate `local_tz_offset_hours` integer column if the
   local-time effect matters (e.g. retail intraday).
2. Detect and warn at FE boundary: list every observed timezone (
   including the tz-naive bucket as `"naive"`) -- this is what our
   helper now does. Don't auto-convert: the caller's downstream code
   might depend on the local-time semantics.
3. Never silently strip tz (`dt.tz_localize(None)`) inside an FE
   helper -- destroys information.

The helper in this module warns with the concrete tz list rather than
raising, because legitimate use cases (single-column FE, intentionally
mixed columns for an ablation) shouldn't be blocked by a guard.

## 5. Other Kaggle-popular date features (NOT in defaults)

These keep showing up in winning solutions but are domain-specific
enough to leave out of the unconditional defaults. Listed for
reference -- callers can add them via `methods={...}` or directly:

- `days_since_<event>` -- distance to a fixed anchor (start of train,
  product launch, last holiday). Hugely predictive on retail.
- `payday_distance` -- abs(day - 15) or abs(day - 28). Mobility /
  banking targets.
- `week_of_month` -- `(day - 1) // 7 + 1`. Retail promotion cycles.
- `is_month_start` / `is_month_end` / `is_quarter_start` / `..._end` /
  `is_year_start` / `..._end` -- pandas built-ins. fastai emits these
  by default.
- `time_of_day` bucketed (morning / afternoon / evening / night) --
  more interpretable than raw hour for linear models.
- `lunar_phase` -- documented to help on EM crime / ER / fishing
  targets but extremely niche.
- `days_to_holiday` -- requires a calendar of holidays; usually shipped
  as a separate FE step.

## 6. Library-level cross-references

Implementations worth cribbing from:

- fastai `add_datepart`: <https://github.com/fastai/fastai2/blob/master/fastai2/tabular/core.py>
  -- the de-facto reference for tabular date FE since 2018.
- feature-engine `CyclicalFeatures`: <https://feature-engine.trainindata.com/en/latest/user_guide/creation/CyclicalFeatures.html>
  -- clean sin/cos API with explicit period dict, sklearn-compatible.
- skforecast cyclical features: <https://skforecast.org/0.10.1/faq/cyclical-features-time-series>
  -- specifically for forecasting pipelines.
- scikit-learn cyclical example: <https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html>
  -- the canonical scikit-learn implementation pattern.

## 7. Recommendations applied to mlframe

The implementation in `basic.py` reflects these conclusions:

- Default `_DEFAULT_DATE_METHODS` now includes `year` (int32),
  `quarter` (int8), `week_of_year` (int8), `day_of_year` (int16),
  `is_weekend` (bool) on top of the legacy `day` / `weekday` / `month`.
- New helper `add_cyclical_date_features(df, cols, periods=(...))`
  emits sin/cos pairs for `hour` / `day` / `weekday` / `month` /
  `day_of_year` by default.
- New `add_cyclical: bool = False` kwarg on `create_date_features`
  lets callers opt in to "everything at once" without two function
  calls.
- Mixed-tz columns trigger a single explicit warning listing every
  observed tz (including `"naive"`), but never auto-convert.
- Sin/cos outputs are float32 normalised to `[-1, 1]`.
- Backward compatibility: every previous default field is still
  emitted; new fields are additive.
