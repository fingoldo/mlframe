# Visual walkthrough - what the charts actually look like

Not a code read. Every finding below was seen in a rendered PNG, on the chart type's OWN defaults with
ordinary inputs (3-4k rows, 6-9 classes/labels/models, feature names of the length this repo really
produces). Reproduce with `matplotlib[png]` through `render_and_save`; the harness is
`profiling/render_every_chart.py`.

The point of rendering rather than reading: several of these fire on the DEFAULT panel template with no
adversarial input at all, which a code read tends to discount as "only at large N".

## WALK-01 [P1] LTR NDCG_BY_QSIZE: rotated tick label crosses out of its panel and lands on the neighbour

`charts/ltr.py` NDCG_BY_QSIZE. With one query-size bin the tick label is
`8-15 (n=300, CI[0.92,0.94])` - 24 chars, rotated 45 deg, anchored under a panel that is not tall enough
for it. It runs past the panel's bottom edge and overprints the axis title of the panel below. The same
happens on NDCG_DIST, whose `all queries (n=300)` collides with its own `query population` axis title.

Two separate causes: the label carries the CI and the support inside the tick text (so it is unbounded in
length), and the figure height is fixed per cell while the rotated text needs height proportional to
`len(label) * sin(45)`.

## WALK-02 [P1] LTR NDCG_BY_QSIZE: a single category is drawn as a full-panel-width bar

Same panel. One bin means one bar, and with no bar-width cap it spans the entire panel - it reads as a
filled background, not as a measurement. Any chart that bins into a variable number of groups can land
here (query size, decile, group count).

## WALK-03 [P1] Legends are drawn inside the axes and cover the data

Seen on four of the seven figures rendered:

* `multilabel` CALIB_GRID - six entries of `label_5_long_descriptive_name (ECE=0.203, n=3,000)`. The legend
  box is WIDER than the panel and covers the whole upper-left half, including the reliability curves it
  labels. Worst instance.
* `regression` decile panel - legend sits over bars D5..D7.
* `binary` SCORE_DIST - legend over the left tail of both histograms.
* `model_comparison` overlay - 8 model names over the curves.

`legend_outside` exists in the matplotlib renderer and is simply not requested by these panels.

## WALK-04 [P1] Point annotations duplicate the legend entry AND overflow the panel

`binary`: the operating point is written twice - once as a legend entry
(`thr=0.50: TPR=0.64 FPR=0.29`) and again as red text next to the star. On the PR panel that second copy
starts near recall 0.64 and runs off the right edge, clipped by the axes. On the THRESHOLD panel
`F1 optimum @ 0.257 (F1=0.574)` is drawn ON the F1 curve, which strikes through the text.

## WALK-05 [P2] Panel titles that carry statistics wrap to two and three lines and eat the plot area

`quantile` INTERVAL_COVERAGE takes three lines
(`Interval coverage (empirical vs nominal): 1 of 1 levels miss their 95% CI; worst at nominal 0.80,
empirical 0.998 -- over-covers (intervals too wide)`), pushing into the panel above. Two-line titles are
routine: binary PR and SCORE_DIST, regression decile, multilabel co-occurrence and cardinality, ltr
NDCG_BY_QSIZE and MRR, model_comparison correlation. The titles are built by string concatenation with no
width budget, so the wrap point depends on the numbers.

## WALK-06 [P2] Spearman heatmap reads as anti-correlation

`model_comparison` correlation panel, confirming the diverging-scale finding from the code audit
(VIS-01): eight models correlate 0.20-0.25, the colour scale spans 0.2-1.0 on `RdBu_r`, and every
off-diagonal cell renders deep blue - the colour a reader takes for strong NEGATIVE correlation. Nothing
pins 0 to white.

## WALK-07 [P2] An empty panel is reserved and captioned for a metric that was never chosen

`model_comparison` with default arguments prints `Leaderboard: metric '' missing on all models` into a
full half-row. The default `metric=""` cannot match anything, so the default invocation always wastes a
panel.

## WALK-08 [P2] Grid cells are unequal and the last row is left ragged

`regression`: the scatter panel renders visibly narrower than its row neighbour. `multilabel` and
`quantile`: an odd panel count leaves a final half-width panel alone on the last row, aligned left.
`model_comparison`: the correlation heatmap occupies a narrow left column against empty space.

## WALK-09 [P2] Heatmap axis labels are full-length on both axes

`multilabel` co-occurrence: `label_N_long_descriptive_name` on rows AND rotated on columns, together
taking roughly half the panel area, squeezing the matrix itself into the remainder. Bars truncate; heatmap
ticks do not (matches LBL-03).

## WALK-10 [P3] Red/green is the only channel on the WoE bars

`category_discriminability`: direction is encoded as red vs green fill and nothing else - no hatch, no
sign in the label, no ordering by sign. Otherwise this chart is the cleanest of the set: labels do not
collide, error bars are present, ordering is by effect size.

## WALK-11 [P3] The suptitle and the single panel title repeat each other

`category_discriminability` prints `Category discriminability` as the suptitle and again as the first
line of the panel title.
