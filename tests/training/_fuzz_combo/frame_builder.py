"""Frame synthesis: turn a FuzzCombo into (df, target_col, cat_feature_names)."""
from __future__ import annotations

from typing import Any  # noqa: F401  (annotation strings under PEP 563)

from .combo import FuzzCombo  # noqa: F401  (annotation strings under PEP 563)


# Column names the grouped_delta / lagged_diff MRMR-FE kinds consume. Emitted into the synthetic frame (and wired into the MRMR ctor via
# builders.build_mrmr_kwargs) only when ``mrmr_fe_ratio_delta_diff_cfg`` selects one of those kinds, so the two FE kinds actually run in the
# sweep instead of being canonicalised to "off" for lack of a group / time column. ``MRMR_FE_GROUP_COL`` is a low-cardinality repeating int key
# (per-group mean/std are well defined); ``MRMR_FE_ORDER_COL`` is a strictly-monotone index so the lagged_diff sort is deterministic. The
# ``*_VAL_COL`` columns are the numeric sources the transform is applied to, engineered so the transform recovers a target-predictive signal.
MRMR_FE_GROUP_COL = "mrmr_fe_group"
MRMR_FE_GROUP_VAL_COL = "mrmr_fe_gval"
MRMR_FE_ORDER_COL = "mrmr_fe_order"
MRMR_FE_LAG_VAL_COL = "mrmr_fe_lval"


def build_frame_for_combo(combo: FuzzCombo):
    """Build a pd / pl DataFrame matching the combo's input spec.

    Returns (df, target_col_name, cat_feature_names: list[str]).

    Text columns (``combo.text_col_count > 0``) are only emitted when
    ``"cb"`` is in ``combo.models`` — CatBoost is the only strategy that
    consumes ``text_features`` (see ``strategies.py``
    ``supports_text_features=True`` for CB; every other model either
    drops them via ``core.py:486-496`` or never looks at them). Same
    gate for embedding columns (``pl.List(pl.Float32)``). We still
    emit them SOMETIMES (not always) because the CB×text_features and
    CB×embeddings paths have their own TF-IDF / feature-dispatch
    edge cases that the earlier fuzz runs never exercised — pin them
    behind the "cb present" gate so a CB-less combo doesn't spuriously
    fail for a reason unrelated to what's being sampled.
    """
    import numpy as np

    rng = np.random.default_rng(combo.seed)
    n = combo.n_rows

    num_cols = {
        f"num_{i}": rng.standard_normal(n).astype("float32") for i in range(4)
    }
    cat_pools = [
        ["A", "B", "C"],
        ["X", "Y", "Z", "W"],
        ["alpha", "beta"],
        ["cat1", "cat2", "cat3", "cat4", "cat5"],
        ["US", "UK", "DE"],
        ["mon", "tue", "wed", "thu"],
        ["P", "Q"],
        ["r1", "r2", "r3"],
    ]
    cat_cols = {}
    cat_names: list[str] = []
    # R3-5 weird_cat_content: substitute specific pool entries with
    # pathological values that historically broke auto-detection, TF-IDF,
    # or encoder dispatch.
    def _apply_weird(pool: list[str], kind: "str | None") -> list[str]:
        if not kind:
            return pool
        pool = list(pool)
        if kind == "empty":
            # replace first entry with empty string
            if pool:
                pool[0] = ""
        elif kind == "unicode":
            # mix in a unicode-heavy value (emoji + CJK + combining marks)
            pool.append("кат́")  # cyrillic + combining acute
            pool.append("\U0001f600\U0001f4ca")        # emoji pair
        elif kind == "null_like":
            # strings that LOOK like nulls but are real string values.
            # Pipeline bugs sometimes treat these as actual nulls.
            pool.extend(["None", "NaN", "null", "NA"])
        return pool

    for i in range(combo.cat_feature_count):
        # Wrap with modulo so cat_feature_count > len(cat_pools) cycles
        # through the pool list rather than IndexError-ing.
        pool = _apply_weird(cat_pools[i % len(cat_pools)], combo.weird_cat_content)
        values = [pool[j % len(pool)] for j in range(n)]
        if combo.null_fraction_cats > 0:
            mask = rng.random(n) < combo.null_fraction_cats
            values = [None if mask[j] else v for j, v in enumerate(values)]
        cat_cols[f"cat_{i}"] = values
        cat_names.append(f"cat_{i}")

    # Target: derive from num_0 + num_1 with noise so models have signal.
    # R3-2 multi_classification_{3,5}: discretise a continuous score into
    # N bins by quantile so distribution is approximately balanced.
    # R3-4 imbalance_ratio: on binary, shift threshold so minority class
    # is 5%/1% of rows instead of ~50/50. Not applied to multi-class
    # (implementation complexity not worth it — balanced multiclass is
    # the useful axis to exercise).
    if combo.target_type == "regression":
        target = 2.0 * num_cols["num_0"] - 1.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
        target_col = "target_reg"
    elif combo.target_type == "multi_target_regression":
        # F-24 audit-pass-9 #8: K=2 independent continuous targets derived
        # from disjoint informative features. Shape (N, K=2) float32 so the
        # estimator's auto-detect at training/neural/base.py:548 takes the
        # multi-target branch (num_classes=K head sharing trunk). Both
        # targets carry distinct signal so the MSE-on-(N,K) loss is
        # well-defined and per-target metrics are meaningfully different.
        #
        # iter633 fix: emit (N, K) ONLY when EVERY model in the combo
        # natively handles a 2-D continuous target. The CANON_KEY at
        # :2306-2313 collapses MTR-without-native-backend to "regression"
        # for dedup but a mixed combo (one native + one non-native, e.g.
        # mlp+linear or cb+lgb+xgb where lgb chokes) still crashes the
        # non-native model with ``Unknown label type: continuous-multioutput``.
        # The "native" gate is restricted to ``cb`` (CatBoost MultiRMSE,
        # configured by ``_ensure_cb_mtr_loss``) because the MLP F-24 (N,K)
        # auto-detect path documented in
        # docs/multi_target_regression_design.md hangs in
        # torch.nn.utils.clip_grad on the smoke harness (observed
        # 2026-05-31 on cb+mlp combo), and the xgb multi_output_tree
        # dispatcher is opt-in. Whenever a non-cb-only combo is detected,
        # downgrade the frame to 1-D regression so the canon's
        # "equivalent to regression" promise holds at the data level
        # instead of crashing or hanging at model.fit.
        _NATIVE_MTR_MODELS = {"cb"}
        if all(m in _NATIVE_MTR_MODELS for m in combo.models):
            t0 = 2.0 * num_cols["num_0"] - 1.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
            t1 = 1.5 * num_cols["num_2"] + 0.8 * num_cols["num_3"] + rng.standard_normal(n) * 0.3
            target = np.column_stack([t0, t1]).astype("float32")
            target_col = "target"  # FTE handles 2-D target via shape sniff
        else:
            target = 2.0 * num_cols["num_0"] - 1.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
            target_col = "target_reg"
    elif combo.target_type == "binary_classification":
        logits = num_cols["num_0"] - 0.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
        # Use the canonical imbalance value (clamped by n_rows via
        # _canonical_imbalance) so we never generate a target whose split
        # would reliably drop a class from val/test.
        imb = combo._canonical_imbalance()
        if imb == "rare_5pct":
            thresh = np.quantile(logits, 0.95)
        elif imb == "rare_1pct":
            thresh = np.quantile(logits, 0.99)
        else:
            thresh = 0.0
        target = (logits > thresh).astype("int32")
        target_col = "target"
    elif combo.target_type == "multiclass_classification":
        # 3-class quantile-cut to balanced classes (Phase H restoration of R3-2).
        score = num_cols["num_0"] + 0.3 * num_cols["num_1"] + rng.standard_normal(n) * 0.4
        k = 3  # default 3 classes; multi_5 deferred (resource-heavy)
        quantiles = [np.quantile(score, i / k) for i in range(1, k)]
        target = np.digitize(score, quantiles).astype("int32")
        target_col = "target"
    elif combo.target_type == "multilabel_classification":
        # K=3 binary labels with deliberate label correlation so chain ensemble
        # has a chance to win. Post-generation guarantee: no all-zero rows
        # (iterstrat / sklearn reject those silently).
        k = 3
        logit0 = num_cols["num_0"] - 0.4 * num_cols["num_1"] + rng.standard_normal(n) * 0.4
        y0 = (logit0 > 0).astype("int8")
        logit1 = 0.5 * y0 + num_cols["num_2"] + rng.standard_normal(n) * 0.4
        y1 = (logit1 > 0).astype("int8")
        logit2 = 0.5 * y0 + 0.5 * y1 + 0.3 * num_cols["num_3"] + rng.standard_normal(n) * 0.4
        y2 = (logit2 > 0.6).astype("int8")  # rarer
        Y = np.column_stack([y0, y1, y2])
        # Guarantee no all-zero rows (iterstrat, MultiOutputClassifier).
        zeros = (Y.sum(axis=1) == 0)
        if zeros.any():
            # flip a random label to 1 in zero rows (deterministic via rng)
            for i in np.where(zeros)[0]:
                Y[i, rng.integers(0, k)] = 1
        target = Y  # (N, K)
        target_col = "target"  # FTE will need to handle 2-D target
    elif combo.target_type == "learning_to_rank":
        # Graded relevance 0..3 derived from the same informative features
        # as regression, then bucketed. Synthetic queries with ~8 docs each
        # — group_field 'qid' is added below for the ranker suite.
        # Post-generation guarantee: every query has at least one positive
        # (some library rankers warn or NDCG goes NaN otherwise).
        score = 1.5 * num_cols["num_0"] - 0.7 * num_cols["num_1"] + rng.standard_normal(n) * 0.4
        # Quantile-cut to 4 levels (0..3) so frame has graded relevance.
        q = [np.quantile(score, i / 4) for i in range(1, 4)]
        target = np.digitize(score, q).astype("int32")
        # Build qid: ~8 docs per query (n_rows / 8).
        n_per_query = max(2, min(10, n // 30))  # at least 2 docs per query
        n_queries = max(1, n // n_per_query)
        # Last query may be short; pad qid array to length n.
        qid = np.repeat(np.arange(n_queries), n_per_query)
        if len(qid) < n:
            qid = np.concatenate([qid, np.full(n - len(qid), n_queries - 1, dtype=qid.dtype)])
        elif len(qid) > n:
            qid = qid[:n]
        # Guarantee at least one positive per query: for any query whose
        # docs are all-zero, flip the highest-score doc to relevance 1.
        for q_id in np.unique(qid):
            mask = qid == q_id
            if (target[mask] == 0).all():
                top_idx = np.where(mask)[0][np.argmax(score[mask])]
                target[top_idx] = 1
        # Add qid as a frame column so downstream FTE.group_field can pick it up.
        num_cols["qid"] = qid.astype("int32")
        target_col = "relevance"
    else:
        raise ValueError(f"unknown target_type: {combo.target_type}")

    # Text columns: only emit when CB will actually consume them. Each
    # "text" row is a 3-word sentence drawn from a shared vocabulary so
    # CB's TF-IDF builds a non-empty dictionary (a single-word-per-row
    # column above the cardinality threshold would otherwise degenerate).
    # Use the canonical text count so combos that would crash CB's
    # text-estimator on a small NaN-heavy fold never see a text column
    # in the data — _canonical_text_col_count returns 0 in that window.
    _eff_text_col_count = combo._canonical_text_col_count()
    # Text columns are only ROUTABLE when auto-detection runs: the suite
    # classifies them as text_features (CB consumes them, every non-CB model
    # excludes them via core/_misc_helpers.py:876 supports_text_features
    # gate). With auto_detect_cats=False the text column is never classified
    # nor excluded, so in a mixed combo (cb + hgb/lgb/xgb) the raw object
    # column reaches the non-CB model's numeric pipeline and crashes with
    # "could not convert string to float". Gate emission on auto_detect_cats
    # so an unroutable text column is never produced. Surfaced by fuzz
    # (2 cb+non-cb combos with auto-detect off).
    want_text = _eff_text_col_count > 0 and "cb" in combo.models and combo.auto_detect_cats
    text_vocab = [
        "python", "rust", "golang", "java", "swift", "kotlin",
        "backend", "frontend", "devops", "mlops", "dataeng", "platform",
        "cloud", "edge", "realtime", "batch", "stream", "vector",
        "search", "nlp", "vision", "audio", "robotics", "quantum",
    ]
    text_cols: dict[str, list] = {}
    if want_text:
        # Vectorised token-row build. The naive per-row loop builds n separate
        # Python lists of 3 ints + n " ".join calls -> ~n * (3 * 28B int + 1 list
        # header + 3 dict lookups) overhead, which OOMed on c0028 at n=200k under
        # concurrent profiler memory pressure (iter536 MemoryError at the
        # ``rows.append(" ".join(...))`` site). Numpy fancy-indexing into a
        # str-array gives the same per-cell strings without ever allocating the
        # Python-int idx-list, and the ``map(" ".join, words)`` builds the joined
        # strings as a streaming iterator the list constructor materialises in
        # one shot.
        vocab_arr = np.asarray(text_vocab)
        for i in range(_eff_text_col_count):
            idxs_arr = rng.integers(0, len(text_vocab), size=(n, 3))
            words = vocab_arr[idxs_arr]  # (n, 3) np.str_ — single buffer
            text_cols[f"text_{i}"] = list(map(" ".join, words))

    # Embedding columns: only Polars inputs support detection via
    # ``pl.List(pl.Float32)``; pandas has no robust native analog the
    # auto-detector recognises — skip for pandas to avoid spurious
    # xfails unrelated to the axis under test.
    want_embedding = (
        combo.embedding_col_count > 0
        and "cb" in combo.models
        and combo.input_type != "pandas"
    )

    # Data-axis injections (2026-04-24 combo extension).
    # inject_inf_nan: drop np.inf/-np.inf/np.nan into num_0's first 3 rows
    if combo.inject_inf_nan and n >= 3:
        num_cols["num_0"][0] = np.inf
        num_cols["num_0"][1] = -np.inf
        num_cols["num_0"][2] = np.nan
    # inject_degenerate_cols (#7): add one constant + one all-null numeric
    # column that the ``remove_constant_columns`` flag should strip.
    # The CB+multilabel canon at canonical_key was retired 2026-04-27
    # (batch 2): the production fix landed in trainer / wrappers.py
    # ensures num_const / num_null aren't mis-promoted to cat_features.
    extra_num_cols: dict = {}
    if combo.inject_degenerate_cols:
        extra_num_cols["num_const"] = np.full(n, 7.5, dtype="float32")
        extra_num_cols["num_null"] = np.full(n, np.nan, dtype="float32")
    # inject_zero_col (#40): add an all-zero numeric column as an
    # uninformative feature. Triggers the per-model "constant feature"
    # handling in CB/XGB/LGB/HGB — not supposed to break anything.
    if combo.inject_zero_col:
        extra_num_cols["num_zero"] = np.zeros(n, dtype="float32")
    # Fix G — adversarial columns.
    # inject_rank_deficient: a colinear pair (num_dep = 2 * num_0).
    # Should NOT crash linear models or destabilise GBDTs — this is a
    # correctness guard, not a performance ask.
    if combo.inject_rank_deficient:
        extra_num_cols["num_dep"] = (2.0 * num_cols["num_0"]).astype("float32")
    # inject_all_nan_col: a column that is 100% NaN. Separate from
    # inject_degenerate_cols (which covers const + null together) so
    # combos can toggle it independently.
    if combo.inject_all_nan_col:
        extra_num_cols["num_all_nan"] = np.full(n, np.nan, dtype="float32")
    # 2026-05-31 audit-pass-8 #10: XOR-synergy pair injection. Two binary
    # cols whose XOR predicts y at high MI but whose individual MI with y
    # is ~0 -- the canonical hard case for greedy MRMR. The new fleuret-
    # mode conditional-MI gate at evaluation.py:596 (_force_cond branch)
    # is what surfaces these survivors in mrmr_gains_. Gate at the canon
    # layer collapses this back to False outside (use_mrmr_fs AND
    # interactions_max_order >= 2) so dedup absorbs phantom variation.
    # The synergy is derived from the target so the conditional-MI test
    # at high n surfaces it -- pre-fix this pair was dropped silently by
    # the absolute-floor branch.
    if combo.inject_xor_synergy_pair_cfg:
        # Draw two independent Bernoulli(0.5) cols from a separate stream so
        # the per-combo seed produces a deterministic pair.
        _xor_rng = np.random.default_rng(combo.seed + 7919)  # 7919 = 1000th prime
        xor_a = _xor_rng.integers(0, 2, size=n).astype("float32")
        # Force XOR(xor_a, xor_b) ~ target where target is binarised. For
        # non-binary targets, binarise via threshold at median so the
        # synergy signal survives the y-discretisation MRMR runs.
        if combo.target_type == "binary_classification":
            y_bin = target.astype("int32")
        elif combo.target_type == "multilabel_classification":
            # Use label-0 as the discriminating y for the XOR pair.
            y_bin = target[:, 0].astype("int32")
        elif combo.target_type == "multi_target_regression" and target.ndim == 2:
            # iter633: native-MTR branch leaves target 2-D; XOR pair needs
            # a 1-D y, use target-0 as the discriminating signal (mirrors
            # the multilabel branch above).
            y_bin = (target[:, 0] > np.median(target[:, 0])).astype("int32")
        else:
            # Regression / multiclass / LTR / non-native MTR (1-D after
            # iter633 downgrade): binarise around median.
            y_bin = (target > np.median(target)).astype("int32")
        # xor_b = xor_a XOR y_bin => XOR(xor_a, xor_b) == y_bin.
        # Pure synergy: marginal MI(xor_a, y) ~ 0, MI(xor_b, y) ~ 0, but
        # conditional MI(xor_a; y | xor_b) >> 0.
        xor_b = np.bitwise_xor(xor_a.astype("int32"), y_bin).astype("float32")
        extra_num_cols["num_xor_a"] = xor_a
        extra_num_cols["num_xor_b"] = xor_b
    # 2026-05-31 audit-pass-8 #9: zero-weight-batch injection. Inserts a
    # contiguous block of far-past timestamps for the last 20% of rows so
    # the recency-weight builder (FTE._build_sample_weights when
    # ``weight_schemas`` includes "recency") produces ~0 weights for that
    # block -- at least one MLP training batch then sees
    # weight_sum < 1e-12 and the once-per-fit WARN at
    # _flat_torch_module.py:233-256 fires. Gate at the canon layer
    # collapses this back to False outside ('mlp' in models AND
    # weight_schemas != ("uniform",)) so non-recency / non-MLP combos
    # don't accumulate phantom variation. The injection unconditionally
    # adds a ``ts`` column when active so the recency builder has
    # something to consume (existing ``with_datetime_col`` axis is
    # orthogonal and may also emit ``ts`` -- the active branch wins).
    _inject_zero_wb = (
        combo.mlp_inject_zero_sample_weight_batch_cfg
        and "mlp" in combo.models
        and combo.weight_schemas != ("uniform",)
    )
    # inject_label_leak: a feature exactly equal to target + tiny noise.
    # A correctly-functioning suite trains on this happily; the val
    # metric must land near-perfect. Deliberately NOT asserted here —
    # the adversarial axis catches pipeline corruption that SILENTLY
    # suppresses the leak (e.g. label-column reordering, caller-frame
    # mutation); any crash is the real bug we're probing for.
    # For multilabel (target is (N, K)): leak label 0 specifically.
    if combo.inject_label_leak:
        # 2-D targets (multilabel_classification, multi_target_regression,
        # quantile_regression) can't be broadcast as a single feature; leak
        # the first column / target only -- still catastrophic for a model
        # that silently mis-uses the first target dimension.
        _target_arr = np.asarray(target)
        if _target_arr.ndim >= 2:
            leak_src = _target_arr[:, 0]
        else:
            leak_src = _target_arr
        leak_col = leak_src.astype("float32") + (rng.standard_normal(n) * 0.01).astype("float32")
        extra_num_cols["num_leak"] = leak_col
    # R3-1 inject_test_drift: perturb the last 15% of rows so test/val
    # slices see a distribution mismatch. Real prod bug surface (unseen
    # categories, out-of-range values, feature shift) — catches pipelines
    # that memoise train stats without guarding against unseen state.
    if combo.inject_test_drift and n >= 20:
        tail = max(3, int(n * 0.15))
        tail_slice = slice(n - tail, n)
        if combo.inject_test_drift == "out_of_range_numeric":
            # scale last 15% of num_0 by 100× (values outside train range)
            num_cols["num_0"][tail_slice] = num_cols["num_0"][tail_slice] * 100.0
        elif combo.inject_test_drift == "shifted_distribution":
            # shift num_0 by +5 sigma (covariate shift)
            num_cols["num_0"][tail_slice] = num_cols["num_0"][tail_slice] + 5.0
        elif combo.inject_test_drift == "unseen_category" and combo.cat_feature_count > 0:
            # overwrite the FIRST cat column's tail values with a string
            # that didn't exist in the training portion.
            # (cat_cols[f"cat_0"] is already populated; mutate in place.)
            cat_cols["cat_0"] = list(cat_cols["cat_0"])
            unseen = "ZZZ_UNSEEN"
            for j in range(n - tail, n):
                cat_cols["cat_0"][j] = unseen

    # grouped_delta / lagged_diff MRMR-FE kinds need a group key and a sortable order column respectively; emit them (plus a matching source
    # column carrying genuine signal) only for the combos that select those kinds, so every other combo's frame and its canonical key is
    # unchanged. The source column is engineered so the FE transform RECOVERS a target-predictive signal the raw column hides -- otherwise the
    # MRMR Tier-1 local-MI gate (correctly) drops the engineered column as a no-uplift near-duplicate and the kind, while it runs, leaves nothing
    # behind. ``MRMR_FE_GROUP_VAL_COL`` = per-group target offset + within-group target signal: the raw column is dominated by the group offset,
    # but ``x - mean(x|group)`` isolates the within-group signal (decorrelated from the raw column, still target-predictive). ``MRMR_FE_LAG_VAL_COL``
    # = slow random-walk drift + a target step: the raw column is dominated by the drift, but ``x - x.shift(p)`` (after the order-column sort)
    # differences out the drift and leaves the target step.
    _kind = combo.mrmr_fe_ratio_delta_diff_cfg if combo.use_mrmr_fs else "off"
    if _kind in ("grouped_delta", "lagged_diff"):
        _t = np.asarray(target)
        _t1d = _t[:, 0] if _t.ndim >= 2 else _t
        _t1d = _t1d.astype("float64")
        _ts = (_t1d - _t1d.mean()) / (_t1d.std() + 1e-9)  # standardised 1-D target signal
    if _kind == "grouped_delta":
        _grp = (np.arange(n) % 8).astype("float32")
        extra_num_cols[MRMR_FE_GROUP_COL] = _grp
        # Per-group offset uncorrelated with target (decorrelates the RAW column) + within-group signal that IS the target -> grouped_delta keeps
        # it. The target coupling is strong (and noise small) so the recovered ``x - mean(x|group)`` column clears the Tier-1 local-MI floor, which
        # is raised by the high-MI default-FE pool already in X by the time this kind runs -- a weak signal is (correctly) gated out.
        _grp_offset = (rng.standard_normal(8) * 5.0)[(np.arange(n) % 8)]
        extra_num_cols[MRMR_FE_GROUP_VAL_COL] = (_grp_offset + 3.0 * _ts + rng.standard_normal(n) * 0.05).astype("float32")
    elif _kind == "lagged_diff":
        extra_num_cols[MRMR_FE_ORDER_COL] = np.arange(n, dtype="float32")
        # Slow random-walk drift (dominates the raw column) + a strong per-row target step that the first difference isolates; coupling chosen so
        # the recovered ``x - x.shift(p)`` column clears the local-MI floor raised by the default-FE pool (see grouped_delta note above).
        _drift = np.cumsum(rng.standard_normal(n) * 0.5)
        extra_num_cols[MRMR_FE_LAG_VAL_COL] = (_drift + 3.0 * _ts + rng.standard_normal(n) * 0.05).astype("float32")

    # 2026-05-12 Wave 30: when no model in the combo supports polars
    # natively (CB/XGB/HGB), build a pandas frame regardless of the
    # axis-sampled input_type. CatBoostEncoder and other sklearn-native
    # transformers reject polars DataFrames with ``ValueError: Unexpected
    # input type: <class 'polars...'>``. Canonicalised in canonical_key
    # so dedup collapses these combos correctly.
    _any_polars_native = any(m in combo.models for m in ("cb", "xgb", "hgb", "mlp", "lstm", "gru", "transformer"))
    _build_input_type = combo.input_type if _any_polars_native else "pandas"

    if _build_input_type == "pandas":
        import pandas as pd
        data = {**num_cols, **extra_num_cols}
        for name, values in cat_cols.items():
            data[name] = pd.Categorical(values)
        for name, values in text_cols.items():
            # pandas object dtype with n_unique > threshold triggers text
            # auto-promotion inside ``_auto_detect_feature_types``.
            data[name] = pd.array(values, dtype="string")
        # with_datetime_col (#11): add a pandas datetime64 column.
        # 2026-05-31 audit-pass-8 #9: when the zero-weight-batch axis is on,
        # force a ts column AND splat the last 20% of rows to a far-past
        # timestamp (year 1900) so recency-weight schemes collapse that
        # contiguous block to ~0 weights.
        if combo.with_datetime_col or _inject_zero_wb:
            ts = pd.date_range("2026-01-01", periods=n, freq="h")
            if _inject_zero_wb and n >= 5:
                tail = max(1, int(n * 0.2))
                far_past = pd.Timestamp("1900-01-01")
                ts = ts.to_series().reset_index(drop=True)
                ts.iloc[n - tail :] = far_past
                ts = pd.DatetimeIndex(ts)
            data["ts"] = ts
        # Multilabel target: 2-D (N, K) stored as an object column of list cells.
        # SimpleFeaturesAndTargetsExtractor unpacks back to (N, K) ndarray at
        # consumption time.
        if combo.target_type == "multilabel_classification":
            data[target_col] = pd.array([row.tolist() for row in target], dtype=object)
        elif combo.target_type == "multi_target_regression" and target.ndim == 2:
            # F-24 audit-pass-9 #8: (N, K) continuous targets stored as a
            # single object column of list cells (mirrors the multilabel
            # pattern). The estimator branch at training/neural/base.py:548
            # auto-detects (N, K>=2) shape and routes through the multi-
            # target regression head.
            # iter633: only when every model in combo natively handles 2-D
            # MTR (see :6788 branch); non-native combos receive a 1-D
            # target so ``target.ndim == 2`` selects the native path.
            data[target_col] = pd.array([row.tolist() for row in target], dtype=object)
        else:
            data[target_col] = target
        return pd.DataFrame(data), target_col, cat_names

    import polars as pl
    data_pl: dict[str, Any] = {**num_cols, **extra_num_cols}
    for name, values in cat_cols.items():
        if _build_input_type == "polars_enum":
            pool_values = [v for v in values if v is not None]
            enum_type = pl.Enum(sorted(set(pool_values)))
            data_pl[name] = pl.Series(values).cast(enum_type)
        elif _build_input_type == "polars_nullable":
            data_pl[name] = pl.Series(values).cast(pl.Categorical)
        else:  # polars_utf8
            data_pl[name] = pl.Series(values, dtype=pl.Utf8)
    for name, values in text_cols.items():
        # Text columns are always pl.Utf8 — the auto-detector routes them
        # to text_features via cardinality threshold (hundreds of unique
        # 3-word sentences on 300+ rows) regardless of combo.input_type.
        data_pl[name] = pl.Series(values, dtype=pl.Utf8)
    if want_embedding:
        emb_dim = 4
        for i in range(combo.embedding_col_count):
            vecs = rng.standard_normal((n, emb_dim)).astype("float32")
            data_pl[f"emb_{i}"] = pl.Series(
                [vecs[j].tolist() for j in range(n)],
                dtype=pl.List(pl.Float32),
            )
    # with_datetime_col (#11): polars datetime64 column.
    # 2026-05-31 audit-pass-8 #9: when the zero-weight-batch axis is on,
    # force a ts column AND splat the last 20% of rows to a far-past
    # timestamp (year 1900) so recency-weight schemes collapse that
    # contiguous block to ~0 weights.
    if combo.with_datetime_col or _inject_zero_wb:
        import datetime as _dt
        start = _dt.datetime(2026, 1, 1)
        far_past = _dt.datetime(1900, 1, 1)
        ts_values = [start + _dt.timedelta(hours=i) for i in range(n)]
        if _inject_zero_wb and n >= 5:
            tail = max(1, int(n * 0.2))
            for i in range(n - tail, n):
                ts_values[i] = far_past
        data_pl["ts"] = pl.Series(ts_values, dtype=pl.Datetime)
    # Multilabel target: 2-D (N, K) stored as pl.List(pl.Int8) column.
    # SimpleFeaturesAndTargetsExtractor unpacks back to (N, K) ndarray.
    if combo.target_type == "multilabel_classification":
        data_pl[target_col] = pl.Series(
            [row.tolist() for row in target],
            dtype=pl.List(pl.Int8),
        )
    elif combo.target_type == "multi_target_regression" and target.ndim == 2:
        # F-24 audit-pass-9 #8: (N, K) continuous targets stored as a
        # pl.List(pl.Float32) column (mirrors multilabel polars wiring).
        # iter633: only when every model in combo natively handles 2-D MTR;
        # non-native combos receive a 1-D target so the else-branch runs.
        data_pl[target_col] = pl.Series(
            [row.tolist() for row in target],
            dtype=pl.List(pl.Float32),
        )
    else:
        data_pl[target_col] = target
    return pl.DataFrame(data_pl), target_col, cat_names


# 2026-05-31 audit-pass-8 #6 verification-probe TODO.
#
# Architectural default-flip dc9723ea (2026-05-30): binary classification on
# PytorchLightningClassifier now silently uses the 1-output sigmoid + BCE
# head whenever ``len(self.classes_) == 2`` (training/neural/base.py:438-443).
# There is no opt-in flag and no back-compat shim.
#
# Today no convenient builder-side hook exists to assert
# ``model._binary_sigmoid_head is True`` after fit -- the fuzz harness does
# not retain the fitted MLP estimator objects in the path that test_iter613
# exercises (the suite builds them inside _phase_train_one_target and
# discards once preds + metrics are stamped). Verification therefore stays
# a TODO until either (a) the suite exposes a per-model fit-hook the fuzz
# harness can opt into, or (b) a dedicated sensor in
# tests/training/test_fuzz_regression_sensors.py spins up a single binary
# MLP fit and inspects ``estimator._binary_sigmoid_head`` directly.
#
# Failure mode being guarded against: a downstream wrapper (calibration
# wrapper, multilabel adapter, recurrent estimator subclass) silently
# overrides the gate to ``False`` and reverts to the legacy 2-output
# softmax head -- pickled-model state-dict incompatibility + (N, 2)
# prediction-shape regressions follow without any other warning.
#
# Until then the gate is exercised indirectly: every binary-classification
# fuzz combo flows through the new branch and a regression at
# training/neural/base.py:438-443 surfaces as either a shape mismatch
# downstream (predict_proba returns (N, 1) instead of (N, 2)) or a loss
# mismatch (CE vs BCE).
