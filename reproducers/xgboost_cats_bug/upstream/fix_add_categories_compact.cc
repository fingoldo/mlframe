// ============================================================================
// Patch for xgboost::common::AddCategories  (option 1: compact codes)
// ----------------------------------------------------------------------------
// File to edit:  src/common/quantile.cc  (function AddCategories, ~L389-402)
// Header change: src/common/categorical.h (add code_to_rank map on metadata)
// ----------------------------------------------------------------------------
// Motivation
//   Existing AddCategories sizes cut_values by (max_cat + 1). For a
//   polars-style sparse dictionary { 0..87, 2_526_058 } that means a ~9.6 MB
//   buffer for 89 real uniques. On Windows, the oversize allocation inside
//   IterativeDMatrix's batched sketch causes STATUS_ACCESS_VIOLATION (0xC0000005)
//   and silent process death. On Linux the fit completes but wastes memory
//   and indexes garbage bins.
//
// Fix strategy
//   Treat the physical category code as an opaque identifier. Build a dense
//   rank mapping (code -> [0..k-1]) during AddCategories. Cut values store
//   actual physical codes (unchanged model semantics). Every downstream site
//   that used the code as a bin index must now look up code_to_rank[code].
//
// Correctness invariants preserved
//   * cut_values still stores physical codes, so serialised models remain
//     byte-compatible when the input already had dense codes starting at 0.
//   * categories.size() == cut_values.size() == code_to_rank.size().
//   * For any observed code c, code_to_rank.at(c) < categories.size().
//
// Model-level impact on well-formed inputs (dense codes starting at 0)
//   code_to_rank is the identity map; indexing sites compute the same value
//   they did before. No retraining needed.
//
// Memory impact on pathological inputs
//   cut_values[i] count drops from (max_cat + 1) to categories.size().
//   For the bug-trigger case: 2_526_059 floats -> 89 floats.
// ============================================================================

#include "xgboost/data.h"
#include "xgboost/logging.h"
#include "categorical.h"
#include "quantile.h"

#include <algorithm>
#include <set>
#include <unordered_map>

namespace xgboost::common {

// ---------------------------------------------------------------------------
// 1) Metadata change (src/common/categorical.h)
//
// Add a per-feature mapping from physical category code to dense bin rank.
// Stored alongside cut_values_ on HistogramCuts.
//
//    struct HistogramCuts {
//      // existing:
//      HostDeviceVector<float>     cut_values_;
//      HostDeviceVector<uint32_t>  cut_ptrs_;
//      HostDeviceVector<FeatureType> feature_types_;
//
//      // NEW: populated only for categorical features. index = feature_idx.
//      // Empty map for non-categorical features.
//      std::vector<std::unordered_map<bst_cat_t, bst_cat_t>> cat_code_to_rank_;
//
//      // NEW helper:
//      inline bst_cat_t CatRank(bst_feature_t fidx, bst_cat_t code) const {
//        auto const& m = cat_code_to_rank_[fidx];
//        auto it = m.find(code);
//        return it == m.end() ? InvalidCat() : it->second;
//      }
//    };
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 2) Patched AddCategories (replaces current body in src/common/quantile.cc)
//
// The function now:
//   (a) still rejects max_cat >= OutOfRangeCat() via CheckMaxCat;
//   (b) emits exactly categories.size() cut_values entries;
//   (c) builds code_to_rank so downstream histogram / prediction code can
//       translate physical code -> dense bin id.
// ---------------------------------------------------------------------------
inline bst_cat_t AddCategories(std::set<float> const& categories,
                               HistogramCuts* cuts,
                               bst_feature_t fidx) {
  if (categories.empty()) {
    return InvalidCat();
  }

  auto& cut_values = cuts->cut_values_.HostVector();
  auto& code_to_rank = cuts->cat_code_to_rank_[fidx];
  code_to_rank.clear();
  code_to_rank.reserve(categories.size());

  // categories is std::set<float>, already sorted ascending.
  // max_cat is the last element, used only for the guard below.
  float max_cat_f = *categories.crbegin();
  CheckMaxCat(max_cat_f, categories.size());

  // Reserve exactly what we need — one bin per unique code.
  cut_values.reserve(cut_values.size() + categories.size());

  bst_cat_t rank = 0;
  for (float cat_f : categories) {
    bst_cat_t code = AsCat(cat_f);

    // Preserve the physical code in cut_values; downstream code that serialises
    // the model or writes cat splits to a Booster still sees the real code.
    cut_values.push_back(cat_f);

    // Dense rank for bin-array indexing. A polars dictionary with codes
    // { 0, 88, 2526058 } now yields ranks { 0, 1, 2 } and a 3-entry cut
    // array rather than a 2.5 M-entry one.
    code_to_rank.emplace(code, rank++);
  }

  return AsCat(max_cat_f);
}

// ---------------------------------------------------------------------------
// 3) Indexing sites that must switch from "bin = code" to "bin = rank".
//
// Grep hits in xgboost 3.2.0 (file : approximate lines):
//
//   src/common/hist_util.cc            : BinIdx for categorical feature
//   src/common/row_set.h               : partitioning predicate
//   src/tree/hist/histogram.cc         : per-feature bin increment loop
//   src/tree/updater_quantile_hist.cc  : node-split value lookup
//   src/predictor/cpu_predictor.cc     : tree traversal for categorical split
//   src/predictor/gpu_predictor.cu     : same, GPU path
//
// Transformation is mechanical:
//
//   // BEFORE
//   bst_bin_t bin = static_cast<bst_bin_t>(AsCat(value));
//
//   // AFTER
//   bst_cat_t code = AsCat(value);
//   bst_bin_t bin  = static_cast<bst_bin_t>(cuts.CatRank(fidx, code));
//   if (XGBOOST_EXPECT(bin == InvalidCat(), false)) {
//     // category seen at predict time but not in training — existing policy
//     // (treat as missing / route left) applies unchanged.
//     bin = kUnknownCategory;
//   }
//
// The CatRank helper returns InvalidCat() for unknown codes, so the
// out-of-training-set guard already present in cat_container.h continues
// to fire where it used to.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 4) Serialisation compatibility
//
// cut_values_ still stores physical codes for categorical features. Any
// saved model from a previous XGBoost version loads unchanged. On load,
// cat_code_to_rank_ is rebuilt from cut_values_ via:
//
//   for (bst_feature_t f = 0; f < num_features; ++f) {
//     if (feature_types_[f] != FeatureType::kCategorical) continue;
//     auto const& [beg, end] = FeatureBounds(f);  // cut_ptrs_
//     auto& m = cat_code_to_rank_[f];
//     m.clear();
//     m.reserve(end - beg);
//     for (size_t i = beg; i < end; ++i) {
//       m.emplace(AsCat(cut_values_[i]), static_cast<bst_cat_t>(i - beg));
//     }
//   }
//
// No on-disk format change. Old snapshots produced with dense codes
// ({0,1,2,...,k-1}) map to identity ranks; behaviour is identical.
// Old snapshots produced with sparse codes (pre-patch, if any survived the
// Windows crash or were trained on Linux) get the correct compact mapping
// at load time and start behaving correctly.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 5) Unit tests to add (tests/cpp/common/test_quantile.cc)
//
// TEST(AddCategories, DenseCodesStartingAtZero_IdentityRank) {
//   std::set<float> cats{0, 1, 2, 3};
//   HistogramCuts cuts; cuts.cat_code_to_rank_.resize(1);
//   AddCategories(cats, &cuts, /*fidx=*/0);
//   ASSERT_EQ(cuts.cut_values_.HostVector().size(), 4u);
//   for (bst_cat_t c = 0; c < 4; ++c) {
//     ASSERT_EQ(cuts.CatRank(0, c), c);
//   }
// }
//
// TEST(AddCategories, SparseCodes_CompactRank) {
//   std::set<float> cats{0, 88, 2526058};
//   HistogramCuts cuts; cuts.cat_code_to_rank_.resize(1);
//   AddCategories(cats, &cuts, /*fidx=*/0);
//   ASSERT_EQ(cuts.cut_values_.HostVector().size(), 3u);          // NOT 2526059
//   ASSERT_EQ(cuts.CatRank(0, 0),        0);
//   ASSERT_EQ(cuts.CatRank(0, 88),       1);
//   ASSERT_EQ(cuts.CatRank(0, 2526058),  2);
//   ASSERT_EQ(cuts.CatRank(0, 99999),    InvalidCat());            // unknown
// }
//
// TEST(AddCategories, MaxCatAboveThreshold_Rejected) {
//   std::set<float> cats{0, static_cast<float>(OutOfRangeCat())};
//   HistogramCuts cuts; cuts.cat_code_to_rank_.resize(1);
//   EXPECT_THROW(AddCategories(cats, &cuts, 0), dmlc::Error);
// }
// ---------------------------------------------------------------------------

}  // namespace xgboost::common
