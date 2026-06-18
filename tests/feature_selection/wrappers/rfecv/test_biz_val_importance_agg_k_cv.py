"""biz_value: ``importance_agg_k_cv`` measurably separates a steady-gain feature from an unstable decoy.

``aggregate_tree`` scores a feature as ``mean / (1 + k_cv * cv)`` with ``cv = std / (|mean| + eps)``. The knob
``importance_agg_k_cv`` controls how hard fold-to-fold instability is penalised. The win it buys is param-isolated:
at ``k_cv=0`` a noisy decoy with the SAME mean gain as a steady true feature is scored identically (ranking ties /
mis-ranks); turning the knob up demotes the unstable decoy below the steady feature, and the separation grows
monotonically with the knob value.
"""
from __future__ import annotations

from mlframe.feature_selection.wrappers._helpers_importance_agg import aggregate_tree


def _steady_vs_noisy_fi():
    # steady true feature: gain 1.0 every fold. noisy decoy: identical mean 1.0 but swings 0<->2 fold to fold.
    return {f"r{k}": {"steady": 1.0, "noisy": (2.0 if k % 2 == 0 else 0.0)} for k in range(6)}


def test_biz_val_rfecv_importance_agg_k_cv_zero_cannot_separate_equal_mean_features():
    fi = _steady_vs_noisy_fi()
    s = aggregate_tree(fi, k_cv=0.0)
    ratio = s["steady"] / max(s["noisy"], 1e-9)
    assert abs(ratio - 1.0) < 1e-6, f"k_cv=0 must NOT separate equal-mean features; ratio {ratio:.4f}"


def test_biz_val_rfecv_importance_agg_k_cv_positive_demotes_unstable_decoy():
    fi = _steady_vs_noisy_fi()
    s = aggregate_tree(fi, k_cv=1.0)
    ratio = s["steady"] / max(s["noisy"], 1e-9)
    # Measured: ratio 2.0 at k_cv=1.0 (steady 1.0 vs noisy 0.5). Floor 1.8 absorbs noise; k_cv regression trips it.
    assert ratio >= 1.8, f"k_cv=1.0 must rank steady >> noisy (ratio {ratio:.2f}); measured ~2.0"


def test_biz_val_rfecv_importance_agg_k_cv_separation_grows_monotonically():
    fi = _steady_vs_noisy_fi()
    ratios = []
    for k_cv in (0.0, 0.5, 1.0, 2.0):
        s = aggregate_tree(fi, k_cv=k_cv)
        ratios.append(s["steady"] / max(s["noisy"], 1e-9))
    # Measured: 1.0 -> 1.5 -> 2.0 -> 3.0. A higher knob must yield strictly stronger demotion of the unstable decoy.
    assert all(b > a + 1e-6 for a, b in zip(ratios, ratios[1:])), f"separation must grow with k_cv; got {ratios}"
    assert ratios[-1] >= 2.7, f"strong k_cv=2.0 must give ratio >=2.7; measured ~3.0, got {ratios[-1]:.2f}"
