"""After the clean-form demotion the prewarp MI read by the later uplift acceptance is capped below the clean MI, so the demoted form cannot be re-promoted."""
from types import SimpleNamespace

import numpy as np

import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_score_step4 as step4

PW = "__prewarp__"


def test_demoted_prewarp_winner_cannot_pass_the_uplift_ratio_again(monkeypatch):
    """prewarp MI 0.105 vs clean 0.100 with |corr| equal: demoted; the uplift gate (ratio 1.02) would still have accepted 0.105 >= 0.100 * 1.02."""
    monkeypatch.setattr(step4, "_config_corr", lambda cfg, *a, **k: 0.5)
    pw_cfg = (((0, PW), (1, "identity")), "mul", 0)
    clean_cfg = (((0, "identity"), (1, "identity")), "mul", 1)
    st = SimpleNamespace(
        _local_times={}, best_config=pw_cfg, best_nonprewarp_config=clean_cfg, best_nonprewarp_mi=0.100, best_mi=0.105, best_prewarp_mi=0.105,
        var_pairs_perf={pw_cfg: 0.105, clean_cfg: 0.100}, messages=[],
    )
    step4._score_one_pair_step4_iteration_serialization_point_part1(
        st, {}, 0, (0, 1), np.ones(10), PW, 0.85, None, False, None, None, None, {}, {}, None, 0.2,
    )
    assert st.best_config is clean_cfg
    assert st.best_prewarp_mi < 0.100 * 1.02
