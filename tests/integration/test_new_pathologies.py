"""SRM, switchback, and quasi-experiment numbers verified against ground truth."""

import numpy as np
import pytest
from scipy import stats as sps

from lab.assignment import assign_users
from lab.pathologies import (
    simulate_quasi,
    simulate_srm,
    simulate_switchback,
    srm_check,
    srm_detection_power,
)
from lab.populations import continuous_users


class TestSrm:
    def test_check_matches_scipy_chisquare(self) -> None:
        for nt, nc in [(50_100, 49_900), (10_000, 10_000), (52_000, 48_000)]:
            ours = srm_check(nt, nc)
            ref = sps.chisquare([nt, nc], f_exp=[(nt + nc) / 2] * 2).pvalue
            assert ours == pytest.approx(float(ref))

    def test_bug_fires_alarm_and_biases_estimate(self) -> None:
        users = continuous_users(200_000, seed=30)
        treated = assign_users(users, seed=31)
        res = simulate_srm(users, treated, effect=0.5, drop_fraction=0.02, seed=32)
        # The alarm fires well below the conventional 0.001 threshold.
        assert res.srm_p_value < 1e-4
        # The estimate matches its exact expectation and is biased upward.
        assert res.estimates["naive"] == pytest.approx(res.estimates["expected_naive"], abs=0.05)
        assert res.estimates["expected_naive"] > res.true_effect + 0.01

    def test_clean_experiment_does_not_alarm(self) -> None:
        # False positive rate of the check at alpha=0.001 is ~0.001.
        fires = srm_detection_power(n_per_arm=100_000, drop_fraction=0.0, n_sims=2000)
        assert fires < 0.01

    def test_detection_power_at_scale(self) -> None:
        # A 2% one-arm loss is essentially always caught at 100k/arm.
        assert srm_detection_power(n_per_arm=100_000, drop_fraction=0.02) > 0.99
        # A 1% loss slips under the strict alpha at 100k/arm...
        assert srm_detection_power(n_per_arm=100_000, drop_fraction=0.01) < 0.5
        # ...and is caught essentially always at 1M/arm (sqrt-of-scale law).
        assert srm_detection_power(n_per_arm=1_000_000, drop_fraction=0.01) > 0.99


class TestSwitchback:
    def test_estimates_match_exact_expectations(self) -> None:
        res = simulate_switchback(n_sims=400, seed=40)
        for _, row in res.by_window.iterrows():
            assert row["naive_estimate"] == pytest.approx(
                row["naive_expected"], abs=3 * row["naive_se"] / np.sqrt(400)
            )

    def test_short_windows_attenuate_long_windows_recover(self) -> None:
        res = simulate_switchback(windows=(1, 24), delta=1.0, gamma=0.4, n_sims=400, seed=41)
        by = res.by_window.set_index("window_hours")
        # One-hour windows: prev arm is ~independent, contrast keeps only
        # (1 - gamma) of the effect plus gamma * (1/2 - 1/2)... measured:
        assert by.loc[1, "naive_estimate"] < 0.75 * res.true_effect
        # Day-long windows sit near the truth even naively.
        assert by.loc[24, "naive_estimate"] == pytest.approx(res.true_effect, abs=0.05)

    def test_burn_in_recovers_at_every_window(self) -> None:
        res = simulate_switchback(n_sims=400, seed=42)
        for _, row in res.by_window.iterrows():
            band = 4 * row["burn_in_se"] / np.sqrt(400) + 0.02
            assert row["burn_in_estimate"] == pytest.approx(res.true_effect, abs=band)

    def test_variance_grows_with_window_length(self) -> None:
        res = simulate_switchback(windows=(2, 24), n_sims=400, seed=43)
        by = res.by_window.set_index("window_hours")
        assert by.loc[24, "burn_in_se"] > by.loc[2, "burn_in_se"]


class TestQuasi:
    def test_estimators_match_exact_expectations(self) -> None:
        res = simulate_quasi(seed=50)
        for name in res.estimates.index:
            assert res.estimates[name] == pytest.approx(res.expected[name], abs=0.15)

    def test_did_recovers_under_parallel_trends(self) -> None:
        res = simulate_quasi(differential_trend=0.0, seed=51)
        assert res.expected["difference_in_differences"] == pytest.approx(res.true_effect)
        assert res.estimates["difference_in_differences"] == pytest.approx(
            res.true_effect, abs=0.15
        )
        # And the two naive estimators are far off, by construction.
        assert res.expected["post_vs_pre"] > res.true_effect + 3.0
        assert res.expected["treated_vs_control"] > res.true_effect + 3.0

    def test_did_bias_equals_differential_trend_and_placebo_catches_it(self) -> None:
        g = 0.3
        res = simulate_quasi(differential_trend=g, n_pre=8, n_post=8, n_cities=200, seed=52)
        dt = 8.0  # mean post period minus mean pre period with 8 + 8
        assert res.expected["difference_in_differences"] == pytest.approx(
            res.true_effect + g * dt
        )
        # The placebo DiD on pre-period data is far from zero.
        assert res.placebo_did == pytest.approx(g * 4.0, abs=0.25)

    def test_placebo_near_zero_under_parallel_trends(self) -> None:
        res = simulate_quasi(differential_trend=0.0, seed=53)
        assert abs(res.placebo_did) < 0.3
