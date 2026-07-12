"""Every pathology's headline numbers verified against ground truth.

Structure per pathology: the naive analysis shows the quantified failure,
the corrected analysis restores the guarantee, and where a closed form
exists the simulation matches it within a Monte Carlo band.
"""

import numpy as np
import pytest

from lab.assignment import assign_clusters, assign_users
from lab.pathologies import (
    simulate_contamination,
    simulate_daily_peeking,
    simulate_interference,
    simulate_many_metrics,
    simulate_novelty,
    simulate_subgroup_fishing,
)
from lab.populations import continuous_users
from lab.populations.graphs import sbm_graph


def binomial_band(rate: float, n: int, sigmas: float = 3.0) -> float:
    return sigmas * np.sqrt(rate * (1.0 - rate) / n)


class TestPeeking:
    def test_naive_daily_peeking_inflates_fpr(self) -> None:
        res = simulate_daily_peeking(n_sims=4000, n_days=14, effect=0.0, seed=1)
        rates = res.rejection_rates()
        # Fixed horizon is calibrated.
        assert abs(rates["fixed_horizon"] - 0.05) < binomial_band(0.05, 4000)
        # Naive daily peeking at 14 looks: literature puts this in the
        # low-to-mid twenties percent; require at least 3x nominal.
        assert rates["naive_peeking"] > 0.15
        # Sequential boundaries restore the nominal rate.
        assert abs(rates["sequential"] - 0.05) < binomial_band(0.05, 4000)

    def test_sequential_keeps_power_and_stops_early(self) -> None:
        res = simulate_daily_peeking(n_sims=2000, n_days=14, effect=0.7, seed=2)
        rates = res.rejection_rates()
        assert rates["sequential"] > 0.8
        # A real effect should usually be caught before the last day.
        assert res.decisions["sequential_stop_day"].median() < 14


class TestMultipleComparisons:
    def test_fwer_inflation_matches_theory(self) -> None:
        res = simulate_many_metrics(n_sims=4000, n_metrics=20, seed=3)
        expected = 1.0 - 0.95**20  # independent metrics
        naive = res.summary.loc["naive", "fwer"]
        assert abs(naive - expected) < binomial_band(expected, 4000)

    def test_corrections_control_their_error_rates(self) -> None:
        res = simulate_many_metrics(n_sims=4000, n_metrics=20, seed=4)
        assert res.summary.loc["bonferroni", "fwer"] <= 0.05 + binomial_band(0.05, 4000)
        assert res.summary.loc["benjamini_hochberg", "fdr"] <= 0.05 + binomial_band(0.05, 4000)

    def test_power_cost_ordering(self) -> None:
        res = simulate_many_metrics(
            n_sims=2000, n_metrics=20, n_true_effects=5, effect=0.4, seed=5
        )
        p = res.summary["power"]
        assert p["naive"] > p["benjamini_hochberg"] > p["bonferroni"]
        assert res.summary.loc["benjamini_hochberg", "fdr"] <= 0.05 + binomial_band(0.05, 2000)


class TestContamination:
    def test_estimators_match_exact_expectations(self) -> None:
        users = continuous_users(100_000, seed=6)
        treated = assign_users(users, seed=7)
        # Small within-user noise so the Monte Carlo error is well inside
        # the tolerance and the exact-expectation identity does the work.
        res = simulate_contamination(users, treated, effect=0.5, within_sd=2.0, seed=8)
        for name in res.estimates.index:
            assert res.estimates[name] == pytest.approx(res.expected[name], abs=0.07)

    def test_itt_attenuated_pp_biased_iv_recovers(self) -> None:
        users = continuous_users(100_000, seed=6)
        treated = assign_users(users, seed=7)
        res = simulate_contamination(
            users,
            treated,
            effect=0.5,
            treat_noncompliance=0.20,
            control_exposure=0.10,
            seed=8,
        )
        # ITT attenuated to ~70% of the true effect.
        assert res.expected["intent_to_treat"] == pytest.approx(0.7 * 0.5, abs=0.02)
        # Per-protocol biased upward by selection (here, substantially).
        assert res.expected["per_protocol"] > res.true_effect + 0.1
        # IV/Wald recovers the true effect.
        assert res.estimates["iv_wald"] == pytest.approx(res.true_effect, abs=0.1)


class TestInterference:
    def test_naive_misses_spillover_cluster_recovers(self) -> None:
        n = 20_000
        users = continuous_users(n, seed=9)
        adjacency, clusters = sbm_graph(n, n_clusters=100, seed=10)
        users["cluster"] = clusters
        user_treated = assign_users(users, seed=11)
        cluster_treated = assign_clusters(users, cluster_col="cluster", seed=11)

        res = simulate_interference(
            users, adjacency, user_treated, cluster_treated, direct=0.5, spillover=0.3, seed=12
        )
        gte = res.global_treatment_effect

        # Estimates match the closed-form expectation for each design.
        for name in res.estimates.index:
            assert res.estimates[name] == pytest.approx(res.expected[name], abs=0.12)
        # User-level randomization: exposure contrast ~0, misses ~all spillover.
        assert abs(res.exposure_contrast["user_randomized"]) < 0.05
        assert abs(res.expected["user_randomized"] - 0.5) < 0.05
        # Cluster randomization: contrast near 1, recovers most of the GTE.
        assert res.exposure_contrast["cluster_randomized"] > 0.85
        assert abs(res.expected["cluster_randomized"] - gte) < 0.06


class TestSubgroups:
    def test_fishing_discovery_rate_matches_theory(self) -> None:
        res = simulate_subgroup_fishing(n_sims=4000, n_segments=20, seed=13)
        expected = 1.0 - 0.95**20
        assert abs(res.summary["fishing_any_discovery"] - expected) < binomial_band(
            expected, 4000
        )
        assert res.summary["fishing_false_discoveries_per_experiment"] == pytest.approx(
            20 * 0.05, abs=0.1
        )

    def test_preregistration_and_bh_stay_controlled(self) -> None:
        res = simulate_subgroup_fishing(n_sims=4000, n_segments=20, seed=14)
        assert abs(res.summary["preregistered_primary_fpr"] - 0.05) < binomial_band(0.05, 4000)
        assert res.summary["bh_corrected_any_discovery"] <= 0.05 + binomial_band(0.05, 4000)


class TestNovelty:
    def test_window_estimates_match_analytic_path_means(self) -> None:
        res = simulate_novelty(seed=15)
        for name in res.estimates.index:
            assert res.estimates[name] == pytest.approx(res.expected[name], abs=0.03)

    def test_first_week_overstates_long_run(self) -> None:
        res = simulate_novelty(seed=15)
        # Analytic: first week averages the novelty spike.
        assert res.expected["first_week"] > 2.5 * res.long_run_effect
        # Post-burn-in window sits close to the long-run truth.
        assert res.expected["post_burn_in"] == pytest.approx(res.long_run_effect, rel=0.10)
