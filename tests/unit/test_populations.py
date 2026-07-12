"""Populations: reproducibility, parameter recovery, and exact ground-truth ATEs."""

import numpy as np
import pandas as pd
import pytest

from lab.assignment import assign_clusters, assign_users
from lab.populations import (
    binary_users,
    continuous_users,
    draw_binary,
    draw_continuous,
    draw_zero_inflated,
    true_binary_ate,
    true_zero_inflated_ate,
    zero_inflated_users,
)

N = 200_000


class TestReproducibility:
    def test_same_seed_same_population(self) -> None:
        a = continuous_users(1000, seed=3)
        b = continuous_users(1000, seed=3)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_different_population(self) -> None:
        a = continuous_users(1000, seed=3)
        b = continuous_users(1000, seed=4)
        assert not a["user_mean"].equals(b["user_mean"])


class TestContinuous:
    def test_moments(self) -> None:
        users = continuous_users(N, mean=10.0, between_sd=2.0, within_sd=5.0, seed=0)
        assert users["user_mean"].mean() == pytest.approx(10.0, abs=0.05)
        assert users["user_mean"].std() == pytest.approx(2.0, abs=0.05)

    def test_pre_post_correlation(self) -> None:
        between_sd, within_sd = 2.0, 5.0
        users = continuous_users(N, between_sd=between_sd, within_sd=within_sd, seed=0)
        treated = np.zeros(N, dtype=bool)
        y = draw_continuous(users, treated, effect=0.0, within_sd=within_sd, seed=1)
        expected_rho = between_sd**2 / (between_sd**2 + within_sd**2)
        rho = np.corrcoef(users["pre_metric"], y)[0, 1]
        assert rho == pytest.approx(expected_rho, abs=0.01)

    def test_empirical_ate_matches_truth(self) -> None:
        users = continuous_users(N, seed=0)
        treated = assign_users(users, seed=1)
        y = draw_continuous(users, treated, effect=0.5, seed=2)
        empirical = y[treated].mean() - y[~treated].mean()
        assert empirical == pytest.approx(0.5, abs=0.06)


class TestBinary:
    def test_rates_valid_and_centered(self) -> None:
        users = binary_users(N, base_rate=0.1, seed=0)
        assert users["user_rate"].between(0, 1).all()
        assert users["user_rate"].mean() == pytest.approx(0.1, abs=0.005)

    def test_empirical_ate_matches_truth(self) -> None:
        users = binary_users(N, base_rate=0.1, seed=0)
        treated = assign_users(users, seed=1)
        y = draw_binary(users, treated, lift=0.02, seed=2)
        empirical = y[treated].mean() - y[~treated].mean()
        assert empirical == pytest.approx(true_binary_ate(users, 0.02), abs=0.005)


class TestZeroInflated:
    def test_zero_fraction(self) -> None:
        users = zero_inflated_users(N, active_rate=0.3, seed=0)
        treated = np.zeros(N, dtype=bool)
        y = draw_zero_inflated(users, treated, spend_multiplier=1.0, seed=1)
        assert (y == 0).mean() == pytest.approx(0.7, abs=0.01)

    def test_empirical_ate_matches_truth(self) -> None:
        users = zero_inflated_users(N, active_rate=0.3, seed=0)
        treated = assign_users(users, seed=1)
        y = draw_zero_inflated(users, treated, spend_multiplier=1.10, seed=2)
        truth = true_zero_inflated_ate(users, 1.10)
        empirical = y[treated].mean() - y[~treated].mean()
        assert empirical == pytest.approx(truth, abs=0.05)


class TestAssignment:
    def test_user_assignment_exact_split(self) -> None:
        users = continuous_users(10_001, seed=0)
        treated = assign_users(users, treat_fraction=0.5, seed=1)
        assert treated.sum() == round(10_001 * 0.5)

    def test_cluster_assignment_keeps_clusters_intact(self) -> None:
        users = continuous_users(10_000, seed=0)
        users["cluster"] = np.arange(10_000) % 100
        treated = assign_clusters(users, cluster_col="cluster", seed=1)
        per_cluster = pd.Series(treated).groupby(users["cluster"]).nunique()
        assert (per_cluster == 1).all()
        assert pd.Series(treated).groupby(users["cluster"]).first().sum() == 50
