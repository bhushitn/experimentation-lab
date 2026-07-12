"""Synthetic populations and outcome generation with known ground truth."""

from lab.populations.outcomes import (
    draw_binary,
    draw_continuous,
    draw_zero_inflated,
    true_binary_ate,
    true_continuous_ate,
    true_zero_inflated_ate,
)
from lab.populations.users import binary_users, continuous_users, zero_inflated_users

__all__ = [
    "binary_users",
    "continuous_users",
    "zero_inflated_users",
    "draw_binary",
    "draw_continuous",
    "draw_zero_inflated",
    "true_binary_ate",
    "true_continuous_ate",
    "true_zero_inflated_ate",
]
