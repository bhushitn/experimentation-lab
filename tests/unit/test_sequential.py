"""Boundary computation validated against published Lan-DeMets values.

Reference: ldbounds R package vignette, "An R Package for Group Sequential
Boundaries Using Alpha Spending Functions",
https://cran.r-project.org/web/packages/ldbounds/vignettes/ldbounds.pdf
(five equal looks, two-sided alpha = 0.05, O'Brien-Fleming-type spending).
"""

import numpy as np
import pytest

from lab.stats.sequential import obf_boundaries, obf_spending

LDBOUNDS_FIVE_LOOKS = [4.8769, 3.3569, 2.6803, 2.2898, 2.0310]


class TestObfBoundaries:
    def test_matches_ldbounds_published_values(self) -> None:
        t = np.linspace(0.2, 1.0, 5)
        ours = obf_boundaries(t, alpha=0.05)
        assert ours == pytest.approx(LDBOUNDS_FIVE_LOOKS, abs=2e-3)

    def test_single_look_reduces_to_fixed_horizon(self) -> None:
        ours = obf_boundaries([1.0], alpha=0.05)
        assert ours[0] == pytest.approx(1.9600, abs=1e-3)

    def test_spending_reaches_alpha_at_one(self) -> None:
        assert obf_spending([1.0], alpha=0.05)[0] == pytest.approx(0.05)

    def test_boundaries_decrease(self) -> None:
        b = obf_boundaries(np.linspace(1 / 14, 1.0, 14), alpha=0.05)
        assert (np.diff(b) < 0).all()

    def test_rejects_bad_fractions(self) -> None:
        with pytest.raises(ValueError):
            obf_boundaries([0.5, 0.4, 1.0])
