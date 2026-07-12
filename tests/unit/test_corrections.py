"""Corrections validated against statsmodels multipletests."""

import numpy as np
from statsmodels.stats.multitest import multipletests

from lab.stats.corrections import benjamini_hochberg, bonferroni

RNG = np.random.default_rng(7)


class TestCorrections:
    def test_bonferroni_matches_statsmodels(self) -> None:
        for _ in range(20):
            p = RNG.uniform(0, 0.2, 30)
            ref = multipletests(p, alpha=0.05, method="bonferroni")[0]
            assert (bonferroni(p, alpha=0.05) == ref).all()

    def test_bh_matches_statsmodels(self) -> None:
        for _ in range(20):
            p = RNG.uniform(0, 0.2, 30)
            ref = multipletests(p, alpha=0.05, method="fdr_bh")[0]
            assert (benjamini_hochberg(p, alpha=0.05) == ref).all()

    def test_bh_no_rejections_when_all_large(self) -> None:
        p = np.full(10, 0.9)
        assert not benjamini_hochberg(p).any()
