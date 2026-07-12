"""Subgroup fishing: post-hoc segment scans manufacture discoveries.

Simulates experiments over a population split into segments (countries,
platforms, tenure bands). The fishing analyst tests every segment at
alpha and reports whichever comes up significant; with 20 null segments
that finds "a significant subgroup" in roughly 1 - 0.95**20 = 64% of
experiments. The pre-registered analyst names one primary segment in
advance and corrects the rest with Benjamini-Hochberg.

Reuses the multiple-comparisons machinery: a segment scan IS a multiple
comparisons problem wearing a trench coat, and the module exists to make
that identity explicit with numbers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from lab.stats.corrections import benjamini_hochberg


@dataclass(frozen=True)
class SubgroupResult:
    """Discovery rates for fishing vs pre-registration.

    Attributes
    ----------
    summary : pd.Series
        fishing_any_discovery: share of experiments where at least one
        null segment tested significant.
        fishing_false_discoveries_per_experiment: mean count of null
        segments flagged.
        preregistered_primary_fpr: rejection rate on the pre-named
        segment (a single test, stays at alpha).
        bh_corrected_any_discovery: share of experiments with any null
        segment surviving Benjamini-Hochberg.
    p_values : np.ndarray
        (n_sims, n_segments) per-segment p-values.
    """

    summary: pd.Series
    p_values: np.ndarray


def simulate_subgroup_fishing(
    *,
    n_sims: int = 4000,
    n_segments: int = 20,
    n_per_segment_per_arm: int = 500,
    effect_segments: int = 0,
    effect: float = 0.0,
    sd: float = 5.4,
    alpha: float = 0.05,
    seed: int = 0,
) -> SubgroupResult:
    """Simulate per-segment tests; the first effect_segments carry `effect`.

    Per-segment z-statistics are exact for the difference in segment
    means at these sample sizes. Segments are independent (users belong
    to exactly one).
    """
    rng = np.random.default_rng(seed)
    is_null = np.arange(n_segments) >= effect_segments
    shift = np.where(is_null, 0.0, effect / (sd * np.sqrt(2.0 / n_per_segment_per_arm)))
    z = rng.normal(0.0, 1.0, (n_sims, n_segments)) + shift
    p = 2.0 * sps.norm.sf(np.abs(z))

    null_p = p[:, is_null]
    bh_null_any = np.array(
        [benjamini_hochberg(row, alpha=alpha)[is_null].any() for row in p]
    )
    summary = pd.Series(
        {
            "fishing_any_discovery": float((null_p < alpha).any(axis=1).mean()),
            "fishing_false_discoveries_per_experiment": float(
                (null_p < alpha).sum(axis=1).mean()
            ),
            "preregistered_primary_fpr": float((p[:, -1] < alpha).mean()),
            "bh_corrected_any_discovery": float(bh_null_any.mean()),
        }
    )
    return SubgroupResult(summary=summary, p_values=p)
