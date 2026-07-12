"""Multiple comparisons: many metrics, one experiment.

Simulates experiments that track m metrics at once, a handful of which
have a real effect. Metrics are independent across metrics and
replications (stated simplification; positive dependence makes naive
inflation worse, not better). Per-metric z-statistics are exact for the
difference-in-means test at these sample sizes.

Reports, per policy (naive, Bonferroni, Benjamini-Hochberg):
family-wise error rate, false discovery rate, and per-true-effect power.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from lab.stats.corrections import benjamini_hochberg, bonferroni


@dataclass(frozen=True)
class MultipleComparisonsResult:
    """Error and power rates per analysis policy.

    Attributes
    ----------
    summary : pd.DataFrame
        Rows: naive, bonferroni, benjamini_hochberg. Columns: fwer
        (any false rejection), fdr (mean fraction of rejections that are
        false), power (mean rejection rate on truly non-null metrics;
        NaN when there are none).
    p_values : np.ndarray
        (n_sims, n_metrics) matrix of two-sided p-values.
    is_null : np.ndarray
        Boolean mask of truly null metrics.
    """

    summary: pd.DataFrame
    p_values: np.ndarray
    is_null: np.ndarray


def simulate_many_metrics(
    *,
    n_sims: int = 4000,
    n_metrics: int = 20,
    n_true_effects: int = 0,
    n_per_arm: int = 2000,
    effect: float = 0.0,
    sd: float = 5.4,
    alpha: float = 0.05,
    seed: int = 0,
) -> MultipleComparisonsResult:
    """Simulate m-metric experiments and apply the three policies.

    The first n_true_effects metrics carry the additive `effect`; the
    rest are null. The per-metric z-statistic is N(shift, 1) with
    shift = effect / (sd * sqrt(2 / n_per_arm)), the exact sampling
    distribution of the standardized difference in means.
    """
    if n_true_effects > n_metrics:
        raise ValueError("n_true_effects cannot exceed n_metrics")
    rng = np.random.default_rng(seed)
    is_null = np.arange(n_metrics) >= n_true_effects
    shift = np.where(is_null, 0.0, effect / (sd * np.sqrt(2.0 / n_per_arm)))
    z = rng.normal(0.0, 1.0, (n_sims, n_metrics)) + shift
    p_values = 2.0 * sps.norm.sf(np.abs(z))

    rejections = {
        "naive": p_values < alpha,
        "bonferroni": np.vstack([bonferroni(row, alpha=alpha) for row in p_values]),
        "benjamini_hochberg": np.vstack(
            [benjamini_hochberg(row, alpha=alpha) for row in p_values]
        ),
    }

    rows = {}
    for name, rej in rejections.items():
        false_rej = rej & is_null[None, :]
        n_rej = rej.sum(axis=1)
        fdr = np.where(n_rej > 0, false_rej.sum(axis=1) / np.maximum(n_rej, 1), 0.0)
        power = float(rej[:, ~is_null].mean()) if n_true_effects > 0 else float("nan")
        rows[name] = {
            "fwer": float(false_rej.any(axis=1).mean()),
            "fdr": float(fdr.mean()),
            "power": power,
        }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    return MultipleComparisonsResult(summary=summary, p_values=p_values, is_null=is_null)
