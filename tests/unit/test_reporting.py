"""Reporting layer: every figure builds and its data export is JSON-clean.

The full regeneration (all seeds, all outputs) runs as a CI job via
scripts/regenerate_figures.py; this is the fast in-suite check.
"""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

from lab.pathologies import (  # noqa: E402
    effect_path,
    simulate_daily_peeking,
    simulate_many_metrics,
    simulate_novelty,
    simulate_subgroup_fishing,
)
from lab.reporting import (  # noqa: E402
    fig_multiple_comparisons,
    fig_novelty,
    fig_peeking,
    fig_subgroups,
)


class TestFiguresBuild:
    def test_all_fast_figures_build_and_serialize(self) -> None:
        figures = [
            fig_peeking(simulate_daily_peeking(n_sims=200, n_days=5, seed=0)),
            fig_multiple_comparisons(simulate_many_metrics(n_sims=200, seed=0)),
            fig_subgroups(simulate_subgroup_fishing(n_sims=200, seed=0)),
            fig_novelty(
                simulate_novelty(n_users_per_arm=2000, seed=0),
                path_days=np.arange(28),
                path=effect_path(
                    np.arange(28), long_run=0.1, novelty_amplitude=0.4, decay_days=5.0
                ),
            ),
        ]
        for fig, data in figures:
            assert fig.axes, "figure has no axes"
            json.dumps(data)  # raises if not serializable
