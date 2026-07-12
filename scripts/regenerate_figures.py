"""Regenerate every figure and its underlying data from fixed seeds.

Usage: python scripts/regenerate_figures.py [--outdir docs/assets]

Writes, per figure: PNG and SVG to <outdir>/figures/, and the figure's data
as JSON to <outdir>/data/ for the D3 versions on the site. Also writes
headline_results.json, the source of the README results table. Runs in CI,
so if any figure stops regenerating, the build fails and the repository's
central claim (every visual is reproducible) fails loudly.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from lab.assignment import assign_clusters, assign_users
from lab.pathologies import (
    effect_path,
    simulate_contamination,
    simulate_daily_peeking,
    simulate_interference,
    simulate_many_metrics,
    simulate_novelty,
    simulate_subgroup_fishing,
)
from lab.populations import continuous_users
from lab.populations.graphs import sbm_graph
from lab.reporting import (
    fig_contamination,
    fig_interference,
    fig_multiple_comparisons,
    fig_novelty,
    fig_peeking,
    fig_subgroups,
)
from lab.reporting.explainer_data import (
    build_cuped,
    build_power_curves,
    build_randomization,
    build_test_choice,
)
from lab.reporting.explainer_figures import (
    fig_cuped,
    fig_power,
    fig_randomization,
    fig_test_choice,
)


def build_all() -> dict[str, tuple[Figure, dict[str, Any]]]:
    """Run every simulation at its canonical seed and build every figure."""
    out: dict[str, tuple[Figure, dict[str, Any]]] = {}

    peek = simulate_daily_peeking(n_sims=4000, n_days=14, effect=0.0, seed=1)
    out["peeking"] = fig_peeking(peek)

    multi = simulate_many_metrics(n_sims=4000, n_metrics=20, seed=3)
    out["multiple_comparisons"] = fig_multiple_comparisons(multi)

    users = continuous_users(100_000, seed=6)
    treated = assign_users(users, seed=7)
    contam = simulate_contamination(users, treated, effect=0.5, seed=8)
    out["contamination"] = fig_contamination(contam)

    n = 20_000
    net_users = continuous_users(n, seed=9)
    adjacency, clusters = sbm_graph(n, n_clusters=100, seed=10)
    net_users["cluster"] = clusters
    interf = simulate_interference(
        net_users,
        adjacency,
        assign_users(net_users, seed=11),
        assign_clusters(net_users, cluster_col="cluster", seed=11),
        direct=0.5,
        spillover=0.3,
        seed=12,
    )
    out["interference"] = fig_interference(interf)

    sub = simulate_subgroup_fishing(n_sims=4000, n_segments=20, seed=13)
    out["subgroups"] = fig_subgroups(sub)

    nov = simulate_novelty(seed=15)
    days = np.arange(28)
    path = effect_path(days, long_run=0.10, novelty_amplitude=0.40, decay_days=5.0)
    out["novelty"] = fig_novelty(nov, path_days=days, path=path)

    rand_data = build_randomization()
    out["randomization"] = (fig_randomization(rand_data), rand_data)
    test_data = build_test_choice()
    out["test_choice"] = (fig_test_choice(test_data), test_data)
    cuped_data = build_cuped()
    out["cuped"] = (fig_cuped(cuped_data), cuped_data)
    power_data = build_power_curves()
    out["power"] = (fig_power(power_data), power_data)

    return out


def headline_results(figures: dict[str, tuple[Figure, dict[str, Any]]]) -> dict[str, Any]:
    """The numbers the README quotes, pulled from the same runs as the figures."""
    peek = figures["peeking"][1]
    multi = figures["multiple_comparisons"][1]
    contam = figures["contamination"][1]
    interf = figures["interference"][1]
    sub = figures["subgroups"][1]
    nov = figures["novelty"][1]
    return {
        "peeking_naive_fpr": peek["naive_cumulative_fpr"][-1],
        "peeking_sequential_fpr": peek["sequential_cumulative_fpr"][-1],
        "peeking_n_sims": peek["n_sims"],
        "multiple_naive_fwer": multi["fwer"][0],
        "multiple_bh_fwer": multi["fwer"][2],
        "contamination_itt": contam["estimates"][0],
        "contamination_per_protocol": contam["estimates"][1],
        "contamination_iv": contam["estimates"][2],
        "contamination_truth": contam["true_effect"],
        "interference_user_randomized": interf["estimates"][0],
        "interference_cluster_randomized": interf["estimates"][1],
        "interference_gte": interf["global_treatment_effect"],
        "subgroup_fishing_discovery_rate": sub["rates"][0],
        "novelty_week1": nov["estimates"]["first_week"],
        "novelty_long_run": nov["long_run_effect"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()

    fig_dir = args.outdir / "figures"
    data_dir = args.outdir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    figures = build_all()
    for name, (fig, data) in figures.items():
        fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight")
        fig.savefig(fig_dir / f"{name}.svg", bbox_inches="tight")
        payload = json.dumps(data, indent=1)
        (data_dir / f"{name}.json").write_text(payload)
        # Same data as a script for the explainers: works over file:// where
        # fetch() of local JSON is blocked, and needs no async plumbing.
        (data_dir / f"{name}.js").write_text(
            f'window.LAB_DATA = window.LAB_DATA || {{}};\nwindow.LAB_DATA["{name}"] = {payload};\n'
        )
        print(f"wrote {name}: figures/{name}.{{png,svg}} data/{name}.{{json,js}}")

    results = headline_results(figures)
    (data_dir / "headline_results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote data/headline_results.json ({len(results)} headline numbers)")


if __name__ == "__main__":
    main()
