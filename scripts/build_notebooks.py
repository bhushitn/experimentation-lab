"""Build and execute the repository notebooks from source.

Usage: python scripts/build_notebooks.py

Notebooks are generated programmatically and executed with nbclient, so
their outputs always come from the current code. Rerunning this script is
the supported way to refresh them; editing notebook JSON by hand is not.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

PREAMBLE = """\
import numpy as np

from lab.assignment import assign_clusters, assign_users
from lab.pathologies import (
    effect_path,
    simulate_contamination,
    simulate_daily_peeking,
    simulate_interference,
    simulate_many_metrics,
    simulate_novelty,
    simulate_quasi,
    simulate_srm,
    simulate_subgroup_fishing,
    simulate_switchback,
    srm_detection_power,
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
)"""


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


def engine_walkthrough() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md(
            "# Engine walkthrough\n\n"
            "The simulation engine in `src/lab/`: synthetic populations with "
            "stored latent parameters, randomization, outcome generation with a "
            "configurable true effect, and the validated statistical toolbox. "
            "Because the truth is stored, every estimate in this notebook can be "
            "compared to the exact answer."
        ),
        code(
            "import numpy as np\n\n"
            "from lab.assignment import assign_users\n"
            "from lab.populations import continuous_users, draw_continuous\n"
            "from lab.stats import cuped_t_test, welch_t_test"
        ),
        md(
            "## A clean experiment, start to finish\n\n"
            "100,000 users, true effect fixed at 0.5 by construction."
        ),
        code(
            "users = continuous_users(100_000, mean=10.0, between_sd=2.0, within_sd=5.0, seed=0)\n"
            "treated = assign_users(users, treat_fraction=0.5, seed=1)\n"
            "y = draw_continuous(users, treated, effect=0.5, within_sd=5.0, seed=2)\n"
            "result = welch_t_test(y[treated], y[~treated])\n"
            "print(f'estimate {result.estimate:.3f}, true effect 0.500')\n"
            "ci = f'[{result.ci_low:.3f}, {result.ci_high:.3f}]'\n"
            "print(f'95% CI {ci}, p = {result.p_value:.2e}')"
        ),
        md(
            "## CUPED with the pre-period covariate\n\n"
            "A persistent engagement metric (user-level signal comparable to "
            "period noise) with a 28-day pre-period aggregate as the covariate. "
            "The pre/post correlation is known analytically, so CUPED's "
            "variance reduction has an exact expected value: rho squared."
        ),
        code(
            "b, w, k = 3.0, 3.0, 28  # between-user sd, within sd, pre-period days\n"
            "users2 = continuous_users(100_000, between_sd=b, within_sd=w,\n"
            "                          pre_period_days=k, seed=3)\n"
            "treated2 = assign_users(users2, seed=4)\n"
            "y2 = draw_continuous(users2, treated2, effect=0.2, within_sd=w, seed=5)\n"
            "raw = welch_t_test(y2[treated2], y2[~treated2])\n"
            "cuped = cuped_t_test(y2[treated2], users2['pre_metric'][treated2],\n"
            "                     y2[~treated2], users2['pre_metric'][~treated2])\n"
            "rho = b**2 / (np.sqrt(b**2 + w**2 / k) * np.sqrt(b**2 + w**2))\n"
            "print(f'measured variance reduction {cuped.variance_reduction:.3f}, '\n"
            "      f'analytic rho^2 {rho**2:.3f}')\n"
            "raw_width = raw.ci_high - raw.ci_low\n"
            "cuped_width = cuped.test.ci_high - cuped.test.ci_low\n"
            "print(f'CI width {raw_width:.4f} raw vs {cuped_width:.4f} CUPED '\n"
            "      f'({1 - cuped_width / raw_width:.0%} narrower)')"
        ),
    ]
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "language": "python",
        "display_name": "Python 3",
    }
    return nb


SECTIONS = [
    (
        "Peeking: checking daily manufactures false positives",
        "A growth PM watches the dashboard every morning and calls the test at "
        "the first green p-value. 4,000 replications of a 14-day experiment with "
        "**no true effect**: the naive policy rejects far above the nominal 5%; "
        "the Lan-DeMets O'Brien-Fleming boundary restores it.",
        "peek = simulate_daily_peeking(n_sims=4000, n_days=14, effect=0.0, seed=1)\n"
        "print(peek.rejection_rates().round(4))",
        "fig, _ = fig_peeking(peek)",
    ),
    (
        "Multiple comparisons: twenty metrics, one experiment",
        "The dashboard tracks 20 metrics, all null here. Testing each at 0.05 "
        "finds 'something significant' in most experiments. Bonferroni and "
        "Benjamini-Hochberg both control their target rates; the difference is "
        "power, shown in the pathology tests with true effects present.",
        "multi = simulate_many_metrics(n_sims=4000, n_metrics=20, seed=3)\n"
        "print(multi.summary.round(4))",
        "fig, _ = fig_multiple_comparisons(multi)",
    ),
    (
        "Contamination: assignment is not receipt",
        "20% of treatment never receives the feature, 10% of control is exposed "
        "(shared devices), and receipt correlates with baseline engagement. "
        "True effect 0.50. Intent-to-treat attenuates; per-protocol inflates via "
        "selection; IV/Wald divides ITT by the compliance gap and recovers it. "
        "Expected values are closed-form from the latent baselines.",
        "users = continuous_users(100_000, seed=6)\n"
        "treated = assign_users(users, seed=7)\n"
        "contam = simulate_contamination(users, treated, effect=0.5, seed=8)\n"
        "print('estimates', dict(contam.estimates.round(3)))\n"
        "print('expected ', dict(contam.expected.round(3)))",
        "fig, _ = fig_contamination(contam)",
    ),
    (
        "Interference: the design decides what the estimator can see",
        "A sharing feature leaks through the follow graph (stochastic block "
        "model, 100 clusters). Direct effect 0.5, spillover 0.3, so the global "
        "treatment effect is 0.8. Under user-level randomization both arms have "
        "~50% treated neighbors and the estimator reads only the direct effect; "
        "cluster randomization pushes the exposure contrast near 1 and recovers "
        "the global effect. The expected row is the exact linear-in-means "
        "identity, not a Monte Carlo average.",
        "net_users = continuous_users(20_000, seed=9)\n"
        "adjacency, clusters = sbm_graph(20_000, n_clusters=100, seed=10)\n"
        "net_users['cluster'] = clusters\n"
        "interf = simulate_interference(\n"
        "    net_users, adjacency,\n"
        "    assign_users(net_users, seed=11),\n"
        "    assign_clusters(net_users, cluster_col='cluster', seed=11),\n"
        "    direct=0.5, spillover=0.3, seed=12,\n"
        ")\n"
        "print('estimates', dict(interf.estimates.round(3)))\n"
        "print('expected ', dict(interf.expected.round(3)))\n"
        "print('exposure contrast', dict(interf.exposure_contrast.round(3)))",
        "fig, _ = fig_interference(interf)",
    ),
    (
        "Subgroup fishing: significant in Brazil, probably noise",
        "Topline flat, so someone scans 20 country segments, all null here. "
        "The scan finds at least one 'significant' subgroup in ~64% of "
        "experiments. A pre-registered primary stays at 5%; a "
        "Benjamini-Hochberg-corrected scan stays controlled.",
        "sub = simulate_subgroup_fishing(n_sims=4000, n_segments=20, seed=13)\n"
        "print(sub.summary.round(4))",
        "fig, _ = fig_subgroups(sub)",
    ),
    (
        "Sample ratio mismatch: the smoke alarm",
        "A logging bug drops 2% of the treatment arm, concentrated among "
        "low-engagement users. The chi-square check on the counts fires long "
        "before any eyeball would, and the estimate on the surviving data is "
        "biased by an exactly computable amount.",
        "srm_users = continuous_users(200_000, seed=30)\n"
        "srm = simulate_srm(srm_users, assign_users(srm_users, seed=31),\n"
        "                   effect=0.5, drop_fraction=0.02, seed=32)\n"
        "print('logged counts:', dict(srm.counts))\n"
        "print('SRM p-value:', f'{srm.srm_p_value:.2e}')\n"
        "print('estimates:', dict(srm.estimates.round(3)), 'truth', srm.true_effect)",
        "print('detection power, 2% loss at 100k/arm:',\n"
        "      srm_detection_power(n_per_arm=100_000, drop_fraction=0.02))\n"
        "print('detection power, 1% loss at 100k/arm:',\n"
        "      srm_detection_power(n_per_arm=100_000, drop_fraction=0.01))",
    ),
    (
        "Switchbacks: carryover versus window length",
        "A marketplace alternates arms in time windows. Carryover dilutes "
        "short windows; day-level shocks inflate the variance of long ones; "
        "a burn-in hour after each flip recovers the truth at every length.",
        "sb = simulate_switchback(seed=40)\n"
        "print(sb.by_window.round(3).to_string(index=False))\n"
        "print('truth:', sb.true_effect, '| carryover:', sb.carryover)",
        "# The naive estimate matches the exact carryover arithmetic per window.\n"
        "print(sb.by_window[['window_hours', 'naive_estimate', 'naive_expected']].round(3)\n"
        "      .to_string(index=False))",
    ),
    (
        "Quasi-experiments: diff-in-diff and the placebo test",
        "A city-by-city launch, big cities first, no randomization. DiD "
        "recovers the truth under parallel trends and overstates it when "
        "treated cities were already growing faster; the pre-period placebo "
        "separates the two cases.",
        "parallel = simulate_quasi(differential_trend=0.0, seed=50)\n"
        "violated = simulate_quasi(differential_trend=0.3, n_cities=200, seed=52)\n"
        "print('truth:', parallel.true_effect)\n"
        "print('parallel trends:', dict(parallel.estimates.round(3)))\n"
        "print('divergent trends:', dict(violated.estimates.round(3)))",
        "print('placebo DiD, parallel: ', round(parallel.placebo_did, 3))\n"
        "print('placebo DiD, divergent:', round(violated.placebo_did, 3))",
    ),
    (
        "Novelty: the first week is not the launch effect",
        "New feed layout: long-run effect 0.10 plus a 0.40 novelty spike "
        "decaying with a 5-day time constant. The week-1 readout "
        "overstates the launch effect 3x; the 4-week average still overstates; "
        "the post-burn-in window sits near the truth. Window expectations are "
        "the analytic means of the effect path.",
        "nov = simulate_novelty(seed=15)\n"
        "print('estimates', dict(nov.estimates.round(3)))\n"
        "print('expected ', dict(nov.expected.round(3)))\n"
        "print('long-run truth', nov.long_run_effect)",
        "days = np.arange(28)\n"
        "path = effect_path(days, long_run=0.10, novelty_amplitude=0.40, decay_days=5.0)\n"
        "fig, _ = fig_novelty(nov, path_days=days, path=path)",
    ),
]


def pathology_gallery() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md(
            "# Pathology gallery\n\n"
            "One section per pathology. Every section follows the same shape: "
            "the vignette (a fictional short-video app; all data synthetic), the "
            "ground truth set by construction, the naive analysis's quantified "
            "error, and the corrected method's recovery. Each figure here is the "
            "same code path as `scripts/regenerate_figures.py`, and each number "
            "is asserted against theory in `tests/integration/test_pathologies.py`."
        ),
        code(PREAMBLE),
    ]
    for title, narrative, sim_code, fig_code in SECTIONS:
        nb.cells += [md(f"## {title}\n\n{narrative}"), code(sim_code), code(fig_code)]
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "language": "python",
        "display_name": "Python 3",
    }
    return nb


def review_agent_eval() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md(
            "# Review agent evaluation\n\n"
            "32 labeled experiment designs (8 clean, 24 flawed; labels correct by "
            "construction, see `eval/build_designs.py`), scored per (case, flaw) "
            "pair with 95% Wilson intervals. Method note: the committed memos are "
            "a reference run, and flaw difficulty is bounded by construction, so "
            "read the numbers as a demonstration of the harness and the memo "
            "format; regenerate live against any model with "
            "`python eval/harness.py --live`."
        ),
        code(
            "import json\n\n"
            "import pandas as pd\n\n"
            "results = json.load(open('eval/results/metrics.json'))\n"
            "micro = results['micro']\n"
            "print('generation:', results['generation']['method'],\n"
            "      results['generation']['model'])\n"
            "print(f\"cases: {results['n_cases']}\")\n"
            "for k in ('precision', 'recall'):\n"
            "    m = micro[k]\n"
            "    print(f\"micro {k}: {m['value']} \"\n"
            "          f\"[{m['ci_low']}, {m['ci_high']}] (n={m['n']})\")"
        ),
        code(
            "rows = {code: {'tp': b['tp'], 'fp': b['fp'], 'fn': b['fn'],\n"
            "               'precision': b['precision']['value'],\n"
            "               'recall': b['recall']['value']}\n"
            "        for code, b in results['per_code'].items()}\n"
            "pd.DataFrame(rows).T"
        ),
        md(
            "## One memo, end to end\n\n"
            "The memo format the agent produces: recommendation first, then "
            "flags grounded in the evidence pack (numbers computed by the lab, "
            "never by the model), then the analysis plan."
        ),
        code(
            "transcripts = json.load(open('eval/results/transcripts.json'))\n"
            "memo = transcripts['memos']['multi_09']\n"
            "print(memo['recommendation'])\n"
            "print()\n"
            "for f in memo['flags']:\n"
            "    print(f\"[{f['severity']}] {f['code']}: {f['quantification']}\")\n"
            "print()\n"
            "print('Analysis plan:', memo['analysis_plan'])"
        ),
    ]
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "language": "python",
        "display_name": "Python 3",
    }
    return nb


def main() -> None:
    out = Path("notebooks")
    out.mkdir(exist_ok=True)
    for name, nb in [
        ("01_engine_walkthrough.ipynb", engine_walkthrough()),
        ("02_pathology_gallery.ipynb", pathology_gallery()),
        ("03_review_agent_eval.ipynb", review_agent_eval()),
    ]:
        client = NotebookClient(nb, timeout=600, resources={"metadata": {"path": "."}})
        client.execute()
        nbformat.write(nb, out / name)
        print(f"built and executed notebooks/{name}")


if __name__ == "__main__":
    main()
