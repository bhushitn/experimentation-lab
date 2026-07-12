"""Deterministic evidence pack: every number the memo may cite.

The agent never does arithmetic. Before the model is called, this module
computes the design's power analysis with the Layer 1 calculators and pulls
the headline pathology rates measured by the simulations. The prompt
instructs the model to quote only these numbers.
"""

import json
from pathlib import Path
from typing import Any

from lab.stats import power_two_sample, sample_size_per_arm
from review_agent.schemas import ExperimentDesign

_HEADLINES_PATH = Path(__file__).resolve().parents[2] / "docs/assets/data/headline_results.json"


def build_evidence(design: ExperimentDesign) -> dict[str, Any]:
    """Layer 1 computations for one design, JSON-serializable."""
    n_per_arm = design.users_per_day_per_arm * design.duration_days
    achieved_power = power_two_sample(
        n_per_arm,
        effect=design.target_effect,
        sd=design.baseline_sd,
        alpha=design.alpha,
    )
    required = sample_size_per_arm(
        effect=design.target_effect,
        sd=design.baseline_sd,
        alpha=design.alpha,
        power=design.power_target,
    )
    evidence: dict[str, Any] = {
        "n_per_arm": n_per_arm,
        "required_n_per_arm_for_target_power": required,
        "achieved_power": round(achieved_power, 3),
        "expected_fwer_if_uncorrected": round(
            1.0 - (1.0 - design.alpha) ** design.n_metrics, 3
        ),
    }
    if _HEADLINES_PATH.exists():
        evidence["measured_pathology_rates"] = json.loads(_HEADLINES_PATH.read_text())
    return evidence
