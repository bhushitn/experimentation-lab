"""Generate the labeled eval set: 32 designs, ground truth by construction.

Each case starts from one of four clean base designs (the fictional
short-video app, per ADR-6) and injects zero or more flaws by modifying the
fields that define them. Labels are therefore correct by construction: the
label is the list of injections applied. Regenerate with:
python eval/build_designs.py
"""

import itertools
import json
import math
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).parent / "labeled_designs"

# Clean bases: powered, single primary metric, pre-registered plans,
# 28-day runs, randomization matched to the feature's exposure pattern.
BASES: list[dict[str, Any]] = [
    {
        "name": "signup_flow_simplification",
        "description": "Removes one step from the signup flow. Individual experience.",
        "metric_name": "signup_rate",
        "metric_type": "binary",
        "baseline_mean": 0.10,
        "baseline_sd": round(math.sqrt(0.10 * 0.90), 4),
        "target_effect": 0.01,
        "users_per_day_per_arm": 2500,
        "duration_days": 28,
    },
    {
        "name": "upload_pipeline_speedup",
        "description": "Faster video upload pipeline. Backend change users "
        "do not visibly notice.",
        "metric_name": "uploads_per_user",
        "metric_type": "continuous",
        "baseline_mean": 4.0,
        "baseline_sd": 6.0,
        "target_effect": 0.15,
        "users_per_day_per_arm": 3000,
        "duration_days": 28,
    },
    {
        "name": "creator_payout_change",
        "description": "New payout formula for creators. Spend-shaped "
        "outcome, mostly zeros with a heavy right tail.",
        "metric_name": "weekly_payout_dollars",
        "metric_type": "zero_inflated",
        "baseline_mean": 2.5,
        "baseline_sd": 12.0,
        "target_effect": 0.30,
        "users_per_day_per_arm": 3500,
        "duration_days": 28,
    },
    {
        "name": "notification_batching",
        "description": "Batches push notifications into a daily digest. "
        "Individual experience.",
        "metric_name": "daily_sessions",
        "metric_type": "continuous",
        "baseline_mean": 3.2,
        "baseline_sd": 2.8,
        "target_effect": 0.08,
        "users_per_day_per_arm": 4000,
        "duration_days": 28,
    },
]

CLEAN_FIELDS: dict[str, Any] = {
    "randomization_unit": "user",
    "monitoring_plan": "Two scheduled looks at day 14 and day 28 using an "
    "O'Brien-Fleming alpha-spending boundary.",
    "n_metrics": 1,
    "multiple_testing_correction": "none",
    "subgroup_plan": "One pre-registered segment: new users (tenure under 30 days).",
    "exposure_notes": "Assignment enforced server-side per account; no known "
    "leakage paths.",
    "alpha": 0.05,
    "power_target": 0.80,
}

# Each injection edits fields; the label is the injection name.
INJECTIONS: dict[str, dict[str, Any]] = {
    "PEEKING": {
        "monitoring_plan": "PM checks the dashboard every morning and will "
        "call the test the first day the p-value drops below 0.05.",
    },
    "MULTIPLE_COMPARISONS": {
        "n_metrics": 20,
        "multiple_testing_correction": "none",
        "monitoring_plan_suffix": " All 20 dashboard metrics are tested for "
        "significance at the final readout.",
    },
    "CONTAMINATION": {
        "exposure_notes": "Roughly 15% of accounts are shared family-tablet "
        "devices, so control users can see the treated experience; treatment "
        "is also unavailable in logged-out sessions. Analysis plan compares "
        "users by what they actually experienced.",
    },
    "INTERFERENCE": {
        "description_suffix": " The launch bundles a share-to-friends "
        "prompt, so treated users send content into their follow graph and "
        "move outcomes for the users they are connected to.",
        "randomization_unit": "user",
    },
    "SUBGROUP_FISHING": {
        "subgroup_plan": "After the readout, scan all 20 country segments "
        "and report whichever segments come out significant at 0.05.",
    },
    "NOVELTY_TOO_SHORT": {
        "description_suffix": " The launch bundles a visible redesign of "
        "the feed layout that users immediately notice.",
        "duration_days": 7,
        "monitoring_plan": "Single scheduled readout at the end of the "
        "seven-day run.",
    },
    "UNDERPOWERED": {
        "users_per_day_per_arm_factor": 0.04,
    },
}


def apply(base: dict[str, Any], flaws: tuple[str, ...], case_id: str) -> dict[str, Any]:
    design: dict[str, Any] = {**base, **CLEAN_FIELDS, "id": case_id}
    # PEEKING last: its monitoring plan must win over NOVELTY's single readout.
    for flaw in sorted(flaws, key=lambda f: f == "PEEKING"):
        edit = INJECTIONS[flaw]
        for key, value in edit.items():
            if key.endswith("_suffix"):
                design[key.removesuffix("_suffix")] += value
            elif key.endswith("_factor"):
                target = key.removesuffix("_factor")
                design[target] = max(1, int(design[target] * value))
            else:
                design[key] = value
    return design


def main() -> None:
    cases: list[tuple[str, tuple[str, ...]]] = []
    # 8 clean: each base twice (second copy varies cosmetically via id only).
    for i, _ in enumerate(BASES * 2):
        cases.append((f"clean_{i:02d}", ()))
    # 14 single-flaw: every flaw on two different bases.
    singles = [(f,) for f in INJECTIONS] * 2
    for i, flaws in enumerate(singles):
        cases.append((f"single_{i:02d}", flaws))
    # 10 multi-flaw combinations.
    combos = list(itertools.combinations(INJECTIONS, 2))[:8] + [
        ("PEEKING", "MULTIPLE_COMPARISONS", "SUBGROUP_FISHING"),
        ("INTERFERENCE", "NOVELTY_TOO_SHORT", "UNDERPOWERED"),
    ]
    for i, flaws in enumerate(combos):
        cases.append((f"multi_{i:02d}", tuple(flaws)))

    designs, labels = [], {}
    for i, (case_id, flaws) in enumerate(cases):
        base = BASES[i % len(BASES)]
        designs.append(apply(base, flaws, case_id))
        labels[case_id] = sorted(flaws)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "designs.json").write_text(json.dumps(designs, indent=1))
    (OUT_DIR / "labels.json").write_text(json.dumps(labels, indent=1))
    n_flawed = sum(1 for _, f in cases if f)
    print(f"wrote {len(designs)} designs ({len(designs) - n_flawed} clean, {n_flawed} flawed)")


if __name__ == "__main__":
    main()
