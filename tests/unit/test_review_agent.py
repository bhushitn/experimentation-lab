"""Review agent: schemas, evidence grounding, and eval scoring.

No API calls; the live path is exercised only via eval/harness.py --live.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from metrics import score  # noqa: E402

from lab.stats import power_two_sample  # noqa: E402
from review_agent.agent import build_user_message  # noqa: E402
from review_agent.evidence import build_evidence  # noqa: E402
from review_agent.schemas import ExperimentDesign, ReviewMemo  # noqa: E402

DESIGN = ExperimentDesign(
    id="t1",
    name="test",
    description="a feature",
    metric_name="rate",
    metric_type="binary",
    baseline_mean=0.1,
    baseline_sd=0.3,
    target_effect=0.01,
    users_per_day_per_arm=1000,
    duration_days=14,
    randomization_unit="user",
    monitoring_plan="final readout only",
    n_metrics=1,
    multiple_testing_correction="none",
    subgroup_plan="none",
    exposure_notes="none",
)


class TestEvidence:
    def test_power_numbers_come_from_lab(self) -> None:
        ev = build_evidence(DESIGN)
        assert ev["n_per_arm"] == 14_000
        expected = power_two_sample(14_000, effect=0.01, sd=0.3, alpha=0.05)
        assert ev["achieved_power"] == pytest.approx(expected, abs=1e-3)

    def test_fwer_formula(self) -> None:
        ev = build_evidence(DESIGN.model_copy(update={"n_metrics": 20}))
        assert ev["expected_fwer_if_uncorrected"] == pytest.approx(1 - 0.95**20, abs=1e-3)

    def test_user_message_embeds_design_and_evidence(self) -> None:
        msg = build_user_message(DESIGN)
        assert "EVIDENCE PACK" in msg and '"achieved_power"' in msg and '"id": "t1"' in msg


class TestEvalArtifacts:
    def test_designs_and_labels_align(self) -> None:
        designs = json.loads((REPO / "eval/labeled_designs/designs.json").read_text())
        labels = json.loads((REPO / "eval/labeled_designs/labels.json").read_text())
        assert len(designs) >= 30
        assert {d["id"] for d in designs} == set(labels)
        for d in designs:
            ExperimentDesign(**d)  # every design validates

    def test_committed_transcripts_validate(self) -> None:
        transcripts = json.loads((REPO / "eval/results/transcripts.json").read_text())
        assert transcripts["generation"]["method"] in {"reference_run", "live_api"}
        for memo in transcripts["memos"].values():
            ReviewMemo(**memo)


class TestScoring:
    def test_hand_computed_example(self) -> None:
        labels = {"a": ["PEEKING"], "b": [], "c": ["PEEKING", "UNDERPOWERED"]}
        preds = {"a": ["PEEKING"], "b": ["CONTAMINATION"], "c": ["PEEKING"]}
        r = score(preds, labels)
        # tp=2 (a:PEEKING, c:PEEKING), fp=1 (b:CONTAMINATION), fn=1 (c:UNDERPOWERED)
        assert r["micro"]["tp"] == 2 and r["micro"]["fp"] == 1 and r["micro"]["fn"] == 1
        assert r["micro"]["precision"]["value"] == pytest.approx(2 / 3, abs=1e-4)
        assert r["micro"]["recall"]["value"] == pytest.approx(2 / 3, abs=1e-4)
        assert r["exact_match"]["value"] == pytest.approx(1 / 3, abs=1e-4)

    def test_wilson_ci_matches_statsmodels(self) -> None:
        from statsmodels.stats.proportion import proportion_confint

        r = score({"a": ["PEEKING"]}, {"a": ["PEEKING"]})
        low, high = proportion_confint(1, 1, alpha=0.05, method="wilson")
        assert r["micro"]["precision"]["ci_low"] == pytest.approx(float(low), abs=1e-4)
        assert r["micro"]["precision"]["ci_high"] == pytest.approx(float(high), abs=1e-4)
