"""Run or replay the review-agent eval and score it.

Default mode replays the committed transcripts in eval/results/, so the
metrics are reproducible with no API key. Pass --live to regenerate the
transcripts with your own ANTHROPIC_API_KEY (see .env.example); the model
and generation method of the committed transcripts are recorded in
eval/results/transcripts.json under "generation".

Usage:
  python eval/harness.py            # replay committed transcripts, score
  python eval/harness.py --live     # call the API, rewrite transcripts, score
"""

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(EVAL_DIR.parent / "src"))

from metrics import score  # noqa: E402

from review_agent.schemas import ExperimentDesign, ReviewMemo  # noqa: E402

DESIGNS_PATH = EVAL_DIR / "labeled_designs" / "designs.json"
LABELS_PATH = EVAL_DIR / "labeled_designs" / "labels.json"
TRANSCRIPTS_PATH = EVAL_DIR / "results" / "transcripts.json"
METRICS_PATH = EVAL_DIR / "results" / "metrics.json"


def run_live(model: str) -> dict:
    from review_agent.agent import review_design

    designs = [ExperimentDesign(**d) for d in json.loads(DESIGNS_PATH.read_text())]
    memos = {}
    for design in designs:
        memo = review_design(design, model=model)
        memos[design.id] = memo.model_dump(mode="json")
        print(f"{design.id}: {[f['code'] for f in memos[design.id]['flags']]}")
    return {
        "generation": {"method": "live_api", "model": model},
        "memos": memos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="call the API instead of replaying")
    parser.add_argument("--model", default="claude-opus-4-8")
    args = parser.parse_args()

    if args.live:
        transcripts = run_live(args.model)
        TRANSCRIPTS_PATH.write_text(json.dumps(transcripts, indent=1))
    else:
        if not TRANSCRIPTS_PATH.exists():
            raise SystemExit("no committed transcripts; run with --live and an API key")
        transcripts = json.loads(TRANSCRIPTS_PATH.read_text())

    labels = json.loads(LABELS_PATH.read_text())
    predictions = {}
    for case_id, memo_dict in transcripts["memos"].items():
        memo = ReviewMemo(**memo_dict)  # validate every transcript against the schema
        predictions[case_id] = [flag.code.value for flag in memo.flags]

    results = score(predictions, labels)
    results["generation"] = transcripts["generation"]
    METRICS_PATH.write_text(json.dumps(results, indent=1))

    micro = results["micro"]
    print(f"\ncases: {results['n_cases']}")
    print(
        f"micro precision {micro['precision']['value']} "
        f"[{micro['precision']['ci_low']}, {micro['precision']['ci_high']}]"
    )
    print(
        f"micro recall    {micro['recall']['value']} "
        f"[{micro['recall']['ci_low']}, {micro['recall']['ci_high']}]"
    )
    print(f"exact match     {results['exact_match']['value']}")
    print(f"wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
