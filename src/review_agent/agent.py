"""The review agent: one model call, structured output, grounded numbers.

Single-agent by decision (DECISIONS.md, ADR-3). The API key is read from the
ANTHROPIC_API_KEY environment variable by the SDK; nothing secret lives in
the repository (see .env.example). The committed eval transcripts are a
reference run; regenerate them live with any key via eval/harness.py --live.
"""

import json
from pathlib import Path

import anthropic

from review_agent.evidence import build_evidence
from review_agent.schemas import ExperimentDesign, ReviewMemo

PROMPT_PATH = Path(__file__).parent / "prompts" / "v1_system.md"
DEFAULT_MODEL = "claude-opus-4-8"


def build_user_message(design: ExperimentDesign) -> str:
    """The exact user message the model sees; also recorded in transcripts."""
    return (
        "Review this experiment design.\n\n"
        f"DESIGN:\n{design.model_dump_json(indent=1)}\n\n"
        f"EVIDENCE PACK (the only numbers you may cite):\n"
        f"{json.dumps(build_evidence(design), indent=1)}"
    )


def review_design(
    design: ExperimentDesign,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
) -> ReviewMemo:
    """One live API call returning a validated ReviewMemo."""
    client = client or anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        max_tokens=2048,
        system=PROMPT_PATH.read_text(),
        messages=[{"role": "user", "content": build_user_message(design)}],
        output_format=ReviewMemo,
    )
    memo = response.parsed_output
    if memo is None:
        raise ValueError(f"model returned no parseable memo for design {design.id}")
    return memo
