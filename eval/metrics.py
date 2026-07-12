"""Precision and recall of flaw detection, with Wilson confidence intervals.

Scoring unit: one (case, flaw_code) pair. A predicted code on a case is a
true positive if the label contains it, a false positive otherwise; a
labeled code the agent missed is a false negative. Wilson intervals come
from statsmodels (validated implementation, per repo policy).
"""

from collections import Counter
from typing import Any

from statsmodels.stats.proportion import proportion_confint

FLAW_CODES = [
    "PEEKING",
    "MULTIPLE_COMPARISONS",
    "CONTAMINATION",
    "INTERFERENCE",
    "SUBGROUP_FISHING",
    "NOVELTY_TOO_SHORT",
    "UNDERPOWERED",
]


def _rate_with_ci(hits: int, total: int) -> dict[str, float | None]:
    if total == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "n": 0}
    low, high = proportion_confint(hits, total, alpha=0.05, method="wilson")
    return {
        "value": round(hits / total, 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
        "n": total,
    }


def score(
    predictions: dict[str, list[str]], labels: dict[str, list[str]]
) -> dict[str, Any]:
    """Micro-averaged and per-code precision/recall with 95% Wilson CIs."""
    counts: Counter[str] = Counter()
    per_code: dict[str, Counter[str]] = {c: Counter() for c in FLAW_CODES}

    for case_id, true_codes in labels.items():
        pred = set(predictions.get(case_id, []))
        true = set(true_codes)
        for code in pred & true:
            counts["tp"] += 1
            per_code[code]["tp"] += 1
        for code in pred - true:
            counts["fp"] += 1
            per_code[code]["fp"] += 1
        for code in true - pred:
            counts["fn"] += 1
            per_code[code]["fn"] += 1

    def block(c: Counter[str]) -> dict[str, Any]:
        return {
            "precision": _rate_with_ci(c["tp"], c["tp"] + c["fp"]),
            "recall": _rate_with_ci(c["tp"], c["tp"] + c["fn"]),
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
        }

    exact = sum(
        1
        for case_id, true_codes in labels.items()
        if set(predictions.get(case_id, [])) == set(true_codes)
    )
    return {
        "n_cases": len(labels),
        "exact_match": _rate_with_ci(exact, len(labels)),
        "micro": block(counts),
        "per_code": {code: block(c) for code, c in per_code.items()},
    }
