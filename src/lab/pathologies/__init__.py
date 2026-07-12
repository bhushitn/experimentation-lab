"""Experiment pathologies, each with ground truth, naive error, and correction."""

from lab.pathologies.contamination import ContaminationResult, simulate_contamination
from lab.pathologies.interference import InterferenceResult, simulate_interference
from lab.pathologies.multiple_comparisons import (
    MultipleComparisonsResult,
    simulate_many_metrics,
)
from lab.pathologies.novelty import NoveltyResult, effect_path, simulate_novelty
from lab.pathologies.peeking import PeekingResult, simulate_daily_peeking
from lab.pathologies.subgroups import SubgroupResult, simulate_subgroup_fishing

__all__ = [
    "ContaminationResult",
    "InterferenceResult",
    "MultipleComparisonsResult",
    "NoveltyResult",
    "PeekingResult",
    "SubgroupResult",
    "effect_path",
    "simulate_contamination",
    "simulate_daily_peeking",
    "simulate_interference",
    "simulate_many_metrics",
    "simulate_novelty",
    "simulate_subgroup_fishing",
]
