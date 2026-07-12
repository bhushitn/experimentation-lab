"""Experiment pathologies, each with ground truth, naive error, and correction."""

from lab.pathologies.contamination import ContaminationResult, simulate_contamination
from lab.pathologies.interference import InterferenceResult, simulate_interference
from lab.pathologies.multiple_comparisons import (
    MultipleComparisonsResult,
    simulate_many_metrics,
)
from lab.pathologies.novelty import NoveltyResult, effect_path, simulate_novelty
from lab.pathologies.peeking import PeekingResult, simulate_daily_peeking
from lab.pathologies.quasi import QuasiResult, simulate_quasi
from lab.pathologies.srm import SrmResult, simulate_srm, srm_check, srm_detection_power
from lab.pathologies.subgroups import SubgroupResult, simulate_subgroup_fishing
from lab.pathologies.switchback import (
    SwitchbackResult,
    simulate_switchback,
    switchback_single_run,
)

__all__ = [
    "ContaminationResult",
    "InterferenceResult",
    "MultipleComparisonsResult",
    "NoveltyResult",
    "PeekingResult",
    "QuasiResult",
    "SrmResult",
    "SubgroupResult",
    "SwitchbackResult",
    "effect_path",
    "simulate_quasi",
    "simulate_srm",
    "simulate_switchback",
    "srm_check",
    "srm_detection_power",
    "switchback_single_run",
    "simulate_contamination",
    "simulate_daily_peeking",
    "simulate_interference",
    "simulate_many_metrics",
    "simulate_novelty",
    "simulate_subgroup_fishing",
]
