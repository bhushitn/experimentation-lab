"""Structured input and output schemas for the review agent.

The agent's contract: an ExperimentDesign in, a ReviewMemo out, validated
by Pydantic on both sides. Flaw codes are a closed vocabulary shared with
the eval labels.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class FlawCode(StrEnum):
    PEEKING = "PEEKING"
    MULTIPLE_COMPARISONS = "MULTIPLE_COMPARISONS"
    CONTAMINATION = "CONTAMINATION"
    INTERFERENCE = "INTERFERENCE"
    SUBGROUP_FISHING = "SUBGROUP_FISHING"
    NOVELTY_TOO_SHORT = "NOVELTY_TOO_SHORT"
    UNDERPOWERED = "UNDERPOWERED"


class MetricType(StrEnum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    ZERO_INFLATED = "zero_inflated"


class ExperimentDesign(BaseModel):
    """A structured experiment design, the agent's input."""

    id: str
    name: str
    description: str = Field(description="What the feature is and who it touches")
    metric_name: str
    metric_type: MetricType
    baseline_mean: float = Field(description="Baseline metric mean, or rate if binary")
    baseline_sd: float = Field(description="Metric sd; for binary pass sqrt(p*(1-p))")
    target_effect: float = Field(description="Absolute effect the team wants to detect")
    users_per_day_per_arm: int
    duration_days: int
    randomization_unit: str = Field(description="user or cluster")
    monitoring_plan: str = Field(description="How and when results will be checked")
    n_metrics: int = Field(description="Metrics tracked for significance")
    multiple_testing_correction: str = Field(description="none, bonferroni, or benjamini_hochberg")
    subgroup_plan: str = Field(description="Planned segment analysis, if any")
    exposure_notes: str = Field(description="Known compliance or exposure issues")
    alpha: float = 0.05
    power_target: float = 0.80


class Flag(BaseModel):
    """One identified threat to validity."""

    code: FlawCode
    severity: str = Field(description="high, medium, or low")
    rationale: str = Field(description="Why this design triggers the flag, one or two sentences")
    quantification: str = Field(
        description="The relevant number from the evidence pack or lab "
        "simulation; never computed by the model"
    )
    fix: str = Field(description="The concrete design or analysis change")


class ReviewMemo(BaseModel):
    """The agent's output: recommendation first, evidence after."""

    recommendation: str = Field(description="One or two sentences, the decision first")
    verdict: str = Field(description="approve, revise, or reject")
    flags: list[Flag]
    analysis_plan: str = Field(
        description="The correct analysis for this design in two or three sentences"
    )
