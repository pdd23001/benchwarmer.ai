"""Pydantic models defining data contracts for Benchwarmer.AI."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Objective(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class RunStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


class Priority(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    NOT_A_CONCERN = "not a concern"
    REPORT_ONLY = "report but don't optimize for"


# ---------------------------------------------------------------------------
# Benchmark configuration models
# ---------------------------------------------------------------------------

class GeneratorConfig(BaseModel):
    """Configuration for a single instance generator."""
    model_config = ConfigDict(populate_by_name=True)

    type: str = Field(..., description="Generator type, e.g. 'erdos_renyi'")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra generator params (e.g. {'p': 0.3})",
        validation_alias=AliasChoices("params", "parameters"),
    )
    sizes: list[int] = Field(..., description="Graph sizes to generate")
    count_per_size: int = Field(default=3, ge=1, description="Instances per size")
    why: str = Field(default="", description="Reason this generator was chosen")


class InstanceConfig(BaseModel):
    """Specifies which instances to generate."""
    generators: list[GeneratorConfig]
    custom_instances: list[dict[str, Any]] = Field(
        default_factory=list,
        description="User-uploaded instance dicts",
    )


class EvaluationPriorities(BaseModel):
    """What the user cares about most."""
    solution_quality: Priority = Priority.PRIMARY
    runtime: str = Field(default="secondary", description="May include hard ceiling")
    memory: Priority = Priority.NOT_A_CONCERN
    consistency: Priority = Priority.REPORT_ONLY


class ExecutionConfig(BaseModel):
    """Resource limits for benchmark runs."""
    timeout_seconds: float = Field(default=60.0, gt=0)
    runs_per_config: int = Field(default=5, ge=1)
    memory_limit_mb: int = Field(default=2048, gt=0)


class SolutionValidation(BaseModel):
    """Describes how to check feasibility and compute objectives."""
    feasibility_check: str = Field(
        default="",
        description="Human-readable description of the feasibility rule",
    )
    objective_function: str = Field(
        default="",
        description="Human-readable description of the objective",
    )


class BenchmarkConfig(BaseModel):
    """Top-level configuration produced by the Intake Agent."""
    problem_class: str
    problem_description: str = ""
    objective: Objective = Objective.MINIMIZE

    instance_config: InstanceConfig
    evaluation_priorities: EvaluationPriorities = Field(
        default_factory=EvaluationPriorities,
    )
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    solution_validation: SolutionValidation = Field(
        default_factory=SolutionValidation,
    )


class AlgorithmSpec(BaseModel):
    """Structured algorithm description extracted by IntakeAgent from papers or user input."""
    name: str = Field(..., description="Snake_case identifier, e.g. 'spectral_partitioning'")
    approach: str = Field(..., description="One-line summary of the algorithm's approach")
    complexity: str = Field(default="unknown", description="Time complexity, e.g. 'O(n^2)'")
    key_steps: list[str] = Field(
        default_factory=list,
        description="Pseudocode-like steps for the algorithm",
    )
    source: str = Field(default="user", description="Where this came from, e.g. 'paper.pdf'")


class IntakeResult(BaseModel):
    """Complete output from IntakeAgent: benchmark config + algorithm specs."""
    config: BenchmarkConfig
    algorithms: list[AlgorithmSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

class BenchmarkResult(BaseModel):
    """A single benchmark measurement (one algorithm × one instance × one run)."""
    algorithm_name: str
    instance_name: str
    instance_generator: str
    problem_size: int
    objective_value: Optional[float] = None
    wall_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    status: RunStatus = RunStatus.SUCCESS
    run_index: int = 0
    feasible: bool = True
    error_message: str = ""
