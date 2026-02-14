"""
Tool functions exposed to the Intake Agent via Claude tool_use.

These are *deterministic* functions that the LLM calls to query the
problem class registry and validate configs.  They can be unit-tested
without any API key.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from benchwarmer.config import BenchmarkConfig
from benchwarmer.problem_classes.registry import (
    get_problem_class,
    list_problem_classes,
)


# ------------------------------------------------------------------
# Tool 1: classify_problem
# ------------------------------------------------------------------

def classify_problem(description: str) -> list[dict[str, Any]]:
    """
    Score every registered problem class against a freeform description.

    Returns a list sorted by confidence (descending), e.g.::

        [
            {"name": "minimum_vertex_cover", "confidence": 0.82,
             "description": "Find the smallest subset …"},
            {"name": "maximum_cut", "confidence": 0.35, …},
        ]
    """
    description_lower = description.lower()
    classes = list_problem_classes()
    scored: list[dict[str, Any]] = []

    for cls in classes:
        best_kw_score = 0.0
        for keyword in cls["keywords"]:
            # Direct substring match → high confidence
            if keyword.lower() in description_lower:
                best_kw_score = max(best_kw_score, 0.90)
            else:
                # Fuzzy match
                ratio = SequenceMatcher(
                    None, keyword.lower(), description_lower,
                ).ratio()
                best_kw_score = max(best_kw_score, ratio)

        # Also match against the class description
        desc_ratio = SequenceMatcher(
            None, cls["description"].lower(), description_lower,
        ).ratio()

        confidence = round(max(best_kw_score, desc_ratio), 2)

        scored.append({
            "name": cls["name"],
            "confidence": confidence,
            "description": cls["description"],
            "objective": cls["objective"],
        })

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    return scored


# ------------------------------------------------------------------
# Tool 2: get_generators
# ------------------------------------------------------------------

def get_generators_for_class(problem_class: str) -> list[str]:
    """
    Return the list of available generator names for a problem class.

    Raises ``ValueError`` if the class is unknown.
    """
    cls = get_problem_class(problem_class)
    return cls.available_generators()


# ------------------------------------------------------------------
# Tool 3: validate_config
# ------------------------------------------------------------------

def validate_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a candidate BenchmarkConfig dict via Pydantic.

    Returns ``{"valid": True, "config": <parsed>}`` on success or
    ``{"valid": False, "errors": <details>}`` on failure.
    """
    try:
        cfg = BenchmarkConfig(**config_dict)
        return {"valid": True, "config": cfg.model_dump(mode="json")}
    except Exception as exc:
        return {"valid": False, "errors": str(exc)}


# ------------------------------------------------------------------
# Anthropic tool-use definitions (JSON schemas)
# ------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "classify_problem",
        "description": (
            "Classify a natural-language problem description against "
            "the registered problem classes. Returns a ranked list of "
            "candidates with confidence scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The user's freeform problem description.",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "get_generators",
        "description": (
            "Given a problem class name, return the list of available "
            "instance generators that can produce test instances for it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "problem_class": {
                    "type": "string",
                    "description": "Name of the problem class, e.g. 'minimum_vertex_cover'.",
                },
            },
            "required": ["problem_class"],
        },
    },
    {
        "name": "validate_config",
        "description": (
            "Validate a candidate BenchmarkConfig JSON object. "
            "Returns {valid: true, config: ...} or {valid: false, errors: ...}. "
            "IMPORTANT: The config must follow the EXACT schema shown below. "
            "Use 'params' (not 'parameters') for generator parameters. "
            "Each generator's 'params' is a flat dict of single values "
            "like {\"p\": 0.3}, NOT arrays. To test multiple parameter "
            "values, create separate generator entries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "A BenchmarkConfig object.",
                    "properties": {
                        "problem_class": {
                            "type": "string",
                            "description": "e.g. 'maximum_cut' or 'minimum_vertex_cover'",
                        },
                        "problem_description": {
                            "type": "string",
                            "description": "Brief human-readable description of the problem",
                        },
                        "objective": {
                            "type": "string",
                            "enum": ["minimize", "maximize"],
                        },
                        "instance_config": {
                            "type": "object",
                            "properties": {
                                "generators": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "description": "Generator name, e.g. 'erdos_renyi'",
                                            },
                                            "params": {
                                                "type": "object",
                                                "description": (
                                                    "Generator-specific params as SINGLE values. "
                                                    "E.g. {'p': 0.3} for erdos_renyi, "
                                                    "{'m': 3} for barabasi_albert. "
                                                    "To sweep multiple param values, create "
                                                    "separate generator entries."
                                                ),
                                            },
                                            "sizes": {
                                                "type": "array",
                                                "items": {"type": "integer"},
                                                "description": "List of graph sizes (node counts)",
                                            },
                                            "count_per_size": {
                                                "type": "integer",
                                                "description": "Number of instances per size (default: 3)",
                                            },
                                            "why": {
                                                "type": "string",
                                                "description": "Reason this generator was chosen",
                                            },
                                        },
                                        "required": ["type", "sizes"],
                                    },
                                },
                                "custom_instances": {
                                    "type": "array",
                                    "description": "Optional user-provided instances",
                                },
                            },
                            "required": ["generators"],
                        },
                        "execution_config": {
                            "type": "object",
                            "properties": {
                                "timeout_seconds": {"type": "number"},
                                "runs_per_config": {"type": "integer"},
                                "memory_limit_mb": {"type": "integer"},
                            },
                        },
                    },
                    "required": ["problem_class", "instance_config"],
                },
            },
            "required": ["config"],
        },
    },
]


# ------------------------------------------------------------------
# Dispatcher — routes tool calls from the Claude response
# ------------------------------------------------------------------

def dispatch_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Execute a tool call and return the JSON-encoded result."""
    if tool_name == "classify_problem":
        result = classify_problem(tool_input["description"])
    elif tool_name == "get_generators":
        result = get_generators_for_class(tool_input["problem_class"])
    elif tool_name == "validate_config":
        result = validate_config(tool_input["config"])
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, indent=2)
