"""
Tests for Intake Agent tool functions.

These test the *deterministic* tool logic (classify, generators, validate)
WITHOUT making any API calls — no ANTHROPIC_API_KEY needed.
"""

import pytest

from benchwarmer.agents.tools import (
    classify_problem,
    dispatch_tool_call,
    get_generators_for_class,
    validate_config,
)


# ── classify_problem ─────────────────────────────────────────────────

class TestClassifyProblem:
    def test_exact_keyword_match(self):
        results = classify_problem("I need a minimum vertex cover for my graph")
        assert results[0]["name"] == "minimum_vertex_cover"
        assert results[0]["confidence"] >= 0.8

    def test_domain_description(self):
        results = classify_problem(
            "I want to place sensors to cover all corridors in a building"
        )
        # "sensor placement" is a keyword for min vertex cover
        assert results[0]["name"] == "minimum_vertex_cover"

    def test_max_cut_keywords(self):
        results = classify_problem(
            "Partition the graph to maximize edges between the two sets"
        )
        top = results[0]
        assert top["name"] == "maximum_cut"
        assert top["confidence"] > 0

    def test_returns_all_classes(self):
        results = classify_problem("some random problem")
        names = {r["name"] for r in results}
        assert "minimum_vertex_cover" in names
        assert "maximum_cut" in names

    def test_sorted_by_confidence(self):
        results = classify_problem("vertex cover sensors")
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)


# ── get_generators_for_class ─────────────────────────────────────────

class TestGetGenerators:
    def test_known_class(self):
        gens = get_generators_for_class("minimum_vertex_cover")
        assert isinstance(gens, list)
        assert len(gens) >= 2
        assert "erdos_renyi" in gens

    def test_unknown_class(self):
        with pytest.raises(ValueError, match="Unknown problem class"):
            get_generators_for_class("does_not_exist")


# ── validate_config ──────────────────────────────────────────────────

class TestValidateConfig:
    def test_valid_config(self):
        cfg = {
            "problem_class": "minimum_vertex_cover",
            "objective": "minimize",
            "instance_config": {
                "generators": [
                    {"type": "erdos_renyi", "sizes": [20, 50]}
                ]
            },
        }
        result = validate_config(cfg)
        assert result["valid"] is True
        assert "config" in result

    def test_invalid_config_missing_required(self):
        result = validate_config({"objective": "minimize"})
        assert result["valid"] is False
        assert "errors" in result

    def test_invalid_generator_sizes(self):
        cfg = {
            "problem_class": "maximum_cut",
            "instance_config": {
                "generators": [
                    {"type": "erdos_renyi"}
                    # missing required 'sizes'
                ]
            },
        }
        result = validate_config(cfg)
        assert result["valid"] is False


# ── dispatch_tool_call ───────────────────────────────────────────────

class TestDispatch:
    def test_classify(self):
        import json
        raw = dispatch_tool_call("classify_problem", {"description": "vertex cover"})
        data = json.loads(raw)
        assert isinstance(data, list)

    def test_get_generators(self):
        import json
        raw = dispatch_tool_call("get_generators", {"problem_class": "maximum_cut"})
        data = json.loads(raw)
        assert isinstance(data, list)

    def test_validate(self):
        import json
        raw = dispatch_tool_call("validate_config", {
            "config": {
                "problem_class": "maximum_cut",
                "instance_config": {
                    "generators": [{"type": "erdos_renyi", "sizes": [10]}]
                },
            }
        })
        data = json.loads(raw)
        assert data["valid"] is True

    def test_unknown_tool(self):
        import json
        raw = dispatch_tool_call("nonexistent", {})
        data = json.loads(raw)
        assert "error" in data
