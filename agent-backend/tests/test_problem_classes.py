"""Tests for problem classes and registry."""

import pytest

from benchwarmer.problem_classes.registry import (
    get_problem_class,
    list_problem_classes,
)


# ── Registry ─────────────────────────────────────────────────────────

def test_list_classes():
    classes = list_problem_classes()
    names = [c["name"] for c in classes]
    assert "maximum_cut" in names
    assert "minimum_vertex_cover" in names


def test_get_unknown():
    with pytest.raises(ValueError, match="Unknown problem class"):
        get_problem_class("nonexistent")


# ── Minimum Vertex Cover ────────────────────────────────────────────

class TestMinVertexCover:
    @pytest.fixture()
    def triangle(self):
        return {
            "nodes": [0, 1, 2],
            "edges": [
                {"source": 0, "target": 1},
                {"source": 1, "target": 2},
                {"source": 0, "target": 2},
            ],
        }

    def test_valid_cover(self, triangle):
        cls = get_problem_class("minimum_vertex_cover")
        sol = {"solution": {"vertices": [0, 1]}}
        result = cls.validate_solution(triangle, sol)
        assert result["feasible"] is True
        assert result["cover_size"] == 2

    def test_invalid_cover(self, triangle):
        cls = get_problem_class("minimum_vertex_cover")
        sol = {"solution": {"vertices": [0]}}
        result = cls.validate_solution(triangle, sol)
        assert result["feasible"] is False
        assert result["uncovered_edges"] >= 1

    def test_objective(self, triangle):
        cls = get_problem_class("minimum_vertex_cover")
        sol = {"solution": {"vertices": [0, 1]}}
        assert cls.compute_objective(triangle, sol) == 2.0


# ── Maximum Cut ──────────────────────────────────────────────────────

class TestMaxCut:
    @pytest.fixture()
    def triangle(self):
        return {
            "nodes": [0, 1, 2],
            "edges": [
                {"source": 0, "target": 1, "weight": 1.0},
                {"source": 1, "target": 2, "weight": 1.0},
                {"source": 0, "target": 2, "weight": 1.0},
            ],
        }

    def test_valid_partition(self, triangle):
        cls = get_problem_class("maximum_cut")
        sol = {"solution": {"partition": [0, 1, 0]}}
        result = cls.validate_solution(triangle, sol)
        assert result["feasible"] is True
        assert result["cut_size"] == 2.0  # edges (0,1) and (1,2) are cut

    def test_wrong_length(self, triangle):
        cls = get_problem_class("maximum_cut")
        sol = {"solution": {"partition": [0, 1]}}
        result = cls.validate_solution(triangle, sol)
        assert result["feasible"] is False

    def test_objective(self, triangle):
        cls = get_problem_class("maximum_cut")
        sol = {"solution": {"partition": [0, 1, 0]}}
        assert cls.compute_objective(triangle, sol) == 2.0
