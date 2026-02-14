"""Minimum Vertex Cover problem class."""

from __future__ import annotations


class ProblemClass:
    name = "minimum_vertex_cover"
    objective = "minimize"
    description = "Find the smallest subset of vertices that covers all edges"

    keywords = [
        "vertex cover", "node cover", "sensor placement",
        "cover all edges", "hitting set on edges",
        "minimum dominating", "facility coverage",
    ]

    @staticmethod
    def validate_solution(instance: dict, solution: dict) -> dict:
        """
        Check if every edge has at least one endpoint in the cover set.

        A Vertex Cover solution is: ``{"vertices": [0, 3, 7, ...]}``
        """
        cover_set = set(solution.get("solution", {}).get("vertices", []))
        uncovered = []
        for edge in instance.get("edges", []):
            if edge["source"] not in cover_set and edge["target"] not in cover_set:
                uncovered.append(edge)
        return {
            "feasible": len(uncovered) == 0,
            "uncovered_edges": len(uncovered),
            "cover_size": len(cover_set),
        }

    @staticmethod
    def compute_objective(instance: dict, solution: dict) -> float:
        """Return the cover size (lower is better)."""
        return float(len(solution.get("solution", {}).get("vertices", [])))

    @staticmethod
    def available_generators() -> list[str]:
        return ["erdos_renyi", "grid_2d", "planar_random", "barabasi_albert"]
