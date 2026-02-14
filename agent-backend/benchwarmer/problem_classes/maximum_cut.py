"""Maximum Cut problem class."""

from __future__ import annotations


class ProblemClass:
    name = "maximum_cut"
    objective = "maximize"
    description = "Partition graph nodes into two sets to maximize edges between sets"

    keywords = [
        "max cut", "maximum cut", "graph partitioning",
        "cross-group connections", "bipartition",
    ]

    @staticmethod
    def validate_solution(instance: dict, solution: dict) -> dict:
        """
        Check if a solution is a valid partition (every node assigned).

        A Max-Cut solution is: ``{"partition": [0, 1, 0, 1, ...]}``
        where each entry is the set assignment for that node.
        """
        partition = solution.get("solution", {}).get("partition", [])
        nodes = instance.get("nodes", [])

        if len(partition) != len(nodes):
            return {
                "feasible": False,
                "reason": f"Partition length {len(partition)} != node count {len(nodes)}",
                "cut_size": 0,
            }

        # Check that partition only contains 0/1
        invalid = [v for v in partition if v not in (0, 1)]
        if invalid:
            return {
                "feasible": False,
                "reason": f"Partition values must be 0 or 1, got {set(invalid)}",
                "cut_size": 0,
            }

        return {"feasible": True, "cut_size": ProblemClass.compute_objective(instance, solution)}

    @staticmethod
    def compute_objective(instance: dict, solution: dict) -> float:
        """Return the cut size (number/weight of edges crossing the partition)."""
        partition = solution.get("solution", {}).get("partition", [])
        cut = 0.0
        for edge in instance.get("edges", []):
            u, v = edge["source"], edge["target"]
            if partition[u] != partition[v]:
                cut += edge.get("weight", 1.0)
        return cut

    @staticmethod
    def available_generators() -> list[str]:
        return ["erdos_renyi", "barabasi_albert", "planted_partition"]
