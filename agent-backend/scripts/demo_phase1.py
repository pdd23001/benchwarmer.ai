"""
Quick demo / smoke test for Phase 1.

Run:  python scripts/demo_phase1.py
"""

import logging
import sys

sys.path.insert(0, ".")

from benchwarmer.algorithms.base import AlgorithmWrapper
from benchwarmer.config import BenchmarkConfig, GeneratorConfig, InstanceConfig
from benchwarmer.engine.runner import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ── Toy algorithms ─────────────────────────────────────────────────────

class GreedyVertexCover(AlgorithmWrapper):
    """Simple greedy: repeatedly pick the endpoint of any uncovered edge."""
    name = "greedy_vc"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        covered: set[int] = set()
        cover: list[int] = []
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                # pick the higher-degree heuristic (approx: pick first)
                cover.append(u)
                covered.add(u)
        # make sure everything is covered
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        return {"solution": {"vertices": cover}, "metadata": {"strategy": "greedy"}}


class RandomVertexCover(AlgorithmWrapper):
    """Picks endpoints of edges at random until all are covered."""
    name = "random_vc"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        import random
        covered: set[int] = set()
        cover: list[int] = []
        edges = list(instance["edges"])
        random.shuffle(edges)
        for edge in edges:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                chosen = random.choice([u, v])
                cover.append(chosen)
                covered.add(chosen)
        # second pass safety net
        for edge in instance["edges"]:
            u, v = edge["source"], edge["target"]
            if u not in covered and v not in covered:
                cover.append(u)
                covered.add(u)
        return {"solution": {"vertices": cover}, "metadata": {"strategy": "random"}}


# ── Run ──────────────────────────────────────────────────────────────

def main():
    config = BenchmarkConfig(
        problem_class="minimum_vertex_cover",
        problem_description="Demo: sensor coverage",
        objective="minimize",
        instance_config=InstanceConfig(
            generators=[
                GeneratorConfig(type="erdos_renyi", sizes=[20, 50], count_per_size=2, params={"p": 0.3}),
                GeneratorConfig(type="grid_2d", sizes=[25, 36], count_per_size=2),
            ]
        ),
        execution_config={"timeout_seconds": 30, "runs_per_config": 2},
    )

    runner = BenchmarkRunner(config)
    runner.register_algorithm(GreedyVertexCover())
    runner.register_algorithm(RandomVertexCover())

    df = runner.run()

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(df.to_string(index=False))

    # Quick summary
    print("\n── Summary ──")
    summary = (
        df[df["status"] == "success"]
        .groupby(["algorithm_name", "problem_size"])
        .agg(
            avg_objective=("objective_value", "mean"),
            avg_time_s=("wall_time_seconds", "mean"),
            avg_mem_mb=("peak_memory_mb", "mean"),
        )
        .round(4)
    )
    print(summary)


if __name__ == "__main__":
    main()
