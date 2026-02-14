"""Tests for the benchmark execution engine."""

import pytest
import pandas as pd

from benchwarmer.algorithms.base import AlgorithmWrapper
from benchwarmer.config import BenchmarkConfig, GeneratorConfig, InstanceConfig
from benchwarmer.engine.runner import BenchmarkRunner


# ── Dummy algorithms ─────────────────────────────────────────────────

class DummyVertexCover(AlgorithmWrapper):
    name = "dummy_vc"

    def solve(self, instance, timeout=60.0):
        # Cover = all nodes (trivially feasible)
        return {"solution": {"vertices": list(instance["nodes"])}, "metadata": {}}


class FailingAlgorithm(AlgorithmWrapper):
    name = "fail_algo"

    def solve(self, instance, timeout=60.0):
        raise RuntimeError("intentional failure")


# ── Tests ────────────────────────────────────────────────────────────

class TestBenchmarkRunner:
    @pytest.fixture()
    def config(self):
        return BenchmarkConfig(
            problem_class="minimum_vertex_cover",
            objective="minimize",
            instance_config=InstanceConfig(
                generators=[
                    GeneratorConfig(type="erdos_renyi", sizes=[10], count_per_size=1, params={"p": 0.3}),
                ]
            ),
            execution_config={"timeout_seconds": 10, "runs_per_config": 2},
        )

    def test_run_produces_dataframe(self, config):
        runner = BenchmarkRunner(config)
        runner.register_algorithm(DummyVertexCover())
        df = runner.run()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 1 algo × 1 instance × 2 runs

    def test_schema_columns(self, config):
        runner = BenchmarkRunner(config)
        runner.register_algorithm(DummyVertexCover())
        df = runner.run()
        expected = {
            "algorithm_name", "instance_name", "instance_generator",
            "problem_size", "objective_value", "wall_time_seconds",
            "peak_memory_mb", "status", "run_index", "feasible",
            "error_message",
        }
        assert expected.issubset(set(df.columns))

    def test_feasibility_checked(self, config):
        runner = BenchmarkRunner(config)
        runner.register_algorithm(DummyVertexCover())
        df = runner.run()
        assert all(df["feasible"])  # all-nodes cover is always feasible

    def test_failing_algorithm(self, config):
        runner = BenchmarkRunner(config)
        runner.register_algorithm(FailingAlgorithm())
        df = runner.run()
        assert all(df["status"] == "error")
        assert all(df["error_message"].str.contains("intentional"))

    def test_no_algorithms_raises(self, config):
        runner = BenchmarkRunner(config)
        with pytest.raises(RuntimeError, match="No algorithms"):
            runner.run()

    def test_generate_instances(self, config):
        runner = BenchmarkRunner(config)
        instances = runner.generate_instances()
        assert len(instances) == 1  # 1 size × 1 count
        assert "nodes" in instances[0]
        assert "edges" in instances[0]
