"""Tests for Modal runner source serialization (no Modal account needed)."""

from __future__ import annotations
import pytest
from benchwarmer.algorithms.base import AlgorithmWrapper
from benchwarmer.engine.modal_runner import _get_algo_source, WORKER_SCRIPT


# ── Fixtures ──────────────────────────────────────────────────────────────

class DummyAlgorithm(AlgorithmWrapper):
    name = "dummy_modal_test"

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        partition = [0] * len(instance["nodes"])
        return {"solution": {"partition": partition}, "metadata": {}}


# ── Tests ─────────────────────────────────────────────────────────────────

class TestAlgoSourceExtraction:
    """Verify algorithm source code can be serialized for sandbox execution."""

    def test_extract_source_contains_class(self):
        algo = DummyAlgorithm()
        source = _get_algo_source(algo)
        assert "class DummyAlgorithm" in source
        assert "def solve" in source

    def test_extracted_source_is_executable(self):
        """The extracted source should be exec()-able in isolation."""
        algo = DummyAlgorithm()
        source = _get_algo_source(algo)

        namespace = {}
        exec(source, namespace)  # Should not raise

        # Should find the class
        assert "DummyAlgorithm" in namespace
        cls = namespace["DummyAlgorithm"]
        instance = cls()
        assert instance.name == "dummy_modal_test"

    def test_extracted_source_solve_works(self):
        """The extracted algo should actually solve a problem."""
        algo = DummyAlgorithm()
        source = _get_algo_source(algo)

        namespace = {}
        exec(source, namespace)

        cls = namespace["DummyAlgorithm"]
        instance = cls()

        test_graph = {
            "nodes": [0, 1, 2],
            "edges": [{"source": 0, "target": 1}],
        }
        result = instance.solve(test_graph)
        assert result["solution"]["partition"] == [0, 0, 0]

    def test_source_includes_base_stub(self):
        """The extracted source should include an AlgorithmWrapper stub."""
        algo = DummyAlgorithm()
        source = _get_algo_source(algo)
        assert "class AlgorithmWrapper" in source

    def test_source_includes_common_imports(self):
        """The extracted source should include common imports."""
        algo = DummyAlgorithm()
        source = _get_algo_source(algo)
        assert "import random" in source
        assert "import math" in source


class TestWorkerScript:
    """Verify the worker script template is valid Python."""

    def test_worker_script_syntax(self):
        """The worker script should be valid Python syntax."""
        compile(WORKER_SCRIPT, "<worker>", "exec")  # Should not raise

    def test_worker_script_has_result_marker(self):
        """The worker script must use __RESULT__ marker for output."""
        assert "__RESULT__" in WORKER_SCRIPT
