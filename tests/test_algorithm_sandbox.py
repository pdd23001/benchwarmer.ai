"""Tests for the algorithm sandbox (no API key required)."""

import pytest

from benchwarmer.utils.algorithm_sandbox import (
    execute_algorithm_code,
    _default_smoke_instance,
)


class TestAlgorithmSandbox:
    """Deterministic tests for algorithm code execution and validation."""

    def test_valid_algorithm(self):
        """A properly defined AlgorithmWrapper subclass passes."""
        code = '''
class MyGreedy(AlgorithmWrapper):
    name = "test_greedy"
    def solve(self, instance, timeout=60.0):
        nodes = instance["nodes"]
        return {"solution": {"vertices": nodes[:2]}, "metadata": {}}
'''
        result = execute_algorithm_code(code, "minimum_vertex_cover")
        assert result["success"] is True
        assert result["name"] == "test_greedy"
        assert result["algorithm"] is not None
        assert "solution" in result["smoke_result"]

    def test_uses_random(self):
        """Code can use the pre-injected random module."""
        code = '''
class RandomAlgo(AlgorithmWrapper):
    name = "random_algo"
    def solve(self, instance, timeout=60.0):
        partition = [random.choice([0, 1]) for _ in instance["nodes"]]
        return {"solution": {"partition": partition}, "metadata": {}}
'''
        result = execute_algorithm_code(code, "maximum_cut")
        assert result["success"] is True

    def test_uses_collections(self):
        """Code can use defaultdict, deque, Counter."""
        code = '''
class CollectionsAlgo(AlgorithmWrapper):
    name = "collections_algo"
    def solve(self, instance, timeout=60.0):
        adj = defaultdict(set)
        for e in instance["edges"]:
            adj[e["source"]].add(e["target"])
        cover = list(adj.keys())[:2]
        return {"solution": {"vertex_cover": cover}, "metadata": {}}
'''
        result = execute_algorithm_code(code, "minimum_vertex_cover")
        assert result["success"] is True

    def test_uses_heapq(self):
        """Code can use heapq."""
        code = '''
class HeapAlgo(AlgorithmWrapper):
    name = "heap_algo"
    def solve(self, instance, timeout=60.0):
        degrees = [(-len(instance["edges"]), n) for n in instance["nodes"]]
        heapq.heapify(degrees)
        top = heapq.heappop(degrees)
        return {"solution": {"vertices": [top[1]]}, "metadata": {}}
'''
        result = execute_algorithm_code(code, "minimum_vertex_cover")
        assert result["success"] is True

    def test_no_subclass_found(self):
        """Code that doesn't define an AlgorithmWrapper subclass fails."""
        code = '''
def my_function():
    pass
'''
        result = execute_algorithm_code(code, "maximum_cut")
        assert result["success"] is False
        assert "No AlgorithmWrapper subclass" in result["error"]

    def test_syntax_error(self):
        """Code with a syntax error fails."""
        code = "class Foo("
        result = execute_algorithm_code(code, "maximum_cut")
        assert result["success"] is False
        assert "execution failed" in result["error"]

    def test_no_name(self):
        """Class without a name attribute fails."""
        code = '''
class BadAlgo(AlgorithmWrapper):
    def solve(self, instance, timeout=60.0):
        return {"solution": {"partition": [0]}, "metadata": {}}
'''
        result = execute_algorithm_code(code, "maximum_cut")
        assert result["success"] is False
        assert "name" in result["error"].lower()

    def test_solve_raises(self):
        """If solve() raises, the smoke test fails with a helpful error."""
        code = '''
class CrashAlgo(AlgorithmWrapper):
    name = "crash_algo"
    def solve(self, instance, timeout=60.0):
        raise RuntimeError("intentional crash")
'''
        result = execute_algorithm_code(code, "maximum_cut")
        assert result["success"] is False
        assert "Smoke test failed" in result["error"]

    def test_bad_return_type(self):
        """solve() returning a non-dict fails."""
        code = '''
class BadReturn(AlgorithmWrapper):
    name = "bad_return"
    def solve(self, instance, timeout=60.0):
        return [1, 2, 3]
'''
        result = execute_algorithm_code(code, "maximum_cut")
        assert result["success"] is False
        assert "must return a dict" in result["error"]

    def test_missing_solution_key(self):
        """solve() returning dict without 'solution' key fails."""
        code = '''
class MissingSolution(AlgorithmWrapper):
    name = "missing_solution"
    def solve(self, instance, timeout=60.0):
        return {"answer": [1, 2]}
'''
        result = execute_algorithm_code(code, "maximum_cut")
        assert result["success"] is False
        assert "solution" in result["error"]

    def test_custom_smoke_instance(self):
        """Can pass a custom smoke instance."""
        custom_instance = {
            "nodes": [0, 1],
            "edges": [{"source": 0, "target": 1, "weight": 1.0}],
            "metadata": {"generator": "test", "size": 2, "params": {}},
        }
        code = '''
class TinyAlgo(AlgorithmWrapper):
    name = "tiny_algo"
    def solve(self, instance, timeout=60.0):
        return {"solution": {"vertices": instance["nodes"]}, "metadata": {}}
'''
        result = execute_algorithm_code(
            code, "minimum_vertex_cover", smoke_instance=custom_instance
        )
        assert result["success"] is True
        assert result["smoke_result"]["solution"]["vertices"] == [0, 1]

    def test_default_smoke_instance(self):
        """Default smoke instance is a valid 5-node graph."""
        inst = _default_smoke_instance()
        assert len(inst["nodes"]) == 5
        assert len(inst["edges"]) == 6
        assert "metadata" in inst
