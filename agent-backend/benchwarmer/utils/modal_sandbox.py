"""
Modal Sandbox — executes LLM-generated algorithm code inside a Modal
sandbox container and smoke-tests it remotely.

This ensures the generated code works in the exact same environment where
benchmarks will run (Modal sandbox with Python 3.12 + scientific deps).
Falls back to local execution if Modal is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import textwrap
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared preamble & image (also used by modal_runner.py)
# ---------------------------------------------------------------------------

ALGO_PREAMBLE = textwrap.dedent("""\
import random
import math
import itertools
import collections
import heapq
import functools
import copy
from abc import ABC, abstractmethod
from collections import defaultdict, deque, Counter
from typing import Any, Optional

inf = float("inf")

try:
    import networkx as nx
except ImportError:
    pass
try:
    import numpy as np
except ImportError:
    pass
try:
    import scipy
    from scipy import optimize, sparse
except ImportError:
    pass
try:
    import cvxpy as cp
except ImportError:
    pass
""")

ALGO_BASE_STUB = textwrap.dedent("""\
class AlgorithmWrapper:
    name = "unnamed"
    def solve(self, instance, timeout=60.0):
        raise NotImplementedError
""")

_modal_image = None


def get_modal_image():
    """Build (or reuse) the Modal image with scientific deps pre-installed."""
    global _modal_image
    if _modal_image is not None:
        return _modal_image

    import modal
    _modal_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("networkx", "numpy", "pandas", "scipy", "cvxpy")
    )
    return _modal_image


def wrap_algo_source(code: str) -> str:
    """Wrap raw algorithm code with preamble + base class stub."""
    return ALGO_PREAMBLE + "\n" + ALGO_BASE_STUB + "\n" + code


# ---------------------------------------------------------------------------
# Benchmark worker script (runs inside sandbox for actual benchmark runs)
# Also used by modal_runner.py
# ---------------------------------------------------------------------------

BENCHMARK_WORKER_SCRIPT = textwrap.dedent('''\
import json
import sys
import time
import tracemalloc
import traceback
import types

def _inject_benchwarmer_stub():
    """Inject minimal benchwarmer package so 'from benchwarmer.algorithms.base import AlgorithmWrapper' works."""
    class AlgorithmWrapper:
        name = "unnamed"
        def solve(self, instance, timeout=60.0):
            raise NotImplementedError
    base = types.ModuleType("base")
    base.AlgorithmWrapper = AlgorithmWrapper
    algorithms = types.ModuleType("algorithms")
    algorithms.base = base
    benchwarmer = types.ModuleType("benchwarmer")
    benchwarmer.algorithms = algorithms
    sys.modules["benchwarmer"] = benchwarmer
    sys.modules["benchwarmer.algorithms"] = algorithms
    sys.modules["benchwarmer.algorithms.base"] = base

def main():
    _inject_benchwarmer_stub()
    # Read inputs from files written into the sandbox
    with open("/tmp/instance.json", "r") as f:
        instance = json.load(f)
    with open("/tmp/algo_source.py", "r") as f:
        algo_source = f.read()
    with open("/tmp/run_config.json", "r") as f:
        run_config = json.load(f)

    timeout = run_config.get("timeout", 60.0)

    # Execute the algorithm source to get the class
    namespace = {}
    try:
        exec(algo_source, namespace)
    except Exception as e:
        result = {
            "solution": None,
            "wall_time": 0.0,
            "peak_memory_mb": 0.0,
            "status": "error",
            "error": f"Failed to load algorithm: {e}",
        }
        print("__RESULT__" + json.dumps(result))
        return

    # Find the algorithm class (look for classes with a 'solve' method)
    # Skip the AlgorithmWrapper base class stub — we want the SUBCLASS.
    algo_class = None
    for name, obj in namespace.items():
        if (isinstance(obj, type)
            and hasattr(obj, "solve")
            and name != "ABC"
            and name != "AlgorithmWrapper"
            and not name.startswith("_")):
            algo_class = obj
            break

    if algo_class is None:
        result = {
            "solution": None,
            "wall_time": 0.0,
            "peak_memory_mb": 0.0,
            "status": "error",
            "error": "No algorithm class with solve() found in source",
        }
        print("__RESULT__" + json.dumps(result))
        return

    # Instantiate and run
    try:
        algo_instance = algo_class()
    except Exception as e:
        result = {
            "solution": None,
            "wall_time": 0.0,
            "peak_memory_mb": 0.0,
            "status": "error",
            "error": f"Failed to instantiate algorithm: {e}",
        }
        print("__RESULT__" + json.dumps(result))
        return

    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        solution = algo_instance.solve(instance, timeout=timeout)
        wall_time = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result = {
            "solution": solution,
            "wall_time": wall_time,
            "peak_memory_mb": peak_mem / (1024 * 1024),
            "status": "success",
            "error": "",
        }
    except Exception as e:
        wall_time = time.perf_counter() - t0
        try:
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        except Exception:
            peak_mem = 0
        result = {
            "solution": None,
            "wall_time": wall_time,
            "peak_memory_mb": peak_mem / (1024 * 1024),
            "status": "error",
            "error": str(e),
        }

    print("__RESULT__" + json.dumps(result))

if __name__ == "__main__":
    main()
''')


# ---------------------------------------------------------------------------
# Smoke-test script that runs INSIDE the Modal sandbox
# ---------------------------------------------------------------------------

SMOKE_TEST_SCRIPT = textwrap.dedent('''\
import json
import sys
import traceback
import types

def _inject_benchwarmer_stub():
    """Inject minimal benchwarmer package so 'from benchwarmer.algorithms.base import AlgorithmWrapper' works."""
    class AlgorithmWrapper:
        name = "unnamed"
        def solve(self, instance, timeout=60.0):
            raise NotImplementedError
    base = types.ModuleType("base")
    base.AlgorithmWrapper = AlgorithmWrapper
    algorithms = types.ModuleType("algorithms")
    algorithms.base = base
    benchwarmer = types.ModuleType("benchwarmer")
    benchwarmer.algorithms = algorithms
    sys.modules["benchwarmer"] = benchwarmer
    sys.modules["benchwarmer.algorithms"] = algorithms
    sys.modules["benchwarmer.algorithms.base"] = base

def main():
    _inject_benchwarmer_stub()
    # Read inputs
    with open("/tmp/algo_source.py", "r") as f:
        algo_source = f.read()
    with open("/tmp/smoke_instance.json", "r") as f:
        smoke_instance = json.load(f)

    # Execute algorithm source
    namespace = {}
    try:
        exec(algo_source, namespace)
    except Exception as e:
        result = {
            "success": False,
            "error": f"Code execution failed: {e}",
            "traceback": traceback.format_exc(),
        }
        print("__RESULT__" + json.dumps(result))
        return

    # Find the AlgorithmWrapper subclass (skip the stub)
    algo_class = None
    for name, obj in namespace.items():
        if (isinstance(obj, type)
            and hasattr(obj, "solve")
            and name != "ABC"
            and name != "AlgorithmWrapper"
            and not name.startswith("_")):
            algo_class = obj
            break

    if algo_class is None:
        result = {
            "success": False,
            "error": (
                "No AlgorithmWrapper subclass found in the generated code. "
                "The code must define a class that extends AlgorithmWrapper "
                "with a `name` attribute and a `solve()` method."
            ),
            "traceback": "",
        }
        print("__RESULT__" + json.dumps(result))
        return

    # Instantiate
    try:
        algo_instance = algo_class()
    except Exception as e:
        result = {
            "success": False,
            "error": f"Failed to instantiate {algo_class.__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        print("__RESULT__" + json.dumps(result))
        return

    # Verify name
    algo_name = getattr(algo_instance, "name", "unnamed")
    if algo_name == "unnamed":
        result = {
            "success": False,
            "error": (
                f"Class {algo_class.__name__} must set a `name` attribute "
                f"(got \\'{algo_name}\\')."
            ),
            "traceback": "",
        }
        print("__RESULT__" + json.dumps(result))
        return

    # Smoke test — run solve() on tiny instance
    try:
        smoke_result = algo_instance.solve(smoke_instance, timeout=10.0)
    except Exception as e:
        result = {
            "success": False,
            "error": f"Smoke test failed — solve() raised: {e}",
            "traceback": traceback.format_exc(),
        }
        print("__RESULT__" + json.dumps(result))
        return

    # Validate result format
    if not isinstance(smoke_result, dict):
        result = {
            "success": False,
            "error": f"solve() must return a dict, got {type(smoke_result).__name__}",
            "traceback": "",
        }
        print("__RESULT__" + json.dumps(result))
        return

    if "solution" not in smoke_result:
        result = {
            "success": False,
            "error": (
                "solve() must return a dict with a \\'solution\\' key. "
                f"Got keys: {list(smoke_result.keys())}"
            ),
            "traceback": "",
        }
        print("__RESULT__" + json.dumps(result))
        return

    # Success!
    result = {
        "success": True,
        "name": algo_name,
        "class_name": algo_class.__name__,
        "smoke_result": smoke_result,
    }
    print("__RESULT__" + json.dumps(result))

if __name__ == "__main__":
    main()
''')


# ---------------------------------------------------------------------------
# Default smoke instance
# ---------------------------------------------------------------------------

def _default_smoke_instance() -> dict:
    """Return a small 5-node graph for smoke testing."""
    return {
        "nodes": [0, 1, 2, 3, 4],
        "edges": [
            {"source": 0, "target": 1, "weight": 1.0},
            {"source": 1, "target": 2, "weight": 1.0},
            {"source": 2, "target": 3, "weight": 1.0},
            {"source": 3, "target": 4, "weight": 1.0},
            {"source": 4, "target": 0, "weight": 1.0},
            {"source": 0, "target": 2, "weight": 1.0},
        ],
        "metadata": {
            "generator": "smoke_test",
            "size": 5,
            "params": {},
        },
    }


# ---------------------------------------------------------------------------
# Main entry point: execute_algorithm_code_modal
# ---------------------------------------------------------------------------

async def _execute_in_modal(
    code: str,
    problem_class: str,
    smoke_instance: dict,
    pool=None,
    algo_name_hint: str | None = None,
) -> dict[str, Any]:
    """
    Run the algorithm code inside a Modal sandbox and smoke-test it.

    Parameters
    ----------
    pool : SandboxPool | None
        If provided, sandbox is created via pool and kept alive.
    algo_name_hint : str | None
        Algorithm name hint for the pool key (used before we know the
        actual name from the smoke test).
    """
    algo_source = wrap_algo_source(code)

    if pool is not None:
        # Use the pool — sandbox stays alive after smoke test
        temp_name = algo_name_hint or f"_pending_{id(code)}"
        await pool.create_sandbox(temp_name)

        # Write smoke test script + algo source
        async with await pool.get_sandbox(temp_name).open.aio(
            "/tmp/smoke_test.py", "w"
        ) as f:
            await f.write.aio(SMOKE_TEST_SCRIPT)

        # Write algo source
        await pool.write_files(temp_name, BENCHMARK_WORKER_SCRIPT, algo_source)

        # Write smoke instance
        result = await pool.exec_in_sandbox(
            temp_name,
            "/tmp/smoke_test.py",
            input_files={"/tmp/smoke_instance.json": json.dumps(smoke_instance)},
        )

        # If smoke test passed and algo name differs from temp_name,
        # re-key the sandbox under the real name
        if result.get("success") and result.get("name") and result["name"] != temp_name:
            real_name = result["name"]
            if real_name != temp_name:
                pool._sandboxes[real_name] = pool._sandboxes.pop(temp_name)
                pool._ready[real_name] = pool._ready.pop(temp_name, True)
                logger.info(
                    "Re-keyed sandbox '%s' → '%s'", temp_name, real_name,
                )

        return result

    # No pool — ephemeral sandbox (create, use, destroy)
    import modal

    image = get_modal_image()
    app = await modal.App.lookup.aio("benchwarmer-dev", create_if_missing=True)

    sb = await modal.Sandbox.create.aio(
        "sleep", "infinity",
        image=image,
        timeout=120,
        app=app,
    )

    try:
        async with await sb.open.aio("/tmp/smoke_test.py", "w") as f:
            await f.write.aio(SMOKE_TEST_SCRIPT)

        async with await sb.open.aio("/tmp/algo_source.py", "w") as f:
            await f.write.aio(algo_source)

        async with await sb.open.aio("/tmp/smoke_instance.json", "w") as f:
            await f.write.aio(json.dumps(smoke_instance))

        process = await sb.exec.aio("python3", "/tmp/smoke_test.py")

        stdout_lines = []
        async for line in process.stdout:
            stdout_lines.append(line)

        await process.wait.aio()

        for line in stdout_lines:
            if line.startswith("__RESULT__"):
                return json.loads(line[len("__RESULT__"):])

        stderr_lines = []
        async for line in process.stderr:
            stderr_lines.append(line)
        stderr_str = "\n".join(stderr_lines)

        return {
            "success": False,
            "error": f"Sandbox produced no result. stderr: {stderr_str[:500]}",
            "traceback": "",
        }

    except Exception as exc:
        return {
            "success": False,
            "error": f"Modal sandbox error: {exc}",
            "traceback": "",
        }
    finally:
        try:
            await sb.terminate.aio()
        except Exception:
            pass


def execute_algorithm_code_modal(
    code: str,
    problem_class: str,
    smoke_instance: Optional[dict] = None,
    pool=None,
) -> dict[str, Any]:
    """
    Execute generated algorithm code inside a Modal sandbox, smoke-test it,
    and return the algorithm instance.

    Parameters
    ----------
    code : str
        Python code defining a class that extends AlgorithmWrapper.
    problem_class : str
        The problem class name (for context in error messages).
    smoke_instance : dict | None
        A small graph instance for smoke testing.
    pool : SandboxPool | None
        If provided, the sandbox is kept alive in the pool for later
        benchmark reuse.

    Returns
    -------
    dict
        On success: {"success": True, "algorithm": <instance>, "name": "...",
                     "smoke_result": {...}, "code": code}
        On failure: {"success": False, "error": "...", "traceback": "..."}
    """
    if smoke_instance is None:
        smoke_instance = _default_smoke_instance()

    logger.info("Running algorithm smoke test in Modal sandbox…")
    result = asyncio.run(
        _execute_in_modal(code, problem_class, smoke_instance, pool=pool)
    )

    if not result["success"]:
        logger.warning("Modal smoke test failed: %s", result["error"])
        return result

    logger.info(
        "Modal smoke test passed for '%s' — creating local instance",
        result["name"],
    )

    # Exec locally to get the algorithm instance for registration
    from benchwarmer.utils.algorithm_sandbox import execute_algorithm_code
    local_result = execute_algorithm_code(code, problem_class, smoke_instance)

    if not local_result["success"]:
        logger.warning(
            "Code passed Modal but failed locally: %s", local_result["error"]
        )
        return local_result

    # Attach source code for Modal benchmark execution
    local_result["algorithm"]._source_code = code

    return local_result

