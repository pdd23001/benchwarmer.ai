"""
Modal Sandbox execution engine.

Runs each (algorithm × instance × run) concurrently in isolated
Modal sandbox containers for faster, parallelized benchmarking.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import textwrap
from typing import Any

import pandas as pd

from benchwarmer.config import (
    BenchmarkConfig,
    BenchmarkResult,
    RunStatus,
)
from benchwarmer.generators import get_generator
from benchwarmer.problem_classes.registry import get_problem_class

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The worker script that runs INSIDE each Modal sandbox.
# It is completely self-contained — no imports from benchwarmer.
# ---------------------------------------------------------------------------

WORKER_SCRIPT = textwrap.dedent('''\
import json
import sys
import time
import tracemalloc
import traceback

def main():
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
# Modal image definition (lazy — only built when needed)
# ---------------------------------------------------------------------------

_modal_image = None


def _get_modal_image():
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


# ---------------------------------------------------------------------------
# Algorithm source code extraction
# ---------------------------------------------------------------------------

def _get_algo_source(algo) -> str:
    """
    Get self-contained source code for an algorithm instance.

    For dynamically generated algorithms (from the Implementation Agent),
    the source is stored on the instance as `_source_code`.
    For file-backed classes, we use inspect.getsource().
    """
    cls = type(algo)

    # 1) Check for stored source (dynamically created via exec)
    if hasattr(algo, "_source_code"):
        source = algo._source_code
    else:
        # 2) Fall back to inspect for file-backed classes
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            raise ValueError(
                f"Cannot extract source for {cls.__name__}. "
                "Modal mode requires algorithm classes whose source is inspectable "
                "or created via the Implementation Agent."
            )

    # Prepend common imports the algorithm might need
    # (must match what algorithm_sandbox.py provides locally)
    preamble = textwrap.dedent("""\
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

    # Include the ABC base class stub so the algo can inherit from it
    base_stub = textwrap.dedent("""\
    class AlgorithmWrapper:
        name = "unnamed"
        def solve(self, instance, timeout=60.0):
            raise NotImplementedError
    """)

    return preamble + "\n" + base_stub + "\n" + source


# ---------------------------------------------------------------------------
# ModalRunner
# ---------------------------------------------------------------------------

class ModalRunner:
    """
    Benchmark runner that executes algorithms in Modal sandboxes concurrently.

    Drop-in replacement for BenchmarkRunner when execution_mode="modal".

    Usage
    -----
    >>> runner = ModalRunner(config)
    >>> runner.register_algorithm(my_algo)
    >>> df = asyncio.run(runner.run())
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        modal_token_id: str | None = None,
        modal_token_secret: str | None = None,
    ) -> None:
        self.config = config
        self.algorithms: list[Any] = []
        self._instances: list[dict] = []

        # BYOK: if the caller provides Modal credentials, inject them
        # into the environment so the Modal SDK picks them up.
        import os
        if modal_token_id and modal_token_secret:
            os.environ["MODAL_TOKEN_ID"] = modal_token_id
            os.environ["MODAL_TOKEN_SECRET"] = modal_token_secret
            logger.info("Using per-request Modal credentials (BYOK)")
        elif not os.environ.get("MODAL_TOKEN_ID"):
            logger.info("Using operator Modal credentials (from `modal token set`)")

    def register_algorithm(self, algorithm) -> None:
        """Add an AlgorithmWrapper instance to the benchmark."""
        self.algorithms.append(algorithm)

    def generate_instances(self) -> list[dict]:
        """Build all graph instances according to the config."""
        instances: list[dict] = []
        for gen_cfg in self.config.instance_config.generators:
            GenClass = get_generator(gen_cfg.type)
            gen = GenClass()
            for size in gen_cfg.sizes:
                for i in range(gen_cfg.count_per_size):
                    inst = gen.generate(size, **gen_cfg.params)
                    inst["instance_name"] = f"{gen_cfg.type}_n{size}_{i}"
                    instances.append(inst)
        for idx, inst in enumerate(self.config.instance_config.custom_instances):
            if "instance_name" not in inst:
                inst["instance_name"] = f"custom_{idx}"
            instances.append(inst)
        self._instances = instances
        logger.info("Generated %d instances", len(instances))
        return instances

    async def run(self) -> pd.DataFrame:
        """
        Execute the full benchmark concurrently on Modal and return results.

        Architecture: one sandbox per algorithm, all instances × runs
        execute sequentially within that sandbox.  Different algorithms
        run in parallel across sandboxes.
        """
        import modal

        if not self.algorithms:
            raise RuntimeError("No algorithms registered.")

        if not self._instances:
            self.generate_instances()

        # Load problem class for validation
        problem_cls = None
        try:
            problem_cls = get_problem_class(self.config.problem_class)
        except ValueError:
            logger.warning(
                "Problem class '%s' not found — skipping validation",
                self.config.problem_class,
            )

        timeout = self.config.execution_config.timeout_seconds
        runs = self.config.execution_config.runs_per_config

        # Pre-extract source for each algorithm
        algo_sources: dict[str, str] = {}
        for algo in self.algorithms:
            algo_sources[algo.name] = _get_algo_source(algo)

        # Build one task per algorithm (each runs all instances × runs)
        tasks = []
        for algo in self.algorithms:
            tasks.append(
                self._run_algorithm_in_sandbox(
                    algo_name=algo.name,
                    algo_source=algo_sources[algo.name],
                    instances=self._instances,
                    runs=runs,
                    timeout=timeout,
                )
            )

        logger.info(
            "Launching %d sandbox(es) on Modal (%d algo × %d inst × %d runs)…",
            len(tasks), len(self.algorithms), len(self._instances), runs,
        )

        # Run all algorithms concurrently (one sandbox each)
        all_algo_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Assemble records
        records: list[BenchmarkResult] = []
        for algo_idx, (algo, algo_results) in enumerate(
            zip(self.algorithms, all_algo_results)
        ):
            if isinstance(algo_results, Exception):
                # Entire sandbox failed — mark all runs as errors
                for inst in self._instances:
                    for run_idx in range(runs):
                        records.append(BenchmarkResult(
                            algorithm_name=algo.name,
                            instance_name=inst.get("instance_name", "unknown"),
                            instance_generator=inst.get("metadata", {}).get("generator", "custom"),
                            problem_size=inst.get("metadata", {}).get("size", len(inst.get("nodes", []))),
                            objective_value=None,
                            wall_time_seconds=0.0,
                            peak_memory_mb=0.0,
                            status=RunStatus.ERROR,
                            run_index=run_idx,
                            feasible=False,
                            error_message=f"Sandbox error: {algo_results}",
                        ))
                continue

            # algo_results is a list of (raw_result, inst, run_idx) tuples
            for raw, inst, run_idx in algo_results:
                status_str = raw.get("status", "error")
                try:
                    status = RunStatus(status_str)
                except ValueError:
                    status = RunStatus.ERROR

                objective_value = None
                feasible = True

                if status == RunStatus.SUCCESS and raw.get("solution") is not None:
                    if problem_cls is not None:
                        val = problem_cls.validate_solution(inst, raw["solution"])
                        feasible = val.get("feasible", True)
                        objective_value = problem_cls.compute_objective(
                            inst, raw["solution"],
                        )
                elif status == RunStatus.SUCCESS:
                    status = RunStatus.ERROR
                    raw["error"] = "Algorithm returned None"

                records.append(BenchmarkResult(
                    algorithm_name=algo.name,
                    instance_name=inst.get("instance_name", "unknown"),
                    instance_generator=inst.get("metadata", {}).get("generator", "custom"),
                    problem_size=inst.get("metadata", {}).get("size", len(inst.get("nodes", []))),
                    objective_value=objective_value,
                    wall_time_seconds=round(raw.get("wall_time", 0.0), 6),
                    peak_memory_mb=round(raw.get("peak_memory_mb", 0.0), 3),
                    status=status,
                    run_index=run_idx,
                    feasible=feasible,
                    error_message=raw.get("error", ""),
                ))

        df = pd.DataFrame([r.model_dump() for r in records])
        logger.info("Modal benchmark complete — %d results collected", len(df))
        return df

    async def _run_algorithm_in_sandbox(
        self,
        algo_name: str,
        algo_source: str,
        instances: list[dict],
        runs: int,
        timeout: float,
    ) -> list[tuple[dict, dict, int]]:
        """
        Run ALL instances × runs for a single algorithm inside ONE sandbox.

        Returns a list of (raw_result, instance, run_index) tuples.
        """
        import modal

        image = _get_modal_image()
        app = await modal.App.lookup.aio("benchwarmer-runner", create_if_missing=True)

        # Total time budget: (instances × runs × timeout) + grace
        total_timeout = int(len(instances) * runs * timeout) + 120

        # Create sandbox with a keep-alive entrypoint
        sb = await modal.Sandbox.create.aio(
            "sleep", "infinity",
            image=image,
            timeout=total_timeout,
            app=app,
        )

        results: list[tuple[dict, dict, int]] = []

        try:
            # Write the worker script (once)
            async with await sb.open.aio("/tmp/worker.py", "w") as f:
                await f.write.aio(WORKER_SCRIPT)

            # Write the algorithm source (once)
            async with await sb.open.aio("/tmp/algo_source.py", "w") as f:
                await f.write.aio(algo_source)

            logger.info(
                "Sandbox for '%s' ready — running %d instance(s) × %d run(s)",
                algo_name, len(instances), runs,
            )

            # Run each instance × run sequentially inside this sandbox
            for inst in instances:
                for run_idx in range(runs):
                    raw = await self._exec_single_run(
                        sb, inst, timeout, algo_name, run_idx,
                    )
                    results.append((raw, inst, run_idx))

        except Exception as exc:
            logger.error("Sandbox for '%s' failed: %s", algo_name, exc)
            # Fill remaining results with errors
            done = len(results)
            total = len(instances) * runs
            for idx in range(done, total):
                inst_idx = idx // runs
                run_idx = idx % runs
                inst = instances[inst_idx] if inst_idx < len(instances) else {}
                results.append((
                    {
                        "solution": None,
                        "wall_time": 0.0,
                        "peak_memory_mb": 0.0,
                        "status": "error",
                        "error": f"Sandbox error: {exc}",
                    },
                    inst,
                    run_idx,
                ))
        finally:
            try:
                await sb.terminate.aio()
            except Exception:
                pass

        return results

    async def _exec_single_run(
        self,
        sb,
        instance: dict,
        timeout: float,
        algo_name: str,
        run_idx: int,
    ) -> dict:
        """Execute a single (instance × run) inside an already-running sandbox."""
        try:
            # Write instance data
            async with await sb.open.aio("/tmp/instance.json", "w") as f:
                await f.write.aio(json.dumps(instance))

            # Write run config
            async with await sb.open.aio("/tmp/run_config.json", "w") as f:
                await f.write.aio(json.dumps({"timeout": timeout}))

            # Execute the worker
            process = await sb.exec.aio("python3", "/tmp/worker.py")

            # Collect stdout
            stdout_lines = []
            async for line in process.stdout:
                stdout_lines.append(line)

            await process.wait.aio()

            # Parse result from stdout
            for line in stdout_lines:
                if line.startswith("__RESULT__"):
                    return json.loads(line[len("__RESULT__"):])

            # No result marker — collect stderr for debugging
            stderr_lines = []
            async for line in process.stderr:
                stderr_lines.append(line)
            stderr_str = "\n".join(stderr_lines)
            return {
                "solution": None,
                "wall_time": 0.0,
                "peak_memory_mb": 0.0,
                "status": "error",
                "error": f"No result marker in stdout. stderr: {stderr_str[:500]}",
            }

        except Exception as exc:
            return {
                "solution": None,
                "wall_time": 0.0,
                "peak_memory_mb": 0.0,
                "status": "error",
                "error": f"Exec error ({algo_name} run {run_idx}): {exc}",
            }
