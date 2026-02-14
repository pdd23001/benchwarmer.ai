"""
Benchmark execution engine.

Generates instances, runs algorithms, collects metrics, validates
solutions, and produces a pandas DataFrame of results.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
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
# Helper that runs inside a worker process
# ---------------------------------------------------------------------------

def _run_one(
    algorithm_wrapper,
    instance: dict,
    timeout: float,
) -> dict:
    """Run a single (algorithm × instance) and return raw measurements."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = algorithm_wrapper.solve(instance, timeout=timeout)
        wall_time = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "solution": result,
            "wall_time": wall_time,
            "peak_memory_mb": peak_mem / (1024 * 1024),
            "status": RunStatus.SUCCESS,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        wall_time = time.perf_counter() - t0
        try:
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        except Exception:
            peak_mem = 0
        return {
            "solution": None,
            "wall_time": wall_time,
            "peak_memory_mb": peak_mem / (1024 * 1024),
            "status": RunStatus.ERROR,
            "error": str(exc),
        }


class BenchmarkRunner:
    """
    Core benchmark execution engine (no LLM dependency).

    Usage
    -----
    >>> runner = BenchmarkRunner(config)
    >>> runner.register_algorithm(my_algo)
    >>> df = runner.run()
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.algorithms: list[Any] = []
        self._instances: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_algorithm(self, algorithm) -> None:
        """Add an :class:`AlgorithmWrapper` instance to the benchmark."""
        self.algorithms.append(algorithm)

    def generate_instances(self) -> list[dict]:
        """
        Build all graph instances according to the config.

        Returns a list of dicts, each augmented with an ``instance_name``
        key for later identification.
        """
        instances: list[dict] = []

        for gen_cfg in self.config.instance_config.generators:
            GenClass = get_generator(gen_cfg.type)
            gen = GenClass()
            for size in gen_cfg.sizes:
                for i in range(gen_cfg.count_per_size):
                    inst = gen.generate(size, **gen_cfg.params)
                    inst["instance_name"] = f"{gen_cfg.type}_n{size}_{i}"
                    instances.append(inst)

        # Append any custom instances
        for idx, inst in enumerate(self.config.instance_config.custom_instances):
            if "instance_name" not in inst:
                inst["instance_name"] = f"custom_{idx}"
            instances.append(inst)

        self._instances = instances
        logger.info("Generated %d instances", len(instances))
        return instances

    def run(self, parallel: bool = False) -> pd.DataFrame:
        """
        Execute the full benchmark and return a results DataFrame.

        Parameters
        ----------
        parallel : bool
            If True, run each (algorithm × instance × run) in a separate
            process.  Defaults to False (sequential) for simplicity and
            easier debugging.
        """
        if not self.algorithms:
            raise RuntimeError("No algorithms registered. Call register_algorithm() first.")

        if not self._instances:
            self.generate_instances()

        # Try to load problem class for solution validation
        problem_cls = None
        try:
            problem_cls = get_problem_class(self.config.problem_class)
        except ValueError:
            logger.warning(
                "Problem class '%s' not found — skipping solution validation",
                self.config.problem_class,
            )

        timeout = self.config.execution_config.timeout_seconds
        runs = self.config.execution_config.runs_per_config

        records: list[BenchmarkResult] = []

        total = len(self.algorithms) * len(self._instances) * runs
        done = 0

        for algo in self.algorithms:
            for inst in self._instances:
                for run_idx in range(runs):
                    done += 1
                    inst_name = inst.get("instance_name", "unknown")
                    gen_name = inst.get("metadata", {}).get("generator", "custom")
                    size = inst.get("metadata", {}).get("size", len(inst.get("nodes", [])))

                    logger.info(
                        "[%d/%d] %s × %s (run %d)",
                        done, total, algo.name, inst_name, run_idx,
                    )

                    # --- execute ---
                    if parallel:
                        raw = self._run_parallel(algo, inst, timeout)
                    else:
                        raw = _run_one(algo, inst, timeout)

                    # --- validate ---
                    objective_value = None
                    feasible = True
                    if raw["status"] == RunStatus.SUCCESS and raw["solution"] is not None:
                        if problem_cls is not None:
                            val = problem_cls.validate_solution(inst, raw["solution"])
                            feasible = val.get("feasible", True)
                            objective_value = problem_cls.compute_objective(
                                inst, raw["solution"],
                            )
                    elif raw["status"] == RunStatus.SUCCESS:
                        raw["status"] = RunStatus.ERROR
                        raw["error"] = "Algorithm returned None"

                    records.append(BenchmarkResult(
                        algorithm_name=algo.name,
                        instance_name=inst_name,
                        instance_generator=gen_name,
                        problem_size=size,
                        objective_value=objective_value,
                        wall_time_seconds=round(raw["wall_time"], 6),
                        peak_memory_mb=round(raw["peak_memory_mb"], 3),
                        status=raw["status"],
                        run_index=run_idx,
                        feasible=feasible,
                        error_message=raw.get("error", ""),
                    ))

        df = pd.DataFrame([r.model_dump() for r in records])
        logger.info("Benchmark complete — %d results collected", len(df))
        return df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _run_parallel(algo, inst: dict, timeout: float) -> dict:
        """Run in a subprocess with hard timeout enforcement."""
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_one, algo, inst, timeout)
            try:
                return future.result(timeout=timeout + 5)  # small grace period
            except FuturesTimeout:
                return {
                    "solution": None,
                    "wall_time": timeout,
                    "peak_memory_mb": 0.0,
                    "status": RunStatus.TIMEOUT,
                    "error": f"Hard timeout after {timeout}s",
                }
            except Exception as exc:
                return {
                    "solution": None,
                    "wall_time": 0.0,
                    "peak_memory_mb": 0.0,
                    "status": RunStatus.ERROR,
                    "error": str(exc),
                }
