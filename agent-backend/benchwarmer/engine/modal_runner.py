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

# Worker script is shared with modal_sandbox.py
from benchwarmer.utils.modal_sandbox import BENCHMARK_WORKER_SCRIPT as WORKER_SCRIPT

# ---------------------------------------------------------------------------
# Modal image definition (lazy — only built when needed)
# ---------------------------------------------------------------------------

def _get_modal_image():
    """Build (or reuse) the Modal image with scientific deps pre-installed."""
    from benchwarmer.utils.modal_sandbox import get_modal_image
    return get_modal_image()


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
    from benchwarmer.utils.modal_sandbox import wrap_algo_source

    cls = type(algo)

    # 1) Check for stored source (dynamically created via exec)
    if hasattr(algo, "_source_code"):
        return wrap_algo_source(algo._source_code)

    # 2) Fall back to inspect for file-backed classes
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        raise ValueError(
            f"Cannot extract source for {cls.__name__}. "
            "Modal mode requires algorithm classes whose source is inspectable "
            "or created via the Implementation Agent."
        )

    return wrap_algo_source(source)


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
        sandbox_pool=None,
    ) -> None:
        self.config = config
        self.algorithms: list[Any] = []
        self._instances: list[dict] = []
        self._pool = sandbox_pool  # Reuse dev sandboxes if provided

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
            # Check if sandbox pool has an existing sandbox for this algo
            existing_sb = None
            if self._pool is not None:
                existing_sb = self._pool.get_sandbox(algo.name)
                if existing_sb:
                    logger.info(
                        "Reusing development sandbox for '%s'", algo.name
                    )

            tasks.append(
                self._run_algorithm_in_sandbox(
                    algo_name=algo.name,
                    algo_source=algo_sources[algo.name],
                    instances=self._instances,
                    runs=runs,
                    timeout=timeout,
                    existing_sandbox=existing_sb,
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
        existing_sandbox=None,
    ) -> list[tuple[dict, dict, int]]:
        """
        Run ALL instances × runs for a single algorithm inside ONE sandbox.

        If existing_sandbox is provided (from SandboxPool), reuse it.
        Otherwise create a new ephemeral sandbox.

        Returns a list of (raw_result, instance, run_index) tuples.
        """
        owns_sandbox = existing_sandbox is None

        if owns_sandbox:
            import modal
            image = _get_modal_image()
            app = await modal.App.lookup.aio("benchwarmer-runner", create_if_missing=True)

            total_timeout = int(len(instances) * runs * timeout) + 120
            sb = await modal.Sandbox.create.aio(
                "sleep", "infinity",
                image=image,
                timeout=total_timeout,
                app=app,
            )
        else:
            sb = existing_sandbox

        results: list[tuple[dict, dict, int]] = []

        try:
            if owns_sandbox:
                # Fresh sandbox — write worker script + algo source
                async with await sb.open.aio("/tmp/worker.py", "w") as f:
                    await f.write.aio(WORKER_SCRIPT)
                async with await sb.open.aio("/tmp/algo_source.py", "w") as f:
                    await f.write.aio(algo_source)
            # else: pool sandbox already has worker.py + algo_source.py

            logger.info(
                "Sandbox for '%s' ready — running %d instance(s) × %d run(s)",
                algo_name, len(instances), runs,
            )

            for inst in instances:
                for run_idx in range(runs):
                    raw = await self._exec_single_run(
                        sb, inst, timeout, algo_name, run_idx,
                    )
                    results.append((raw, inst, run_idx))

        except Exception as exc:
            logger.error("Sandbox for '%s' failed: %s", algo_name, exc)
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
            if owns_sandbox:
                try:
                    await sb.terminate.aio()
                except Exception:
                    pass
            # Pool sandboxes are NOT terminated here — pool manages lifecycle

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
