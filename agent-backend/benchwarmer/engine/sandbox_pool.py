"""
Sandbox Pool — manages long-lived Modal sandboxes that persist from
algorithm development through benchmark execution.

Each algorithm gets its own sandbox. During development, the sandbox is
created and the algorithm code is smoke-tested. The sandbox stays alive
so it can be reused for benchmark runs (modal mode), avoiding the cost
of recreating containers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SandboxPool:
    """
    Manages a pool of persistent Modal sandboxes, one per algorithm.

    Lifecycle:
      1. create_sandbox(algo_name) — spin up a sandbox and store it
      2. write_files(algo_name, ...) — write worker script + algo source
      3. exec_smoke_test(algo_name, ...) — run smoke test inside sandbox
      4. exec_benchmark_run(algo_name, ...) — run a benchmark trial
      5. teardown(algo_name) or teardown_all() — terminate sandboxes
    """

    def __init__(self) -> None:
        self._sandboxes: dict[str, Any] = {}  # algo_name → sandbox
        self._ready: dict[str, bool] = {}     # algo_name → worker+algo written
        self._app = None
        self._image = None

    async def _ensure_app(self):
        """Lazily create a Modal app and image."""
        if self._app is not None:
            return

        import modal
        from benchwarmer.utils.modal_sandbox import get_modal_image

        self._image = get_modal_image()
        self._app = await modal.App.lookup.aio(
            "benchwarmer-pool", create_if_missing=True,
        )

    async def create_sandbox(self, algo_name: str, timeout: int = 600) -> Any:
        """
        Create a new sandbox for the given algorithm.

        The sandbox uses `sleep infinity` as its entrypoint so it stays
        alive until explicitly terminated.
        """
        import modal

        await self._ensure_app()

        if algo_name in self._sandboxes:
            logger.info("Sandbox for '%s' already exists, reusing", algo_name)
            return self._sandboxes[algo_name]

        sb = await modal.Sandbox.create.aio(
            "sleep", "infinity",
            image=self._image,
            timeout=timeout,
            app=self._app,
        )
        self._sandboxes[algo_name] = sb
        self._ready[algo_name] = False
        logger.info("Created sandbox for '%s'", algo_name)
        return sb

    async def write_files(
        self,
        algo_name: str,
        worker_script: str,
        algo_source: str,
    ) -> None:
        """Write the worker script and algorithm source into the sandbox."""
        sb = self._sandboxes[algo_name]

        async with await sb.open.aio("/tmp/worker.py", "w") as f:
            await f.write.aio(worker_script)

        async with await sb.open.aio("/tmp/algo_source.py", "w") as f:
            await f.write.aio(algo_source)

        self._ready[algo_name] = True
        logger.debug("Wrote files to sandbox for '%s'", algo_name)

    async def exec_in_sandbox(
        self,
        algo_name: str,
        script_path: str,
        input_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a script inside the sandbox and parse __RESULT__ from stdout.

        Parameters
        ----------
        algo_name : str
            Name of the algorithm (must have an existing sandbox).
        script_path : str
            Path to the script to execute inside the sandbox.
        input_files : dict | None
            Additional files to write before execution.
            Maps file path → content string.

        Returns
        -------
        dict
            Parsed JSON from the __RESULT__ line in stdout, or error dict.
        """
        sb = self._sandboxes[algo_name]

        # Write any additional input files
        if input_files:
            for path, content in input_files.items():
                async with await sb.open.aio(path, "w") as f:
                    await f.write.aio(content)

        # Execute
        process = await sb.exec.aio("python3", script_path)

        # Collect stdout
        stdout_lines = []
        async for line in process.stdout:
            stdout_lines.append(line)

        await process.wait.aio()

        # Parse result
        for line in stdout_lines:
            if line.startswith("__RESULT__"):
                return json.loads(line[len("__RESULT__"):])

        # No result marker — collect stderr
        stderr_lines = []
        async for line in process.stderr:
            stderr_lines.append(line)
        stderr_str = "\n".join(stderr_lines)

        return {
            "success": False,
            "status": "error",
            "solution": None,
            "wall_time": 0.0,
            "peak_memory_mb": 0.0,
            "error": f"No __RESULT__ in stdout. stderr: {stderr_str[:500]}",
            "traceback": "",
        }

    def get_sandbox(self, algo_name: str) -> Any | None:
        """Return an existing sandbox, or None."""
        return self._sandboxes.get(algo_name)

    def is_ready(self, algo_name: str) -> bool:
        """Check if the sandbox has worker + algo source written."""
        return self._ready.get(algo_name, False)

    @property
    def algo_names(self) -> list[str]:
        """List of algorithms with active sandboxes."""
        return list(self._sandboxes.keys())

    async def teardown(self, algo_name: str) -> None:
        """Terminate a single sandbox."""
        sb = self._sandboxes.pop(algo_name, None)
        self._ready.pop(algo_name, None)
        if sb is not None:
            try:
                await sb.terminate.aio()
                logger.info("Terminated sandbox for '%s'", algo_name)
            except Exception:
                pass

    async def teardown_all(self) -> None:
        """Terminate all sandboxes."""
        names = list(self._sandboxes.keys())
        for name in names:
            await self.teardown(name)
        logger.info("All sandboxes terminated")

    def teardown_all_sync(self) -> None:
        """Synchronous wrapper for teardown_all (for use in finally blocks)."""
        try:
            loop = asyncio.get_running_loop()
            # Already in an event loop — schedule teardown
            loop.create_task(self.teardown_all())
        except RuntimeError:
            # No event loop — safe to use asyncio.run
            asyncio.run(self.teardown_all())
