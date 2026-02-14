"""Execution engine: run generated or fetched code in sandboxed environments."""

from benchwarmer.engine.modal_runner import run_code_remotely

__all__ = ["run_code_remotely"]
