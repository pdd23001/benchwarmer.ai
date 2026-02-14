"""
Modal sandbox runner: execute Python code in an isolated container.

Receives code as a string, optionally installs pip dependencies, runs the code,
and returns stdout, stderr, exit code, and runtime.
"""

import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from benchwarmer.schemas.search import ExecutionResult


def _extract_pip_packages(code: str) -> List[str]:
    """Extract package names from # pip install ... comments in code."""
    packages: List[str] = []
    for m in re.finditer(r"#\s*pip\s+install\s+(.+)", code, re.IGNORECASE):
        for pkg in m.group(1).split():
            pkg = pkg.strip().rstrip(",;")
            if pkg and not pkg.startswith("-"):
                packages.append(pkg)
    return list(dict.fromkeys(packages))


def run_code_locally(
    code: str,
    pip_packages: Optional[List[str]] = None,
    timeout_seconds: int = 120,
) -> ExecutionResult:
    """
    Run Python code locally (for testing or when Modal is not available).
    Writes code to a temp file, optionally installs packages, runs with subprocess.
    """
    if pip_packages is None:
        pip_packages = _extract_pip_packages(code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        start = time.perf_counter()
        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=None,
        )
        runtime = time.perf_counter() - start
        return ExecutionResult(
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            exit_code=proc.returncode or 0,
            runtime_seconds=round(runtime, 3),
            error=None if proc.returncode == 0 else (proc.stderr or f"Exit code {proc.returncode}"),
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-1,
            runtime_seconds=timeout_seconds,
            error=f"Execution timed out after {timeout_seconds}s",
        )
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr="",
            exit_code=-1,
            runtime_seconds=0.0,
            error=str(e),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _run_code_impl(code: str, pip_packages: List[str], timeout_seconds: int) -> dict:
    """
    Implementation run inside the Modal container.
    Writes code to a file, optionally pip installs, runs Python, returns result as dict.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    if pip_packages:
        subprocess.run(
            ["pip", "install", "-q"] + pip_packages,
            capture_output=True,
            timeout=60,
            check=False,
        )

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=None,
        )
        runtime = time.perf_counter() - start
        return {
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "exit_code": proc.returncode or 0,
            "runtime_seconds": round(runtime, 3),
            "error": None if proc.returncode == 0 else (proc.stderr or f"Exit code {proc.returncode}"),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "runtime_seconds": float(timeout_seconds),
            "error": f"Execution timed out after {timeout_seconds}s",
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": -1,
            "runtime_seconds": 0.0,
            "error": str(e),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Modal app and function for deployment (modal deploy) or modal run (local_entrypoint)
_modal_available = False
_run_code_modal_fn = None  # in-process function reference for use when app is running via modal run

try:
    import modal

    _image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("numpy", "scipy", "networkx")
    )
    _app = modal.App("benchwarmer-runner")

    @_app.function(image=_image, timeout=150)
    def _run_code_modal(code: str, pip_packages: List[str], timeout_seconds: int) -> dict:
        """Runs on a remote Modal worker. Returns dict with stdout, stderr, exit_code, runtime_seconds, error."""
        return _run_code_impl(code, pip_packages, timeout_seconds)

    _run_code_modal_fn = _run_code_modal
    _modal_available = True

    @_app.local_entrypoint()
    def main(pdf_path: str):
        """
        Run the full PDF pipeline with code execution on Modal.
        Usage: modal run benchwarmer/engine/modal_runner.py --pdf-path path/to/paper.pdf
        """
        from benchwarmer.agents import process_pdf
        result = process_pdf(pdf_path, run_code=True)
        print("\n--- Result ---")
        print(f"Source: {result.source}")
        if result.execution_result:
            ex = result.execution_result
            print(f"Exit code: {ex.exit_code}, Runtime: {ex.runtime_seconds}s")
            if ex.stdout:
                print("Stdout:", ex.stdout[:2000] + ("..." if len(ex.stdout) > 2000 else ""))
            if ex.error:
                print("Error:", ex.error)
except ImportError:
    _app = None


def _get_modal_function():
    """
    Return the Modal function if the app is deployed and reachable.
    Uses Function.from_name() so we only call .remote() when Modal is active (deployed).
    """
    if not _modal_available:
        return None
    try:
        import modal
        return modal.Function.from_name("benchwarmer-runner", "_run_code_modal")
    except Exception:
        return None


def run_code_remotely(
    code: str,
    pip_packages: Optional[List[str]] = None,
    timeout_seconds: int = 120,
    use_gpu: bool = False,
) -> ExecutionResult:
    """
    Run Python code in a Modal container when the app is deployed.

    Checks if Modal is active (deployed via `modal deploy benchwarmer/engine/modal_runner.py`)
    before calling .remote(). Otherwise runs locally.
    """
    import sys

    if pip_packages is None:
        pip_packages = _extract_pip_packages(code)

    if not _modal_available:
        print("  (Running locally — install 'modal' and run 'modal setup')", file=sys.stderr)
        return run_code_locally(code, pip_packages=pip_packages, timeout_seconds=timeout_seconds)

    modal_fn = _get_modal_function()
    if modal_fn is None and _run_code_modal_fn is not None:
        # Not deployed, but we might be inside "modal run" (local_entrypoint) — try in-process
        modal_fn = _run_code_modal_fn
    if modal_fn is None:
        print("  (Modal not active — running locally. Use: modal run benchwarmer/engine/modal_runner.py --pdf-path <pdf> or modal deploy)", file=sys.stderr)
        return run_code_locally(code, pip_packages=pip_packages, timeout_seconds=timeout_seconds)

    try:
        result_dict = modal_fn.remote(code, pip_packages, timeout_seconds)
        return ExecutionResult(
            stdout=result_dict.get("stdout", ""),
            stderr=result_dict.get("stderr", ""),
            exit_code=result_dict.get("exit_code", -1),
            runtime_seconds=result_dict.get("runtime_seconds", 0.0),
            error=result_dict.get("error"),
        )
    except Exception as e:
        print(f"  (Modal call failed — running locally: {str(e)[:60]})", file=sys.stderr)
        return run_code_locally(code, pip_packages=pip_packages, timeout_seconds=timeout_seconds)
