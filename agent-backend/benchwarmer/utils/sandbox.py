"""
Sandboxed execution environment for LLM-generated plot code.

Executes matplotlib / pandas / numpy code strings in a restricted
namespace with the results DataFrame pre-injected.
"""

from __future__ import annotations

import io
import os
import traceback
import math
import json
from typing import Any, Optional


def execute_plot_code(
    code: str,
    df: Any,
    output_dir: str = "plots",
    plot_index: int = 0,
) -> dict[str, Any]:
    """
    Execute generated plot code in a sandboxed namespace.

    Parameters
    ----------
    code : str
        Python code string (matplotlib / pandas / numpy).
    df : pandas.DataFrame
        The benchmark results DataFrame.
    output_dir : str
        Directory to save generated plots.
    plot_index : int
        Counter used for default filenames.

    Returns
    -------
    dict
        ``{"success": True, "output_path": "...", "stdout": "..."}`` on
        success, or ``{"success": False, "error": "...", "traceback": "..."}``
        on failure.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    default_path = os.path.join(output_dir, f"plot_{plot_index:03d}.png")

    # Build safe builtins — only what plotting code needs
    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "min": min,
        "max": max,
        "round": round,
        "abs": abs,
        "sum": sum,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "isinstance": isinstance,
        "hasattr": hasattr,
        "getattr": getattr,
        "True": True,
        "False": False,
        "None": None,
    }

    # Build the restricted namespace (df is the benchmark results DataFrame)
    namespace: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "matplotlib": matplotlib,
        "math": math,
        "json": json,
        "output_path": default_path,
    }

    # Capture stdout
    stdout_capture = io.StringIO()

    try:
        # Close any existing figures to avoid leaks
        plt.close("all")

        # Execute the code
        exec(code, namespace)  # noqa: S102

        # Determine the output path (code may have overridden it)
        actual_path = namespace.get("output_path", default_path)

        # If there's a current figure that hasn't been saved, save it
        if plt.get_fignums():
            plt.savefig(actual_path, dpi=150, bbox_inches="tight")
            plt.close("all")
        elif not os.path.exists(actual_path):
            # No figure created and no file saved
            actual_path = None

        return {
            "success": True,
            "output_path": actual_path,
            "stdout": stdout_capture.getvalue(),
        }

    except Exception as exc:
        plt.close("all")
        return {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
