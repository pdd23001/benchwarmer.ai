"""
Plot Agent — generates matplotlib visualisation code from natural-language
requests using Claude Sonnet 4.

The agent receives the DataFrame schema + sample rows and the user's
request, then returns executable Python code.  If execution fails the
error is fed back for self-correction (up to 2 retries).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import pandas as pd

from benchwarmer.utils.sandbox import execute_plot_code

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the visualization agent for Benchwarmer.AI, an algorithm benchmarking platform.

You will receive:
1. The schema and sample rows of a benchmark results DataFrame
2. A natural-language visualization request from the user

Your job is to write Python code using matplotlib and pandas that
creates the requested visualization.

AVAILABLE VARIABLES (pre-injected into the execution namespace):
- `df`          — pandas DataFrame with the benchmark results
- `pd`          — pandas module
- `np`          — numpy module
- `plt`         — matplotlib.pyplot (Agg backend, non-interactive)
- `output_path` — file path where the plot should be saved

KEY DataFrame COLUMNS:
- algorithm_name (str): Name of the algorithm
- instance_name (str): Unique instance identifier
- instance_generator (str): Generator type used (e.g. "erdos_renyi")
- problem_size (int): Number of nodes in the graph
- objective_value (float): The algorithm's objective score
- wall_time_seconds (float): Execution time
- peak_memory_mb (float): Peak memory usage
- status (str): "success", "timeout", or "error"
- run_index (int): Run number within repeated trials
- feasible (bool): Whether the solution is valid

RULES:
1. Always filter to status == "success" unless specifically asked about failures.
2. When comparing algorithms, group by algorithm_name.
3. For scaling plots, use problem_size on the x-axis.
4. Use clear labels, titles, and legends.  Be professional.
5. Use a clean, modern style.  Prefer `plt.style.use('seaborn-v0_8-darkgrid')`
   or similar.  Use a good color palette.
6. Always call `plt.tight_layout()` before saving.
7. Save the figure to `output_path` using `plt.savefig(output_path, dpi=150, bbox_inches='tight')`.
8. For tables, print them to stdout using pandas `.to_string()` or `.to_markdown()`.
9. If the user asks for a "summary", generate a summary table, not a plot.
10. NEVER use plt.show() — the backend is non-interactive.

OUTPUT FORMAT:
Return ONLY the Python code inside a ```python ... ``` code block.
No explanations outside the code block.
"""


class PlotAgent:
    """
    Conversational agent that generates matplotlib code from NL requests.

    Parameters
    ----------
    api_key : str | None
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    model : str
        Model to use.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for the Plot Agent. "
                "Install it with: pip install anthropic"
            ) from e

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set ANTHROPIC_API_KEY."
            )

        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self._conversation_history: list[dict[str, str]] = []
        self._df_context: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Set the results DataFrame and pre-compute the schema context."""
        info_parts = [
            f"DataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns",
            "",
            "Columns and dtypes:",
        ]
        for col in df.columns:
            info_parts.append(f"  - {col} ({df[col].dtype})")

        info_parts.append("")
        info_parts.append("Unique values per key column:")
        for col in ["algorithm_name", "instance_generator", "problem_size", "status"]:
            if col in df.columns:
                vals = df[col].unique().tolist()
                info_parts.append(f"  - {col}: {vals}")

        info_parts.append("")
        info_parts.append("Sample rows (first 5):")
        info_parts.append(df.head().to_string())

        self._df_context = "\n".join(info_parts)

    def generate_and_execute(
        self,
        user_request: str,
        df: pd.DataFrame,
        output_dir: str = "plots",
        plot_index: int = 0,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """
        Generate plot code from user request, execute it, and return result.

        Parameters
        ----------
        user_request : str
            Natural-language visualization request.
        df : pd.DataFrame
            Results DataFrame.
        output_dir : str
            Directory to save plots.
        plot_index : int
            Counter for default filenames.
        max_retries : int
            Number of self-correction retries on execution failure.

        Returns
        -------
        dict
            Result from sandbox execution, plus the generated ``code``.
        """
        if not self._df_context:
            self.set_dataframe(df)

        # Build the user message with DataFrame context
        full_request = (
            f"DataFrame info:\n{self._df_context}\n\n"
            f"User request: {user_request}"
        )

        self._conversation_history.append({
            "role": "user",
            "content": full_request,
        })

        for attempt in range(max_retries + 1):
            logger.info("Plot agent attempt %d", attempt + 1)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=self._conversation_history,
            )

            text = self._extract_text(response.content)
            code = self._extract_code(text)

            if not code:
                # Agent didn't produce code — return the text as explanation
                self._conversation_history.append({
                    "role": "assistant",
                    "content": text,
                })
                return {
                    "success": True,
                    "output_path": None,
                    "message": text,
                    "code": None,
                }

            # Execute the code in sandbox
            result = execute_plot_code(code, df, output_dir, plot_index)
            result["code"] = code

            if result["success"]:
                self._conversation_history.append({
                    "role": "assistant",
                    "content": text,
                })
                return result

            # Execution failed — feed error back for self-correction
            if attempt < max_retries:
                logger.warning(
                    "Plot code failed (attempt %d): %s",
                    attempt + 1,
                    result["error"],
                )
                self._conversation_history.append({
                    "role": "assistant",
                    "content": text,
                })
                self._conversation_history.append({
                    "role": "user",
                    "content": (
                        f"The code raised an error:\n"
                        f"```\n{result['traceback']}\n```\n"
                        f"Please fix the code and try again."
                    ),
                })
            else:
                self._conversation_history.append({
                    "role": "assistant",
                    "content": text,
                })
                return result

        return {"success": False, "error": "Max retries exceeded", "code": code}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(content: list) -> str:
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _extract_code(text: str) -> Optional[str]:
        """Extract Python code from a ```python ... ``` block."""
        import re

        match = re.search(r"```python\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return None
