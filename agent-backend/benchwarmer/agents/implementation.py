"""
Implementation Agent — generates AlgorithmWrapper subclasses from
natural language descriptions using Claude Sonnet 4.

The agent receives the problem class context (solution format, validation
rules) and the AlgorithmWrapper API, then generates working Python code
that implements the user's described algorithm.  Generated code is
smoke-tested on a tiny graph instance, with self-correction on failure.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from benchwarmer.utils.algorithm_sandbox import execute_algorithm_code

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an algorithm implementation agent for Benchwarmer.AI, an algorithm \
benchmarking platform for graph optimization problems.

Your job is to write a working Python class that implements the user's \
described algorithm. The class MUST extend `AlgorithmWrapper`.

## AlgorithmWrapper API

```python
class AlgorithmWrapper(ABC):
    name: str = "unnamed"  # MUST be set to a unique identifier

    def solve(self, instance: dict, timeout: float = 60.0) -> dict:
        \"\"\"
        Solve the given problem instance.

        Parameters
        ----------
        instance : dict
            A graph dict with:
              - "nodes": list[int] — e.g. [0, 1, 2, ...]
              - "edges": list[dict] — e.g. [{"source": 0, "target": 1, "weight": 1.0}, ...]
              - "metadata": dict — generator info

        Returns
        -------
        dict with:
          - "solution": dict — problem-specific (see below)
          - "metadata": dict — optional (iterations, etc.)
        \"\"\"
```

<<<PROBLEM_CONTEXT>>>

## Available imports (pre-injected, do NOT import them):
- `AlgorithmWrapper` — the base class to extend
- `random` — Python's random module
- `math` — Python's math module
- `collections` — including `defaultdict`, `deque`, `Counter`
- `itertools` — Python's itertools module
- `heapq` — Python's heapq module
- `inf` — float("inf")

## CRITICAL RULES:
1. Define exactly ONE class that extends `AlgorithmWrapper`
2. Set `name` to a descriptive snake_case identifier (e.g. "greedy_vertex_cover")
3. Implement the `solve(self, instance, timeout=60.0)` method
4. Return the solution in the EXACT format specified above for this problem
5. Do NOT import anything — all allowed modules are pre-injected
6. Do NOT use `open()`, `exec()`, `eval()`, or `__import__()`
7. Handle edge cases (empty graphs, disconnected graphs)
8. Use efficient algorithms — O(n²) is fine, O(n³) may be too slow for n>200

## OUTPUT:
Return ONLY the Python code inside a ```python ... ``` block. No explanation \
outside the code block.
"""


# ------------------------------------------------------------------
# Problem-specific context templates
# ------------------------------------------------------------------

PROBLEM_CONTEXTS = {
    "maximum_cut": """\
## Problem: Maximum Cut
**Objective:** Maximize the number/weight of edges crossing the partition.

**Solution format:**
```python
return {
    "solution": {"partition": [0, 1, 0, 1, ...]},  # 0 or 1 for each node
    "metadata": {}
}
```

The `partition` list must have the same length as `instance["nodes"]`.
Each entry is 0 or 1, indicating which set the node belongs to.
The objective is the total weight of edges where the endpoints are in \
different sets.

**Example algorithm ideas:**
- Random partition (baseline)
- Greedy: assign each node to the set that maximizes cut edges
- Local search: start random, flip nodes that improve the cut
- Simulated annealing
- Semi-definite programming (SDP) relaxation + rounding
""",

    "minimum_vertex_cover": """\
## Problem: Minimum Vertex Cover
**Objective:** Minimize the number of vertices that cover all edges.

**Solution format:**
```python
return {
    "solution": {"vertex_cover": sorted(list_of_node_ids)},
    "metadata": {}
}
```

The `vertex_cover` list contains node IDs (ints) from `instance["nodes"]`.
Every edge in `instance["edges"]` must have at least one endpoint in the cover.

**Example algorithm ideas:**
- Greedy: repeatedly pick the highest-degree uncovered node
- 2-approximation: for each uncovered edge, add both endpoints
- LP relaxation + rounding
- Local search: start with all nodes, try removing them
- Branch and bound (exact, but slow for large instances)
""",
}

DEFAULT_CONTEXT = """\
## Problem: {problem_class}
**Solution format:**
```python
return {{
    "solution": {{...}},  # problem-specific solution dict
    "metadata": {{}}
}}
```

Look at the instance structure (nodes, edges) and return a solution dict \
appropriate for this problem type. Include a "solution" key and a "metadata" key.
"""


class ImplementationAgent:
    """
    Conversational agent that generates AlgorithmWrapper implementations
    from natural language descriptions.

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
                "The 'anthropic' package is required. "
                "Install it with: pip install anthropic"
            ) from e

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set ANTHROPIC_API_KEY."
            )

        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model

    def generate(
        self,
        description: str,
        problem_class: str,
        additional_context: str | None = None,
        pdf_paths: list[str] | None = None,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """
        Generate an AlgorithmWrapper from a natural language description.

        Parameters
        ----------
        description : str
            NL description or instruction.
        problem_class : str
            The problem class name.
        additional_context : str | None
            Optional extra text context.
        pdf_paths : list[str] | None
            List of paths to PDF files to attach for analysis.
        max_retries : int
            Self-correction retries.

        Returns
        -------
        dict
            Result dict with code or error.
        """
        import base64

        # Build problem-specific context
        problem_context = PROBLEM_CONTEXTS.get(
            problem_class,
            DEFAULT_CONTEXT.format(problem_class=problem_class),
        )

        system = SYSTEM_PROMPT.replace("<<<PROBLEM_CONTEXT>>>", problem_context)

        user_content = []

        # 1. Add PDF documents if provided
        if pdf_paths:
            for i, path in enumerate(pdf_paths):
                try:
                    with open(path, "rb") as f:
                        pdf_data = base64.b64encode(f.read()).decode("utf-8")
                    
                    user_content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        }
                    })
                except Exception as e:
                    logger.error(f"Failed to read PDF {path}: {e}")
                    # Continue with other PDFs if one fails, or maybe we should raise?
                    # For now, let's just log and continue, but append a note.
                    if additional_context:
                        additional_context += f"\n[Error reading {path}: {e}]"
                    else:
                        additional_context = f"[Error reading {path}: {e}]"
            
            # Augment description to reference the PDFs
            description += f"\n\n(Refer to the attached {len(pdf_paths)} PDF document(s) for details.)"

        # 2. Add text content (description + additional context)
        text_part = (
            f"Implement this algorithm:\n\n{description}\n\n"
            f"Problem class: {problem_class}"
        )

        if additional_context:
            text_part += (
                f"\n\n--- ADDITIONAL CONTEXT ---\n"
                f"{additional_context}\n"
                f"--------------------------"
            )

        user_content.append({"type": "text", "text": text_part})

        messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        last_code = None

        for attempt in range(max_retries + 1):
            logger.info("Implementation agent attempt %d", attempt + 1)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system,
                messages=messages,
            )

            text = self._extract_text(response.content)
            code = self._extract_code(text)

            if not code:
                return {
                    "success": False,
                    "error": "Agent did not produce code.",
                    "code": None,
                    "raw_response": text,
                }

            last_code = code

            # Execute and smoke test
            result = execute_algorithm_code(code, problem_class)

            if result["success"]:
                logger.info(
                    "Algorithm '%s' generated successfully (attempt %d)",
                    result["name"],
                    attempt + 1,
                )
                return {
                    "success": True,
                    "algorithm": result["algorithm"],
                    "code": code,
                    "name": result["name"],
                    "smoke_result": result["smoke_result"],
                }

            # Failed — feed error back for self-correction
            if attempt < max_retries:
                logger.warning(
                    "Algorithm code failed (attempt %d): %s",
                    attempt + 1,
                    result["error"],
                )
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"The generated code failed validation:\n\n"
                        f"**Error:** {result['error']}\n\n"
                        f"```\n{result.get('traceback', '')}\n```\n\n"
                        f"Please fix the code and try again. Remember:\n"
                        f"- Extend AlgorithmWrapper\n"
                        f"- Set a unique `name` attribute\n"
                        f"- Return the solution in the correct format\n"
                        f"- Do NOT use import statements"
                    ),
                })
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "code": code,
                    "traceback": result.get("traceback", ""),
                }

        return {
            "success": False,
            "error": "Max retries exceeded",
            "code": last_code,
        }

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
        match = re.search(r"```python\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        return None
