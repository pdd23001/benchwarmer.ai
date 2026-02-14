"""
Intake Agent — converts natural-language problem descriptions into
structured BenchmarkConfig objects using Claude Sonnet 4 with tool_use.

Usage
-----
>>> from benchwarmer.agents.intake import IntakeAgent
>>> agent = IntakeAgent()                       # reads ANTHROPIC_API_KEY from env
>>> config = agent.run("I'm trying to find …")  # may ask clarifying questions
>>> print(config)                                # BenchmarkConfig
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from benchwarmer.agents.tools import (
    TOOL_DEFINITIONS,
    dispatch_tool_call,
)
from benchwarmer.config import BenchmarkConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# System prompt — closely follows the architecture doc
# ------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the intake agent for Benchwarmer.AI, an algorithm benchmarking platform.

The user will describe their optimization problem in natural language.
Your job is to:

1. UNDERSTAND the problem — ask clarifying questions if needed
2. CLASSIFY it into a known problem class (or flag it as custom)
3. INFER the right benchmarking setup:
   - What graph types match their real-world scenario?
   - What sizes to test at?
   - What matters more: speed, quality, memory, consistency?
   - Any hard constraints (e.g., "must run under 60 seconds")?
4. OUTPUT a structured BenchmarkConfig JSON

You have access to the following tools:
- classify_problem(description) → returns candidate problem classes with confidence
- get_generators(problem_class) → returns available instance generators
- validate_config(config) → checks if a config is valid and complete

IMPORTANT BEHAVIORS:
- If you see "[SYSTEM: This is an API request]" in the user's message, NEVER ask clarifying questions. Go directly to generating the config with sensible defaults.
- If the problem clearly maps to a known class, don't over-ask. Confirm and move on.
- If it's ambiguous (could be Max-Cut OR graph partitioning) and NOT an API request, ask ONE clarifying question.
- Always infer instance generators from the user's domain description:
    - "social networks" → Barabási-Albert, planted partition
    - "road networks" → grid-like graphs, planar graphs
    - "molecular structures" → sparse, bounded-degree graphs
    - "internet topology" → power-law graphs
    - "random benchmarks" → Erdős-Rényi
- Extract any implicit constraints the user mentioned.
- Don't ask about things you can set sensible defaults for.
- If the user gives a short/clear description, just proceed with sensible defaults.
  Do NOT ask clarifying questions unless the problem is truly ambiguous.

WORKFLOW:
1. First, call classify_problem with the user's description.
2. If confidence is high (≥ 0.7) OR this is an API request, proceed WITHOUT asking questions.
   Set sensible defaults for anything the user didn't specify.
3. Call get_generators to see what's available for the matched class.
4. Build a BenchmarkConfig JSON using the EXACT schema below.
5. Call validate_config to verify it's valid.
6. Present the final config to the user in a clear, readable way.

BENCHMARK CONFIG SCHEMA (you MUST follow this exactly):
```json
{
  "problem_class": "maximum_cut",
  "problem_description": "Brief description",
  "objective": "maximize",
  "instance_config": {
    "generators": [
      {
        "type": "erdos_renyi",
        "params": {"p": 0.3},
        "sizes": [50, 100, 200],
        "count_per_size": 2,
        "why": "General random benchmark graphs"
      }
    ]
  },
  "execution_config": {
    "timeout_seconds": 60,
    "runs_per_config": 3,
    "memory_limit_mb": 2048
  }
}
```

FOR API REQUESTS (when you see [SYSTEM: This is an API request]):
- Use SMALLER benchmarks for faster results
- Max 3 sizes (e.g., [50, 100, 200])
- count_per_size: 2 (not 3 or more)
- runs_per_config: 3 (not 5 or more)
- Use 1-2 generator configurations (not 3+)

CRITICAL RULES FOR THE CONFIG:
- Use "params" (NOT "parameters") for generator params.
- Each "params" value must be a SINGLE value (e.g. {"p": 0.3}), NOT a list.
- To test different parameter values, create SEPARATE generator entries
  (e.g. one with {"p": 0.3} and another with {"p": 0.7}).
- Required fields: problem_class, instance_config.generators (each with type + sizes).

When you have the final config ready, output it inside a JSON code block like:
```json
{ ... }
```

Keep your responses concise and helpful. You are an expert who gets things
right quickly — the user shouldn't have to answer more than 1–2 questions.
"""


class IntakeAgent:
    """
    Conversational agent that maps NL problem descriptions to
    :class:`BenchmarkConfig` objects.

    Parameters
    ----------
    api_key : str | None
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    model : str
        Model to use.  Defaults to ``claude-sonnet-4-20250514``.
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
                "The 'anthropic' package is required for the Intake Agent. "
                "Install it with: pip install anthropic"
            ) from e

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set ANTHROPIC_API_KEY."
            )

        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_description: str,
        interactive: bool = True,
    ) -> BenchmarkConfig:
        """
        Run the intake conversation.

        Parameters
        ----------
        user_description : str
            The user's natural-language problem description.
        interactive : bool
            If True (default), prompt the user on stdin when the agent
            asks clarifying questions.  If False, the agent must resolve
            the problem in a single turn (useful for testing).

        Returns
        -------
        BenchmarkConfig
            The validated benchmark configuration.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_description},
        ]

        max_turns = 10  # safety rail
        for turn in range(max_turns):
            logger.info("Intake agent turn %d", turn + 1)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            logger.debug("Stop reason: %s", response.stop_reason)

            # ── Handle tool use ──────────────────────────────────
            if response.stop_reason == "tool_use":
                # There may be text + tool_use blocks mixed together
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        logger.info(
                            "Tool call: %s(%s)",
                            block.name,
                            json.dumps(block.input, indent=2),
                        )
                        result_str = dispatch_tool_call(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })

                messages.append({"role": "user", "content": tool_results})
                continue

            # ── Handle end_turn (text response) ──────────────────
            if response.stop_reason == "end_turn":
                text = self._extract_text(response.content)

                # Try to extract a JSON config from the response
                config = self._try_parse_config(text)
                if config is not None:
                    print(f"\n🤖 Intake Agent:\n{text}")
                    return config

                # No config yet — agent is asking a clarifying question
                print(f"\n🤖 Intake Agent:\n{text}")

                if not interactive:
                    raise RuntimeError(
                        "Agent asked a clarifying question but interactive=False. "
                        f"Question was: {text}"
                    )

                while True:
                    user_reply = input("\n📝 Your answer: ").strip()
                    if user_reply:
                        break
                    print("   (Please type a response, or type 'defaults' to let the agent decide)")
                if user_reply.lower() == "defaults":
                    user_reply = "Use your best judgment, go with sensible defaults."

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": user_reply})
                continue

            # ── Unexpected stop reason ───────────────────────────
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        raise RuntimeError(
            f"Intake agent did not produce a config within {max_turns} turns."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(content: list) -> str:
        """Pull out the text from a Claude response content list."""
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _try_parse_config(text: str) -> Optional[BenchmarkConfig]:
        """
        Try to extract a BenchmarkConfig JSON from the agent's text.

        Looks for a ```json … ``` code block first, then tries the
        entire text as JSON.
        """
        import re

        # Look for JSON inside a code fence
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if json_match:
            candidate = json_match.group(1)
        else:
            # Maybe the entire message is JSON
            candidate = text.strip()

        try:
            data = json.loads(candidate)
            return BenchmarkConfig(**data)
        except Exception:
            return None
