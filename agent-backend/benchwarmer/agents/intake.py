"""
Intake Agent — converts natural-language problem descriptions into
structured IntakeResult objects (BenchmarkConfig + AlgorithmSpecs)
using Claude Sonnet 4 with tool_use.

Usage
-----
>>> from benchwarmer.agents.intake import IntakeAgent
>>> agent = IntakeAgent()
>>> result = agent.run("I'm trying to find …", pdf_paths=["paper.pdf"])
>>> print(result.config)       # BenchmarkConfig
>>> print(result.algorithms)   # [AlgorithmSpec, ...]
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Optional

from benchwarmer.agents.tools import (
    TOOL_DEFINITIONS,
    dispatch_tool_call,
)
from benchwarmer.config import AlgorithmSpec, BenchmarkConfig, IntakeResult

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
4. If PDF papers are attached, EXTRACT algorithm descriptions from them
5. OUTPUT a structured IntakeResult JSON (config + algorithms)

You have access to the following tools:
- classify_problem(description) → returns candidate problem classes with confidence
- get_generators(problem_class) → returns available instance generators
- validate_config(config) → checks if a config is valid and complete

IMPORTANT BEHAVIORS:
- If the problem clearly maps to a known class, don't over-ask. Confirm and move on.
- If it's ambiguous (could be Max-Cut OR graph partitioning), ask ONE clarifying question.
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
2. If confidence is high (≥ 0.7), proceed WITHOUT asking questions.
   Set sensible defaults for anything the user didn't specify.
3. Call get_generators to see what's available for the matched class.
4. If papers are attached, analyze each paper and extract the algorithms described.
5. Build an IntakeResult JSON containing BOTH the benchmark config AND algorithm specs.
6. Present the final result to the user in a clear, readable way.

OUTPUT SCHEMA (you MUST follow this exactly):
```json
{
  "config": {
    "problem_class": "maximum_cut",
    "problem_description": "Brief description",
    "objective": "maximize",
    "instance_config": {
      "generators": [
        {
          "type": "erdos_renyi",
          "params": {"p": 0.3},
          "sizes": [50, 100, 200, 500],
          "count_per_size": 3,
          "why": "General random benchmark graphs"
        }
      ]
    },
    "execution_config": {
      "timeout_seconds": 60,
      "runs_per_config": 5,
      "memory_limit_mb": 2048
    }
  },
  "algorithms": [
    {
      "name": "greedy_max_cut",
      "approach": "Greedy vertex assignment to maximize cut edges",
      "complexity": "O(n*m)",
      "key_steps": [
        "Start with empty partitions S and T",
        "For each vertex, assign to the partition that maximizes cut",
        "Repeat until no improvement"
      ],
      "source": "paper_1.pdf"
    }
  ]
}
```

ALGORITHM EXTRACTION RULES:
- If PDF papers are attached, extract EVERY distinct algorithm from them.
- Each algorithm needs: name (snake_case), approach (one-line), key_steps (pseudocode-like).
- Set "source" to the PDF filename.
- If NO papers are attached, the "algorithms" list should be EMPTY [].

CRITICAL RULES FOR THE CONFIG:
- Use "params" (NOT "parameters") for generator params.
- Each "params" value must be a SINGLE value (e.g. {"p": 0.3}), NOT a list.
- To test different parameter values, create SEPARATE generator entries
  (e.g. one with {"p": 0.3} and another with {"p": 0.7}).
- Required fields: config.problem_class, config.instance_config.generators (each with type + sizes).

When you have the final result ready, output it inside a JSON code block like:
```json
{ ... }
```

Keep your responses concise and helpful. You are an expert who gets things
right quickly — the user shouldn't have to answer more than 1–2 questions.
"""


class IntakeAgent:
    """
    Conversational agent that maps NL problem descriptions (and optional
    PDF papers) to :class:`IntakeResult` objects containing both
    :class:`BenchmarkConfig` and :class:`AlgorithmSpec` entries.

    Parameters
    ----------
    backend : AbstractLLMBackend | None
        The LLM backend to use. If None, defaults to ClaudeBackend.
    api_key : str | None
        Anthropic API key (legacy/convenience). Used if backend is None.
    model : str
        Model to use (legacy/convenience). Used if backend is None.
    """

    def __init__(
        self,
        backend=None,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        from benchwarmer.agents.backends import ClaudeBackend

        if backend is not None:
            self.backend = backend
        else:
            # Default to Claude
            self.backend = ClaudeBackend(api_key=api_key, model=model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_description: str,
        pdf_paths: list[str] | None = None,
        interactive: bool = True,
    ) -> IntakeResult:
        """
        Run the intake conversation.

        Parameters
        ----------
        user_description : str
            The user's natural-language problem description.
        pdf_paths : list[str] | None
            Optional list of PDF file paths to analyze for algorithms.
        interactive : bool
            If True (default), prompt the user on stdin when the agent
            asks clarifying questions.  If False, the agent must resolve
            the problem in a single turn (useful for testing).

        Returns
        -------
        IntakeResult
            The validated benchmark config + extracted algorithm specs.
        """
        from benchwarmer.agents.backends import ClaudeBackend

        use_tools = isinstance(self.backend, ClaudeBackend)

        # Build system prompt — enriched with tool results for non-Claude backends
        if use_tools:
            system_prompt = SYSTEM_PROMPT
            tools = TOOL_DEFINITIONS
        else:
            system_prompt = self._build_enriched_prompt(user_description)
            tools = None

        # Build user message content — may include PDFs
        user_content = self._build_user_content(
            user_description, pdf_paths, use_claude=use_tools
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]
        logger.info("IntakeAgent run() called with: %r (PDFs: %s)",
                    user_description, pdf_paths)

        max_turns = 10  # safety rail
        for turn in range(max_turns):
            logger.info("Intake agent turn %d", turn + 1)

            response = self.backend.generate(
                messages=messages,
                system=system_prompt,
                tools=tools,
                max_tokens=4096,
            )

            logger.debug("Stop reason: %s", response.stop_reason)

            # ── Handle tool use (Claude only) ────────────────────
            if response.stop_reason == "tool_use":
                # Convert dataclass objects to plain dicts for re-serialization
                from dataclasses import asdict
                assistant_content = [asdict(block) for block in response.content]
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in response.content:  # iterate originals for attribute access
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
            if response.stop_reason in ("end_turn", "stop"):
                text = self._extract_text(response.content)

                # Try to extract an IntakeResult from the response
                result = self._try_parse_result(text)
                if result is not None:
                    print(f"\n[Intake Agent]:\n{text}")
                    return result

                # No result yet — agent is asking a clarifying question
                print(f"\n[Intake Agent]:\n{text}")

                if not interactive:
                    raise RuntimeError(
                        "Agent asked a clarifying question but interactive=False. "
                        f"Question was: {text}"
                    )

                while True:
                    user_reply = input("\n[Your answer]: ").strip()
                    if user_reply:
                        break
                    print("   (Please type a response, or type 'defaults' to let the agent decide)")
                if user_reply.lower() == "defaults":
                    user_reply = "Use your best judgment, go with sensible defaults."

                # Convert dataclass objects to plain dicts for re-serialization
                from dataclasses import asdict
                serializable_content = [asdict(block) for block in response.content]
                messages.append({"role": "assistant", "content": serializable_content})
                messages.append({"role": "user", "content": user_reply})
                continue

            # ── Unexpected stop reason ───────────────────────────
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        raise RuntimeError(
            f"Intake agent did not produce a result within {max_turns} turns."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_user_content(
        self,
        user_description: str,
        pdf_paths: list[str] | None,
        use_claude: bool,
    ) -> str | list[dict[str, Any]]:
        """
        Build user message content, optionally including PDFs.

        PDFs are converted to text via PyMuPDF for all backends.
        This avoids needing the Anthropic PDF beta API.
        """
        if not pdf_paths:
            return user_description

        # Extract text from all PDFs
        pdf_texts = []
        for path in pdf_paths:
            text = self._extract_pdf_text(path)
            if text:
                filename = os.path.basename(path)
                pdf_texts.append(f"--- PAPER: {filename} ---\n{text}\n--- END ---")
                logger.info("Extracted text from PDF: %s (%d chars)", filename, len(text))

        if pdf_texts:
            papers_section = "\n\n".join(pdf_texts)
            return (
                f"{user_description}\n\n"
                f"The following papers have been provided for algorithm extraction:\n\n"
                f"{papers_section}"
            )
        return user_description

    @staticmethod
    def _extract_pdf_text(path: str) -> str:
        """Extract text from a PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning(
                "PyMuPDF not installed. Install with: pip install PyMuPDF"
            )
            return ""

        try:
            doc = fitz.open(path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            return "\n".join(pages)
        except Exception as e:
            logger.error("Failed to extract text from %s: %s", path, e)
            return ""

    def _build_enriched_prompt(self, user_description: str) -> str:
        """
        Build a system prompt with pre-computed tool results embedded.

        Used for backends that don't support native tool calling (e.g.
        Nemotron). We call classify_problem and get_generators up front
        and inject results directly into the prompt.
        """
        from benchwarmer.agents.tools import classify_problem, get_generators_for_class
        from benchwarmer.problem_classes.registry import list_problem_classes

        # 1. Classify the problem
        classifications = classify_problem(user_description)
        classification_text = json.dumps(classifications, indent=2)

        # 2. Get generators for all problem classes
        generators_info = {}
        all_unique_generators = set()
        for cls in list_problem_classes():
            try:
                gens = get_generators_for_class(cls["name"])
                generators_info[cls["name"]] = gens
                all_unique_generators.update(gens)
            except Exception:
                pass
        generators_text = json.dumps(generators_info, indent=2)
        all_gens_list = sorted(all_unique_generators)

        enriched = f"""{SYSTEM_PROMPT}

--- PRE-COMPUTED TOOL RESULTS ---

You do NOT have access to tools. The following results have been
pre-computed for you. Use them directly.

## classify_problem result:
```json
{classification_text}
```

## Available generators per problem class:
```json
{generators_text}
```

## ALL unique generators: {all_gens_list}

CRITICAL INSTRUCTIONS:
1. Pick the problem class with the highest confidence from the classification.
2. You MUST include ALL {len(all_gens_list)} unique generators ({', '.join(all_gens_list)})
   in your config. Create a separate generator entry for EACH generator type.
   Do NOT skip any generator — use every single one listed above.
3. Use sensible default params for each generator:
   - erdos_renyi: {{"p": 0.3}} for sparse, {{"p": 0.7}} for dense
   - barabasi_albert: {{"m": 3}}
   - planted_partition: {{"p_in": 0.7, "p_out": 0.1}}
   - grid_2d: {{}} (no params needed)
   - planar_random: {{}} (no params needed)
   - watts_strogatz: {{"k": 4, "p": 0.3}}
4. If papers are included in the message, extract algorithms from them.
5. Output the final IntakeResult JSON inside a ```json code block.
"""
        return enriched

    @staticmethod
    def _extract_text(content: list) -> str:
        """Pull out the text from a Claude response content list."""
        parts = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    @staticmethod
    def _try_parse_result(text: str) -> Optional[IntakeResult]:
        """
        Try to extract an IntakeResult JSON from the agent's text.

        Supports two JSON formats:
        1. IntakeResult format: {"config": {...}, "algorithms": [...]}
        2. Legacy BenchmarkConfig format: {"problem_class": ...}
           (auto-wrapped into IntakeResult with empty algorithms)

        Handles various LLM output formats including:
        - ```json ... ``` code blocks
        - ``` ... ``` code blocks (no language tag)
        - <think>...</think> reasoning tags (Nemotron)
        - Raw JSON in the response
        """
        import re

        # Strip <think>...</think> reasoning blocks (Nemotron)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        def _try_parse(candidate: str) -> Optional[IntakeResult]:
            """Try to parse a JSON string as IntakeResult or BenchmarkConfig."""
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                return None

            # Format 1: IntakeResult with "config" key
            if "config" in data and isinstance(data["config"], dict):
                try:
                    return IntakeResult(**data)
                except Exception:
                    pass

            # Format 2: Legacy BenchmarkConfig with "problem_class" key
            if "problem_class" in data:
                try:
                    config = BenchmarkConfig(**data)
                    return IntakeResult(config=config, algorithms=[])
                except Exception:
                    pass

            return None

        # 1. Look for JSON inside a ```json code fence
        json_match = re.search(r"```json\s*\n(.*?)\n\s*```", cleaned, re.DOTALL)
        if json_match:
            result = _try_parse(json_match.group(1).strip())
            if result:
                return result

        # 2. Look for JSON inside a bare ``` code fence
        bare_match = re.search(r"```\s*\n(.*?)\n\s*```", cleaned, re.DOTALL)
        if bare_match:
            result = _try_parse(bare_match.group(1).strip())
            if result:
                return result

        # 3. Look for any JSON object containing "config" or "problem_class"
        brace_match = re.search(
            r'\{[^{}]*"(?:config|problem_class)".*\}', cleaned, re.DOTALL
        )
        if brace_match:
            result = _try_parse(brace_match.group(0))
            if result:
                return result

        # 4. Maybe the entire cleaned text is JSON
        return _try_parse(cleaned)
