"""
Orchestrator Agent — conversational CLI powered by Claude Sonnet 4.

Replaces the rigid step-by-step pipeline with a free-form NL loop.
Users can navigate freely, go back, undo, and modify any part of the
benchmarking pipeline using natural language.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from benchwarmer.utils.loader import load_algorithm_from_file

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Pipeline State
# ──────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """Mutable state for the entire benchmarking pipeline."""
    config: Any = None                  # BenchmarkConfig
    algo_specs: list = field(default_factory=list)  # AlgorithmSpec list from PDFs
    algorithms: list = field(default_factory=list)   # registered AlgorithmWrappers
    results: Any = None                 # pandas DataFrame
    execution_mode: str = "local"       # local | modal
    pool: Any = None                    # SandboxPool (if modal)
    pdf_paths: list = field(default_factory=list)
    custom_algo_path: str | None = None
    custom_algo_name: str | None = None
    instance_source: str | None = None  # "generator" | "custom" | "suite" | None
    # Plot path from latest analyze_results call (consumed by server after each turn)
    last_plot_path: str | None = None
    # Preferences extracted from the user's initial message (so we don't ask again)
    preferred_runs_per_instance: int | None = None
    preferred_algo_spec_indices: list[int] | None = None  # e.g. [0, 2]
    preferred_plot_requests: list[str] | None = None  # NL descriptions to plot after benchmark

    def summary(self) -> str:
        """Human-readable summary for the LLM system prompt."""
        lines = []
        if self.config:
            c = self.config
            lines.append(f"Problem: {c.problem_class}")

            if self.instance_source:
                lines.append(f"Instance source: {self.instance_source}")
            else:
                lines.append("Instance source: NOT CHOSEN — YOU MUST ASK: 'Use generators, custom JSON file, or benchmark suite?' Do NOT call use_generators until user chooses.")

            gens = c.instance_config.generators
            if gens:
                gen_strs = [f"{g.type}({json.dumps(g.params)}, sizes={g.sizes}, count={g.count_per_size})" for g in gens]
                lines.append(f"Generators: {'; '.join(gen_strs)}")
            custom = getattr(c.instance_config, 'custom_instances', None)
            if custom:
                lines.append(f"Custom instances: {len(custom)} loaded")
            ec = c.execution_config
            lines.append(f"Runs per instance: {ec.runs_per_config}")
            lines.append(f"Timeout: {ec.timeout_seconds}s")
        else:
            lines.append("Config: NOT SET (run intake first)")

        if self.preferred_runs_per_instance is not None:
            lines.append(f"User requested runs per instance: {self.preferred_runs_per_instance} (already applied)")
        if self.preferred_plot_requests:
            lines.append(f"User requested plots (generate after benchmark): {', '.join(self.preferred_plot_requests)}")

        if self.algo_specs:
            specs = [f"[{i}] {s.name} ({s.source})" for i, s in enumerate(self.algo_specs)]
            lines.append(f"Paper algorithms (extracted, not yet coded): {', '.join(specs)}")
            if self.preferred_algo_spec_indices is not None:
                lines.append(f"User requested algorithms (by index): {self.preferred_algo_spec_indices} — call code_algorithm with these indices without asking.")
            elif not self.algorithms or len(self.algorithms) <= (1 if self.custom_algo_name else 0):
                lines.append("ACTION REQUIRED: Ask user 'Which of these do you want me to implement? (Reply by index, e.g. 0, 1, or 0 2, or all)' BEFORE asking about instance source.")

        if self.algorithms:
            algo_names = [a.name for a in self.algorithms]
            lines.append(f"Registered algorithms: {', '.join(algo_names)}")
        else:
            lines.append("Registered algorithms: NONE")

        if self.results is not None:
            lines.append(f"Benchmark results: {len(self.results)} rows")
        else:
            lines.append("Benchmark results: NOT RUN")

        lines.append(f"Execution mode: {self.execution_mode}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Tool Definitions (Claude tool_use format)
# ──────────────────────────────────────────────────────────────

ORCHESTRATOR_TOOLS = [
    {
        "name": "run_intake",
        "description": "Run the IntakeAgent to analyze the user's problem description (and optional PDFs) to produce a BenchmarkConfig and extract algorithm specs. Pass the full user message as description — it may also include runs per instance, which algorithms to implement, and which plots they want; these are extracted and carried forward so you do not ask later.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The user's full message (problem description and optionally: runs/trials per instance, which paper algorithms to implement, what plots they want).",
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "modify_generators",
        "description": "Modify a generator's params, sizes, or count. Use when the user wants to change graph density, sizes, or counts. Can accept natural language like 'make graphs denser'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "generator_index": {
                    "type": "integer",
                    "description": "0-based index of the generator to modify.",
                },
                "new_params": {
                    "type": "object",
                    "description": "New params dict for the generator (e.g. {\"p\": 0.7}).",
                },
                "new_sizes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "New sizes list (e.g. [50, 100, 200, 500]).",
                },
                "new_count": {
                    "type": "integer",
                    "description": "New count_per_size value.",
                },
            },
            "required": ["generator_index"],
        },
    },
    {
        "name": "modify_execution_config",
        "description": "Change runs_per_instance, timeout, or memory limit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "runs_per_instance": {"type": "integer", "description": "Number of runs per instance."},
                "timeout_seconds": {"type": "number", "description": "Timeout per run in seconds."},
                "memory_limit_mb": {"type": "number", "description": "Memory limit in MB."},
            },
        },
    },
    {
        "name": "code_algorithm",
        "description": "Generate and register code for paper-extracted algorithm(s). Call ONLY after the user has told you which algorithm(s) to implement (by index). Do not call before asking the user which of the listed paper algorithms they want.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "0-based indices into algo_specs (e.g. [0, 2] for first and third).",
                },
            },
            "required": ["spec_indices"],
        },
    },
    {
        "name": "remove_algorithm",
        "description": "Remove a registered algorithm by name or index.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the algorithm to remove."},
                "index": {"type": "integer", "description": "0-based index of the algorithm to remove."},
            },
        },
    },
    {
        "name": "show_status",
        "description": "Display the current pipeline state: config, generators, algorithms, results.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "run_benchmark",
        "description": "Execute the benchmark with the current config and algorithms. Requires at least one algorithm to be registered.",
        "input_schema": {
            "type": "object",
            "properties": {
                "execution_mode": {
                    "type": "string",
                    "enum": ["local", "modal"],
                    "description": "Execution mode. Defaults to current state setting.",
                },
            },
        },
    },
    {
        "name": "analyze_results",
        "description": "Generate a visualization or analysis of benchmark results. Call ONLY when the user explicitly asks for a plot, chart, or analysis (e.g. 'plot a bar chart', 'show me runtime comparison'). Do NOT call automatically after run_benchmark — wait for the user to request a visualization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The user's visualization/analysis request (e.g. 'bar chart comparing algorithms'). Use what the user asked for, not a default.",
                },
            },
            "required": ["request"],
        },
    },
    {
        "name": "go_back",
        "description": "Undo/reset a specific part of the pipeline. For example, clear algorithms to re-select them, or clear config to re-run intake.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["config", "algorithms", "results", "all"],
                    "description": "What to reset. 'config' clears config+algorithms+results. 'algorithms' clears just algorithms. 'results' clears just results. 'all' resets everything.",
                },
            },
            "required": ["target"],
        },
    },
    {
        "name": "export_results",
        "description": "Export benchmark results to a CSV file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Output filename. Defaults to 'benchmark_results.csv'.",
                    "default": "benchmark_results.csv",
                },
            },
        },
    },
    {
        "name": "set_execution_mode",
        "description": "Switch between local and modal (remote sandbox) execution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["local", "modal"],
                    "description": "Execution mode to use.",
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "use_generators",
        "description": "Set instance source to generators. ONLY call this when the user has explicitly said they want to use the proposed generators. Do NOT call by default after intake — ask the user first to choose: generators, custom file, or suite.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "load_custom_instances",
        "description": "Set instance source to custom and load graph instances from a JSON file. Call when the user wants to provide their own instance file (not the default).",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the JSON file containing graph instances.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "load_suite",
        "description": "Set instance source to suite and load instances from a benchmark suite (DIMACS, Biq Mac, SNAP). First call with suite_key and empty instance_names to list instances; then WAIT for the user to say which instances they want (by name). Only call again with instance_names when the user has specified which to load — do NOT pass all instance names unless the user explicitly asked for all.",
        "input_schema": {
            "type": "object",
            "properties": {
                "list_suites": {
                    "type": "boolean",
                    "description": "If true, just list available suites without loading anything.",
                },
                "suite_key": {
                    "type": "string",
                    "description": "Key of the suite to browse (e.g. biqmac).",
                },
                "instance_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Instance names to load. Leave empty or omit to only list instances. Only fill with names AFTER the user has told you which ones they want (e.g. g05_60.0, g05_80.0). Never pass the full list unless the user said 'all' or 'load all'.",
                },
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the Benchwarmer.AI orchestrator — a conversational assistant that guides \
users through setting up and running algorithm benchmarks.

## Your Role
You manage the benchmarking pipeline conversationally. The user talks to you in \
natural language and you use your tools to take actions. You should be helpful, \
concise, and proactive.

## Pipeline Steps (order matters after intake)
1. **Intake**: Analyze the problem description (+ optional PDFs) → BenchmarkConfig + AlgorithmSpecs. The user may also specify in the same message: runs per instance, which algorithms to implement, and which plots they want — these are carried forward so you do not ask again.
2. **Which algorithms to implement**: If state shows "User requested algorithms (by index)", call code_algorithm with those indices immediately — do NOT ask. If paper algorithms exist AND the user uploaded PDFs, implement ALL paper algorithm after asking, and confirming— call code_algorithm with all indicec. If more than 1 algorithm is present, always prompt the user to confirm which algorithms to run.
3. **Instance source** (REQUIRED before benchmark): The user MUST choose one: generator, custom, or suite (see tools).
4. **Configure**: Modify generators/instances, execution settings
5. **Benchmark**: Run the benchmark → you receive a **summary table** with avg runtime, avg memory, success rate per algorithm. Present this table to the user verbatim (keep the markdown table format). No charts are generated at this stage.
6. **Analysis**: After presenting the benchmark summary table, ask: "Would you like any plots or visualizations of these results?" Only call analyze_results when the user explicitly says yes or requests specific plots. If state shows "User requested plots (generate after benchmark)", call analyze_results for EACH requested plot after asking the user, then ask if they want more.

## Key Behaviors
- **Short replies like "0", "1", "0 2", "all", "generators", "yes", "run" are ANSWERS to your previous question.** Interpret them in context of what you last asked. For example, if you asked "Which algorithms do you want me to implement?" and the user replies "0", that means "implement algorithm at index 0" — call code_algorithm with spec_indices=[0]. If you asked about instance source and the user replies "generators", call use_generators. NEVER ask for clarification when the user gives a clear index or keyword.
- When "User requested algorithms (by index)" is in state, call code_algorithm with that list without asking. When "User requested plots" is in state, after run_benchmark call analyze_results for each listed request, then ask if they want more.
- When "Instance source: NOT CHOSEN" appears in state, ask the user to choose: generator, custom JSON file, or benchmark suite. Do NOT call use_generators until the user explicitly chooses generators.
- When using load_suite: after you list instances (first call with empty instance_names), you MUST wait for the user to tell you which instance names to load. Do NOT call load_suite again with all names — only pass the instance_names the user asked for. If the user says "all" or "load all", then you may pass the full list.
- Do NOT call analyze_results unless the user explicitly asks for a plot/chart/visualization. Exception: if state lists "User requested plots", ask the user to confirm, then call analyze_results for each.
- After run_benchmark, you get a markdown summary table. Present it to the user EXACTLY (keep the | table format). Then ask "Would you like any plots or visualizations?" Do NOT generate plots automatically.
- The user can go back to any step at any time
- If the user says "go back", "undo", "restart", or "change X", use the appropriate tool
- If the user describes a problem, run intake automatically
- Always be concise — don't repeat information the user already knows
- If unsure what the user wants, ask a brief clarifying question
- When showing status, be brief — just key facts

## Current Pipeline State
{state_summary}
"""


# ──────────────────────────────────────────────────────────────
# User preference extraction (from initial message)
# ──────────────────────────────────────────────────────────────

def _extract_user_preferences(backend: Any, user_message: str, algo_specs: list) -> dict[str, Any]:
    """
    Extract runs_per_instance, algo_indices, and plot_requests from the user's
    message so we can carry them forward without asking again.
    Returns dict with keys: runs_per_instance (int|None), algo_indices (list[int]|None), plot_requests (list[str]|None).
    """
    if not user_message.strip():
        return {"runs_per_instance": None, "algo_indices": None, "plot_requests": None}

    algo_list = "\n".join(f"  [{i}] {s.name}" for i, s in enumerate(algo_specs)) if algo_specs else "(none)"

    system = """You extract structured preferences from the user's message. Reply with ONLY a single JSON object, no markdown, no explanation.
Use these exact keys:
- "runs_per_instance": integer or null (e.g. 5, 10 if they said "5 runs", "10 trials per instance")
- "algo_indices": list of 0-based integers or null (which paper algorithms to implement; if they say "all" use null to mean ask later; if they name algorithms match by name to index)
- "plot_requests": list of strings or null (each string is a natural-language plot/visualization request, e.g. "bar chart of runtimes", "scatter objective vs size")
If something is not mentioned, use null for that key."""

    user_content = f"""User message:
{user_message}

Paper algorithms (by index):
{algo_list}

Extract runs_per_instance, algo_indices, and plot_requests. Output only the JSON object."""

    try:
        response = backend.generate(
            messages=[{"role": "user", "content": user_content}],
            system=system,
            tools=None,
            max_tokens=512,
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        text = text.strip()
        # Strip markdown code block if present
        if "```" in text:
            for marker in ("```json", "```"):
                if marker in text:
                    start = text.index(marker) + len(marker)
                    end = text.find("```", start)
                    text = text[start:(end if end != -1 else None)].strip()
                    break
        data = json.loads(text)
        runs = data.get("runs_per_instance")
        if runs is not None and not isinstance(runs, int):
            runs = int(runs) if isinstance(runs, (float, str)) and str(runs).isdigit() else None
        indices = data.get("algo_indices")
        if indices is not None and not isinstance(indices, list):
            indices = None
        if indices is not None:
            indices = [int(i) for i in indices if isinstance(i, (int, float)) or (isinstance(i, str) and i.isdigit())]
        plots = data.get("plot_requests")
        if plots is not None and not isinstance(plots, list):
            plots = [str(plots)] if plots else None
        if plots is not None:
            plots = [str(p).strip() for p in plots if p]
        return {"runs_per_instance": runs, "algo_indices": indices if indices else None, "plot_requests": plots or None}
    except Exception as e:
        logger.warning("Could not extract user preferences: %s", e)
        return {"runs_per_instance": None, "algo_indices": None, "plot_requests": None}


# ──────────────────────────────────────────────────────────────
# Orchestrator Agent
# ──────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """Conversational orchestrator for the benchmarking pipeline."""

    def __init__(
        self,
        execution_mode: str = "local",
        intake_backend: str = "claude",
        pdf_paths: list[str] | None = None,
        custom_algo_path: str | None = None,
        nemotron_url: str | None = None,
        nemotron_model: str | None = None,
    ):
        from benchwarmer.agents.backends import ClaudeBackend, OpenAIBackend

        # Same LLM backend for both orchestrator and intake (claude or nemotron)
        if intake_backend == "nemotron":
            self.backend = OpenAIBackend(
                base_url=nemotron_url or "https://integrate.api.nvidia.com/v1",
                model=nemotron_model or "nvidia/nemotron-3-nano-30b-a3b",
            )
        else:
            self.backend = ClaudeBackend()
        self.state = PipelineState(
            execution_mode=execution_mode,
            pdf_paths=pdf_paths or [],
            custom_algo_path=custom_algo_path,
        )
        self.intake_backend_name = intake_backend
        self.nemotron_url = nemotron_url
        self.nemotron_model = nemotron_model
        self.plot_index = 0
        self._plot_agent = None
        self._impl_agent = None

        # Load custom algorithm if provided
        if self.state.custom_algo_path:
            if os.path.exists(self.state.custom_algo_path):
                try:
                    algo = load_algorithm_from_file(self.state.custom_algo_path)
                    self.state.algorithms.append(algo)
                    self.state.custom_algo_name = algo.name
                    print(f"✅ Loaded custom algorithm: {algo.name}")
                except Exception as e:
                    print(f"❌ Failed to load custom algorithm: {e}")
            else:
                print(f"❌ Custom algorithm file not found: {self.state.custom_algo_path}")

    # ------------------------------------------------------------------
    # One-turn API (for web frontend multi-turn chat)
    # ------------------------------------------------------------------

    def run_one_turn(
        self,
        messages: list[dict[str, Any]],
        user_message: str,
        max_tool_turns: int = 15,
        progress_cb: Callable[[dict], None] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Append *user_message*, run LLM tool loop until final text reply.
        Returns ``(updated_messages, assistant_reply_text)``.

        If *progress_cb* is provided it is called with event dicts:
            {"type": "thinking"}
            {"type": "tool_start", "tool": <name>, "input": <dict>}
            {"type": "tool_end",   "tool": <name>, "result": <str>}
        """
        messages.append({"role": "user", "content": user_message})

        system = ORCHESTRATOR_SYSTEM_PROMPT.format(
            state_summary=self.state.summary(),
        )
        reply_text = ""

        for _ in range(max_tool_turns):
            if progress_cb:
                progress_cb({"type": "thinking"})

            try:
                response = self.backend.generate(
                    messages=messages,
                    system=system,
                    tools=ORCHESTRATOR_TOOLS,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.exception("LLM generate error: %s", e)
                reply_text = f"Error: {e}"
                break

            if response.stop_reason == "tool_use":
                assistant_content = [asdict(b) for b in response.content]
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        logger.info("Tool call: %s", block.name)
                        if progress_cb:
                            progress_cb({"type": "tool_start", "tool": block.name, "input": block.input})
                        result_str = self._dispatch_tool(block.name, block.input)
                        if progress_cb:
                            progress_cb({"type": "tool_end", "tool": block.name, "result": result_str[:1500]})
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })
                messages.append({"role": "user", "content": tool_results})

                # Refresh system prompt (state may have changed)
                system = ORCHESTRATOR_SYSTEM_PROMPT.format(
                    state_summary=self.state.summary(),
                )
                continue

            # Final text reply
            if response.stop_reason in ("end_turn", "stop"):
                for block in response.content:
                    if hasattr(block, "text"):
                        reply_text += block.text
                serialized = [asdict(b) for b in response.content]
                messages.append({"role": "assistant", "content": serialized})
                break

            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        return messages, reply_text.strip()

    # ------------------------------------------------------------------
    # Main loop (CLI)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the conversational orchestrator loop."""
        print()
        print("=" * 60)
        print("  🏋️  Benchwarmer.AI — Conversational Mode")
        print("=" * 60)
        print()
        print("Describe your optimization problem, or type a command.")
        print("You can modify anything at any step — just ask!")
        print("Type 'exit' or 'quit' to stop.\n")

        messages: list[dict[str, Any]] = []

        # If PDF paths were provided via CLI, mention them in the initial context
        if self.state.pdf_paths:
            pdf_names = [os.path.basename(p) for p in self.state.pdf_paths]
            messages.append({
                "role": "user",
                "content": f"I have these PDF papers to analyze: {', '.join(pdf_names)}. "
                           f"I'll describe my problem now.",
            })
            messages.append({
                "role": "assistant",
                "content": "Great! I can see your papers. Go ahead and describe "
                           "your optimization problem, and I'll analyze both your "
                           "description and the papers together.",
            })

        while True:
            try:
                user_input = input("➤ ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                self._cleanup()
                break

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q", "bye"):
                print("👋 Goodbye!")
                self._cleanup()
                break

            messages.append({"role": "user", "content": user_input})

            # Build system prompt with current state
            system = ORCHESTRATOR_SYSTEM_PROMPT.format(
                state_summary=self.state.summary(),
            )

            # Conversational loop — handle tool calls
            max_turns = 10
            for _ in range(max_turns):
                try:
                    response = self.backend.generate(
                        messages=messages,
                        system=system,
                        tools=ORCHESTRATOR_TOOLS,
                        max_tokens=4096,
                    )
                except Exception as e:
                    print(f"\n❌ LLM error: {e}\n")
                    # Remove the last user message to avoid stuck state
                    if messages and messages[-1]["role"] == "user":
                        messages.pop()
                    break

                # Handle tool use
                if response.stop_reason == "tool_use":
                    assistant_content = [asdict(b) for b in response.content]
                    messages.append({"role": "assistant", "content": assistant_content})

                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            logger.info("Orchestrator tool: %s", block.name)
                            result_str = self._dispatch_tool(block.name, block.input)
                            print(f"  {result_str}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_str,
                            })

                    messages.append({"role": "user", "content": tool_results})
                    continue

                # Handle text response
                if response.stop_reason in ("end_turn", "stop"):
                    text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            text += block.text
                    if text:
                        print(f"\n{text}\n")
                    serialized = [asdict(b) for b in response.content]
                    messages.append({"role": "assistant", "content": serialized})
                    break

                # Unexpected
                logger.warning("Unexpected stop_reason: %s", response.stop_reason)
                break

    # ------------------------------------------------------------------
    # Tool Dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, name: str, input_data: dict) -> str:
        """Route tool calls to handlers and return result strings."""
        handlers = {
            "run_intake": self._tool_run_intake,
            "modify_generators": self._tool_modify_generators,
            "modify_execution_config": self._tool_modify_execution_config,
            "code_algorithm": self._tool_code_algorithm,

            "remove_algorithm": self._tool_remove_algorithm,
            "show_status": self._tool_show_status,
            "run_benchmark": self._tool_run_benchmark,
            "analyze_results": self._tool_analyze_results,
            "go_back": self._tool_go_back,
            "export_results": self._tool_export_results,
            "set_execution_mode": self._tool_set_execution_mode,
            "use_generators": self._tool_use_generators,
            "load_custom_instances": self._tool_load_custom_instances,
            "load_suite": self._tool_load_suite,
        }
        handler = handlers.get(name)
        if not handler:
            return f"❌ Unknown tool: {name}"
        try:
            return handler(input_data)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e, exc_info=True)
            return f"❌ Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool Handlers
    # ------------------------------------------------------------------

    def _tool_run_intake(self, data: dict) -> str:
        description = data.get("description", "")
        if not description:
            return "❌ No description provided."

        from benchwarmer.agents.intake import IntakeAgent

        # Use same backend as orchestrator (claude or nemotron)
        agent = IntakeAgent(backend=self.backend)

        # Validate PDFs
        pdf_paths = None
        if self.state.pdf_paths:
            pdf_paths = [p for p in self.state.pdf_paths if os.path.exists(p)]
            if pdf_paths:
                names = [os.path.basename(p) for p in pdf_paths]
                print(f"  📄 Analyzing papers: {', '.join(names)}")

        result = agent.run(description, pdf_paths=pdf_paths, interactive=False)
        self.state.config = result.config
        self.state.algo_specs = result.algorithms
        self.state.instance_source = None  # Reset — user must choose

        # Extract preferences from the same user message so we don't ask later
        prefs = _extract_user_preferences(self.backend, description, self.state.algo_specs)
        self.state.preferred_runs_per_instance = prefs.get("runs_per_instance")
        self.state.preferred_algo_spec_indices = prefs.get("algo_indices")
        self.state.preferred_plot_requests = prefs.get("plot_requests")
        if self.state.preferred_runs_per_instance is not None:
            self.state.config.execution_config.runs_per_config = self.state.preferred_runs_per_instance

        lines = [f"✅ IntakeAgent complete! Problem class: {result.config.problem_class}"]

        gens = result.config.instance_config.generators
        if gens:
            lines.append(f"   Proposed generators: {', '.join(g.type for g in gens)}")

        ec = result.config.execution_config
        lines.append(f"   Runs: {ec.runs_per_config}, Timeout: {ec.timeout_seconds}s")

        if self.state.preferred_runs_per_instance is not None:
            lines.append(f"   (Using {self.state.preferred_runs_per_instance} runs per instance from your message)")
        if self.state.preferred_algo_spec_indices is not None:
            names = [self.state.algo_specs[i].name for i in self.state.preferred_algo_spec_indices if 0 <= i < len(self.state.algo_specs)]
            lines.append(f"   Will implement: {', '.join(names)} (from your message)")
        if self.state.preferred_plot_requests:
            lines.append(f"   Will generate these plots after benchmark: {', '.join(self.state.preferred_plot_requests)}")

        if self.state.algo_specs:
            lines.append(f"\n📋 Extracted {len(self.state.algo_specs)} algorithm(s) from papers:")
            for i, spec in enumerate(self.state.algo_specs):
                lines.append(f"   [{i}] {spec.name}: {spec.approach} (source: {spec.source})")

        if not self.state.instance_source:
            lines.append("\nHow would you like to provide instances? (generator / custom JSON / benchmark suite)")

        return "\n".join(lines)

    def _tool_modify_generators(self, data: dict) -> str:
        if not self.state.config:
            return "❌ No config — run intake first."

        gens = self.state.config.instance_config.generators
        idx = data.get("generator_index", 0)
        if idx < 0 or idx >= len(gens):
            return f"❌ Invalid generator index {idx}. Have {len(gens)} generators (0-{len(gens)-1})."

        g = gens[idx]
        changes = []

        if "new_params" in data and data["new_params"]:
            g.params = data["new_params"]
            changes.append(f"params → {json.dumps(g.params)}")

        if "new_sizes" in data and data["new_sizes"]:
            g.sizes = data["new_sizes"]
            changes.append(f"sizes → {g.sizes}")

        if "new_count" in data and data["new_count"] is not None:
            g.count_per_size = data["new_count"]
            changes.append(f"count → {g.count_per_size}")

        if changes:
            return f"✨ Generator #{idx} ({g.type}): {', '.join(changes)}"
        return f"ℹ️ No changes made to generator #{idx}."

    def _tool_modify_execution_config(self, data: dict) -> str:
        if not self.state.config:
            return "❌ No config — run intake first."

        ec = self.state.config.execution_config
        changes = []

        if "runs_per_instance" in data and data["runs_per_instance"] is not None:
            ec.runs_per_config = data["runs_per_instance"]
            changes.append(f"runs_per_instance → {ec.runs_per_config}")

        if "timeout_seconds" in data and data["timeout_seconds"] is not None:
            ec.timeout_seconds = data["timeout_seconds"]
            changes.append(f"timeout → {ec.timeout_seconds}s")

        if "memory_limit_mb" in data and data["memory_limit_mb"] is not None:
            ec.memory_limit_mb = data["memory_limit_mb"]
            changes.append(f"memory_limit → {ec.memory_limit_mb} MB")

        if changes:
            return f"✨ Updated: {', '.join(changes)}"
        return "ℹ️ No changes specified."

    def _tool_code_algorithm(self, data: dict) -> str:
        if not self.state.config:
            return "❌ No config — run intake first."

        if self._impl_agent is None:
            from benchwarmer.agents.implementation import ImplementationAgent
            self._impl_agent = ImplementationAgent()

        results_msgs = []

        # Code from spec indices
        spec_indices = data.get("spec_indices", [])
        if spec_indices:
            for idx in spec_indices:
                if idx < 0 or idx >= len(self.state.algo_specs):
                    results_msgs.append(f"❌ Invalid spec index {idx}")
                    continue
                spec = self.state.algo_specs[idx]
                prompt = (
                    f"Implement this algorithm: {spec.name}\n\n"
                    f"Approach: {spec.approach}\n"
                    f"Complexity: {spec.complexity}\n\n"
                    f"Key steps:\n"
                )
                for i, step in enumerate(spec.key_steps, 1):
                    prompt += f"  {i}. {step}\n"

                print(f"  🤖 Coding '{spec.name}'…")
                result = self._impl_agent.generate(
                    description=prompt,
                    problem_class=self.state.config.problem_class,
                    pool=self.state.pool,
                )
                if result["success"]:
                    self.state.algorithms.append(result["algorithm"])
                    results_msgs.append(f"✅ {result['name']} coded and tested!")
                else:
                    results_msgs.append(f"❌ {spec.name} failed: {result['error']}")



        return "\n".join(results_msgs) if results_msgs else "ℹ️ No algorithms specified to code."



    def _tool_remove_algorithm(self, data: dict) -> str:
        name = data.get("name")
        index = data.get("index")

        if index is not None:
            if 0 <= index < len(self.state.algorithms):
                removed = self.state.algorithms.pop(index)
                return f"🗑️ Removed: {removed.name}"
            return f"❌ Invalid index {index}. Have {len(self.state.algorithms)} algorithms."

        if name:
            for i, algo in enumerate(self.state.algorithms):
                if algo.name == name:
                    self.state.algorithms.pop(i)
                    return f"🗑️ Removed: {name}"
            return f"❌ Algorithm '{name}' not found."

        return "❌ Specify name or index to remove."

    def _tool_show_status(self, _data: dict) -> str:
        return f"📋 Current State:\n{self.state.summary()}"

    def _tool_run_benchmark(self, data: dict) -> str:
        if not self.state.config:
            return "❌ No config — run intake first."
        if not self.state.instance_source:
            return "❌ Instance source not chosen. Ask the user: use generators (use_generators), custom JSON file (load_custom_instances), or benchmark suite (load_suite)."
        if not self.state.algorithms:
            return "❌ No algorithms registered. Code or add baselines first."

        mode = data.get("execution_mode", self.state.execution_mode)

        from benchwarmer.engine.runner import BenchmarkRunner

        runner = BenchmarkRunner(self.state.config)
        for algo in self.state.algorithms:
            runner.register_algorithm(algo)

        # Modal pool
        pool = self.state.pool
        if mode == "modal" and pool is None:
            from benchwarmer.engine.sandbox_pool import SandboxPool
            pool = SandboxPool()
            self.state.pool = pool

        algo_names = [a.name for a in self.state.algorithms]
        total_inst = sum(
            len(g.sizes) * g.count_per_size
            for g in self.state.config.instance_config.generators
        )
        runs = self.state.config.execution_config.runs_per_config
        print(f"  🚀 Running benchmark ({mode}): {len(algo_names)} algos × {total_inst} instances × {runs} runs")

        try:
            df = runner.run(execution_mode=mode, sandbox_pool=pool)
        except Exception as e:
            return f"❌ Benchmark failed: {e}"
        finally:
            if pool and mode == "modal":
                try:
                    pool.teardown_all_sync()
                except Exception:
                    pass

        self.state.results = df

        # ── Build CLI-style summary table ──────────────────────────────
        import pandas as _pd

        success_df = df[df["status"] == "success"] if "status" in df.columns else df
        total_rows = len(df)
        success_rows = len(success_df)

        lines = [f"✅ Benchmark complete! {total_rows} result rows ({success_rows} successful)."]

        if not success_df.empty and "algorithm_name" in success_df.columns:
            # Build per-algorithm stats
            table_lines = []
            table_lines.append("")
            table_lines.append("| Algorithm | Avg Objective | Avg Runtime (s) | Avg Memory (MB) | Success Rate |")
            table_lines.append("|-----------|--------------|-----------------|-----------------|--------------|")

            for algo_name in df["algorithm_name"].unique():
                algo_all = df[df["algorithm_name"] == algo_name]
                algo_ok = algo_all[algo_all["status"] == "success"] if "status" in algo_all.columns else algo_all
                total = len(algo_all)
                ok = len(algo_ok)
                rate = f"{ok}/{total} ({100 * ok // max(total, 1)}%)"

                avg_obj = f"{algo_ok['objective_value'].mean():.1f}" if "objective_value" in algo_ok.columns and not algo_ok.empty else "—"
                avg_rt = f"{algo_ok['wall_time_seconds'].mean():.4f}" if "wall_time_seconds" in algo_ok.columns and not algo_ok.empty else "—"
                avg_mem = f"{algo_ok['peak_memory_mb'].mean():.1f}" if "peak_memory_mb" in algo_ok.columns and not algo_ok.empty else "—"

                table_lines.append(f"| {algo_name} | {avg_obj} | {avg_rt} | {avg_mem} | {rate} |")

            table_lines.append("")
            lines.extend(table_lines)

        # Error sample if any
        if "status" in df.columns:
            error_df = df[df["status"] != "success"]
            if not error_df.empty:
                for col_name in ("error_message", "error", "error_msg"):
                    if col_name in error_df.columns:
                        sample = error_df[col_name].dropna()
                        sample = sample[sample.astype(str).str.strip() != ""]
                        if not sample.empty:
                            lines.append(f"❌ Sample error: {sample.iloc[0]}")
                        break

        msg = "\n".join(lines)

        if self.state.preferred_plot_requests:
            return msg + "\nPresent this summary table to the user EXACTLY as shown above (keep the markdown table formatting). Then generate the requested plots (call analyze_results for each item in state 'User requested plots'), then ask if they would like any other visualizations."
        return msg + "\nPresent this summary table to the user EXACTLY as shown above (keep the markdown table formatting). Then ask: 'Would you like any plots or visualizations of these results?' Do NOT call analyze_results until the user asks."

    def _tool_analyze_results(self, data: dict) -> str:
        if self.state.results is None:
            return "❌ No results — run benchmark first."

        request = data.get("request", "summary")

        if self._plot_agent is None:
            from benchwarmer.agents.plot import PlotAgent
            self._plot_agent = PlotAgent()

        # Filter: Exclude baselines/external algos, keep only Schema-derived + Custom
        whitelist = set()
        if self.state.custom_algo_name:
            whitelist.add(self.state.custom_algo_name)
        for spec in self.state.algo_specs:
            whitelist.add(spec.name)
        
        # Also include any algo currently in self.state.algorithms that matches a spec
        # (Just in case names differ slightly, but usually they match spec.name)
        
        df = self.state.results
        if whitelist and "algorithm_name" in df.columns:
            # Only filter if we actually have a whitelist (i.e. intake run or custom loaded)
            # If user manually added everything without intake, filtering might hide everything.
            # But user specific request: "remove ability to add any other... than papers... and custom"
            filtered_df = df[df["algorithm_name"].isin(whitelist)]
            if not filtered_df.empty:
                df = filtered_df
        
        self._plot_agent.set_dataframe(df)

        print(f"  📊 Generating visualization for {len(df)} rows ({df['algorithm_name'].nunique()} algos)…")
        result = self._plot_agent.generate_and_execute(
            user_request=request,
            df=df,
            output_dir="plots",
            plot_index=self.plot_index,
        )

        if result["success"]:
            self.plot_index += 1
            if result.get("output_path") and os.path.exists(result["output_path"]):
                self.state.last_plot_path = result["output_path"]
                return f"✅ Plot generated and displayed to the user. File: {result['output_path']}"
            elif result.get("message"):
                return result["message"]
            if result.get("stdout"):
                return result["stdout"]
            return "✅ Analysis complete."
        else:
            return f"❌ Error: {result.get('error', 'Unknown error')}"

    def _tool_go_back(self, data: dict) -> str:
        target = data.get("target", "algorithms")

        if target == "all":
            self.state.config = None
            self.state.algo_specs = []
            self.state.algorithms = []
            self.state.results = None
            return "↩️ Reset everything. Describe your problem to start fresh."

        if target == "config":
            self.state.config = None
            self.state.algo_specs = []
            self.state.algorithms = []
            self.state.results = None
            return "↩️ Cleared config, algorithms, and results. Describe your problem again."

        if target == "algorithms":
            self.state.algorithms = []
            self.state.results = None
            msg = "↩️ Cleared all registered algorithms."
            if self.state.algo_specs:
                specs = [f"[{i}] {s.name}" for i, s in enumerate(self.state.algo_specs)]
                msg += f"\n   Paper algorithms still available: {', '.join(specs)}"
            return msg

        if target == "results":
            self.state.results = None
            return "↩️ Cleared benchmark results. You can re-run with current algorithms."

        return f"❌ Unknown target '{target}'. Use: config, algorithms, results, or all."

    def _tool_export_results(self, data: dict) -> str:
        if self.state.results is None:
            return "❌ No results to export."
        filename = data.get("filename", "benchmark_results.csv")
        self.state.results.to_csv(filename, index=False)
        return f"📁 Exported {len(self.state.results)} rows to {filename}"

    def _tool_set_execution_mode(self, data: dict) -> str:
        mode = data.get("mode", "local")
        self.state.execution_mode = mode
        return f"✨ Execution mode set to: {mode}"

    def _tool_use_generators(self, _data: dict) -> str:
        if not self.state.config:
            return "❌ No config — run intake first."
        gens = self.state.config.instance_config.generators
        if not gens:
            return "❌ No generators were proposed. Try custom instances or a suite."
        self.state.instance_source = "generator"
        total = sum(len(g.sizes) * g.count_per_size for g in gens)
        gen_info = [f"{g.type}({json.dumps(g.params)}, sizes={g.sizes}, count={g.count_per_size})" for g in gens]
        return f"✅ Using generators ({total} total instances):\n   " + "\n   ".join(gen_info)

    def _tool_load_custom_instances(self, data: dict) -> str:
        if not self.state.config:
            return "❌ No config — run intake first."
        file_path = data.get("file_path", "")
        if not file_path:
            return "❌ No file path provided."
        if not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"
        try:
            from benchwarmer.utils.instance_loader import load_instances
            instances = load_instances(file_path)
            self.state.config.instance_config.generators = []
            self.state.config.instance_config.custom_instances = instances
            self.state.instance_source = "custom"
            lines = [f"✅ Loaded {len(instances)} instance(s) from {file_path}:"]
            for inst in instances:
                n_nodes = len(inst.get('nodes', []))
                n_edges = len(inst.get('edges', []))
                name = inst.get('instance_name', '?')
                lines.append(f"   • {name}: {n_nodes} nodes, {n_edges} edges")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error loading instances: {e}"

    def _tool_load_suite(self, data: dict) -> str:
        if not self.state.config:
            return "❌ No config — run intake first."
        try:
            from benchwarmer.utils.benchmark_suites import (
                list_suites, list_instances, fetch_instance,
            )
        except ImportError:
            return "❌ Benchmark suites module not available."

        # List suites mode
        if data.get("list_suites"):
            suites = list_suites(problem_class=self.state.config.problem_class)
            if not suites:
                suites = list_suites()
            if not suites:
                return "ℹ️ No benchmark suites available."
            lines = ["📦 Available benchmark suites:"]
            for s in suites:
                lines.append(f"   • {s['key']}: {s['name']} ({s['instance_count']} instances)")
                lines.append(f"     {s['description']}")
            return "\n".join(lines)

        suite_key = data.get("suite_key")
        if not suite_key:
            return "❌ Provide suite_key, or set list_suites=true to see available suites."

        instance_names = data.get("instance_names", [])

        # List instances in suite — wait for user to choose which to load
        if not instance_names:
            instances = list_instances(suite_key)
            lines = [f"📋 Instances in '{suite_key}':"]
            for inst in instances:
                lines.append(f"   • {inst['name']} ({inst.get('nodes', '?')} nodes)")
            lines.append("")
            lines.append("Which instances would you like to load? (Reply with names, e.g. g05_60.0, g05_80.0 — or 'all' for all.)")
            return "\n".join(lines)

        # Fetch specific instances
        loaded = []
        for name in instance_names:
            try:
                parsed = fetch_instance(suite_key, name)
                loaded.append(parsed)
                print(f"  ⬇️ {name}: {len(parsed['nodes'])} nodes, {len(parsed['edges'])} edges")
            except Exception as e:
                print(f"  ❌ {name}: {e}")

        if not loaded:
            return "❌ No instances were loaded."

        self.state.config.instance_config.generators = []
        self.state.config.instance_config.custom_instances = loaded
        self.state.instance_source = "suite"
        return f"✅ Loaded {len(loaded)} benchmark instance(s) from '{suite_key}'!"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self):
        """Clean up resources."""
        if self.state.pool:
            try:
                self.state.pool.teardown_all_sync()
            except Exception:
                pass
