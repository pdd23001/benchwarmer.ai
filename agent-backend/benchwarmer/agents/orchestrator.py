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
from typing import Any, Optional

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

    def summary(self) -> str:
        """Human-readable summary for the LLM system prompt."""
        lines = []
        if self.config:
            c = self.config
            lines.append(f"Problem: {c.problem_class}")

            if self.instance_source:
                lines.append(f"Instance source: {self.instance_source}")
            else:
                lines.append("Instance source: NOT CHOSEN (ask user: generator, custom JSON, or benchmark suite)")

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

        if self.algo_specs:
            specs = [f"{s.name} ({s.source})" for s in self.algo_specs]
            lines.append(f"Paper algorithms (extracted, not yet coded): {', '.join(specs)}")

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
        "description": "Run the IntakeAgent to analyze the user's problem description (and optional PDFs) to produce a BenchmarkConfig and extract algorithm specs from papers. Call this first to set up the benchmark.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The user's natural-language problem description (e.g. 'max cut', 'vertex cover on sparse graphs').",
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
        "description": "Use the ImplementationAgent to generate code for an algorithm. Can ONLY be called with an index referencing a paper algorithm spec.",
        "input_schema": {
            "type": "object",
            "properties": {
                "spec_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "0-based indices into algo_specs to code from paper extractions.",
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
        "description": "Generate a visualization or analysis of benchmark results using the PlotAgent. Requires benchmark to have been run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The visualization/analysis request (e.g. 'bar chart comparing algorithms').",
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
        "description": "Confirm using the proposed generators as the instance source (this is set after intake). Call this when the user chooses generators.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "load_custom_instances",
        "description": "Load custom graph instances from a JSON file. Use when user wants to provide their own instances.",
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
        "description": "Browse and load instances from a benchmark suite (DIMACS, Biq Mac, SNAP). Use when user wants standard benchmark instances.",
        "input_schema": {
            "type": "object",
            "properties": {
                "list_suites": {
                    "type": "boolean",
                    "description": "If true, just list available suites without loading anything.",
                },
                "suite_key": {
                    "type": "string",
                    "description": "Key of the suite to browse (returned by list_suites).",
                },
                "instance_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific instance names to fetch from the suite. If empty, lists available instances.",
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

## Pipeline Steps (flexible order)
1. **Intake**: Analyze the problem description (+ optional PDFs) → BenchmarkConfig + AlgorithmSpecs
2. **Instance source**: Ask the user how they want to provide instances:
   - **generator** — use the AI-proposed generators (can modify params/sizes)
   - **custom** — load instances from a JSON file
   - **suite** — pull from benchmark suites (DIMACS, Biq Mac, SNAP)
3. **Configure**: Modify generators/instances, execution settings (runs, timeout)
4. **Algorithms**: Code extracted algorithms from papers
5. **Benchmark**: Run the benchmark
6. **Analysis**: Visualize and analyze results

## Key Behaviors
- After intake, ALWAYS ask the user how they want to provide instances (generator/custom/suite) \
unless they already specified
- The user can go back to any step at any time
- If the user says "go back", "undo", "restart", or "change X", use the appropriate tool
- If the user describes a problem, run intake automatically
- If paper algorithms were extracted, list them and ask which to code
- Always be concise — don't repeat information the user already knows
- If unsure what the user wants, ask a brief clarifying question
- When showing status, be brief — just key facts

## Current Pipeline State
{state_summary}
"""


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
        from benchwarmer.agents.backends import ClaudeBackend

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
    # Main loop
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

        # Build backend
        if self.intake_backend_name == "nemotron":
            from benchwarmer.agents.backends import OpenAIBackend
            backend = OpenAIBackend(
                base_url=self.nemotron_url or "https://integrate.api.nvidia.com/v1",
                model=self.nemotron_model or "nvidia/nemotron-3-nano-30b-a3b",
            )
        else:
            backend = None  # IntakeAgent will use Claude default

        agent = IntakeAgent(backend=backend)

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

        lines = [f"✅ IntakeAgent complete! Problem class: {result.config.problem_class}"]

        gens = result.config.instance_config.generators
        if gens:
            lines.append(f"   Proposed generators: {', '.join(g.type for g in gens)}")

        ec = result.config.execution_config
        lines.append(f"   Runs: {ec.runs_per_config}, Timeout: {ec.timeout_seconds}s")

        if self.state.algo_specs:
            lines.append(f"\n📋 Extracted {len(self.state.algo_specs)} algorithm(s) from papers:")
            for i, spec in enumerate(self.state.algo_specs):
                lines.append(f"   [{i}] {spec.name}: {spec.approach} (source: {spec.source})")

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
        
        # Summarize results
        stats = []
        error_sample = None
        if "algorithm_name" in df.columns and "status" in df.columns:
            for algo_name in df["algorithm_name"].unique():
                sub = df[df["algorithm_name"] == algo_name]
                success = len(sub[sub["status"] == "success"])
                failed = len(sub) - success
                stats.append(f"{algo_name}: {success} OK, {failed} ERR")
                
                if failed > 0 and error_sample is None:
                    # Capture the first error message we find
                    try:
                        errors = pd.Series()
                        if "error" in sub.columns:
                            errors = sub[sub["status"] == "error"]["error"]
                        elif "error_message" in sub.columns:
                            errors = sub[sub["status"] == "error"]["error_message"]
                        elif "error_msg" in sub.columns:
                            errors = sub[sub["status"] == "error"]["error_msg"]
                        
                        if not errors.empty:
                            error_sample = f"Error in {algo_name}: {errors.iloc[0]}"
                    except Exception:
                        error_sample = f"Error in {algo_name}: (Could not extract message)"
        
        stats_str = "; ".join(stats)
        msg = f"✅ Benchmark complete! {len(df)} rows. ({stats_str})."
        if error_sample:
            msg += f"\n❌ {error_sample}"
        return msg + "\nAsk to analyze/plot these results."

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
                return f"✅ Plot saved to: {result['output_path']}"
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

        # List instances in suite
        if not instance_names:
            instances = list_instances(suite_key)
            lines = [f"📋 Instances in '{suite_key}':"]
            for inst in instances:
                lines.append(f"   • {inst['name']} ({inst.get('nodes', '?')} nodes)")
            lines.append("\nTell me which instances to load (by name).")
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
