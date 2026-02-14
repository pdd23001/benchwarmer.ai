#!/usr/bin/env python3
"""
Benchwarmer.AI — Full end-to-end CLI.

1. Intake Agent: NL description → BenchmarkConfig (proposed)
2. Instance selection: generator (confirm/modify params) OR custom JSON
3. Execution Engine: runs benchmarks → Results DataFrame
4. Plot Agent: interactive analysis & visualization loop

Usage
-----
    ANTHROPIC_API_KEY=sk-... python scripts/run_benchmark.py
"""

from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    print("=" * 60)
    print("  🏋️  Benchwarmer.AI — Algorithm Benchmarking Platform")
    print("=" * 60)
    print()

    # ── Step 1: Get problem description ──────────────────────
    print("Describe your optimization problem in natural language.")
    print("(Type your description, then press Enter twice to submit)\n")

    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines and lines[-1] == "":
            lines.pop()
            break
        lines.append(line)

    description = "\n".join(lines).strip()
    if not description:
        print("No description provided. Exiting.")
        return

    print(f"\n📩 Received description ({len(description)} chars)")
    print("-" * 60)

    # ── Step 2: Intake Agent → BenchmarkConfig (proposal) ────
    from benchwarmer.agents.intake import IntakeAgent

    agent = IntakeAgent()
    config = agent.run(description)

    print("\n✅ BenchmarkConfig proposed!")
    print("-" * 60)

    # ── Step 3: Instance source selection ────────────────────
    config = _instance_selection(config)

    # ── Step 4: Register algorithms ─────────────────────────
    algorithms = _algorithm_registration(config)

    if not algorithms:
        print("⚠️  No algorithms registered. Exiting.")
        return

    from benchwarmer.engine.runner import BenchmarkRunner

    runner = BenchmarkRunner(config)
    for algo in algorithms:
        runner.register_algorithm(algo)

    print(f"\n🚀 Running benchmark…")
    df = runner.run()

    print(f"\n✅ Benchmark complete — {len(df)} result rows")
    print("-" * 60)

    _print_summary(df)

    # ── Step 5: Interactive analysis loop ────────────────────
    _analysis_loop(df, config)


# ──────────────────────────────────────────────────────────────
# Instance selection: generator vs custom
# ──────────────────────────────────────────────────────────────

def _instance_selection(config):
    """Ask user whether to use generators or custom instances."""
    from benchwarmer.config import BenchmarkConfig, InstanceConfig, GeneratorConfig

    print()
    print("How would you like to provide problem instances?")
    print("  1) generator  — use the proposed generators (you can modify params)")
    print("  2) custom     — load your own instances from a JSON file")
    print()

    while True:
        choice = input("Choose [1/generator or 2/custom]: ").strip().lower()
        if choice in ("1", "generator", "g", "gen"):
            return _generator_flow(config)
        elif choice in ("2", "custom", "c"):
            return _custom_flow(config)
        else:
            print("   Please enter '1' (generator) or '2' (custom).")


def _generator_flow(config):
    """Show proposed generators, let user confirm or modify."""
    from benchwarmer.config import GeneratorConfig

    generators = config.instance_config.generators

    while True:
        # Display generator table
        print("\n📋 Proposed Generators:")
        print("-" * 70)
        print(f"  {'#':<4} {'Type':<20} {'Params':<25} {'Sizes':<20} {'Count'}")
        print("-" * 70)
        for i, g in enumerate(generators):
            params_str = json.dumps(g.params) if g.params else "{}"
            sizes_str = str(g.sizes)
            print(f"  {i:<4} {g.type:<20} {params_str:<25} {sizes_str:<20} {g.count_per_size}")
            if g.why:
                print(f"       ↳ {g.why}")
        print("-" * 70)

        total_instances = sum(
            len(g.sizes) * g.count_per_size for g in generators
        )
        runs_per = config.execution_config.runs_per_config
        print(f"\n  Total instances: {total_instances}")
        print(f"  Runs per instance: {runs_per}")
        print(f"  Timeout: {config.execution_config.timeout_seconds}s")
        print()

        action = input("Accept this configuration? (yes/modify/add/remove): ").strip().lower()

        if action in ("yes", "y", ""):
            break
        elif action in ("modify", "m"):
            _modify_generator(generators)
        elif action in ("add", "a"):
            _add_generator(generators)
        elif action in ("remove", "r", "delete"):
            _remove_generator(generators)
        else:
            print("   Please enter 'yes', 'modify', 'add', or 'remove'.")

    print("✅ Generator configuration confirmed!")
    return config


def _modify_generator(generators: list):
    """Let user modify params of a specific generator."""
    from benchwarmer.config import GeneratorConfig

    if not generators:
        print("   No generators to modify.")
        return

    try:
        idx = int(input(f"   Which generator # to modify? (0-{len(generators)-1}): ").strip())
        if idx < 0 or idx >= len(generators):
            print("   Invalid index.")
            return
    except ValueError:
        print("   Invalid number.")
        return

    g = generators[idx]
    print(f"\n   Editing generator #{idx}: {g.type}")
    print(f"   Current params: {json.dumps(g.params)}")
    print(f"   Current sizes:  {g.sizes}")
    print(f"   Current count:  {g.count_per_size}")
    print()

    # Modify params
    new_params = input("   New params (JSON, or Enter to keep): ").strip()
    if new_params:
        try:
            g.params = json.loads(new_params)
        except json.JSONDecodeError:
            print("   ⚠️  Invalid JSON — keeping existing params.")

    # Modify sizes
    new_sizes = input("   New sizes (comma-separated, or Enter to keep): ").strip()
    if new_sizes:
        try:
            g.sizes = [int(s.strip()) for s in new_sizes.split(",")]
        except ValueError:
            print("   ⚠️  Invalid sizes — keeping existing.")

    # Modify count
    new_count = input("   New count_per_size (or Enter to keep): ").strip()
    if new_count:
        try:
            g.count_per_size = int(new_count)
        except ValueError:
            print("   ⚠️  Invalid count — keeping existing.")

    print(f"   ✅ Generator #{idx} updated.")


def _add_generator(generators: list):
    """Add a new generator to the config."""
    from benchwarmer.config import GeneratorConfig
    from benchwarmer.generators import list_generators

    available = list_generators()
    print(f"\n   Available generators: {', '.join(available)}")

    gen_type = input("   Generator type: ").strip()
    if gen_type not in available:
        print(f"   ⚠️  Unknown generator '{gen_type}'. Available: {', '.join(available)}")
        return

    params_str = input("   Params (JSON, e.g. {\"p\": 0.3}): ").strip() or "{}"
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        print("   ⚠️  Invalid JSON — using empty params.")
        params = {}

    sizes_str = input("   Sizes (comma-separated, e.g. 50,100,200): ").strip() or "50,100,200"
    try:
        sizes = [int(s.strip()) for s in sizes_str.split(",")]
    except ValueError:
        print("   ⚠️  Invalid sizes — using [50, 100, 200].")
        sizes = [50, 100, 200]

    count_str = input("   Count per size (default 3): ").strip() or "3"
    try:
        count = int(count_str)
    except ValueError:
        count = 3

    new_gen = GeneratorConfig(type=gen_type, params=params, sizes=sizes, count_per_size=count)
    generators.append(new_gen)
    print(f"   ✅ Added generator: {gen_type}")


def _remove_generator(generators: list):
    """Remove a generator from the config."""
    if not generators:
        print("   No generators to remove.")
        return

    try:
        idx = int(input(f"   Which generator # to remove? (0-{len(generators)-1}): ").strip())
        if idx < 0 or idx >= len(generators):
            print("   Invalid index.")
            return
    except ValueError:
        print("   Invalid number.")
        return

    removed = generators.pop(idx)
    print(f"   ✅ Removed generator: {removed.type}")

    if not generators:
        print("   ⚠️  No generators left! Add at least one, or switch to custom instances.")


def _custom_flow(config):
    """Load custom instances from a JSON file."""
    from benchwarmer.utils.instance_loader import load_instances

    print("\n📂 Custom Instance Loading")
    print("   Your JSON file should contain one or more graph instances.")
    print("   Each instance needs 'nodes' and 'edges' keys.")
    print("   Example format:")
    print('   {"nodes": [0, 1, 2], "edges": [{"source": 0, "target": 1}]}')
    print()

    all_instances = []
    while True:
        path = input("   Path to JSON file (or 'done' to finish): ").strip()

        if path.lower() in ("done", "d", ""):
            if all_instances:
                break
            else:
                print("   ⚠️  No instances loaded yet. Provide at least one file.")
                continue

        try:
            instances = load_instances(path)
            all_instances.extend(instances)
            print(f"   ✅ Loaded {len(instances)} instance(s) from {path}")
            for inst in instances:
                n_nodes = len(inst.get("nodes", []))
                n_edges = len(inst.get("edges", []))
                name = inst.get("instance_name", "?")
                print(f"      • {name}: {n_nodes} nodes, {n_edges} edges")
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"   ❌ Error: {e}")

    # Clear generators (custom-only mode) and inject custom instances
    config.instance_config.generators = []
    config.instance_config.custom_instances = all_instances

    print(f"\n✅ {len(all_instances)} custom instance(s) loaded!")
    return config

# ──────────────────────────────────────────────────────────────
# Algorithm registration (Implementation Agent + built-in baselines)
# ──────────────────────────────────────────────────────────────

def _algorithm_registration(config) -> list:
    """Interactive loop to register algorithms — built-in and LLM-generated."""
    algorithms = _get_builtin_algorithms(config.problem_class)

    print()
    print("=" * 60)
    print("  🧠 Algorithm Registration")
    print("=" * 60)
    print()

    if algorithms:
        print(f"Built-in baselines for '{config.problem_class}':")
        for algo in algorithms:
            print(f"   • {algo.name}")
        print()

    print("You can now describe additional algorithms in natural language.")
    print("The AI will implement them as working code and register them.")
    print()
    print("Examples:")
    if config.problem_class == "maximum_cut":
        print('  • "local search that flips nodes improving the cut"')
        print('  • "simulated annealing with temperature cooling"')
    elif config.problem_class == "minimum_vertex_cover":
        print('  • "2-approximation: pick both endpoints of each uncovered edge"')
        print('  • "LP relaxation rounding for vertex cover"')
    else:
        print('  • "greedy heuristic"')
        print('  • "randomized local search"')
    print()
    print("Commands:")
    print("  run       — start the benchmark with current algorithms")
    print("  list      — show registered algorithms")
    print("  remove #  — remove algorithm by number")
    print()

    from benchwarmer.agents.implementation import ImplementationAgent
    impl_agent = ImplementationAgent()

    while True:
        try:
            user_input = input("🧠 Describe algorithm (or 'run'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("run", "start", "go", "benchmark"):
            if not algorithms:
                print("   ⚠️  No algorithms registered yet. Describe at least one.")
                continue
            break

        if lower in ("list", "ls", "show"):
            if algorithms:
                print("\n   Registered algorithms:")
                for i, algo in enumerate(algorithms):
                    print(f"   {i}. {algo.name}")
                print()
            else:
                print("   No algorithms registered yet.\n")
            continue

        if lower.startswith("remove") or lower.startswith("rm"):
            parts = lower.split()
            if len(parts) < 2:
                print("   Usage: remove <number>")
                continue
            try:
                idx = int(parts[1])
                if 0 <= idx < len(algorithms):
                    removed = algorithms.pop(idx)
                    print(f"   ✅ Removed: {removed.name}")
                else:
                    print(f"   Invalid index. Range: 0-{len(algorithms)-1}")
            except ValueError:
                print("   Invalid number.")
            continue

        # Treat as algorithm description → send to Implementation Agent
        print("🤖 Generating algorithm implementation…")
        result = impl_agent.generate(
            description=user_input,
            problem_class=config.problem_class,
        )

        if result["success"]:
            print(f"\n✅ Generated: {result['name']}")
            print("-" * 40)
            print(result["code"])
            print("-" * 40)
            print(f"   Smoke test passed: {result['smoke_result']}")

            algorithms.append(result["algorithm"])
            print(f"\n   Registered! ({len(algorithms)} total algorithms)\n")
        else:
            print(f"\n❌ Failed: {result['error']}")
            if result.get("code"):
                print(f"\n   Generated code:")
                print("-" * 40)
                print(result["code"])
                print("-" * 40)
            if result.get("traceback"):
                print(result["traceback"])
            print("   Try a different description.\n")

    print(f"\n📋 Final algorithm list ({len(algorithms)}):")
    for algo in algorithms:
        print(f"   • {algo.name}")

    return algorithms


# ──────────────────────────────────────────────────────────────
# Built-in baseline algorithms
# ──────────────────────────────────────────────────────────────

def _get_builtin_algorithms(problem_class: str) -> list:
    """Return simple baseline algorithms for the given problem class."""
    from benchwarmer.algorithms.base import AlgorithmWrapper
    import random

    algorithms = []

    if problem_class == "maximum_cut":
        class RandomMaxCut(AlgorithmWrapper):
            name = "random_cut"
            def solve(self, instance: dict, timeout: float = 60.0) -> dict:
                nodes = instance["nodes"]
                partition = [random.choice([0, 1]) for _ in nodes]
                return {"solution": {"partition": partition}, "metadata": {}}

        class GreedyMaxCut(AlgorithmWrapper):
            name = "greedy_cut"
            def solve(self, instance: dict, timeout: float = 60.0) -> dict:
                nodes = instance["nodes"]
                edges = instance["edges"]
                adj = {n: [] for n in nodes}
                for e in edges:
                    adj[e["source"]].append(e["target"])
                    adj[e["target"]].append(e["source"])

                partition = [0] * len(nodes)
                for i, node in enumerate(nodes):
                    count_0, count_1 = 0, 0
                    for neighbor in adj[node]:
                        idx = nodes.index(neighbor)
                        if partition[idx] == 0:
                            count_0 += 1
                        else:
                            count_1 += 1
                    partition[i] = 0 if count_1 >= count_0 else 1
                return {"solution": {"partition": partition}, "metadata": {}}

        algorithms = [GreedyMaxCut(), RandomMaxCut()]

    elif problem_class == "minimum_vertex_cover":
        class GreedyVertexCover(AlgorithmWrapper):
            name = "greedy_cover"
            def solve(self, instance: dict, timeout: float = 60.0) -> dict:
                nodes = instance["nodes"]
                edges = instance["edges"]

                adj: dict[int, set[int]] = {n: set() for n in nodes}
                for e in edges:
                    adj[e["source"]].add(e["target"])
                    adj[e["target"]].add(e["source"])

                cover: set[int] = set()
                degree = {n: len(adj[n]) for n in nodes}

                while True:
                    best_node = max(
                        (n for n in nodes if n not in cover and degree[n] > 0),
                        key=lambda n: degree[n],
                        default=None,
                    )
                    if best_node is None:
                        break

                    cover.add(best_node)
                    for neighbor in adj[best_node]:
                        if neighbor not in cover:
                            degree[neighbor] -= 1
                    degree[best_node] = 0

                return {"solution": {"vertex_cover": sorted(cover)}, "metadata": {}}

        class RandomVertexCover(AlgorithmWrapper):
            name = "random_cover"
            def solve(self, instance: dict, timeout: float = 60.0) -> dict:
                edges = instance["edges"]
                cover: set[int] = set()
                for e in edges:
                    u, v = e["source"], e["target"]
                    if u not in cover and v not in cover:
                        cover.add(random.choice([u, v]))
                return {"solution": {"vertex_cover": sorted(cover)}, "metadata": {}}

        algorithms = [GreedyVertexCover(), RandomVertexCover()]

    return algorithms


# ──────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────

def _print_summary(df) -> None:
    """Print a concise summary table of benchmark results."""
    import pandas as pd

    success_df = df[df["status"] == "success"]
    if success_df.empty:
        print("⚠️  No successful runs to summarize.")
        return

    summary = (
        success_df
        .groupby("algorithm_name")
        .agg(
            avg_objective=("objective_value", "mean"),
            std_objective=("objective_value", "std"),
            avg_time=("wall_time_seconds", "mean"),
            avg_memory=("peak_memory_mb", "mean"),
            success_rate=("status", "count"),
        )
    )
    total_runs = len(df.groupby("algorithm_name").size())
    summary["success_rate"] = (
        summary["success_rate"] / len(df) * len(summary) * 100
    ).round(1)

    print("\n📊 Summary Table:")
    print(summary.to_string())
    print()


# ──────────────────────────────────────────────────────────────
# Interactive analysis loop
# ──────────────────────────────────────────────────────────────

def _analysis_loop(df, config) -> None:
    """Interactive visualization & analysis with the Plot Agent."""
    from benchwarmer.agents.plot import PlotAgent

    print("=" * 60)
    print("  📈 Interactive Analysis — powered by Plot Agent")
    print("=" * 60)
    print()
    print("Ask for visualizations or analysis in natural language.")
    print("Examples:")
    print('  • "Show a bar chart comparing average objective by algorithm"')
    print('  • "Box plot of wall time by algorithm for each graph size"')
    print('  • "Summary table of results"')
    print('  • "How does objective scale with problem size?"')
    print()
    print("Commands:")
    print("  exit      — quit the analysis loop")
    print("  export    — save the results DataFrame as CSV")
    print()

    plot_agent = PlotAgent()
    plot_agent.set_dataframe(df)
    plot_index = 0

    while True:
        try:
            request = input("📊 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not request:
            continue

        if request.lower() in ("exit", "quit", "q", "done", "stop", "bye"):
            print("👋 Goodbye!")
            break

        # Detect natural-language exit intent
        exit_phrases = [
            "that's enough", "thats enough", "that is enough",
            "no more", "i'm done", "im done", "all done",
            "thank you", "thanks", "thank", "enough",
            "nothing else", "no thanks", "stop",
        ]
        if any(phrase in request.lower() for phrase in exit_phrases):
            print("👋 Thanks for using Benchwarmer.AI! Goodbye!")
            break

        if request.lower() in ("export", "export csv", "export raw"):
            path = "benchmark_results.csv"
            df.to_csv(path, index=False)
            print(f"📁 Exported to {path}")
            continue

        # Generate and execute the visualization
        print("🤖 Generating visualization…")
        result = plot_agent.generate_and_execute(
            user_request=request,
            df=df,
            output_dir="plots",
            plot_index=plot_index,
        )

        if result.get("code"):
            print(f"\n💻 Generated code:")
            print("-" * 40)
            print(result["code"])
            print("-" * 40)

        if result["success"]:
            if result.get("output_path") and os.path.exists(result["output_path"]):
                print(f"✅ Plot saved to: {result['output_path']}")
                plot_index += 1
            elif result.get("message"):
                print(f"\n{result['message']}")
            if result.get("stdout"):
                print(result["stdout"])
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
            if result.get("traceback"):
                print(result["traceback"])

        print()


if __name__ == "__main__":
    main()
